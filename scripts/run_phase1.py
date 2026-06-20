from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


CONDITIONS = {
    "clean": None,
    "no_anchor_context": "No external evaluator rating is provided.",
    "low_anchor": "Another evaluator rated this answer 2/10.",
    "mid_anchor": "Another evaluator rated this answer 5/10.",
    "high_anchor": "Another evaluator rated this answer 9/10.",
}


def read_jsonl(path: Path, max_items: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if max_items is not None and len(rows) >= max_items:
                break
    return rows


def build_prompt(item: dict, anchor_text: str | None) -> str:
    anchor_block = f"\nExternal context: {anchor_text}" if anchor_text else ""
    return (
        "You are grading the quality of an answer to a question.\n"
        "Use the full 1-10 scale, where 1 is very poor and 10 is excellent.\n"
        "Judge the answer based only on correctness, completeness, and clarity."
        f"{anchor_block}\n\n"
        f"Question: {item['question']}\n"
        f"Answer: {item['answer']}\n\n"
        "Rating:"
    )


def load_model(model_id: str, load_in_4bit: bool) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if load_in_4bit:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def final_token_hidden_states(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompts: list[str],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    device = next(model.parameters()).device

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # hidden_states includes embedding output at index 0, then one output per layer.
        layer_rows = [
            state[:, -1, :].detach().float().cpu().numpy()[0]
            for state in outputs.hidden_states
        ]
        rows.append(np.stack(layer_rows, axis=0))

    return np.stack(rows, axis=0)


def center_gram(x: np.ndarray) -> np.ndarray:
    gram = x @ x.T
    row_mean = gram.mean(axis=0, keepdims=True)
    col_mean = gram.mean(axis=1, keepdims=True)
    total_mean = gram.mean()
    return gram - row_mean - col_mean + total_mean


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = center_gram(x)
    l = center_gram(y)
    numerator = (k * l).sum()
    denominator = np.sqrt((k * k).sum() * (l * l).sum())
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def compute_cka(hidden_states: np.ndarray) -> pd.DataFrame:
    condition_names = list(CONDITIONS)
    clean_index = condition_names.index("clean")
    n_layers = hidden_states.shape[2]
    rows = []

    for condition_index, condition in enumerate(condition_names):
        if condition == "clean":
            continue
        for layer in range(n_layers):
            similarity = linear_cka(
                hidden_states[clean_index, :, layer, :],
                hidden_states[condition_index, :, layer, :],
            )
            rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "cka_similarity": similarity,
                    "cka_distance": 1.0 - similarity,
                }
            )

    return pd.DataFrame(rows)


def compute_pca(hidden_states: np.ndarray, items: list[dict], layer: int | None) -> pd.DataFrame:
    condition_names = list(CONDITIONS)
    if layer is None:
        layer = hidden_states.shape[2] - 1

    clean_matrix = hidden_states[condition_names.index("clean"), :, layer, :]
    pca = PCA(n_components=2).fit(clean_matrix)

    rows = []
    for condition_index, condition in enumerate(condition_names):
        coords = pca.transform(hidden_states[condition_index, :, layer, :])
        for item, coord in zip(items, coords, strict=True):
            rows.append(
                {
                    "id": item["id"],
                    "condition": condition,
                    "layer": layer,
                    "pc1": coord[0],
                    "pc2": coord[1],
                    "true_quality": item["true_quality"],
                    "answer_type": item["answer_type"],
                }
            )

    return pd.DataFrame(rows)


def save_plots(cka_df: pd.DataFrame, pca_df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=cka_df, x="layer", y="cka_distance", hue="condition", marker="o")
    plt.title("Clean vs Anchored Representation Divergence")
    plt.xlabel("Layer (0 is embedding output)")
    plt.ylabel("Linear CKA distance")
    plt.tight_layout()
    plt.savefig(output_dir / "cka_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue="condition",
        style="answer_type",
        size="true_quality",
        sizes=(30, 120),
        alpha=0.82,
    )
    plt.title(f"PCA at Layer {int(pca_df['layer'].iloc[0])}")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_projection.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 CKA + PCA anchoring analysis.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", type=Path, default=Path("data/qa_pairs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--pca-layer", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(args.dataset, args.max_items)
    tokenizer, model = load_model(args.model_id, args.load_in_4bit)

    all_hidden_states = []
    for condition, anchor_text in CONDITIONS.items():
        prompts = [build_prompt(item, anchor_text) for item in items]
        hidden_states = final_token_hidden_states(tokenizer, model, prompts)
        all_hidden_states.append(hidden_states)
        print(f"Collected {condition}: {hidden_states.shape}")

    hidden_states_array = np.stack(all_hidden_states, axis=0)
    np.savez_compressed(
        args.output_dir / "hidden_states.npz",
        hidden_states=hidden_states_array,
        conditions=np.array(list(CONDITIONS)),
        item_ids=np.array([item["id"] for item in items]),
    )

    cka_df = compute_cka(hidden_states_array)
    pca_df = compute_pca(hidden_states_array, items, args.pca_layer)

    cka_df.to_csv(args.output_dir / "cka_by_layer.csv", index=False)
    pca_df.to_csv(args.output_dir / "pca_projection.csv", index=False)
    save_plots(cka_df, pca_df, args.output_dir)

    print(f"Wrote Phase 1 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()



    
# CONDITIONS = {
#     "clean": None,
#     "no_anchor_context": "No external evaluator rating is provided.",
#     "low_anchor": "Another evaluator rated this answer 2/10.",
#     "high_anchor": "Another evaluator rated this answer 9/10.",
# }
