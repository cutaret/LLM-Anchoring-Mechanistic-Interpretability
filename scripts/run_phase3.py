from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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
        "Use the full 1-9 scale, where 1 is very poor and 9 is excellent.\n"
        "Judge the answer based only on correctness, completeness, and clarity."
        f"{anchor_block}\n\n"
        f"Question: {item['question']}\n"
        f"Answer: {item['answer']}\n\n"
        "Rating:"
    )


def load_model(model_id: str, load_in_4bit: bool) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def get_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def get_transformer_layers(model: Any) -> torch.nn.ModuleList:
    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.decoder.layers", lambda m: m.model.decoder.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ("layers", lambda m: m.layers),
    ]
    for _, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList) or isinstance(layers, list):
            return layers
    raise ValueError("Could not find transformer layers for this model architecture.")


def parse_layer_list(layer_text: str | None, n_layers: int) -> list[int] | None:
    if not layer_text:
        return None

    layers: set[int] = set()
    for part in layer_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    valid_layers = sorted(layer for layer in layers if 1 <= layer <= n_layers)
    if not valid_layers:
        raise ValueError(f"No valid patch layers found. Use layer numbers from 1 to {n_layers}.")
    return valid_layers


def layers_from_phase2_summary(path: Path, top_k: int, n_layers: int) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    layers = []
    for row in summary.get("best_layers", [])[:top_k]:
        layer = int(row["layer"])
        if 1 <= layer <= n_layers:
            layers.append(layer)

    if not layers:
        raise ValueError(f"No valid layers found in {path}.")
    return sorted(set(layers))


def choose_layers(
    requested_layers: str | None,
    phase2_summary: Path | None,
    phase2_top_k: int,
    n_layers: int,
) -> list[int]:
    parsed = parse_layer_list(requested_layers, n_layers)
    if parsed is not None:
        return parsed
    if phase2_summary is not None:
        return layers_from_phase2_summary(phase2_summary, phase2_top_k, n_layers)
    return list(range(1, n_layers + 1))


def score_token_ids(tokenizer: Any) -> pd.DataFrame:
    rows = []
    used_token_ids: set[int] = set()
    for score in range(1, 10):
        chosen_text = None
        chosen_ids = None
        for text in (f" {score}", str(score)):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                chosen_text = text
                chosen_ids = token_ids
                break

        if chosen_ids is None:
            raise ValueError(
                f"Rating {score} is not a single token for this tokenizer. "
                "This Phase 3 script currently expects ratings 1-9 to each be one token."
            )

        token_id = int(chosen_ids[0])
        if token_id in used_token_ids:
            raise ValueError(
                "Two rating labels mapped to the same token id. "
                "Try a different prompt format or tokenizer."
            )
        used_token_ids.add(token_id)
        rows.append(
            {
                "score": score,
                "token_text": chosen_text,
                "token_id": token_id,
                "decoded": tokenizer.decode([token_id]),
            }
        )

    return pd.DataFrame(rows)


def rating_distribution_from_logits(
    logits: torch.Tensor,
    score_tokens: pd.DataFrame,
) -> dict:
    token_ids = torch.tensor(score_tokens["token_id"].to_numpy(), device=logits.device)
    score_values = torch.tensor(score_tokens["score"].to_numpy(), device=logits.device, dtype=torch.float32)
    score_logits = logits[0, token_ids].float()
    probabilities = torch.softmax(score_logits, dim=-1)
    expected_rating = float(torch.sum(probabilities * score_values).detach().cpu())
    probability_values = probabilities.detach().cpu().numpy()
    top_index = int(np.argmax(probability_values))

    return {
        "expected_rating": expected_rating,
        "top_rating": int(score_tokens.iloc[top_index]["score"]),
        "top_probability": float(probability_values[top_index]),
        "probabilities": {
            str(int(score)): float(probability)
            for score, probability in zip(score_tokens["score"], probability_values, strict=True)
        },
    }


@torch.inference_mode()
def run_unpatched(
    tokenizer: Any,
    model: Any,
    prompt: str,
    score_tokens: pd.DataFrame,
    capture_hidden_states: bool,
) -> tuple[dict, list[torch.Tensor] | None]:
    inputs = tokenizer(prompt, return_tensors="pt").to(get_input_device(model))
    outputs = model(
        **inputs,
        output_hidden_states=capture_hidden_states,
        use_cache=False,
    )
    distribution = rating_distribution_from_logits(outputs.logits[:, -1, :], score_tokens)
    hidden_states = None
    if capture_hidden_states:
        hidden_states = [
            state[:, -1, :].detach()
            for state in outputs.hidden_states
        ]
    return distribution, hidden_states


def patch_layer_output(output: Any, clean_state: torch.Tensor, position: int) -> Any:
    if isinstance(output, tuple):
        hidden = output[0].clone()
        replacement = clean_state.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, position, :] = replacement
        return (hidden,) + output[1:]

    hidden = output.clone()
    replacement = clean_state.to(device=hidden.device, dtype=hidden.dtype)
    hidden[:, position, :] = replacement
    return hidden


@torch.inference_mode()
def run_patched(
    tokenizer: Any,
    model: Any,
    layer_modules: torch.nn.ModuleList,
    prompt: str,
    score_tokens: pd.DataFrame,
    phase_layer: int,
    clean_hidden_states: list[torch.Tensor],
) -> dict:
    if phase_layer <= 0:
        raise ValueError("Phase 3 patches transformer block outputs, so phase_layer must be >= 1.")

    inputs = tokenizer(prompt, return_tensors="pt").to(get_input_device(model))
    patch_position = int(inputs["input_ids"].shape[1] - 1)
    clean_state = clean_hidden_states[phase_layer]

    def hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> Any:
        return patch_layer_output(output, clean_state, patch_position)

    handle = layer_modules[phase_layer - 1].register_forward_hook(hook)
    try:
        outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()

    return rating_distribution_from_logits(outputs.logits[:, -1, :], score_tokens)


def restoration_metrics(
    clean_expected: float,
    anchored_expected: float,
    patched_expected: float,
    min_anchor_shift: float,
) -> dict:
    anchor_shift = anchored_expected - clean_expected
    patched_shift = patched_expected - clean_expected
    if abs(anchor_shift) < min_anchor_shift:
        directional = float("nan")
        absolute = float("nan")
    else:
        directional = (anchored_expected - patched_expected) / anchor_shift
        absolute = (abs(anchor_shift) - abs(patched_shift)) / abs(anchor_shift)

    return {
        "anchor_shift": anchor_shift,
        "patched_shift": patched_shift,
        "directional_restoration": directional,
        "absolute_restoration": absolute,
        "patched_moves_toward_clean": abs(patched_shift) < abs(anchor_shift),
    }


def probability_rows(
    item_id: str,
    condition: str,
    layer: int | None,
    run_type: str,
    distribution: dict,
) -> list[dict]:
    return [
        {
            "id": item_id,
            "condition": condition,
            "layer": layer,
            "run_type": run_type,
            "score": int(score),
            "probability": probability,
        }
        for score, probability in distribution["probabilities"].items()
    ]


def aggregate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["condition", "layer"], as_index=False)
        .agg(
            mean_clean_expected_rating=("clean_expected_rating", "mean"),
            mean_anchored_expected_rating=("anchored_expected_rating", "mean"),
            mean_patched_expected_rating=("patched_expected_rating", "mean"),
            mean_anchor_shift=("anchor_shift", "mean"),
            mean_patched_shift=("patched_shift", "mean"),
            mean_directional_restoration=("directional_restoration", "mean"),
            mean_absolute_restoration=("absolute_restoration", "mean"),
            patch_success_rate=("patched_moves_toward_clean", "mean"),
        )
        .sort_values(["condition", "layer"])
    )


def save_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as error:
        print(f"Skipping Phase 3 plots because plotting dependency is missing: {error.name}")
        return

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=summary_df,
        x="layer",
        y="mean_directional_restoration",
        hue="condition",
        marker="o",
    )
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    plt.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    plt.title("Activation Patching Restoration by Layer")
    plt.xlabel("Patched layer")
    plt.ylabel("Mean directional restoration")
    plt.tight_layout()
    plt.savefig(output_dir / "patching_restoration_by_layer.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=summary_df,
        x="layer",
        y="patch_success_rate",
        hue="condition",
        marker="o",
    )
    plt.ylim(-0.02, 1.02)
    plt.title("Patch Success Rate by Layer")
    plt.xlabel("Patched layer")
    plt.ylabel("Fraction moved toward clean")
    plt.tight_layout()
    plt.savefig(output_dir / "patch_success_rate_by_layer.png", dpi=180)
    plt.close()


def summarize_best_layers(summary_df: pd.DataFrame, top_k: int) -> list[dict]:
    ranking = summary_df.sort_values(
        ["mean_directional_restoration", "patch_success_rate"],
        ascending=False,
        na_position="last",
    )
    return [
        {
            "condition": row["condition"],
            "layer": int(row["layer"]),
            "mean_directional_restoration": None
            if pd.isna(row["mean_directional_restoration"])
            else float(row["mean_directional_restoration"]),
            "mean_absolute_restoration": None
            if pd.isna(row["mean_absolute_restoration"])
            else float(row["mean_absolute_restoration"]),
            "patch_success_rate": None
            if pd.isna(row["patch_success_rate"])
            else float(row["patch_success_rate"]),
        }
        for row in ranking.head(top_k).to_dict(orient="records")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 activation patching.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", type=Path, default=Path("data/qa_pairs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase3"))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--anchor-conditions",
        nargs="+",
        default=["low_anchor", "mid_anchor", "high_anchor"],
        help="Anchored conditions to patch back toward clean.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma/range layer list, e.g. 8,10-16,24. Defaults to all block-output layers.",
    )
    parser.add_argument(
        "--phase2-summary",
        type=Path,
        default=None,
        help="Optional phase2_summary.json. If provided and --layers is omitted, patch top Phase 2 layers.",
    )
    parser.add_argument("--phase2-top-k", type=int, default=8)
    parser.add_argument("--min-anchor-shift", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-probabilities", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(args.dataset, args.max_items)
    tokenizer, model = load_model(args.model_id, args.load_in_4bit)
    layer_modules = get_transformer_layers(model)
    patch_layers = choose_layers(
        requested_layers=args.layers,
        phase2_summary=args.phase2_summary,
        phase2_top_k=args.phase2_top_k,
        n_layers=len(layer_modules),
    )

    missing_conditions = [condition for condition in args.anchor_conditions if condition not in CONDITIONS]
    if missing_conditions:
        raise ValueError(f"Unknown anchor conditions: {missing_conditions}")

    score_tokens = score_token_ids(tokenizer)
    score_tokens.to_csv(args.output_dir / "score_token_ids.csv", index=False)

    results = []
    probabilities = []
    for item_index, item in enumerate(items, start=1):
        clean_prompt = build_prompt(item, CONDITIONS["clean"])
        clean_distribution, clean_hidden_states = run_unpatched(
            tokenizer=tokenizer,
            model=model,
            prompt=clean_prompt,
            score_tokens=score_tokens,
            capture_hidden_states=True,
        )
        if clean_hidden_states is None:
            raise RuntimeError("Clean hidden states were not captured.")

        if args.save_probabilities:
            probabilities.extend(
                probability_rows(item["id"], "clean", None, "clean", clean_distribution)
            )

        for condition in args.anchor_conditions:
            anchored_prompt = build_prompt(item, CONDITIONS[condition])
            anchored_distribution, _ = run_unpatched(
                tokenizer=tokenizer,
                model=model,
                prompt=anchored_prompt,
                score_tokens=score_tokens,
                capture_hidden_states=False,
            )
            if args.save_probabilities:
                probabilities.extend(
                    probability_rows(item["id"], condition, None, "anchored", anchored_distribution)
                )

            for phase_layer in patch_layers:
                patched_distribution = run_patched(
                    tokenizer=tokenizer,
                    model=model,
                    layer_modules=layer_modules,
                    prompt=anchored_prompt,
                    score_tokens=score_tokens,
                    phase_layer=phase_layer,
                    clean_hidden_states=clean_hidden_states,
                )
                metrics = restoration_metrics(
                    clean_expected=clean_distribution["expected_rating"],
                    anchored_expected=anchored_distribution["expected_rating"],
                    patched_expected=patched_distribution["expected_rating"],
                    min_anchor_shift=args.min_anchor_shift,
                )
                results.append(
                    {
                        "id": item["id"],
                        "condition": condition,
                        "layer": phase_layer,
                        "true_quality": item["true_quality"],
                        "answer_type": item["answer_type"],
                        "clean_expected_rating": clean_distribution["expected_rating"],
                        "anchored_expected_rating": anchored_distribution["expected_rating"],
                        "patched_expected_rating": patched_distribution["expected_rating"],
                        "clean_top_rating": clean_distribution["top_rating"],
                        "anchored_top_rating": anchored_distribution["top_rating"],
                        "patched_top_rating": patched_distribution["top_rating"],
                        "clean_top_probability": clean_distribution["top_probability"],
                        "anchored_top_probability": anchored_distribution["top_probability"],
                        "patched_top_probability": patched_distribution["top_probability"],
                        **metrics,
                    }
                )
                if args.save_probabilities:
                    probabilities.extend(
                        probability_rows(
                            item["id"],
                            condition,
                            phase_layer,
                            "patched",
                            patched_distribution,
                        )
                    )

        print(f"Patched item {item_index}/{len(items)}: {item['id']}")

    results_df = pd.DataFrame(results)
    summary_df = aggregate_summary(results_df)

    results_df.to_csv(args.output_dir / "patching_results.csv", index=False)
    summary_df.to_csv(args.output_dir / "patching_summary_by_layer.csv", index=False)
    if args.save_probabilities:
        pd.DataFrame(probabilities).to_csv(args.output_dir / "rating_probabilities.csv", index=False)

    summary = {
        "model_id": args.model_id,
        "dataset": str(args.dataset),
        "num_items": len(items),
        "anchor_conditions": args.anchor_conditions,
        "patched_layers": patch_layers,
        "phase2_summary": None if args.phase2_summary is None else str(args.phase2_summary),
        "score_tokens": score_tokens.to_dict(orient="records"),
        "best_layers": summarize_best_layers(summary_df, args.top_k),
        "notes": [
            "directional_restoration near 1 means patching that layer moved the anchored prediction back to clean.",
            "directional_restoration near 0 means patching that layer did little.",
            "negative restoration means the patch moved the rating farther from clean.",
            "This is the first causal test: it asks whether clean activations at a layer can undo anchor influence.",
        ],
    }
    with (args.output_dir / "phase3_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_plots(summary_df, args.output_dir)
    print(f"Wrote Phase 3 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
