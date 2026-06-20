from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_phase1_npz(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    hidden_states = data["hidden_states"]
    conditions = [str(condition) for condition in data["conditions"]]
    item_ids = [str(item_id) for item_id in data["item_ids"]]
    return hidden_states, conditions, item_ids


def anchor_conditions(conditions: list[str]) -> list[str]:
    return [
        condition
        for condition in conditions
        if condition.endswith("_anchor") and condition != "no_anchor_context"
    ]


def build_stratified_folds(y: np.ndarray, cv_splits: int) -> list[np.ndarray]:
    min_class_count = int(np.bincount(y).min())
    usable_splits = min(cv_splits, min_class_count)
    if usable_splits < 2:
        return []

    rng = np.random.default_rng(0)
    folds = [[] for _ in range(usable_splits)]
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        for fold_index, split in enumerate(np.array_split(indices, usable_splits)):
            folds[fold_index].extend(split.tolist())

    return [np.array(sorted(fold), dtype=int) for fold in folds]


def fit_mean_difference_probe(x: np.ndarray, y: np.ndarray) -> dict:
    clean = x[y == 0]
    anchored = x[y == 1]
    direction = normalize(anchored.mean(axis=0) - clean.mean(axis=0))
    clean_scores = clean @ direction
    anchor_scores = anchored @ direction
    threshold = float((clean_scores.mean() + anchor_scores.mean()) / 2.0)
    return {
        "coefficient": direction,
        "intercept": -threshold,
    }


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[y_true == 1]
    negative = scores[y_true == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")

    wins = 0.0
    for positive_score in positive:
        wins += float(np.sum(positive_score > negative))
        wins += 0.5 * float(np.sum(positive_score == negative))
    return wins / float(len(positive) * len(negative))


def fit_probe(x: np.ndarray, y: np.ndarray, cv_splits: int) -> dict:
    folds = build_stratified_folds(y, cv_splits)

    if folds:
        predictions = np.zeros_like(y)
        scores = np.zeros(len(y), dtype=float)
        all_indices = np.arange(len(y))
        for test_indices in folds:
            train_indices = np.setdiff1d(all_indices, test_indices)
            probe = fit_mean_difference_probe(x[train_indices], y[train_indices])
            fold_scores = x[test_indices] @ probe["coefficient"] + probe["intercept"]
            scores[test_indices] = fold_scores
            predictions[test_indices] = (fold_scores >= 0).astype(int)

        accuracy = float(np.mean(predictions == y))
        auc = auc_score(y, scores)
    else:
        accuracy = float("nan")
        auc = float("nan")

    probe = fit_mean_difference_probe(x, y)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "coefficient": probe["coefficient"],
        "intercept": probe["intercept"],
    }


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def compute_phase2(
    hidden_states: np.ndarray,
    conditions: list[str],
    item_ids: list[str],
    cv_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    clean_index = conditions.index("clean")
    anchors = anchor_conditions(conditions)
    if not anchors:
        raise ValueError("No anchor conditions found. Expected names like low_anchor or high_anchor.")

    anchor_indices = [conditions.index(condition) for condition in anchors]
    n_items = hidden_states.shape[1]
    n_layers = hidden_states.shape[2]

    probe_rows = []
    projection_rows = []
    probe_directions = []
    mean_anchor_directions = []

    for layer in range(n_layers):
        clean_x = hidden_states[clean_index, :, layer, :]
        anchor_x = np.concatenate(
            [hidden_states[condition_index, :, layer, :] for condition_index in anchor_indices],
            axis=0,
        )
        x = np.concatenate([clean_x, anchor_x], axis=0)
        y = np.concatenate(
            [np.zeros(n_items, dtype=int), np.ones(n_items * len(anchor_indices), dtype=int)]
        )

        probe_info = fit_probe(x, y, cv_splits)
        probe_direction = normalize(probe_info["coefficient"])
        mean_anchor_direction = normalize(anchor_x.mean(axis=0) - clean_x.mean(axis=0))
        probe_directions.append(probe_direction)
        mean_anchor_directions.append(mean_anchor_direction)

        clean_center = clean_x.mean(axis=0)
        condition_means = {}
        for condition_index, condition in enumerate(conditions):
            values = hidden_states[condition_index, :, layer, :]
            projections = (values - clean_center) @ probe_direction
            condition_means[condition] = float(np.mean(projections))

            for item_id, projection in zip(item_ids, projections, strict=True):
                projection_rows.append(
                    {
                        "layer": layer,
                        "condition": condition,
                        "item_id": item_id,
                        "anchor_score": float(projection),
                    }
                )

        probe_rows.append(
            {
                "layer": layer,
                "cv_accuracy": probe_info["accuracy"],
                "cv_auc": probe_info["auc"],
                "probe_norm": float(np.linalg.norm(probe_info["coefficient"])),
                "probe_intercept": probe_info["intercept"],
                "mean_clean_projection": condition_means.get("clean", float("nan")),
                "mean_no_anchor_projection": condition_means.get("no_anchor_context", float("nan")),
                "mean_low_anchor_projection": condition_means.get("low_anchor", float("nan")),
                "mean_mid_anchor_projection": condition_means.get("mid_anchor", float("nan")),
                "mean_high_anchor_projection": condition_means.get("high_anchor", float("nan")),
            }
        )

    return (
        pd.DataFrame(probe_rows),
        pd.DataFrame(projection_rows),
        {
            "probe_direction": np.stack(probe_directions, axis=0),
            "mean_anchor_direction": np.stack(mean_anchor_directions, axis=0),
        },
    )


def summarize_best_layers(probe_df: pd.DataFrame, top_k: int) -> list[dict]:
    ranking = probe_df.sort_values(["cv_auc", "cv_accuracy"], ascending=False, na_position="last")
    rows = []
    for row in ranking.head(top_k).to_dict(orient="records"):
        rows.append(
            {
                "layer": int(row["layer"]),
                "cv_accuracy": None if pd.isna(row["cv_accuracy"]) else float(row["cv_accuracy"]),
                "cv_auc": None if pd.isna(row["cv_auc"]) else float(row["cv_auc"]),
                "mean_low_anchor_projection": None
                if pd.isna(row["mean_low_anchor_projection"])
                else float(row["mean_low_anchor_projection"]),
                "mean_mid_anchor_projection": None
                if pd.isna(row["mean_mid_anchor_projection"])
                else float(row["mean_mid_anchor_projection"]),
                "mean_high_anchor_projection": None
                if pd.isna(row["mean_high_anchor_projection"])
                else float(row["mean_high_anchor_projection"]),
            }
        )
    return rows


def save_plots(probe_df: pd.DataFrame, projection_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as error:
        print(f"Skipping plots because plotting dependency is missing: {error.name}")
        return

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=probe_df, x="layer", y="cv_auc", marker="o")
    plt.ylim(0.0, 1.05)
    plt.title("Anchor Probe Separability by Layer")
    plt.xlabel("Layer (0 is embedding output)")
    plt.ylabel("Cross-validated AUC")
    plt.tight_layout()
    plt.savefig(output_dir / "probe_auc_by_layer.png", dpi=180)
    plt.close()

    mean_projection = (
        projection_df.groupby(["layer", "condition"], as_index=False)["anchor_score"].mean()
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=mean_projection, x="layer", y="anchor_score", hue="condition", marker="o")
    plt.title("Projection onto Learned Anchor Direction")
    plt.xlabel("Layer (0 is embedding output)")
    plt.ylabel("Mean anchor score")
    plt.tight_layout()
    plt.savefig(output_dir / "anchor_projection_by_layer.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 representation engineering.")
    parser.add_argument(
        "--phase1-hidden-states",
        type=Path,
        default=Path("outputs/phase1/hidden_states.npz"),
        help="Path to Phase 1 hidden_states.npz.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"))
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hidden_states, conditions, item_ids = load_phase1_npz(args.phase1_hidden_states)
    probe_df, projection_df, directions = compute_phase2(
        hidden_states=hidden_states,
        conditions=conditions,
        item_ids=item_ids,
        cv_splits=args.cv_splits,
    )

    probe_df.to_csv(args.output_dir / "probe_metrics_by_layer.csv", index=False)
    projection_df.to_csv(args.output_dir / "anchor_scores_by_item.csv", index=False)
    np.savez_compressed(
        args.output_dir / "anchor_directions.npz",
        probe_direction=directions["probe_direction"],
        mean_anchor_direction=directions["mean_anchor_direction"],
        conditions=np.array(conditions),
    )

    summary = {
        "source": str(args.phase1_hidden_states),
        "conditions": conditions,
        "best_layers": summarize_best_layers(probe_df, args.top_k),
        "notes": [
            "cv_auc close to 1.0 means anchored and clean activations are easy to separate.",
            "anchor_score is projection onto the learned anchor direction after centering on clean activations.",
            "Use high-scoring layers as candidates for Phase 3 activation patching.",
        ],
    }
    with (args.output_dir / "phase2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_plots(probe_df, projection_df, args.output_dir)
    print(f"Wrote Phase 2 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
