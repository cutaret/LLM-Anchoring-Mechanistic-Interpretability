# LLM Anchoring Mechanistic Interpretability

This repository contains the first two stages of an anchoring mechanistic interpretability workflow:

- **Phase 1: correlational geometry** with CKA and PCA.
- **Phase 2: representation engineering** with linear probes and anchor-direction projections.
- **Phase 3: activation patching** to test whether clean activations causally undo anchor influence.

The first experiment compares model representations across four prompt conditions:

- `clean`: the model rates an answer with no external-context line at all.
- `no_anchor_context`: the model is explicitly told that no external evaluator rating is provided.
- `low_anchor`: the same prompt includes `Another evaluator rated this answer 2/10`.
- `mid_anchor`: the same prompt includes `Another evaluator rated this answer 5/10`.
- `high_anchor`: the same prompt includes `Another evaluator rated this answer 9/10`.

For each condition, the script extracts per-layer hidden states at the final prompt token, then computes:

- **Linear CKA distance by layer**, showing where clean and anchored representations diverge.
- **PCA projections**, showing whether anchored activations shift along a quality-related direction.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For a GPU with limited VRAM, use `--load-in-4bit` after installing `bitsandbytes`. On Windows this may be easier in WSL or Colab.

## Run Phase 1

```powershell
python scripts/run_phase1.py `
  --model-id Qwen/Qwen2.5-3B-Instruct `
  --dataset data/qa_pairs.jsonl `
  --output-dir outputs/phase1 `
  --max-items 30
```

Useful smaller smoke test:

```powershell
python scripts/run_phase1.py `
  --model-id Qwen/Qwen2.5-0.5B-Instruct `
  --dataset data/qa_pairs.jsonl `
  --output-dir outputs/smoke `
  --max-items 3
```

## Phase 1 Outputs

The Phase 1 script writes:

- `hidden_states.npz`: hidden states with shape `[condition, item, layer, hidden_dim]`.
- `hidden_states_by_layer.json`: the same activations as inspectable JSON, grouped by condition, item, and layer.
- `cka_by_layer.csv`: CKA similarity and distance curves comparing `clean` to `no_anchor_context`, `low_anchor`, `mid_anchor`, and `high_anchor`.
- `pca_projection.csv`: 2D PCA coordinates for all examples and conditions.
- `cka_distance.png`: layer divergence plot.
- `pca_projection.png`: PCA scatter plot.

The JSON file is intentionally human-inspectable, but it can become large for the 3B model. Use `--json-precision` to control rounding:

```powershell
python scripts/run_phase1.py `
  --model-id Qwen/Qwen2.5-0.5B-Instruct `
  --dataset data/qa_pairs.jsonl `
  --output-dir outputs/smoke `
  --max-items 3 `
  --json-precision 4
```

## Run Phase 2

Phase 2 uses the saved Phase 1 activations. It does not need to reload the LLM.

```powershell
python scripts/run_phase2.py `
  --phase1-hidden-states outputs/phase1/hidden_states.npz `
  --output-dir outputs/phase2
```

Smoke-test version using the small Phase 1 output:

```powershell
python scripts/run_phase2.py `
  --phase1-hidden-states outputs/smoke/hidden_states.npz `
  --output-dir outputs/smoke_phase2
```

## Phase 2 Outputs

The Phase 2 script writes:

- `probe_metrics_by_layer.csv`: how well a linear probe separates clean vs anchored activations at each layer.
- `anchor_scores_by_item.csv`: each item projected onto the learned anchor direction at each layer.
- `anchor_directions.npz`: learned anchor-direction vectors for each layer.
- `phase2_summary.json`: top candidate layers for Phase 3 activation patching.
- `probe_auc_by_layer.png`: where anchor information is easiest to detect.
- `anchor_projection_by_layer.png`: how strongly each condition points in the anchor direction.

## Run Phase 3

Phase 3 reruns the model and performs activation patching. For each anchored prompt, it replaces one layer's anchored hidden state with the clean hidden state from the same question-answer pair, then checks whether the predicted rating moves back toward the clean rating.

Patch all transformer block layers:

```powershell
python scripts/run_phase3.py `
  --model-id Qwen/Qwen2.5-3B-Instruct `
  --dataset data/qa_pairs.jsonl `
  --output-dir outputs/phase3 `
  --max-items 30 `
  --load-in-4bit
```

Patch only selected layers:

```powershell
python scripts/run_phase3.py `
  --model-id Qwen/Qwen2.5-3B-Instruct `
  --dataset data/qa_pairs.jsonl `
  --output-dir outputs/phase3_selected `
  --layers 8,10-16,24 `
  --max-items 30 `
  --load-in-4bit
```

Use the best layers from Phase 2:

```powershell
python scripts/run_phase3.py `
  --model-id Qwen/Qwen2.5-3B-Instruct `
  --dataset data/qa_pairs.jsonl `
  --output-dir outputs/phase3_from_phase2 `
  --phase2-summary outputs/phase2/phase2_summary.json `
  --phase2-top-k 8 `
  --max-items 30 `
  --load-in-4bit
```

## Phase 3 Outputs

The Phase 3 script writes:

- `patching_results.csv`: item-level clean, anchored, and patched expected ratings.
- `patching_summary_by_layer.csv`: average restoration by condition and patched layer.
- `phase3_summary.json`: best causal patch layers and run metadata.
- `score_candidates.csv`: the exact rating candidate strings and token IDs used for scoring `1` through `10`, including multi-token ratings like `10`.
- `patching_restoration_by_layer.png`: where clean activations most restore clean behavior, if plotting dependencies are installed.
- `patch_success_rate_by_layer.png`: fraction of examples moved toward the clean rating, if plotting dependencies are installed.

Use `--save-probabilities` to also save `rating_probabilities.csv`, which contains the full 1-10 rating distribution for clean, anchored, and patched runs.

## Interpretation

Look for a layer range where:

1. `clean` vs `low_anchor` or `clean` vs `high_anchor` CKA distance increases more than `clean` vs `no_anchor_context`.
2. Anchored examples shift coherently in PCA space relative to the clean and no-anchor examples.
3. Phase 2 probe AUC is high, meaning a simple linear boundary can tell clean and anchored activations apart.
4. Phase 2 anchor scores rise in order across `low_anchor`, `mid_anchor`, and `high_anchor`, or show one anchor value as especially dominant.
5. Phase 3 directional restoration is high, meaning clean activations at that layer causally move anchored predictions back toward clean predictions.

Together, these results identify the strongest candidate layers where the anchor is not just represented, but can causally affect the model's rating behavior.
