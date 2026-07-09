#!/usr/bin/env bash
# Train + evaluate + diagnose one SAE experiment into a named results dir.
set -euo pipefail

RUN_NAME="${1:?usage: run_experiment.sh <run_name> <config_yaml>}"
CONFIG="${2:?usage: run_experiment.sh <run_name> <config_yaml>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

RESULTS="sae/results/${RUN_NAME}"
CKPT_DIR="sae/checkpoints/${RUN_NAME}"
mkdir -p "$RESULTS" "$CKPT_DIR"

echo "=== TRAIN ${RUN_NAME} ==="
set +e
python sae/scripts/train_sae.py \
  --config "$CONFIG" \
  --checkpoint-dir "$CKPT_DIR" \
  --results-dir "$RESULTS" \
  --checkpoint-name model.pt \
  --device cuda \
  2>&1 | tee "$RESULTS/train.log"
train_rc=${PIPESTATUS[0]}
set -e
if [[ "$train_rc" -eq 2 ]]; then
  echo "=== EARLY_ABORT ${RUN_NAME} (severe collapse) — skipping eval ==="
  echo "{\"run\":\"${RUN_NAME}\",\"status\":\"early_abort\"}" > "$RESULTS/comparison_row.json"
  echo "=== DONE ${RUN_NAME} (failed) ==="
  exit 0
elif [[ "$train_rc" -ne 0 ]]; then
  echo "=== TRAIN FAILED ${RUN_NAME} rc=${train_rc} ==="
  exit "$train_rc"
fi

echo "=== EVAL ${RUN_NAME} ==="
python sae/scripts/evaluate_sae.py \
  --checkpoint "$CKPT_DIR/model.pt" \
  --data-dir sae/data \
  --basis-matrices PRISM/basis_matrices.pt \
  --run-key PART2_K10_seed42 \
  --split test \
  --results-dir "$RESULTS" \
  --device cuda \
  2>&1 | tee "$RESULTS/eval.log"

echo "=== DIAGNOSE ${RUN_NAME} ==="
python sae/scripts/diagnose_sae.py \
  --checkpoint "$CKPT_DIR/model.pt" \
  --data-dir sae/data \
  --results-dir "$RESULTS" \
  --device cuda \
  2>&1 | tee "$RESULTS/diagnose.log"

python - <<PY
import json, csv
from pathlib import Path
run = "${RUN_NAME}"
results = Path("sae/results") / run
ev = json.loads((results / "sae_eval_summary.json").read_text())
diag = json.loads((results / "sae_diagnostics_summary.json").read_text())
train_split = next(s for s in diag["splits"] if s["split"] == "train")
cfg = json.loads((results / "train_config_resolved.json").read_text())
train = cfg.get("training", {})
row = {
    "run": run,
    "dict_size": ev.get("dict_size"),
    "k": ev.get("k"),
    "sparsity_mode": ev.get("sparsity_mode", train.get("sparsity_mode")),
    "center_inputs": train.get("center_inputs"),
    "normalize_decoder": train.get("normalize_decoder"),
    "aux_k_coef": train.get("aux_k_coef"),
    "basis_score_coef": train.get("basis_score_coef"),
    "max_steps": train.get("max_steps"),
    "live_train": train_split.get("live_features"),
    "dead_train": train_split.get("dead_feature_rate"),
    "gini_train_all": train_split.get("gini_activation_frequency_all"),
    "gini_train_live": train_split.get("gini_activation_frequency_live"),
    "live_test": ev.get("live_features"),
    "dead_test": ev.get("dead_feature_rate"),
    "gini_test_all": ev.get("gini_activation_frequency_all"),
    "gini_test_live": ev.get("gini_activation_frequency_live"),
    "mean_basis_r": ev.get("mean_basis_score_pearson"),
    "min_basis_r": ev.get("min_basis_score_pearson"),
    "mean_pair_r": ev.get("mean_pair_score_pearson"),
    "lore_acc_drop": ev.get("lore_accuracy_drop"),
    "ev": ev.get("explained_variance"),
    "mse": ev.get("mse"),
}
(results / "comparison_row.json").write_text(json.dumps(row, indent=2) + "\n")
print(json.dumps(row, indent=2))
PY

echo "=== DONE ${RUN_NAME} ==="
