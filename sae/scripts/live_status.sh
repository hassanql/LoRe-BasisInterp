#!/usr/bin/env bash
# Print one-line live status for SAE experiments on this machine (run on VM).
# Usage (on VM):  bash sae/scripts/live_status.sh
# Or loop:        watch -n 15 bash sae/scripts/live_status.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || echo "n/a")

if pgrep -f "python sae/scripts/train_sae.py" >/dev/null 2>&1; then
  train_state="TRAINING"
elif pgrep -f "python sae/scripts/evaluate_sae.py" >/dev/null 2>&1; then
  train_state="EVAL"
elif pgrep -f "python sae/scripts/diagnose_sae.py" >/dev/null 2>&1; then
  train_state="DIAGNOSE"
else
  train_state="IDLE"
fi

# Detect active results dir from newest train_log
active="-"
step="-"
mse="-"
dead="-"
live="-"
if ls sae/results/*/train_log.csv >/dev/null 2>&1; then
  newest=$(ls -t sae/results/*/train_log.csv | head -1)
  active=$(basename "$(dirname "$newest")")
  last=$(tail -1 "$newest")
  # step,train_mse,train_aux_mse,train_basis_mse,train_loss,dead,live,val_mse,val_ev
  step=$(echo "$last" | cut -d, -f1)
  mse=$(echo "$last" | cut -d, -f2)
  dead=$(echo "$last" | cut -d, -f6)
  live=$(echo "$last" | cut -d, -f7)
fi

phase="-"
if [ -f sae/results/c_sweep_master.log ]; then
  phase=$(grep -E "=== TRAIN|=== DONE|=== EVAL|ALL_" sae/results/c_sweep_master.log | tail -1 | tr -d '\r')
fi

printf "LIVE %s | %s | gpu=%s | run=%s step=%s mse=%s dead=%s live=%s | %s\n" \
  "$ts" "$train_state" "$gpu" "$active" "$step" "$mse" "$dead" "$live" "$phase"
