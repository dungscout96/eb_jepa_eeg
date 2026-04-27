#!/bin/bash
# Submit 5-seed Exp 13 probe_eval on Delta. Run from repo root on Delta.
set -euo pipefail

SEEDS=(42 123 456 789 2025)
for SEED in "${SEEDS[@]}"; do
  echo "Submitting probe_eval seed=$SEED"
  SEED="$SEED" sbatch scripts/eval_exp13_probes_delta.sbatch
  sleep 2
done
echo "Submitted ${#SEEDS[@]} jobs."
