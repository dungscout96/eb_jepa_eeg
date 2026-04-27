#!/bin/bash
# Submit the 60-job Exp 6 sweep on Delta.
# Runs this script ON Delta in the repo root (expects the branch
# kkokate/exp6-sweep checked out).
#
# Grid:
#   STD_COEFF = COV_COEFF = {0.1, 0.25, 1.0, 4.0}
#   PRED_DIM  = {8, 16, 24, 32, 48}
#   SEED      = {42, 123, 2025}

set -euo pipefail

STDS=(0.1 0.25 1.0 4.0)
PRED_DIMS=(8 16 24 32 48)
SEEDS=(42 123 2025)

N=0
for STD in "${STDS[@]}"; do
  for PD in "${PRED_DIMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      N=$((N+1))
      echo "[$N] std=$STD pd=$PD seed=$SEED"
      STD_COEFF="$STD" PRED_DIM="$PD" SEED="$SEED" \
        sbatch scripts/train_exp6_sweep.sbatch
      # Stagger 2 s to avoid git/wandb races on job startup.
      sleep 2
    done
  done
done

echo "Submitted $N jobs."
