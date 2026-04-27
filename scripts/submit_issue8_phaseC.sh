#!/bin/bash
# Issue #8 Phase C — confirm Phase B winner (nw2_ws4) across the 5 Exp 6 seeds.
# Same stack as Exp 6 (per-rec + CorrCA, std=0.25, pd=24, smooth_l1) but with
# n_windows=2 / window_size=4 (8s context, fewer windows × longer windows).

set -euo pipefail

SEEDS=(42 123 456 789 2025)

mkdir -p logs

N=0
for SEED in "${SEEDS[@]}"; do
  N=$((N+1))
  EXP_TAG="phaseC_corrca_nw2ws4_s${SEED}"
  echo "[${N}/${#SEEDS[@]}] ${EXP_TAG}"
  NW=2 WS=4 SEED="${SEED}" BATCH_SIZE=64 USE_CORRCA=1 EXP_TAG="${EXP_TAG}" \
    sbatch scripts/train_issue8.sbatch
  sleep 2
done

echo "Submitted ${N} Phase C jobs."
