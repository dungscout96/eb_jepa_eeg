#!/bin/bash
# Issue #8 Phase A — paired CorrCA ablation across the 5 Exp 6 seeds.
# Each cell: VICReg + per-rec + nw4_ws2 + NO CorrCA.
#
# Run on Delta from the repo root with the issue-8 branch checked out.

set -euo pipefail

SEEDS=(42 123 456 789 2025)

mkdir -p logs

N=0
for SEED in "${SEEDS[@]}"; do
  N=$((N+1))
  EXP_TAG="ablation_nocorrca_nw4ws2_s${SEED}"
  echo "[${N}/${#SEEDS[@]}] ${EXP_TAG}"
  NW=4 WS=2 SEED="${SEED}" BATCH_SIZE=64 USE_CORRCA=0 EXP_TAG="${EXP_TAG}" \
    sbatch scripts/train_issue8.sbatch
  sleep 2  # avoid git/wandb startup races
done

echo "Submitted ${N} Phase A jobs."
