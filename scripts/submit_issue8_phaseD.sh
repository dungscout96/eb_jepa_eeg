#!/bin/bash
# Issue #8 Phase D — 15 encoder pretrainings (3 configs × 5 seeds), patience=0.
#
# Configs:
#   nw4_ws2 (Exp 6 baseline re-run, lr=5e-4, warmup=5)  — for clean comparison
#   nw2_ws4 (8s, 2x4s, lr=5e-4, warmup=5)
#   nw4_ws4 (16s, lr=3e-4, warmup=10)  — lower LR + longer warmup for 2x token count

set -euo pipefail

SEEDS=(42 123 456 789 2025)

# (NW WS LR WARMUP BATCH_SIZE TAG_PREFIX)
CONFIGS=(
  "4 2 5e-4 5  64 nw4ws2_baseline"
  "2 4 5e-4 5  64 nw2ws4"
  "4 4 3e-4 10 64 nw4ws4"
)

mkdir -p logs

N=0
for CFG in "${CONFIGS[@]}"; do
  read -r NW WS LR WARMUP BS TAG_PREFIX <<<"${CFG}"
  for SEED in "${SEEDS[@]}"; do
    N=$((N+1))
    EXP_TAG="phaseD_${TAG_PREFIX}_s${SEED}"
    echo "[${N}/15] ${EXP_TAG}  nw=${NW} ws=${WS} lr=${LR} warmup=${WARMUP} bs=${BS}"
    NW="${NW}" WS="${WS}" SEED="${SEED}" BATCH_SIZE="${BS}" \
      USE_CORRCA=1 LR="${LR}" WARMUP="${WARMUP}" EPOCHS=100 EXP_TAG="${EXP_TAG}" \
      sbatch scripts/train_issue8_phaseD.sbatch
    sleep 2
  done
done

echo "Submitted ${N} Phase D encoder pretrainings."
