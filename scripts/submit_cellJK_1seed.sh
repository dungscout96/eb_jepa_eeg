#!/bin/bash
# Cell J + Cell K — 1 encoder seed each, 100 epochs, ThePresent.
# Phase D-comparable random sampler (no stim_aligned), so steps_per_epoch is
# implicit (~11 batches/epoch from len(train_set)=700 recordings, B=64).
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=2024

# Cell J: cross-time mask
sbatch \
    --export=ALL,SEED=${SEED},EXP_TAG=cellJ_1seed_${SEED},CELL=j,EPOCHS=100,GROUP_TAG=issue10_cellJ \
    scripts/train_cellJK.sbatch
echo "submitted Cell J seed=${SEED}"

# Cell K: PARS Δt regression
sbatch \
    --export=ALL,SEED=${SEED},EXP_TAG=cellK_1seed_${SEED},CELL=k,EPOCHS=100,GROUP_TAG=issue10_cellK \
    scripts/train_cellJK.sbatch
echo "submitted Cell K seed=${SEED}"
