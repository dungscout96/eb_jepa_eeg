#!/bin/bash
# Phase D nw2_ws4 — 5 seeds with current code (dual-checkpoint discipline).
# Replaces the issue8-era checkpoints which only saved latest.pth.tar.
# Goal: have best_by_online_probe.pth.tar AND latest.pth.tar for every
# Phase D nw2_ws4 seed, matching the canonical eval protocol.
set -euo pipefail
cd "$(dirname "$0")/.."

for SEED in 42 123 456 789 2025; do
    TAG="phaseD_nw2ws4_canonical_${SEED}"
    sbatch \
        --export=ALL,SEED=${SEED},EXP_TAG=${TAG},NW=2,WS=4,EPOCHS=100,GROUP_TAG=issue10_phaseD_canonical_nw2ws4 \
        scripts/train_phaseD.sbatch
    echo "submitted phaseD_nw2ws4 seed=${SEED}"
done
