#!/bin/bash
# Cell L 1-seed test — multi-horizon latent rollout
# Default: mf=0.25 (mask 1 of 4 windows from tail; predict last 2s from first 6s).
# This gives the encoder 6s of past context (matching narrative timescale 5–10s)
# and the horizon embedding lets predictor differentiate near-future patches in
# masked window 3 from the rest. horizon_max=3 (= NW-1).
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=2024
TAG="cellL_1seed_${SEED}"

sbatch \
    --export=ALL,SEED=${SEED},EXP_TAG=${TAG},EPOCHS=100,GROUP_TAG=issue10_cellL \
    scripts/train_cellL.sbatch
