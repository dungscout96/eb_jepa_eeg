#!/bin/bash
# Cell B smoke job: 1 seed, 30 epochs, ThePresent only.
# Goal: verify xsub_pred_loss decreases below trivial floor and probes track.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=2024
TAG="cellB_smoke_seed${SEED}"

sbatch \
    --export=ALL,SEED=${SEED},EXP_TAG=${TAG},EPOCHS=30,GROUP_TAG=issue10_cellB_smoke \
    scripts/train_cellB_xsub.sbatch
