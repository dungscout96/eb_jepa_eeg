#!/bin/bash
# Cell B full run: 5 enc seeds × 100 epochs, ThePresent only.
# Run AFTER smoke confirms xsub_pred_loss is trending down.
set -euo pipefail
cd "$(dirname "$0")/.."

for SEED in 2024 2025 2026 2027 2028; do
    TAG="cellB_5seed_${SEED}"
    sbatch \
        --export=ALL,SEED=${SEED},EXP_TAG=${TAG},EPOCHS=100,GROUP_TAG=issue10_cellB_5seed \
        scripts/train_cellB_xsub.sbatch
    echo "submitted cellB seed=${SEED} tag=${TAG}"
done
