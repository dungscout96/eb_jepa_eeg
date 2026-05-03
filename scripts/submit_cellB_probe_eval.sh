#!/bin/bash
# Cell B probe eval: 1 probe seed per encoder seed, on BOTH
# best_by_online_probe.pth.tar AND latest.pth.tar (when present).
# Run AFTER all 5 training jobs complete.
set -euo pipefail
cd "$(dirname "$0")/.."

NW=4
WS=2
BATCH_SIZE=64
PROBE_SEED=42
CKPT_ROOT=/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue10

for SEED in 2024 2025 2026 2027 2028; do
    EXP_DIR="${CKPT_ROOT}/cellB_5seed_${SEED}"
    for FLAVOR in best_by_online_probe latest; do
        CKPT="${EXP_DIR}/${FLAVOR}.pth.tar"
        TAG="cellB_${SEED}_${FLAVOR}"
        if [ ! -f "$CKPT" ]; then
            echo "skip: $CKPT (not found)"
            continue
        fi
        sbatch \
            --export=ALL,CKPT=${CKPT},NW=${NW},WS=${WS},BATCH_SIZE=${BATCH_SIZE},PROBE_SEED=${PROBE_SEED},EXP_TAG=${TAG} \
            scripts/probe_eval_cellB.sbatch
        echo "submitted probe eval: ${TAG}"
    done
done
