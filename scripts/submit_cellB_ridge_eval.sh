#!/bin/bash
# Cell B kc+Ridge probe eval (PR #15 protocol).
# Reuses the existing probe_lever1_ridge.sbatch driver. 1 probe seed per
# encoder seed × 2 checkpoints (best_by_online_probe + latest) = 10 jobs.
set -euo pipefail
cd "$(dirname "$0")/.."

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
            --export=ALL,CKPT=${CKPT},EXP_TAG=${TAG},PROBE_SEED=${PROBE_SEED},NW=4,WS=2,N_PASSES=20 \
            scripts/probe_lever1_ridge.sbatch
        echo "submitted ridge eval: ${TAG}"
    done
done
