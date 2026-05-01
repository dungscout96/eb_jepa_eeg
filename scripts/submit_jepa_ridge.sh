#!/bin/bash
# Submit JEPA-into-Ridge runs: 5 encoder seeds × 2 keep_channels variants.
# Reuses the existing PhaseD nw4ws2 keep-channels checkpoints.
set -euo pipefail

CKPT_ROOT="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8"
ENC_SEEDS=(42 123 456 789 2025)
PROBE_SEED=42

for ENC in "${ENC_SEEDS[@]}"; do
    CKPT="${CKPT_ROOT}/phaseD_nw4ws2_baseline_s${ENC}/latest.pth.tar"
    if [ ! -f "$CKPT" ]; then
        echo "MISSING $CKPT"; continue
    fi
    for KC in false true; do
        echo "Submitting enc=$ENC keep_channels=$KC"
        CKPT="$CKPT" ENC_SEED="$ENC" PROBE_SEED="$PROBE_SEED" KEEP_CHANNELS="$KC" \
        sbatch --job-name="jepa_ridge_kc${KC}_e${ENC}" scripts/jepa_ridge.sbatch
    done
done
