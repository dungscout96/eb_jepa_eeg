#!/bin/bash
# Submit the linear-movie-probe sweep on the same nw4ws2_baseline checkpoints
# that produced PR #15's --keep_channels MLP results. 5 enc seeds × 5 probe
# seeds = 25 jobs. Replaces MovieFeatureHead with a single Linear(D, n_features),
# nothing else changes — same Adam, same lr=1e-3, 20 epochs, batch=64, joint
# reg+cls training, --keep_channels on.

set -euo pipefail

CKPT_BASE="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8"
ENC_SEEDS=(42 123 456 789 2025)
PROBE_SEEDS=(7 13 42 1234 2025)
NW=4
WS=2
BATCH_SIZE=64

mkdir -p logs

echo "=== checkpoint presence check ==="
for SEED in "${ENC_SEEDS[@]}"; do
    CKPT="${CKPT_BASE}/phaseD_nw4ws2_baseline_s${SEED}/latest.pth.tar"
    if [ ! -f "$CKPT" ]; then
        echo "MISSING: $CKPT"; exit 1
    fi
done
echo "all 5 checkpoints found"

N=0
TOTAL=$(( ${#ENC_SEEDS[@]} * ${#PROBE_SEEDS[@]} ))
for ENC_SEED in "${ENC_SEEDS[@]}"; do
    CKPT="${CKPT_BASE}/phaseD_nw4ws2_baseline_s${ENC_SEED}/latest.pth.tar"
    for PROBE_SEED in "${PROBE_SEEDS[@]}"; do
        N=$((N+1))
        EXP_TAG="nw4ws2_linear_enc${ENC_SEED}_p${PROBE_SEED}"
        echo "[${N}/${TOTAL}] ${EXP_TAG}"
        CKPT="${CKPT}" NW="${NW}" WS="${WS}" BATCH_SIZE="${BATCH_SIZE}" \
            PROBE_SEED="${PROBE_SEED}" EXP_TAG="${EXP_TAG}" \
            sbatch --job-name="linprobe_e${ENC_SEED}_p${PROBE_SEED}" \
                   scripts/probe_eval_linear.sbatch
    done
done
echo "Submitted ${N} linear-probe jobs"
