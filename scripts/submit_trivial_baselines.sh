#!/bin/bash
# Submit 5 trivial-stats baselines × 5 probe seeds = 25 jobs.
#
# Probe seeds match the JEPA --keep_channels 5x5 sweep
# (docs/probe_eval_keep_channels.md): {7, 13, 42, 1234, 2025}.
#
# Each job dumps per-recording prediction npzs into
#   /projects/bbnv/kkokate/eb_jepa_eeg/tier1/predictions/<baseline>_seed<seed>/
# which are consumed by scripts/bootstrap_trivial_perseed.py for View 2 + View 3.
#
# Usage on Delta:
#   bash scripts/submit_trivial_baselines.sh

set -euo pipefail

BASELINES=(
    trivial_corrca_per_chan
    trivial_corrca_chan1_only
    trivial_raw_per_chan
    trivial_corrca_pooled35
    trivial_raw_pooled903
)
SEEDS=(7 13 42 1234 2025)
PRED_ROOT="/projects/bbnv/kkokate/eb_jepa_eeg/tier1/predictions"
TIER1_ROOT="/projects/bbnv/kkokate/eb_jepa_eeg/tier1"

mkdir -p "$PRED_ROOT"

for BASELINE in "${BASELINES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TAG="${BASELINE}_seed${SEED}"
        SAVE_PREDS_DIR="${PRED_ROOT}/${TAG}"
        OUT_JSON="${TIER1_ROOT}/${TAG}.json"
        echo "Submitting BASELINE=${BASELINE} SEED=${SEED}"
        BASELINE="$BASELINE" \
        SEED="$SEED" \
        TAG="$TAG" \
        SAVE_PREDS_DIR="$SAVE_PREDS_DIR" \
        sbatch --job-name="trivial_${BASELINE}_s${SEED}" \
               scripts/tier1_baseline.sbatch
    done
done
