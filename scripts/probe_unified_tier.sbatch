#!/bin/bash
#SBATCH --job-name=PUtier
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/PUtier_%j.out
#SBATCH --error=logs/PUtier_%j.err
# ---------------------------------------------------------------
# Canonical eval driver via unified_probe_eval.py with --feature_source.
# Routes Tier 1 (raw_corrca, psd_band, random_init) and Tier 3 (biot, luna)
# through the SAME canonical Ridge α=1 / LogReg lbfgs C=1 pipeline as Phase D.
#
# Required env: FEATURE_SOURCE, ENC_SEED, EXP_TAG.
# Optional: NW (2), WS (4), N_PASSES (20), PROBE_SEED (42), CKPT (only for jepa).
# ---------------------------------------------------------------

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${FEATURE_SOURCE:?need FEATURE_SOURCE (jepa|raw_corrca|psd_band|random_init)}"
: "${ENC_SEED:?need ENC_SEED}"
: "${EXP_TAG:?need EXP_TAG}"
NW="${NW:-2}"
WS="${WS:-4}"
N_PASSES="${N_PASSES:-20}"
PROBE_SEED="${PROBE_SEED:-42}"
CKPT="${CKPT:-}"

# raw-EEG baselines (raw_stats, raw_stats_pooled, psd_band) run on UNPROJECTED 129ch;
# others run on CorrCA-5 projected EEG.
case "$FEATURE_SOURCE" in
    psd_band|raw_stats|raw_stats_pooled) CORRCA_ARG="" ;;
    *) CORRCA_ARG="--corrca_filters=corrca_filters.npz" ;;
esac

CKPT_ARG=""
if [ "$FEATURE_SOURCE" = "jepa" ]; then
    : "${CKPT:?CKPT required for feature_source=jepa}"
    CKPT_ARG="--checkpoint=${CKPT}"
fi

cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

OUT_JSON="results/canonical/${EXP_TAG}/seed${ENC_SEED}/metrics.json"
PRED_DIR="predictions/canonical/${EXP_TAG}/seed${ENC_SEED}"
mkdir -p "$(dirname "$OUT_JSON")" "$PRED_DIR"

echo "Cfg: source=${FEATURE_SOURCE} enc_seed=${ENC_SEED} probe_seed=${PROBE_SEED} nw=${NW} ws=${WS} n_passes=${N_PASSES}"
echo "Out: ${OUT_JSON}"

# ENC_SEED selects the encoder weights (which checkpoint to load).
# PROBE_SEED is the probe-side RNG (n_passes random clip selection); MUST be
# fixed across encoder seeds so the comparison is apples-to-apples — different
# encoders see the same clip set, only encoder weights differ.
PYTHONPATH=. python scripts/unified_probe_eval.py \
    ${CKPT_ARG} \
    --feature_source="${FEATURE_SOURCE}" \
    --n_windows="${NW}" --window_size_seconds="${WS}" \
    --norm_mode=per_recording \
    ${CORRCA_ARG} \
    --n_passes="${N_PASSES}" \
    --seed="${PROBE_SEED}" \
    --output_json="${OUT_JSON}" \
    --save_predictions_dir="${PRED_DIR}"

echo "PUtier ${FEATURE_SOURCE} enc_seed=${ENC_SEED} complete"
