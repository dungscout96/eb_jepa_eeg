#!/bin/bash
#SBATCH --job-name=trivial_ridge
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/trivial_ridge_%j.out
#SBATCH --error=logs/trivial_ridge_%j.err
# ---------------------------------------------------------------
# Ridge probe trivial baseline (matches the draft's exact procedure).
# Parameterized by env:
#   BASELINE  ∈ {trivial_ridge_corrca35, trivial_ridge_raw903, trivial_ridge_chan1_only}
#   SEED      (default 42)
#   N_PASSES  (default 20)
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"

: "${BASELINE:?need BASELINE}"
SEED="${SEED:-42}"
N_PASSES="${N_PASSES:-20}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Baseline: $BASELINE  Seed: $SEED"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

OUT_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/tier1"
PRED_DIR="${OUT_DIR}/predictions/${BASELINE}_seed${SEED}"
OUT_JSON="${OUT_DIR}/${BASELINE}_seed${SEED}.json"
mkdir -p "$PRED_DIR"

# Pick CorrCA on/off + channel-routing options based on baseline name
CORRCA_ARG=""
EXTRA_PY="--n_windows=4 --window_size_seconds=2 --norm_mode=per_recording"
case "$BASELINE" in
    trivial_ridge_corrca35)
        CORRCA_ARG="--corrca_filters=corrca_filters.npz"
        ;;
    trivial_ridge_chan1_only)
        CORRCA_ARG="--corrca_filters=corrca_filters.npz"
        EXTRA_PY="$EXTRA_PY --chan_slice=0"
        ;;
    trivial_ridge_corrca_pooled35)
        CORRCA_ARG="--corrca_filters=corrca_filters.npz"
        EXTRA_PY="$EXTRA_PY --pooled"
        ;;
    trivial_ridge_raw903)
        ;;
    trivial_ridge_raw_pooled903)
        EXTRA_PY="$EXTRA_PY --pooled"
        ;;
    *) echo "Unknown BASELINE=$BASELINE"; exit 2;;
esac

PYTHONPATH=. /u/kkokate/.conda/envs/eb_jepa/bin/python scripts/trivial_ridge_baseline.py \
    $EXTRA_PY \
    --n_passes="${N_PASSES}" \
    --seed="${SEED}" \
    --baseline="${BASELINE}" \
    --save_predictions_dir="${PRED_DIR}" \
    --output_json="${OUT_JSON}" \
    ${CORRCA_ARG}

echo "Trivial ridge ${BASELINE} seed=${SEED} complete: ${OUT_JSON}"
