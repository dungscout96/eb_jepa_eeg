#!/bin/bash
#SBATCH --job-name=probe_ridge
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ridge_method_%j.out
#SBATCH --error=logs/ridge_method_%j.err
# probe_eval.py with --probe_method=ridge --keep_channels --input_batchnorm.
# Closed-form Ridge per-window with per-feature L2 sweep on val (CLIP protocol).

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"

: "${CKPT:?need CKPT}"; : "${NW:?need NW}"; : "${WS:?need WS}"
: "${BATCH_SIZE:?need BATCH_SIZE}"; : "${PROBE_SEED:?need PROBE_SEED}"
: "${EXP_TAG:?need EXP_TAG}"

if [ ! -f "$CKPT" ]; then echo "ERROR: CKPT not found: $CKPT"; exit 2; fi

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  TAG: $EXP_TAG"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

CONDA_PY="/u/kkokate/.conda/envs/eb_jepa/bin/python"
PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/ridge_method/${EXP_TAG}"
mkdir -p "$PRED_DIR"

PYTHONPATH=. "$CONDA_PY" experiments/eeg_jepa/probe_eval.py \
    --checkpoint="${CKPT}" \
    --n_windows="${NW}" --window_size_seconds="${WS}" \
    --batch_size="${BATCH_SIZE}" --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording --corrca_filters=corrca_filters.npz \
    --keep_channels \
    --probe_method=ridge \
    --input_batchnorm \
    --seed="${PROBE_SEED}" \
    --save_predictions_dir="${PRED_DIR}" \
    --wandb_group="ridge_method_keepch"

echo "ridge_method ${EXP_TAG} done"
