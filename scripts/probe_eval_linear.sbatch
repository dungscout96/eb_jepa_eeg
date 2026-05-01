#!/bin/bash
#SBATCH --job-name=probe_linear
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/linear_probe_%j.out
#SBATCH --error=logs/linear_probe_%j.err
# ---------------------------------------------------------------
# probe_eval.py with --keep_channels --linear_movie_probe.
# Tests whether MovieFeatureHead's 2-layer MLP underperforms a
# single-linear movie probe (the MLP is a strict superset, so
# any underperformance reflects training, not capacity).
#
# Required env:
#   CKPT, NW, WS, BATCH_SIZE, PROBE_SEED, EXP_TAG
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"

: "${CKPT:?need CKPT}"
: "${NW:?need NW}"
: "${WS:?need WS}"
: "${BATCH_SIZE:?need BATCH_SIZE}"
: "${PROBE_SEED:?need PROBE_SEED}"
: "${EXP_TAG:?need EXP_TAG}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: CKPT not found: $CKPT"
    exit 2
fi

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "CKPT=$CKPT NW=$NW WS=$WS BS=$BATCH_SIZE PROBE_SEED=$PROBE_SEED TAG=$EXP_TAG"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi

# Use the eb_jepa conda env directly. The /projects/.../.venv torchaudio
# was bumped to require CUDA 13 (libcudart.so.13 missing). The conda env's
# torchaudio is 2.6.0+cu124 and works on this cluster.
CONDA_PY="/u/kkokate/.conda/envs/eb_jepa/bin/python"
if [ ! -x "$CONDA_PY" ]; then
    echo "ERROR: conda env python not found at $CONDA_PY"
    exit 3
fi

PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/linear_probe/${EXP_TAG}"
mkdir -p "$PRED_DIR"

PYTHONPATH=. "$CONDA_PY" experiments/eeg_jepa/probe_eval.py \
    --checkpoint="${CKPT}" \
    --n_windows="${NW}" \
    --window_size_seconds="${WS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --keep_channels \
    --linear_movie_probe \
    --seed="${PROBE_SEED}" \
    --save_predictions_dir="${PRED_DIR}" \
    --wandb_group="linear_probe_keepch"

echo "linear-probe ${EXP_TAG} done"
