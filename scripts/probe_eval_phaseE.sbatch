#!/bin/bash
#SBATCH --job-name=issue8E
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/issue8E_%j.out
#SBATCH --error=logs/issue8E_%j.err
# ---------------------------------------------------------------
# Issue #8 Phase E — single probe-eval at a fixed (encoder, probe-seed)
# pair. Operates on the latest.pth.tar produced by Phase D.
#
# Required env: CKPT, NW, WS, BATCH_SIZE, PROBE_SEED, EXP_TAG.
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

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
echo "Cfg: ckpt=${CKPT}  nw=${NW} ws=${WS} bs=${BATCH_SIZE} probe_seed=${PROBE_SEED} tag=${EXP_TAG}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="${CKPT}" \
    --n_windows="${NW}" \
    --window_size_seconds="${WS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --seed="${PROBE_SEED}" \
    --wandb_group="issue8E_${EXP_TAG}"

echo "Phase E ${EXP_TAG} complete!"
