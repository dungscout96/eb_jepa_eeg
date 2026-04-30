#!/bin/bash
#SBATCH --job-name=v10stage
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/v10stage_%j.out
#SBATCH --error=logs/v10stage_%j.err
# ---------------------------------------------------------------
# v10 stage-diagnostic — single probe-eval at a fixed
# (config, encoder, probe-seed, stage). Re-uses the Phase D
# checkpoints from the issue8 worktree.
#
# Required env: CKPT, NW, WS, BATCH_SIZE, PROBE_SEED, EXP_TAG, STAGE.
# STAGE in {context_enc, target_enc, patches, target_patches}.
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
: "${STAGE:?need STAGE}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: CKPT not found: $CKPT"
    exit 2
fi

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: ckpt=${CKPT}  nw=${NW} ws=${WS} bs=${BATCH_SIZE} probe_seed=${PROBE_SEED} stage=${STAGE} tag=${EXP_TAG}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# v10 worktree's uv.lock pulled a newer torchaudio that requires CUDA 13.
# Our code changes added no new deps, so reuse the issue8 worktree's
# working .venv (torch 2.6.0+cu124, torchaudio 2.6.0+cu124) directly.
ISSUE8_VENV_PY="/projects/bbnv/kkokate/eb_jepa_eeg/.venv/bin/python"
if [ ! -x "$ISSUE8_VENV_PY" ]; then
    echo "ERROR: issue8 venv python not found at $ISSUE8_VENV_PY"
    exit 3
fi

PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/v10_stage/${EXP_TAG}"

PYTHONPATH=. "$ISSUE8_VENV_PY" experiments/eeg_jepa/probe_eval.py \
    --checkpoint="${CKPT}" \
    --n_windows="${NW}" \
    --window_size_seconds="${WS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --probe_stage="${STAGE}" \
    --seed="${PROBE_SEED}" \
    --save_predictions_dir="${PRED_DIR}" \
    --wandb_group="v10stage_${STAGE}"

echo "v10 stage-diag ${EXP_TAG} (${STAGE}) complete!"
