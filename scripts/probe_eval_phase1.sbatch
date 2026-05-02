#!/bin/bash
#SBATCH --job-name=p1diag
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/p1diag_%j.out
#SBATCH --error=logs/p1diag_%j.err
# ---------------------------------------------------------------
# Issue #10 Phase 1 — encoder-diagnostic probe at (CKPT, PROBE_SEED, COND).
# COND parameters (env): PROBE_LAYER, USE_TEACHER, KEEP_CHANNELS,
# PROBE_CHANNEL, PREPRED. Defaults match a no-flag baseline run.
#
# Required env: CKPT, NW, WS, BATCH_SIZE, PROBE_SEED, EXP_TAG.
# Optional env: PROBE_LAYER (default: final), USE_TEACHER (default: 0),
#   KEEP_CHANNELS (default: 0), PROBE_CHANNEL (default: -1),
#   PREPRED (default: 0).
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

PROBE_LAYER="${PROBE_LAYER:-final}"
USE_TEACHER="${USE_TEACHER:-0}"
KEEP_CHANNELS="${KEEP_CHANNELS:-0}"
PROBE_CHANNEL="${PROBE_CHANNEL:--1}"
PREPRED="${PREPRED:-0}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: CKPT not found: $CKPT"
    exit 2
fi

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: ckpt=${CKPT}  nw=${NW} ws=${WS} bs=${BATCH_SIZE}"
echo "     probe_seed=${PROBE_SEED} tag=${EXP_TAG}"
echo "     layer=${PROBE_LAYER} teacher=${USE_TEACHER} keep_chan=${KEEP_CHANNELS}"
echo "     probe_channel=${PROBE_CHANNEL} prepred=${PREPRED}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_ENTITY="${WANDB_ENTITY:-sccn}"
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# Build flag args (booleans need explicit --flag=True/False for fire)
USE_TEACHER_ARG=$([ "$USE_TEACHER" = "1" ] && echo "--use_teacher=True" || echo "--use_teacher=False")
KEEP_CHANNELS_ARG=$([ "$KEEP_CHANNELS" = "1" ] && echo "--keep_channels=True" || echo "--keep_channels=False")
PREPRED_ARG=$([ "$PREPRED" = "1" ] && echo "--prepred=True" || echo "--prepred=False")

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
    --probe_layer="${PROBE_LAYER}" \
    ${USE_TEACHER_ARG} \
    ${KEEP_CHANNELS_ARG} \
    --probe_channel="${PROBE_CHANNEL}" \
    ${PREPRED_ARG} \
    --wandb_group="issue10p1_${EXP_TAG}"

echo "Phase 1 diag ${EXP_TAG} probe_seed=${PROBE_SEED} done."
