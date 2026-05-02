#!/bin/bash
#SBATCH --job-name=mlp_phaseE
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/mlp_phaseE_%j.out
#SBATCH --error=logs/mlp_phaseE_%j.err
# ---------------------------------------------------------------
# Phase E follow-up — re-evaluate Phase D encoders with a 2-layer MLP
# probe head (vs. the linear baseline previously run in probe_eval_phaseE).
# Same (encoder-seed, probe-seed) grid; saves per-recording predictions
# for bootstrap CIs.
#
# Required env: CKPT, NW, WS, BATCH_SIZE, PROBE_SEED, EXP_TAG.
# Optional env: PROBE_LR (default 5e-4), SUBJECT_PROBE_LR (default 5e-4),
#               HIDDEN_DIM (default 128), MOVIE_HIDDEN (default 128).
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

PROBE_LR="${PROBE_LR:-5e-4}"
SUBJECT_PROBE_LR="${SUBJECT_PROBE_LR:-5e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
MOVIE_HIDDEN="${MOVIE_HIDDEN:-128}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: CKPT not found: $CKPT"
    exit 2
fi

PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/issue8E_mlp/${EXP_TAG}"
mkdir -p "$PRED_DIR"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: ckpt=${CKPT}  nw=${NW} ws=${WS} bs=${BATCH_SIZE} probe_seed=${PROBE_SEED} tag=${EXP_TAG}"
echo "MLP: hidden=${HIDDEN_DIM} movie_hidden=${MOVIE_HIDDEN} probe_lr=${PROBE_LR} subject_lr=${SUBJECT_PROBE_LR}"
echo "Predictions dir: ${PRED_DIR}"

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
    --probe_type=mlp \
    --probe_hidden_dim="${HIDDEN_DIM}" \
    --movie_probe_hidden_dim="${MOVIE_HIDDEN}" \
    --probe_lr="${PROBE_LR}" \
    --subject_probe_lr="${SUBJECT_PROBE_LR}" \
    --save_predictions_dir="${PRED_DIR}" \
    --wandb_group="issue8E_mlp_${EXP_TAG}"

echo "MLP probe ${EXP_TAG} complete!"
