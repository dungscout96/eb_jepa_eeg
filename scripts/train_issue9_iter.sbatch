#!/bin/bash
#SBATCH --job-name=issue9_iter
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/issue9iter_%j.out
#SBATCH --error=logs/issue9iter_%j.err
# ---------------------------------------------------------------
# Issue #9 iteration loop — generic encoder trainer.
# Required env: SEED, EXP_TAG, ITER_TAG.
# Optional env: STD_COEFF (0.25), COV_COEFF (0.25), LR (5e-4),
#               PROJ_SPEC (default = expander 64-256-256), USE_PROJECTOR (true).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${SEED:?need SEED}"
: "${EXP_TAG:?need EXP_TAG}"
: "${ITER_TAG:?need ITER_TAG}"

NW="${NW:-4}"
WS="${WS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
USE_CORRCA="${USE_CORRCA:-1}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP:-5}"
EPOCHS="${EPOCHS:-100}"
STD_COEFF="${STD_COEFF:-0.25}"
COV_COEFF="${COV_COEFF:-0.25}"
PRED_LOSS="${PRED_LOSS:-smooth_l1}"
USE_PROJECTOR="${USE_PROJECTOR:-true}"
PROJ_SPEC="${PROJ_SPEC:-}"
DETACH_ENCODER="${DETACH_ENCODER:-false}"
DINO_COEFF="${DINO_COEFF:-0.0}"
ENV_COEFF="${ENV_COEFF:-0.0}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Iter: ${ITER_TAG}  Tag: ${EXP_TAG}  Seed: ${SEED}"
echo "VC: std=${STD_COEFF} cov=${COV_COEFF} use_proj=${USE_PROJECTOR} spec='${PROJ_SPEC}'"
echo "Optim: lr=${LR} warmup=${WARMUP} epochs=${EPOCHS} pred_loss=${PRED_LOSS}"

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

CORRCA_TRAIN_ARG=""
if [ "${USE_CORRCA}" = "1" ]; then
    CORRCA_TRAIN_ARG="--data.corrca_filters=corrca_filters.npz"
fi

PROJ_ARG=""
if [ -n "${PROJ_SPEC}" ]; then
    PROJ_ARG="--loss.proj_spec=${PROJ_SPEC}"
fi

EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue9/${EXP_TAG}"
mkdir -p "$EXP_DIR"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs="${EPOCHS}" \
    --optim.early_stopping_patience=0 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --optim.lr="${LR}" \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs="${WARMUP}" \
    --data.num_workers=2 \
    --data.batch_size="${BATCH_SIZE}" \
    --data.n_windows="${NW}" \
    --data.window_size_seconds="${WS}" \
    --data.norm_mode=per_recording \
    ${CORRCA_TRAIN_ARG} \
    --loss.std_coeff="${STD_COEFF}" \
    --loss.cov_coeff="${COV_COEFF}" \
    --loss.use_projector="${USE_PROJECTOR}" \
    --loss.detach_encoder="${DETACH_ENCODER}" \
    --loss.dino_coeff="${DINO_COEFF}" \
    --loss.env_coeff="${ENV_COEFF}" \
    ${PROJ_ARG} \
    --loss.pred_loss_type="${PRED_LOSS}" \
    --logging.wandb_group="issue9_${ITER_TAG}_${EXP_TAG}"

CKPT="${EXP_DIR}/latest.pth.tar"
[ -f "$CKPT" ] || { echo "ERROR: no latest.pth.tar at $CKPT"; exit 1; }
echo "Issue #9 iter ${ITER_TAG} ${EXP_TAG} complete!  ckpt=${CKPT}"
