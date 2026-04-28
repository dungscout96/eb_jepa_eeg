#!/bin/bash
#SBATCH --job-name=issue8D
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/issue8D_%j.out
#SBATCH --error=logs/issue8D_%j.err
# ---------------------------------------------------------------
# Issue #8 Phase D — encoder pretrainings WITHOUT broken early-stop.
# Always trains the full epoch budget (patience=0). Saves latest.pth.tar
# every epoch; probe eval (Phase E) consumes that.
#
# Required env: NW, WS, SEED, BATCH_SIZE, USE_CORRCA (0|1),
#               LR, WARMUP, EPOCHS, EXP_TAG.
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${NW:?need NW}"
: "${WS:?need WS}"
: "${SEED:?need SEED}"
: "${BATCH_SIZE:?need BATCH_SIZE}"
: "${USE_CORRCA:?need USE_CORRCA (0 or 1)}"
: "${LR:?need LR}"
: "${WARMUP:?need WARMUP}"
: "${EPOCHS:=100}"
: "${EXP_TAG:?need EXP_TAG}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: nw=${NW} ws=${WS} seed=${SEED} bs=${BATCH_SIZE} corrca=${USE_CORRCA} lr=${LR} warmup=${WARMUP} epochs=${EPOCHS} tag=${EXP_TAG}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

CORRCA_TRAIN_ARG=""
if [ "${USE_CORRCA}" = "1" ]; then
    CORRCA_TRAIN_ARG="--data.corrca_filters=corrca_filters.npz"
fi

EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8/${EXP_TAG}"
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
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="issue8D_${EXP_TAG}"

CKPT="${EXP_DIR}/latest.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no latest.pth.tar at $CKPT"
    exit 1
fi
echo "Phase D ${EXP_TAG} complete!  ckpt=${CKPT}"
