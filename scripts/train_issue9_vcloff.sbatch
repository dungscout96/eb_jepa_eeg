#!/bin/bash
#SBATCH --job-name=issue9A
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/issue9A_%j.out
#SBATCH --error=logs/issue9A_%j.err
# ---------------------------------------------------------------
# Issue #9 Phase A — VCLoss ablation. Same Exp 6 stack but with
#   --loss.std_coeff and --loss.cov_coeff env-driven (default 0/0 = OFF).
# Default LR = 5e-4 (Exp 6 / Phase D baseline). Paired with Phase D nw4_ws2
# seeds {42, 123, 456, 789, 2025} for a clean Δ on VCLoss removal.
#
# Required env: NW, WS, SEED, BATCH_SIZE, USE_CORRCA, EXP_TAG.
# Optional env: LR (default 5e-4), WARMUP (default 5), EPOCHS (default 100),
#               STD_COEFF (default 0.0), COV_COEFF (default 0.0),
#               PRED_LOSS (default smooth_l1).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${NW:?need NW}"
: "${WS:?need WS}"
: "${SEED:?need SEED}"
: "${BATCH_SIZE:?need BATCH_SIZE}"
: "${USE_CORRCA:?need USE_CORRCA (0 or 1)}"
: "${EXP_TAG:?need EXP_TAG}"

LR="${LR:-5e-4}"
WARMUP="${WARMUP:-5}"
EPOCHS="${EPOCHS:-100}"
STD_COEFF="${STD_COEFF:-0.0}"
COV_COEFF="${COV_COEFF:-0.0}"
PRED_LOSS="${PRED_LOSS:-smooth_l1}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: nw=${NW} ws=${WS} seed=${SEED} bs=${BATCH_SIZE} corrca=${USE_CORRCA}"
echo "      lr=${LR} warmup=${WARMUP} epochs=${EPOCHS}"
echo "      std=${STD_COEFF} cov=${COV_COEFF} pred_loss=${PRED_LOSS}"
echo "      tag=${EXP_TAG}"

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
    --loss.pred_loss_type="${PRED_LOSS}" \
    --logging.wandb_group="issue9A_${EXP_TAG}"

CKPT="${EXP_DIR}/latest.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no latest.pth.tar at $CKPT"
    exit 1
fi
echo "Issue #9 Phase A ${EXP_TAG} complete!  ckpt=${CKPT}"
