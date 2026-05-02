#!/bin/bash
#SBATCH --job-name=lejepa
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lejepa_%j.out
#SBATCH --error=logs/lejepa_%j.err
# ---------------------------------------------------------------
# Issue #10 LeJEPA — pretraining with NO EMA target encoder + SIGRegLoss.
# Tests whether dropping the moving-target setup makes pred_loss actually
# decrease across training (Phase-D pred_loss is bounded ≈ noise floor).
#
# Required env: SEED, EXP_TAG.
# Optional env: NW (4), WS (2), BATCH_SIZE (64), LR (5e-4), WARMUP (5),
#               EPOCHS (100), SIGREG_COEFF (0.1).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${SEED:?need SEED}"
: "${EXP_TAG:?need EXP_TAG}"
NW="${NW:-4}"
WS="${WS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP:-5}"
EPOCHS="${EPOCHS:-100}"
SIGREG_COEFF="${SIGREG_COEFF:-0.1}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: seed=${SEED} tag=${EXP_TAG} nw=${NW} ws=${WS} bs=${BATCH_SIZE}"
echo "     lr=${LR} warmup=${WARMUP} epochs=${EPOCHS} sigreg=${SIGREG_COEFF}"

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

EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue10/${EXP_TAG}"
mkdir -p "$EXP_DIR"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs="${EPOCHS}" \
    --optim.early_stopping_patience=0 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --model.use_ema=False \
    --optim.lr="${LR}" \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs="${WARMUP}" \
    --data.num_workers=2 \
    --data.batch_size="${BATCH_SIZE}" \
    --data.n_windows="${NW}" \
    --data.window_size_seconds="${WS}" \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --loss.regularizer=sigreg \
    --loss.use_projector=False \
    --loss.sigreg.coeff="${SIGREG_COEFF}" \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="issue10_lejepa"

CKPT="${EXP_DIR}/latest.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no latest.pth.tar at $CKPT"
    exit 1
fi
echo "LeJEPA ${EXP_TAG} complete!  ckpt=${CKPT}"
