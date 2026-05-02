#!/bin/bash
#SBATCH --job-name=lev1
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/lever1_%j.out
#SBATCH --error=logs/lever1_%j.err
# ---------------------------------------------------------------
# Issue #10 Lever 1 — pretraining with cross-subject stim-aligned InfoNCE.
# Same Phase-D config (nw=4, ws=2, depth=2, pred_dim=24, smooth_l1,
# VCLoss(0.25, 0.25), CorrCA-5, per-recording norm) PLUS the new
# auxiliary loss with coeff=0.5, tau=0.1, position bucket = 4 s.
#
# Required env: SEED, EXP_TAG.
# Optional env: NW (4), WS (2), BATCH_SIZE (64), LR (5e-4), WARMUP (5),
#               EPOCHS (100), STIM_COEFF (0.5), STIM_TAU (0.1),
#               STIM_BUCKET (4.0).
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
STIM_COEFF="${STIM_COEFF:-0.5}"
STIM_TAU="${STIM_TAU:-0.1}"
STIM_BUCKET="${STIM_BUCKET:-4.0}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: seed=${SEED} tag=${EXP_TAG} nw=${NW} ws=${WS} bs=${BATCH_SIZE}"
echo "     lr=${LR} warmup=${WARMUP} epochs=${EPOCHS}"
echo "     stim_coeff=${STIM_COEFF} stim_tau=${STIM_TAU} bucket=${STIM_BUCKET}"

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
    --optim.lr="${LR}" \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs="${WARMUP}" \
    --data.num_workers=2 \
    --data.batch_size="${BATCH_SIZE}" \
    --data.n_windows="${NW}" \
    --data.window_size_seconds="${WS}" \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --data.stim_position_bucket_size="${STIM_BUCKET}" \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --loss.stim_nce_coeff="${STIM_COEFF}" \
    --loss.stim_nce_tau="${STIM_TAU}" \
    --logging.wandb_group="issue10_lever1"

CKPT="${EXP_DIR}/latest.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no latest.pth.tar at $CKPT"
    exit 1
fi
echo "Lever 1 ${EXP_TAG} complete!  ckpt=${CKPT}"
