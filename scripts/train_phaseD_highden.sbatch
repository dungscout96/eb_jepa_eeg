#!/bin/bash
#SBATCH --job-name=PDhd
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/PDhd_%j.out
#SBATCH --error=logs/PDhd_%j.err
# ---------------------------------------------------------------
# Phase D at HIGH gradient density — control for Cell B high-density.
# Same architecture, same epochs, same batch density, NO xsub aux.
# Disambiguates "Cell B objective helps narrative" from "more grad
# steps help narrative".
#
# Trick: enable data.stim_aligned_flat_index=True so __len__ returns
# 68k flat clips (instead of 700 recordings) → standard random sampler
# yields ~1073 batches/epoch automatically. NOT using stim_aligned_sampler
# (no K=2 pairing, no xsub).
#
# Required env: SEED, EXP_TAG.
# Optional: NW (4), WS (2), BATCH_SIZE (64), LR (5e-4), EPOCHS (30 to
#           match Cell B high-density 32k steps), GROUP_TAG.
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
EPOCHS="${EPOCHS:-30}"
TASKS="${TASKS:-ThePresent}"
TASKS="${TASKS//+/,}"
GROUP_TAG="${GROUP_TAG:-issue10_phaseD_highden}"

echo "Cfg: PHASE D HIGH-DENSITY CONTROL"
echo "     seed=${SEED} tag=${EXP_TAG} nw=${NW} ws=${WS} bs=${BATCH_SIZE}"
echo "     lr=${LR} epochs=${EPOCHS}  (matches Cell B high-density 30 ep × ~1073 batches/ep ≈ 32k steps)"

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
    --data.tasks="${TASKS}" \
    --data.stim_aligned_flat_index=True \
    --data.stim_aligned_sampler=False \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="${GROUP_TAG}"

CKPT="${EXP_DIR}/latest.pth.tar"
[ -f "$CKPT" ] || { echo "ERROR: no latest.pth.tar at $CKPT"; exit 1; }
echo "Phase D HIGH-DENSITY ${EXP_TAG} complete!  ckpt=${CKPT}"
