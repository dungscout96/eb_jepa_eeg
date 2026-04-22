#!/bin/bash
#SBATCH --job-name=exp7_clip_align
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# ---------------------------------------------------------------
# Experiment 7: Exp 6 baseline + CLIP-style EEG↔movie alignment.
#
# Identical to Exp 6 (CorrCA 5-component preprocessing, per-recording
# normalization, pred_dim=24 bottleneck, VCLoss(0.25, 0.25), smooth_l1)
# with an added auxiliary InfoNCE loss that pulls EEG window embeddings
# toward frozen CLIP ViT-L/14 embeddings of the corresponding movie
# frames. Target file: movie_annotation/output/The_Present/clip_embeddings.npz
# (4877 × 512, frame-indexed) — already present on Delta.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/train_exp7_clip_align.sbatch
#
# Override with: sbatch --export=ALL,EXTRA_ARGS="--optim.epochs=50" ...
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

echo "========================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $SLURM_NODELIST"
echo "GPUs:           $SLURM_GPUS_ON_NODE"
echo "Preprocess dir: $HBN_PREPROCESS_DIR"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

GIT_BRANCH="${GIT_BRANCH:-kkokate/exp7-clip-align}"
git fetch origin && git checkout "$GIT_BRANCH" && git pull --ff-only origin "$GIT_BRANCH"

CLIP_NPZ="${CLIP_NPZ:-movie_annotation/output/The_Present/clip_embeddings.npz}"
CLIP_COEFF="${CLIP_COEFF:-0.5}"
CLIP_TEMP="${CLIP_TEMP:-0.1}"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --optim.epochs=100 \
    --optim.early_stopping_patience=20 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.num_workers=2 \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --data.clip_embeddings="${CLIP_NPZ}" \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --loss.clip_align_coeff="${CLIP_COEFF}" \
    --loss.clip_align.temperature="${CLIP_TEMP}" \
    --logging.wandb_group=exp7_clip_align \
    ${EXTRA_ARGS:-}

echo "Training complete!"
