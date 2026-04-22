#!/bin/bash
#SBATCH --job-name=exp7d_siglip
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
# Experiment 7d: SigLIP pairwise sigmoid alignment loss (Zhai 2023).
#   - Identical to Exp 7b (CLIP target, per-window, coeff=1.5) except
#     loss_type=sigmoid with learnable bias init=-10.
#
# Hypothesis: sigmoid loss should converge more stably at B=256 and
# lift alignment accuracy — InfoNCE plateaued at 6× chance in Exp 7b.
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

GIT_BRANCH="${GIT_BRANCH:-kkokate/exp7d-siglip}"
git fetch origin && git checkout "$GIT_BRANCH" && git pull --ff-only origin "$GIT_BRANCH"

CLIP_NPZ="${CLIP_NPZ:-movie_annotation/output/The_Present/clip_embeddings.npz}"
CLIP_COEFF="${CLIP_COEFF:-1.5}"
CLIP_TEMP="${CLIP_TEMP:-0.1}"
SEED="${SEED:-2025}"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
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
    --loss.clip_align.per_window=true \
    --loss.clip_align.loss_type=sigmoid \
    --logging.wandb_group=exp7d_siglip \
    ${EXTRA_ARGS:-}

echo "Training complete!"
