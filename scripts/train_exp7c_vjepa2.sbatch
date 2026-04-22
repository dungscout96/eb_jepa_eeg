#!/bin/bash
#SBATCH --job-name=exp7c_vjepa2
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
# Experiment 7c: V-JEPA 2 alignment target instead of CLIP.
#   - target: movie_annotation/output/The_Present/vjepa2_embeddings.npz
#     (406 tokens × 1408-d @ 2Hz, timestamp-keyed)
#   - per-window token match via window-center time + nearest timestamp
#   - everything else identical to Exp 7b (CorrCA, per-recording norm,
#     predictor bottleneck, VCLoss, per_window InfoNCE, coeff=1.5)
#
# Hypothesis: video-native target should rescue narrative and contrast
# where single-frame CLIP under-weighted temporal/fast-local structure.
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

GIT_BRANCH="${GIT_BRANCH:-kkokate/exp7c-vjepa2}"
git fetch origin && git checkout "$GIT_BRANCH" && git pull --ff-only origin "$GIT_BRANCH"

VJEPA2_NPZ="${VJEPA2_NPZ:-movie_annotation/output/The_Present/vjepa2_embeddings.npz}"
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
    --data.vjepa2_embeddings="${VJEPA2_NPZ}" \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --loss.clip_align_coeff="${CLIP_COEFF}" \
    --loss.clip_align.temperature="${CLIP_TEMP}" \
    --loss.clip_align.per_window=true \
    --logging.wandb_group=exp7c_vjepa2 \
    ${EXTRA_ARGS:-}

echo "Training complete!"
