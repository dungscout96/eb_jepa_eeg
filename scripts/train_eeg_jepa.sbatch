#!/bin/bash
#SBATCH --job-name=eeg_jepa_train
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
# EEG JEPA Training on Delta (NCSA)
#
# Trains a masked JEPA encoder (V-JEPA style) on preprocessed HBN
# EEG data (R2/R3/R4 train, R1 val, R6 test).
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/train_eeg_jepa.sbatch
#
# Override parameters (append to sbatch or set env vars):
#   sbatch --export=ALL,EXTRA_ARGS="--optim.epochs=200" scripts/train_eeg_jepa.sbatch
# ---------------------------------------------------------------

set -euo pipefail

# --- Paths ---
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

echo "========================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $SLURM_NODELIST"
echo "GPUs:           $SLURM_GPUS_ON_NODE"
echo "Preprocess dir: $HBN_PREPROCESS_DIR"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"

# Load conda and activate environment
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

# wandb: source API key if saved in ~/.wandb_key
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# Pull latest code
git fetch origin && git checkout main && git pull --ff-only

# Experiment 6: CorrCA spatial filters (129 → 5 ISC-maximizing components)
# + per-recording normalization. Requires offline CorrCA computation:
#   sbatch scripts/compute_corrca.sbatch
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
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group=exp6_corrca \
    ${EXTRA_ARGS:-}

echo "Training complete!"
