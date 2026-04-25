#!/bin/bash
#SBATCH --job-name=exp13_res
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp13_%j.out
#SBATCH --error=logs/exp13_%j.err
# ---------------------------------------------------------------
# Exp 13 — Co-trained CorrCA-residual probe.
# Same JEPA stack as Exp 6 sweep winner (pd=16, std=0.25, smooth_l1,
# per-rec norm, CorrCA-5). The change is *probe input only*: concat raw
# CorrCA-5 box-pool features (D=500) with encoder output (D=64) before
# the online MovieFeatureHead. Probe co-trains with JEPA, so this is
# apples-to-apples vs the experiments.md Exp 6 5-seed online-probe ref.
#
# Required env: SEED.
# Optional:     STD_COEFF (default 0.25), PRED_DIM (default 16).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${SEED:?need SEED}"
STD_COEFF="${STD_COEFF:-0.25}"
PRED_DIM="${PRED_DIM:-16}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  cfg: std=${STD_COEFF} pd=${PRED_DIM} seed=${SEED}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

EXP_ID="exp13_std${STD_COEFF}_pd${PRED_DIM}_seed${SEED}"
EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/exp13_residual/${EXP_ID}"
mkdir -p "$EXP_DIR"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs=100 \
    --optim.early_stopping_patience=20 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim="${PRED_DIM}" \
    --model.probe_corrca_residual=true \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.num_workers=2 \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --loss.std_coeff="${STD_COEFF}" \
    --loss.cov_coeff="${STD_COEFF}" \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="exp13_residual_${EXP_ID}"

echo "Cell ${EXP_ID} complete!"
