#!/bin/bash
#SBATCH --job-name=cellL
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cellL_%j.out
#SBATCH --error=logs/cellL_%j.err
# ---------------------------------------------------------------
# Cell L — Multi-horizon latent rollout (V-JEPA 2 / TD-JEPA mold).
# Cross-Time mask geometry + horizon-conditioned predictor (learned
# embedding per masked target encoding window-distance from last context
# window). Per-horizon loss decomposition logged to wandb.
#
# Required env: SEED, EXP_TAG.
# Optional: NW (4), WS (2), BATCH_SIZE (64), LR (5e-4), EPOCHS (100),
#           MASK_FRAC (0.25 default — gives more horizons), MASK_POS (tail),
#           HORIZON_MAX (auto = NW-1), DECOMPOSE (true), TASKS (ThePresent),
#           GROUP_TAG.
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
MASK_FRAC="${MASK_FRAC:-0.25}"
MASK_POS="${MASK_POS:-tail}"
HORIZON_MAX="${HORIZON_MAX:-$((NW - 1))}"
DECOMPOSE="${DECOMPOSE:-true}"
TASKS="${TASKS:-ThePresent}"
TASKS="${TASKS//+/,}"
GROUP_TAG="${GROUP_TAG:-issue10_cellL}"

echo "Cfg: seed=${SEED} tag=${EXP_TAG} nw=${NW} ws=${WS} bs=${BATCH_SIZE}"
echo "     epochs=${EPOCHS} mask_frac=${MASK_FRAC} mask_pos=${MASK_POS}"
echo "     horizon_max=${HORIZON_MAX} decompose=${DECOMPOSE}"

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
    --masking.geometry=cross_time \
    --masking.mask_window_fraction="${MASK_FRAC}" \
    --masking.mask_position="${MASK_POS}" \
    --masking.horizon_embed_max="${HORIZON_MAX}" \
    --masking.horizon_loss_decompose="${DECOMPOSE}" \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="${GROUP_TAG}"

CKPT="${EXP_DIR}/latest.pth.tar"
[ -f "$CKPT" ] || { echo "ERROR: no latest.pth.tar at $CKPT"; exit 1; }
echo "Cell L ${EXP_TAG} complete!  ckpt=${CKPT}"
