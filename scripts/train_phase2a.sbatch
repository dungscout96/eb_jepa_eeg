#!/bin/bash
#SBATCH --job-name=p2a
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/p2a_%j.out
#SBATCH --error=logs/p2a_%j.err
# ---------------------------------------------------------------
# Exp 6 Phase 2A — OFAT hyperparameter screening.
# Anchor config = Phase 1 best (std=cov=0.25, pd=16, seed=42).
# One varying knob per job (set via env var, others fall back to
# anchor). All Phase 2A knobs listed below; missing env vars take
# the anchor default.
#
# Required: KNOB (name of the varying knob), SEED.
# All KNOB_* env vars optional; defaults = anchor values.
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${KNOB:?need KNOB identifier (e.g. patch_size, lr, ...)}"
: "${VALUE_TAG:?need VALUE_TAG (short string used in EXP_ID, must be filesystem-safe)}"
: "${SEED:?need SEED}"

# --- Anchor defaults (Phase 1 best cell) ---
: "${STD_COEFF:=0.25}"
: "${COV_COEFF:=0.25}"
: "${PRED_DIM:=16}"
: "${ENCODER_DEPTH:=2}"
: "${PREDICTOR_DEPTH:=2}"
: "${PATCH_SIZE:=50}"
: "${PATCH_OVERLAP:=20}"
: "${N_PRED_MASKS_LONG:=2}"
: "${LONG_PATCH_SCALE:=[0.5,1.0]}"
: "${LR:=5e-4}"
: "${EMA_MOMENTUM_END:=1.0}"
: "${CORRCA_FILE:=corrca_filters.npz}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "cfg: knob=${KNOB} seed=${SEED} std=${STD_COEFF} cov=${COV_COEFF} pd=${PRED_DIM} enc_depth=${ENCODER_DEPTH} pred_depth=${PREDICTOR_DEPTH} patch_size=${PATCH_SIZE} patch_overlap=${PATCH_OVERLAP} n_masks_long=${N_PRED_MASKS_LONG} long_patch=${LONG_PATCH_SCALE} lr=${LR} ema_end=${EMA_MOMENTUM_END} corrca=${CORRCA_FILE}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# NOTE: no git checkout here. Submitter must pre-checkout the branch on Delta.
# (Lesson from commit 2eaf5c0: .git/index.lock race with concurrent jobs.)

# Each OFAT cell varies exactly one knob from anchor, so KNOB + VALUE_TAG +
# SEED is sufficient to disambiguate output dirs and wandb groups.
EXP_ID="p2a_${KNOB}_${VALUE_TAG}_s${SEED}"
EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/phase2a/${EXP_ID}"
mkdir -p "$EXP_DIR"

# ---- Train ----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs=100 \
    --optim.early_stopping_patience=20 \
    --optim.lr="${LR}" \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.num_workers=2 \
    --data.norm_mode=per_recording \
    --data.corrca_filters="${CORRCA_FILE}" \
    --model.encoder_depth="${ENCODER_DEPTH}" \
    --model.predictor_depth="${PREDICTOR_DEPTH}" \
    --model.predictor_embed_dim="${PRED_DIM}" \
    --model.patch_size="${PATCH_SIZE}" \
    --model.patch_overlap="${PATCH_OVERLAP}" \
    --model.ema_momentum_end="${EMA_MOMENTUM_END}" \
    --masking.n_pred_masks_long="${N_PRED_MASKS_LONG}" \
    --masking.long_patch_scale="${LONG_PATCH_SCALE}" \
    --loss.std_coeff="${STD_COEFF}" \
    --loss.cov_coeff="${COV_COEFF}" \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="phase2a_${EXP_ID}"

CKPT="${EXP_DIR}/best.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no best.pth.tar at $CKPT"
    exit 1
fi

# ---- Eval ----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="$CKPT" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters="${CORRCA_FILE}" \
    --wandb_group="phase2a_eval_${EXP_ID}"

echo "Cell ${EXP_ID} complete!"
