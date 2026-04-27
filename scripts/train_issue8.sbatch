#!/bin/bash
#SBATCH --job-name=issue8
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/issue8_%j.out
#SBATCH --error=logs/issue8_%j.err
# ---------------------------------------------------------------
# GitHub issue #8 — CorrCA ablation + temporal-context sweep.
# Phase A: 5-seed paired ablation of Exp 6 minus CorrCA at nw4_ws2.
# Phase B: single-seed temporal sweep in the chosen post-decision config.
#
# Required env: NW, WS, SEED, BATCH_SIZE, USE_CORRCA (0|1), EXP_TAG.
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${NW:?need NW}"
: "${WS:?need WS}"
: "${SEED:?need SEED}"
: "${BATCH_SIZE:?need BATCH_SIZE}"
: "${USE_CORRCA:?need USE_CORRCA (0 or 1)}"
: "${EXP_TAG:?need EXP_TAG}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: nw=${NW} ws=${WS} seed=${SEED} bs=${BATCH_SIZE} corrca=${USE_CORRCA} tag=${EXP_TAG}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# CorrCA flag: pass corrca_filters.npz only when USE_CORRCA=1.
CORRCA_TRAIN_ARG=""
CORRCA_EVAL_ARG=""
if [ "${USE_CORRCA}" = "1" ]; then
    CORRCA_TRAIN_ARG="--data.corrca_filters=corrca_filters.npz"
    CORRCA_EVAL_ARG="--corrca_filters=corrca_filters.npz"
fi

EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8/${EXP_TAG}"
mkdir -p "$EXP_DIR"

# ---- Train ----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs=100 \
    --optim.early_stopping_patience=20 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.num_workers=2 \
    --data.batch_size="${BATCH_SIZE}" \
    --data.n_windows="${NW}" \
    --data.window_size_seconds="${WS}" \
    --data.norm_mode=per_recording \
    ${CORRCA_TRAIN_ARG} \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="issue8_${EXP_TAG}"

CKPT="${EXP_DIR}/best.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no best.pth.tar at $CKPT"
    exit 1
fi

# ---- Eval ----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="$CKPT" \
    --n_windows="${NW}" \
    --window_size_seconds="${WS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    ${CORRCA_EVAL_ARG} \
    --wandb_group="issue8_eval_${EXP_TAG}"

echo "Cell ${EXP_TAG} complete!"
