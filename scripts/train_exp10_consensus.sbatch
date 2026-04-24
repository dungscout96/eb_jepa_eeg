#!/bin/bash
#SBATCH --job-name=exp10_cons
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp10_%j.out
#SBATCH --error=logs/exp10_%j.err
# ---------------------------------------------------------------
# Exp 10 — Consensus-target JEPA.
# Teacher encoder sees an across-subject ISC-averaged CorrCA-5 signal
# at matched movie time (scripts/compute_consensus.py), student sees the
# subject's own CorrCA-5. JEPA prediction loss unchanged. Rest of the
# stack = Exp 6 sweep winner (pd=16, VCLoss 0.25/0.25, smooth_l1,
# per-recording norm, CorrCA-5).
#
# Required env: SEED.
# Optional:     STD_COEFF (default 0.25), PRED_DIM (default 16),
#               CONSENSUS_PATH (default consensus.npz).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${SEED:?need SEED}"
STD_COEFF="${STD_COEFF:-0.25}"
PRED_DIM="${PRED_DIM:-16}"
CONSENSUS_PATH="${CONSENSUS_PATH:-consensus.npz}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  cfg: std=${STD_COEFF} pd=${PRED_DIM} seed=${SEED} consensus=${CONSENSUS_PATH}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# Per-cell output dir keyed to seed so parallel seed jobs don't collide.
EXP_ID="exp10_std${STD_COEFF}_pd${PRED_DIM}_seed${SEED}"
EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/exp10_consensus/${EXP_ID}"
mkdir -p "$EXP_DIR"

# ---- Train ----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs=100 \
    --optim.early_stopping_patience=20 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim="${PRED_DIM}" \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.num_workers=2 \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --data.consensus_path="${CONSENSUS_PATH}" \
    --loss.std_coeff="${STD_COEFF}" \
    --loss.cov_coeff="${STD_COEFF}" \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="exp10_consensus_${EXP_ID}"

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
    --corrca_filters=corrca_filters.npz \
    --wandb_group="exp10_consensus_eval_${EXP_ID}"

echo "Cell ${EXP_ID} complete!"
