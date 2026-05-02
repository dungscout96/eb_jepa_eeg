#!/bin/bash
#SBATCH --job-name=phaseD_diag
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/phaseD_diag_%j.out
#SBATCH --error=logs/phaseD_diag_%j.err
# ---------------------------------------------------------------
# Diagnostic re-run of phaseD nw4_ws2_baseline (seed 2025) with the
# pred-target diagnostic metrics added in eb_jepa/jepa.py:
#   train_step/pred_target_cosim   — scale-invariant prediction quality
#   train_step/target_var          — EMA target per-dim variance
#   train_step/pred_var            — predictor output per-dim variance
#   train_step/pred_loss_norm      — pred_loss / target_var
#
# Question: pred_loss rises while val regression improves — is the
# pretext task meaningful, and should we scale pretraining?
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

SEED=2025
EXP_TAG="phaseD_nw4ws2_baseline_s${SEED}_diagnosis"
EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/${EXP_TAG}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Tag: $EXP_TAG"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$EXP_DIR"

# dtyoung Delta env: uv-managed .venv (no conda env). uv lives in ~/.local/bin.
export PATH="$HOME/.local/bin:$PATH"

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

GIT_BRANCH="${GIT_BRANCH:-main}"
git fetch origin && git checkout "$GIT_BRANCH" && git pull --ff-only origin "$GIT_BRANCH"

# Mirrors phaseD nw4_ws2_baseline (commit 8391ad6) exactly: nw=4 ws=2,
# lr=5e-4, warmup=5, bs=64, encoder_depth=2, predictor_embed_dim=24,
# per_recording norm, CorrCA filters, VC(0.25,0.25), smooth_l1, patience=0.
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed=${SEED} \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs=100 \
    --optim.early_stopping_patience=0 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.num_workers=2 \
    --data.batch_size=64 \
    --data.n_windows=4 \
    --data.window_size_seconds=2 \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group=phaseD_nw4ws2_baseline_s2025_diagnosis

CKPT="${EXP_DIR}/latest.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no latest.pth.tar at $CKPT"
    exit 1
fi
echo "Training complete!  ckpt=${CKPT}"
