#!/bin/bash
#SBATCH --job-name=canonical_repl
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/canonical_repl_%j.out
#SBATCH --error=logs/canonical_repl_%j.err
# ---------------------------------------------------------------
# Spec-faithful single-seed replication: pretrain → canonical probe_eval → bootstrap.
#
# Spec (from refactor verification, 2026-05):
#   nw=2 ws=4 (8s clips) | encoder_depth=2 | embed_dim=64 | predictor_embed_dim=24
#   lr=5e-4 cosine→1e-6 | warmup=5 ep | 100 epochs | bs=64 | early_stop_patience=0
#   per_recording norm | CorrCA 129→5 filters | VCLoss(0.25, 0.25) | smooth_l1
#   probe heads: sklearn Ridge(α=1) / LogisticRegression(C=1, lbfgs)
#   n_passes=20 augmentation | probe_seed=42 fixed across encoder seeds
#   bootstrap B=2000 | L1 = single raw r/AUC on flat 2160 | L2 = bootstrap mean+CI
#
# Required env vars:
#   SEED (int)  — encoder seed (one of {42, 123, 456, 789, 2025})
# ---------------------------------------------------------------

set -euo pipefail

if [ -z "${SEED:-}" ]; then
    echo "ERROR: SEED env var must be set" >&2
    exit 1
fi

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

EXP_TAG="canonical_nw2ws4_s${SEED}"
EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/${EXP_TAG}"
PREDS_DIR="${EXP_DIR}/canonical_preds"
L2_JSON="${EXP_DIR}/canonical_L2.json"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Seed: $SEED  Tag: $EXP_TAG"

# Caller (neurolab wrapper or direct sbatch from repo root) is responsible for
# cwd. Don't trust SLURM_SUBMIT_DIR -- under neurolab it points at HOME, not the
# repo. Fail loudly if we're not in the right place.
if [ ! -f experiments/eeg_jepa/train.py ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd), expected to contain experiments/eeg_jepa/train.py)" >&2
    exit 1
fi

mkdir -p logs "$EXP_DIR" "$PREDS_DIR"

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# Inner sbatch deliberately does NOT touch git. The outer sweep launcher
# expects the repo on Delta to already be on the target branch (manually
# pulled once before submission) to avoid concurrent ref-lock races when
# multiple seeds launch at once. Print the SHA we're running on for the log.
echo "Running on commit: $(git rev-parse HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

# ---------- Stage 1: pretraining --------------------------------
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/train.py \
    --meta.seed=${SEED} \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs=100 \
    --optim.early_stopping_patience=0 \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --data.num_workers=4 \
    --data.batch_size=64 \
    --data.n_windows=2 \
    --data.window_size_seconds=4 \
    --data.task=ThePresent \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --eval.auto_run=false \
    --logging.wandb_group=canonical_nw2ws4

CKPT="${EXP_DIR}/latest.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: pretraining produced no checkpoint at $CKPT" >&2
    exit 1
fi
echo "Pretraining complete: ${CKPT}"

# ---------- Stage 2: canonical probe_eval (sklearn) -------------
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.probe_eval_canonical \
    --checkpoint="${CKPT}" \
    --n_windows=2 \
    --window_size_seconds=4 \
    --batch_size=64 \
    --num_workers=2 \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --n_passes=20 \
    --probe_seed=42 \
    --save_predictions_dir="${PREDS_DIR}" \
    --wandb_group=canonical_nw2ws4_probe \
    --seed=${SEED}

PRED_NPZ="${PREDS_DIR}/preds_seed${SEED}.npz"
if [ ! -f "$PRED_NPZ" ]; then
    echo "ERROR: probe_eval produced no NPZ at $PRED_NPZ" >&2
    exit 1
fi
echo "Probe eval complete: ${PRED_NPZ}"

# ---------- Stage 3: recording-level bootstrap → L2 JSON --------
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.bootstrap_canonical \
    --predictions_npz="${PRED_NPZ}" \
    --n_bootstrap=2000 \
    --seed=${SEED} \
    --out_json="${L2_JSON}"

echo "Bootstrap complete: ${L2_JSON}"
echo "Done seed=${SEED}"
