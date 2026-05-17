#!/bin/bash
#SBATCH --job-name=canonical_probe
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/canonical_probe_%j.out
#SBATCH --error=logs/canonical_probe_%j.err
# ---------------------------------------------------------------
# Probe + bootstrap only — for re-running the canonical eval stage
# against an existing latest.pth.tar without redoing pretraining.
#
# Required env var:
#   SEED (int)
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
CKPT="${EXP_DIR}/latest.pth.tar"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Seed: $SEED  Tag: $EXP_TAG"

if [ ! -f experiments/eeg_jepa/train.py ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint at $CKPT — run canonical_replication.sbatch first" >&2
    exit 1
fi

mkdir -p logs "$PREDS_DIR"

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

echo "Running on commit: $(git rev-parse HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Checkpoint: $CKPT"

# ---------- canonical probe_eval (sklearn) ----------------------
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

# ---------- recording-level bootstrap → L2 JSON ----------------
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.bootstrap_canonical \
    --predictions_npz="${PRED_NPZ}" \
    --n_bootstrap=2000 \
    --seed=${SEED} \
    --out_json="${L2_JSON}"

echo "Bootstrap complete: ${L2_JSON}"
echo "Done seed=${SEED}"
