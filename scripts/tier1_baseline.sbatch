#!/bin/bash
#SBATCH --job-name=tier1_baseline
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tier1_%j.out
#SBATCH --error=logs/tier1_%j.err
# ---------------------------------------------------------------
# Tier 1 baseline runner on Delta. Parameterized by env:
#
#   BASELINE ∈ {raw_corrca, psd_band, random_init}
#   SEED     (default 42)
#
# Writes metrics JSON to:
#   /projects/bbnv/kkokate/eb_jepa_eeg/tier1/<baseline>_seed<seed>.json
#
# Expected wall time per cell:
#   raw_corrca : ~15 min (CPU-bound feature extraction, small probe)
#   psd_band   : ~30 min (129-ch Welch x 28k clips)
#   random_init: ~30 min (full transformer forward, no gradients)
#
# The submitter is expected to have the correct branch checked out
# before calling sbatch (do NOT checkout here — races the index lock).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${BASELINE:?need BASELINE (raw_corrca|psd_band|random_init|exp6|exp6_residual)}"
SEED="${SEED:-42}"
# Optional: path to Exp-6 checkpoint, required for BASELINE=exp6 or exp6_residual.
CKPT="${CKPT:-}"

echo "========================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Baseline: $BASELINE"
echo "Seed:     $SEED"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi

OUT_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/tier1"
EMB_DIR="${OUT_DIR}/embeddings"
mkdir -p "$OUT_DIR" "$EMB_DIR"
TAG="${TAG:-}"
if [ -n "$TAG" ]; then
    OUT_JSON="${OUT_DIR}/${TAG}_${BASELINE}_seed${SEED}.json"
else
    OUT_JSON="${OUT_DIR}/${BASELINE}_seed${SEED}.json"
fi

CHECKPOINT_ARG=""
if [ "$BASELINE" = "exp6" ] || [ "$BASELINE" = "exp6_residual" ]; then
    : "${CKPT:?BASELINE=$BASELINE needs CKPT=/path/to/best.pth.tar}"
    CHECKPOINT_ARG="--checkpoint=${CKPT}"
fi

# Exp-6-matching hyperparams:
#   n_windows=4, window_size_seconds=2, per_recording norm,
#   CorrCA-5 for raw_corrca + random_init + exp6, off for psd_band.
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/tier1_baselines.py \
    --baseline="${BASELINE}" \
    --seed="${SEED}" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --batch_size=64 \
    --num_workers=8 \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --probe_epochs=20 \
    --probe_lr=1e-3 \
    --subject_probe_epochs=100 \
    --subject_probe_lr=1e-3 \
    --splits=val,test \
    --output_json="${OUT_JSON}" \
    --save_embeddings="${EMB_DIR}" \
    ${CHECKPOINT_ARG}

echo "Tier 1 ${BASELINE} seed=${SEED} complete: ${OUT_JSON}"
