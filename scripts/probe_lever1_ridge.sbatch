#!/bin/bash
#SBATCH --job-name=lev1pr
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/lever1pr_%j.out
#SBATCH --error=logs/lever1pr_%j.err
# ---------------------------------------------------------------
# Issue #10 Lever 1 — JEPA + Ridge --keep_channels probe-eval against a
# Lever 1 pretrained checkpoint. Uses the same Ridge protocol as PR #15
# (n_passes=20 random clip draws, sklearn Ridge(α=1.0) per stim feature).
#
# Required env: CKPT, EXP_TAG, PROBE_SEED.
# Optional env: NW (4), WS (2), N_PASSES (20).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${CKPT:?need CKPT}"
: "${EXP_TAG:?need EXP_TAG}"
: "${PROBE_SEED:?need PROBE_SEED}"
NW="${NW:-4}"
WS="${WS:-2}"
N_PASSES="${N_PASSES:-20}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: CKPT not found: $CKPT"
    exit 2
fi

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: ckpt=${CKPT}  tag=${EXP_TAG}  probe_seed=${PROBE_SEED}"
echo "     nw=${NW} ws=${WS} n_passes=${N_PASSES}"

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

OUT_JSON="/projects/bbnv/kkokate/eb_jepa_eeg/results/lever1/${EXP_TAG}_seed${PROBE_SEED}.json"
PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/lever1/${EXP_TAG}_seed${PROBE_SEED}"
mkdir -p "$(dirname "$OUT_JSON")" "$PRED_DIR"

PYTHONPATH=. uv run --group eeg python scripts/trivial_ridge_baseline.py \
    --n_windows="${NW}" \
    --window_size_seconds="${WS}" \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --checkpoint="${CKPT}" \
    --keep_channels=True \
    --n_passes="${N_PASSES}" \
    --seed="${PROBE_SEED}" \
    --baseline="lever1_${EXP_TAG}" \
    --output_json="${OUT_JSON}" \
    --save_predictions_dir="${PRED_DIR}"

echo "Lever 1 ${EXP_TAG} probe_seed=${PROBE_SEED} done."
echo "Results JSON: ${OUT_JSON}"
