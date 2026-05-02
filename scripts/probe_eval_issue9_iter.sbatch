#!/bin/bash
#SBATCH --job-name=issue9_probe
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/issue9probe_%j.out
#SBATCH --error=logs/issue9probe_%j.err
# Required env: CKPT, EXP_TAG, PROBE_SEED, ITER_TAG.

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${CKPT:?need CKPT}"
: "${EXP_TAG:?need EXP_TAG}"
: "${PROBE_SEED:?need PROBE_SEED}"
: "${ITER_TAG:?need ITER_TAG}"

NW="${NW:-4}"
WS="${WS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"

[ -f "$CKPT" ] || { echo "ERROR: CKPT not found: $CKPT"; exit 2; }

PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/issue9_${ITER_TAG}/${EXP_TAG}"
mkdir -p "$PRED_DIR"

echo "Job ID: $SLURM_JOB_ID  Iter: ${ITER_TAG}  Tag: ${EXP_TAG}  ProbeSeed: ${PROBE_SEED}"
echo "Predictions: ${PRED_DIR}"

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

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="${CKPT}" \
    --n_windows="${NW}" \
    --window_size_seconds="${WS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --seed="${PROBE_SEED}" \
    --save_predictions_dir="${PRED_DIR}" \
    --wandb_group="issue9_${ITER_TAG}_probe_${EXP_TAG}"

echo "Issue #9 ${ITER_TAG} probe ${EXP_TAG} complete!"
