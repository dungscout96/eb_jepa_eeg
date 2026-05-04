#!/bin/bash
#SBATCH --job-name=PDeval
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/PDeval_%j.out
#SBATCH --error=logs/PDeval_%j.err
# ---------------------------------------------------------------
# Canonical unified_probe_eval driver — runs scripts/unified_probe_eval.py
# on a JEPA encoder checkpoint. Outputs a single JSON with all 18 headline
# metrics (4 stim reg/cls + subject age/sex + movie_id top-1/5).
#
# Required env: CKPT, EXP_TAG.
# Optional: NW (4), WS (2), N_PASSES (20), PROBE_SEED (42), GROUP_TAG.
# ---------------------------------------------------------------

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${CKPT:?need CKPT}"
: "${EXP_TAG:?need EXP_TAG}"
NW="${NW:-4}"
WS="${WS:-2}"
N_PASSES="${N_PASSES:-20}"
PROBE_SEED="${PROBE_SEED:-42}"

[ -f "$CKPT" ] || { echo "ERROR: CKPT not found: $CKPT"; exit 2; }

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

OUT_JSON="/projects/bbnv/kkokate/eb_jepa_eeg/results/unified/${EXP_TAG}_seed${PROBE_SEED}.json"
PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/unified/${EXP_TAG}_seed${PROBE_SEED}"
mkdir -p "$(dirname "$OUT_JSON")" "$PRED_DIR"

echo "Cfg: ckpt=${CKPT} tag=${EXP_TAG} nw=${NW} ws=${WS} n_passes=${N_PASSES} probe_seed=${PROBE_SEED}"
echo "Out: ${OUT_JSON}"

PYTHONPATH=. uv run --group eeg python scripts/unified_probe_eval.py \
    --checkpoint="${CKPT}" \
    --n_windows="${NW}" --window_size_seconds="${WS}" \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --n_passes="${N_PASSES}" \
    --seed="${PROBE_SEED}" \
    --output_json="${OUT_JSON}" \
    --save_predictions_dir="${PRED_DIR}"

echo "PDeval ${EXP_TAG} probe_seed=${PROBE_SEED} complete: ${OUT_JSON}"
