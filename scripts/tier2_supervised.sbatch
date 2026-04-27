#!/bin/bash
#SBATCH --job-name=tier2_sup
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/tier2_%j.out
#SBATCH --error=logs/tier2_%j.err
# ---------------------------------------------------------------
# Tier 2 supervised end-to-end. Parameterized by:
#   MODEL ∈ {shallow, deep4, eegnet, eegnex}
#   SEED  (default 42)
# ---------------------------------------------------------------
set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${MODEL:?need MODEL (shallow|deep4|eegnet|eegnex)}"
SEED="${SEED:-42}"

cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

OUT_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/tier2"
EMB_DIR="${OUT_DIR}/embeddings"
mkdir -p "$OUT_DIR" "$EMB_DIR"
OUT_JSON="${OUT_DIR}/${MODEL}_seed${SEED}.json"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/tier2_supervised.py \
    --model="${MODEL}" \
    --seed="${SEED}" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --batch_size=64 \
    --num_workers=8 \
    --norm_mode=per_recording \
    --epochs=50 \
    --early_stop_patience=8 \
    --lr=1e-3 \
    --splits=val,test \
    --output_json="${OUT_JSON}" \
    --save_embeddings="${EMB_DIR}"

echo "tier2 ${MODEL} seed=${SEED} done: ${OUT_JSON}"
