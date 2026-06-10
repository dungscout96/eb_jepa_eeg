#!/bin/bash
#SBATCH --job-name=tier2_d4_nat
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/t2d4nat_%j.out
#SBATCH --error=logs/t2d4nat_%j.err
# ---------------------------------------------------------------
# Deep4Net with NATIVE preprocessing (Schirrmeister 2017 spec):
#   - Single 4-s window per clip (n_windows=1, window_size_seconds=4)
#   - In-graph resample 200 Hz -> 250 Hz
#   - In-graph 4-38 Hz bandpass
#   - In-graph per-window per-channel z-score
#
# Removes the unfair n_times (400 < 441 minimum) disadvantage from
# the original Tier 2 run. Output JSONs land in tier2/native/.
#
# Parameterized by SEED (default 42).
# ---------------------------------------------------------------
set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy
SEED="${SEED:-42}"

cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

OUT_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/tier2/native"
EMB_DIR="${OUT_DIR}/embeddings"
mkdir -p "$OUT_DIR" "$EMB_DIR"
OUT_JSON="${OUT_DIR}/deep4_native_seed${SEED}.json"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/tier2_supervised.py \
    --model=deep4 \
    --seed="${SEED}" \
    --n_windows=1 \
    --window_size_seconds=4 \
    --native_preproc=True \
    --target_sfreq=250.0 \
    --batch_size=64 \
    --num_workers=8 \
    --norm_mode=per_recording \
    --epochs=50 \
    --early_stop_patience=8 \
    --lr=1e-3 \
    --splits=val,test \
    --output_json="${OUT_JSON}" \
    --save_embeddings="${EMB_DIR}"

echo "deep4 native seed=${SEED} done: ${OUT_JSON}"
