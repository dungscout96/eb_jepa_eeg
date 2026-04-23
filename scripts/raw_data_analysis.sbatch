#!/bin/bash
#SBATCH --job-name=raw_data_an
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/raw_data_an_%j.out
#SBATCH --error=logs/raw_data_an_%j.err

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

GIT_BRANCH="${GIT_BRANCH:-kkokate/exp6-embedding-analysis}"
git fetch origin
git checkout "$GIT_BRANCH"
git pull --ff-only origin "$GIT_BRANCH"

OUT_DIR="${OUT_DIR:-raw_data_analysis_val}"
N_TIME_BINS="${N_TIME_BINS:-20}"
MAX_RECS="${MAX_RECS:-100}"

PYTHONPATH=. uv run --group eeg python scripts/raw_data_analysis.py \
    --out_dir="$OUT_DIR" \
    --n_time_bins="$N_TIME_BINS" \
    --max_recordings="$MAX_RECS" \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz

echo "Raw data analysis complete!"
