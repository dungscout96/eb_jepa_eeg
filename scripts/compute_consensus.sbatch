#!/bin/bash
#SBATCH --job-name=consensus
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=0
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/consensus_%j.out
#SBATCH --error=logs/consensus_%j.err

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

PYTHONPATH=. uv run --group eeg python scripts/compute_consensus.py \
    --corrca_path corrca_filters.npz \
    --output_path consensus.npz \
    --task ThePresent
