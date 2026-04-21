#!/bin/bash
#SBATCH --job-name=corrca
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/corrca_%j.out
#SBATCH --error=logs/corrca_%j.err

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

echo "========================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $SLURM_NODELIST"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

PYTHONPATH=. uv run --group eeg python scripts/compute_corrca.py \
    --output_path corrca_filters.npz \
    --n_components 5 \
    --task ThePresent \
    --n_time_bins 100

echo "CorrCA complete!"
