#!/bin/bash
#SBATCH --job-name=tier1_ext_analysis
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tier1ext_%j.out
#SBATCH --error=logs/tier1ext_%j.err
# ---------------------------------------------------------------
# Run the extended Tier-1 analyses (bootstrap CIs, permutation
# tests, CKA, stacked probe, data-efficiency curve) over the
# embeddings dumped by tier1_baselines.py --save_embeddings.
#
# Outputs JSON at /projects/bbnv/kkokate/eb_jepa_eeg/tier1/extended_analysis.json
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

EMB_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/tier1/embeddings"
OUT_JSON="/projects/bbnv/kkokate/eb_jepa_eeg/tier1/extended_analysis.json"

PYTHONPATH=. uv run --group eeg python \
    experiments/eeg_jepa/tier1_extended_analysis.py \
    --emb_dir="${EMB_DIR}" \
    --out_json="${OUT_JSON}" \
    --n_boot=1000 \
    --n_perm=1000

echo "Extended analysis written to ${OUT_JSON}"
