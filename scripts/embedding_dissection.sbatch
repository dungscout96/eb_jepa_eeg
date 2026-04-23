#!/bin/bash
#SBATCH --job-name=emb_diss
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/emb_diss_%j.out
#SBATCH --error=logs/emb_diss_%j.err

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

CKPT="${CKPT:?Provide CKPT=/path/to/best.pth.tar}"
OUT_DIR="${OUT_DIR:-embedding_dissection_exp6}"
CLIPS_PER_REC="${CLIPS_PER_REC:-8}"

PYTHONPATH=. uv run --group eeg python scripts/embedding_dissection.py \
    --checkpoint="$CKPT" \
    --out_dir="$OUT_DIR" \
    --clips_per_rec="$CLIPS_PER_REC" \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz

echo "Embedding dissection complete!"
