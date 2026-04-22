#!/bin/bash
#SBATCH --job-name=exp8_eval
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp8_eval_%j.out
#SBATCH --error=logs/exp8_eval_%j.err

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi

GIT_BRANCH="${GIT_BRANCH:-kkokate/exp8-xsubj-jepa}"
git fetch origin
git checkout "$GIT_BRANCH"
git pull --ff-only origin "$GIT_BRANCH"

CKPT="${CKPT:?Provide CKPT=/path/to/best.pth.tar}"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="$CKPT" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --wandb_group=exp8_xsubj_eval

echo "Eval complete!"
