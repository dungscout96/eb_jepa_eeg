#!/bin/bash
#SBATCH --job-name=exp13_pe
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/exp13_pe_%j.out
#SBATCH --error=logs/exp13_pe_%j.err
# ---------------------------------------------------------------
# Frozen-encoder probe_eval on a single Exp 13 seed checkpoint.
# Encoder-only protocol (no CorrCA residual concat), matching the
# Exp 6 5-seed probe_eval used for the experiments.md comparison.
#
# Required env: SEED.
# Optional:     CKPT (default = canonical exp13_residual path).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${SEED:?need SEED}"
CKPT_DEFAULT="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/exp13_residual/exp13_std0.25_pd16_seed${SEED}/best.pth.tar"
CKPT="${CKPT:-$CKPT_DEFAULT}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  seed=${SEED}"
echo "Checkpoint: $CKPT"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="${CKPT}" \
    --corrca_filters=corrca_filters.npz \
    --norm_mode=per_recording \
    --n_windows=4 \
    --window_size_seconds=2 \
    --batch_size=64 \
    --num_workers=4 \
    --probe_epochs=20 \
    --subject_probe_epochs=100 \
    --splits=val,test \
    --wandb_group=probe_eval_exp13 \
    --seed="${SEED}"

echo "Probe eval seed=${SEED} complete."
