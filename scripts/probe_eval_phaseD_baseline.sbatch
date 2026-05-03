#!/bin/bash
#SBATCH --job-name=phaseD_base_probe
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/phaseD_base_probe_%j.out
#SBATCH --error=logs/phaseD_base_probe_%j.err
# ---------------------------------------------------------------
# Post-hoc probe eval for one phaseD nw4ws2 baseline checkpoint.
# Required env: SEED, WANDB_RUN_ID
# ---------------------------------------------------------------

set -euo pipefail

: "${SEED:?need SEED}"
: "${WANDB_RUN_ID:?need WANDB_RUN_ID}"

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

CKPT="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8/phaseD_nw4ws2_baseline_s${SEED}/latest.pth.tar"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Seed: $SEED  Ckpt: $CKPT  WandB: $WANDB_RUN_ID"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.probe_eval \
    --checkpoint="${CKPT}" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --batch_size=64 \
    --num_workers=2 \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --splits=val,test \
    --wandb_run_id="${WANDB_RUN_ID}" \
    --wandb_group=phaseD_nw4ws2_baseline \
    --seed=2025

echo "Probe eval seed=$SEED complete!"
