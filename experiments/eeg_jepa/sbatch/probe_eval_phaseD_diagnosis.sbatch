#!/bin/bash
#SBATCH --job-name=phaseD_diag_probe
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phaseD_diag_probe_%j.out
#SBATCH --error=logs/phaseD_diag_probe_%j.err
# ---------------------------------------------------------------
# Post-hoc probe eval on the phaseD nw4ws2 diagnostic checkpoint.
# Uses latest.pth.tar (full 100-ep budget, matching Phase E pattern).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

CKPT="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/phaseD_nw4ws2_baseline_s2025_diagnosis/latest.pth.tar"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Ckpt: $CKPT"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

GIT_BRANCH="${GIT_BRANCH:-main}"
git fetch origin && git checkout "$GIT_BRANCH" && git pull --ff-only origin "$GIT_BRANCH"

PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.probe_eval \
    --checkpoint="${CKPT}" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --batch_size=64 \
    --num_workers=2 \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --splits=val,test \
    --wandb_run_id=n97mlk9k \
    --wandb_group=phaseD_nw4ws2_baseline_s2025_diagnosis \
    --seed=2025

echo "Probe eval complete!"
