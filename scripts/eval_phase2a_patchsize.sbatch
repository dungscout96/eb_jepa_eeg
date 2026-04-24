#!/bin/bash
#SBATCH --job-name=p2a_eval
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/p2a_eval_%j.out
#SBATCH --error=logs/p2a_eval_%j.err
# ---------------------------------------------------------------
# Re-run probe_eval for the two Phase 2A patch_size cells that
# trained successfully but failed at eval (before the probe_eval
# patch_size-inference fix in commit 1be9394).
# ---------------------------------------------------------------

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
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# Both cells sequentially.
for EXP_ID in p2a_patch_size_25_s42 p2a_patch_size_100_s42; do
    echo "=== Re-eval: ${EXP_ID} ==="
    CKPT="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/phase2a/${EXP_ID}/best.pth.tar"
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: checkpoint missing: $CKPT"
        exit 1
    fi
    # cfg: header line so aggregate_phase2a.py can parse. Reconstruct the
    # original Phase 2A cfg for each cell.
    if [ "$EXP_ID" = "p2a_patch_size_25_s42" ]; then
        echo "cfg: knob=patch_size seed=42 std=0.25 cov=0.25 pd=16 enc_depth=2 pred_depth=2 patch_size=25 patch_overlap=20 n_masks_long=2 long_patch=[0.5,1.0] lr=5e-4 ema_end=1.0 corrca=corrca_filters.npz"
    else
        echo "cfg: knob=patch_size seed=42 std=0.25 cov=0.25 pd=16 enc_depth=2 pred_depth=2 patch_size=100 patch_overlap=20 n_masks_long=2 long_patch=[0.5,1.0] lr=5e-4 ema_end=1.0 corrca=corrca_filters.npz"
    fi

    PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
        --checkpoint="$CKPT" \
        --n_windows=4 \
        --window_size_seconds=2 \
        --num_workers=2 \
        --splits=val,test \
        --norm_mode=per_recording \
        --corrca_filters=corrca_filters.npz \
        --wandb_group="phase2a_eval_${EXP_ID}"
    echo "=== Done: ${EXP_ID} ==="
done

echo "Both re-evals complete."
