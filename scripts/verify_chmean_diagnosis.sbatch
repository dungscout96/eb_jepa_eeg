#!/bin/bash
#SBATCH --job-name=verify_chmean
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/verify_chmean_%j.out
#SBATCH --error=logs/verify_chmean_%j.err
# Run peer's canonical_nw2ws4_s42 checkpoint under BOTH pool readouts.
# If our hypothesis is right:
#   jepa (320-D kc-pool) → narr ~ 0.13-0.16
#   jepa_chmean (64-D)   → narr ~ 0.037 (matches peer's report)

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

CKPT=/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/canonical_nw2ws4_s42/latest.pth.tar

for SRC in jepa jepa_chmean; do
    EXP_TAG="verify_${SRC}_s42"
    OUT_JSON="results/canonical/${EXP_TAG}/seed42/metrics.json"
    PRED_DIR="predictions/canonical/${EXP_TAG}/seed42"
    mkdir -p "$(dirname "$OUT_JSON")" "$PRED_DIR"

    echo "=== Running ${SRC} on ${CKPT} ==="
    PYTHONPATH=. python scripts/unified_probe_eval.py \
        --checkpoint="${CKPT}" \
        --feature_source="${SRC}" \
        --n_windows=2 --window_size_seconds=4 \
        --norm_mode=per_recording \
        --corrca_filters=corrca_filters.npz \
        --n_passes=20 \
        --seed=42 \
        --output_json="${OUT_JSON}" \
        --save_predictions_dir="${PRED_DIR}"
done

echo "verify_chmean done"
