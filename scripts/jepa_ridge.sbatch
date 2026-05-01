#!/bin/bash
#SBATCH --job-name=jepa_ridge
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jepa_ridge_%j.out
#SBATCH --error=logs/jepa_ridge_%j.err
# ---------------------------------------------------------------
# Ridge probe on JEPA encoder embeddings (matches the draft's exact
# Ridge procedure but feeds JEPA per-clip embeddings instead of trivial
# stats). Parameterized by env:
#   CKPT            (full path to .pth.tar)
#   ENC_SEED        (encoder seed used for output naming)
#   PROBE_SEED      (clip-sampling seed; default 42)
#   KEEP_CHANNELS   (true|false; default false)
# ---------------------------------------------------------------

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"

: "${CKPT:?need CKPT=/path/to/checkpoint}"
: "${ENC_SEED:?need ENC_SEED}"
PROBE_SEED="${PROBE_SEED:-42}"
KEEP_CHANNELS="${KEEP_CHANNELS:-false}"

if [ "$KEEP_CHANNELS" = "true" ]; then
    KC_ARG="--keep_channels"
    BASELINE_TAG="jepa_ridge_keep_channels"
else
    KC_ARG="--keep_channels=False"
    BASELINE_TAG="jepa_ridge_default"
fi

echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "CKPT=$CKPT  ENC_SEED=$ENC_SEED  PROBE_SEED=$PROBE_SEED  KEEP=$KEEP_CHANNELS"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

OUT_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/tier1"
PRED_DIR="${OUT_DIR}/predictions/${BASELINE_TAG}_enc${ENC_SEED}"
OUT_JSON="${OUT_DIR}/${BASELINE_TAG}_enc${ENC_SEED}.json"
mkdir -p "$PRED_DIR"

PYTHONPATH=. /u/kkokate/.conda/envs/eb_jepa/bin/python scripts/trivial_ridge_baseline.py \
    --n_windows=4 --window_size_seconds=2 --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --n_passes=20 \
    --seed="${PROBE_SEED}" \
    --baseline="${BASELINE_TAG}" \
    --checkpoint="${CKPT}" \
    ${KC_ARG} \
    --save_predictions_dir="${PRED_DIR}" \
    --output_json="${OUT_JSON}"

# Stage the JSON under the expected per-seed name so bootstrap_trivial_perseed
# can find it via {baseline}_seed{seed}.json. Use enc seed as the seed id.
ln -sf "${OUT_JSON}" "${OUT_DIR}/${BASELINE_TAG}_seed${ENC_SEED}.json" 2>/dev/null || true
ln -sf "${PRED_DIR}" "${OUT_DIR}/predictions/${BASELINE_TAG}_seed${ENC_SEED}" 2>/dev/null || true
# Inside the pred dir, bootstrap looks for test_seed${ENC_SEED}.npz; rename
# files saved under the probe-seed name to the encoder-seed name.
for split in val test; do
    if [ -f "${PRED_DIR}/${split}_seed${PROBE_SEED}.npz" ]; then
        cp -f "${PRED_DIR}/${split}_seed${PROBE_SEED}.npz" \
              "${PRED_DIR}/${split}_seed${ENC_SEED}.npz"
    fi
done

echo "JEPA-ridge ${BASELINE_TAG} enc=${ENC_SEED} probe=${PROBE_SEED} done"
