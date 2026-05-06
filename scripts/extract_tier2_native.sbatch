#!/bin/bash
#SBATCH --job-name=t2nat
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:30:00
#SBATCH --output=logs/t2nat_%j.out
#SBATCH --error=logs/t2nat_%j.err
set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"

: "${MODEL:?need MODEL (deep4|shallow|eegnet|eegnex)}"
: "${SEED:?need SEED}"
NW="${NW:-2}"; WS="${WS:-4}"; N_PASSES="${N_PASSES:-20}"
EPOCHS="${EPOCHS:-30}"; EARLY_STOP="${EARLY_STOP:-16}"; LR="${LR:-1e-3}"; BS="${BS:-64}"

cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
module reset; module load miniforge3-python
VENV_PY=/u/kkokate/.conda/envs/eb_jepa/bin/python
test -x "$VENV_PY" || { echo "venv missing"; exit 1; }

OUT_DIR=/projects/bbnv/kkokate/eb_jepa_eeg/predictions/tier2_native/${MODEL}_seed${SEED}
mkdir -p "$OUT_DIR"

PYTHONPATH=. "$VENV_PY" scripts/extract_tier2_native.py \
    --model="${MODEL}" --seed="${SEED}" --out_dir="${OUT_DIR}" \
    --n_windows="${NW}" --window_size_seconds="${WS}" --n_passes="${N_PASSES}" \
    --epochs="${EPOCHS}" --early_stop_patience="${EARLY_STOP}" --lr="${LR}" --batch_size="${BS}"

echo "tier2 native end-to-end ${MODEL} seed=${SEED} → ${OUT_DIR}"
