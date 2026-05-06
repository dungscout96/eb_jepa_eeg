#!/bin/bash
#SBATCH --job-name=t3probe
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/t3probe_%j.out
#SBATCH --error=logs/t3probe_%j.err
# ---------------------------------------------------------------
# Canonical Ridge eval on externally-extracted features (Tier 3 / Tier 2).
# Reads NPZs from EXT_DIR matching {train,val,test}_seed{S, S+1, S+2}.npz with
# keys embs [n_rec, n_passes, D] and labels [n_rec, n_passes, n_features].
# Same Ridge/LogReg/bootstrap protocol as JEPA Phase D.
#
# Required env: EXT_DIR, EXP_TAG, PROBE_SEED.
# Optional:     NW (2), WS (4), N_PASSES (20).
# ---------------------------------------------------------------
set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${EXT_DIR:?need EXT_DIR}"
: "${EXP_TAG:?need EXP_TAG}"
: "${PROBE_SEED:?need PROBE_SEED}"
NW="${NW:-2}"
WS="${WS:-4}"
N_PASSES="${N_PASSES:-20}"

[ -d "$EXT_DIR" ] || { echo "ERROR: EXT_DIR not found: $EXT_DIR"; exit 2; }

cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_ENTITY="${WANDB_ENTITY:-sccn}"
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

OUT_JSON="/projects/bbnv/kkokate/eb_jepa_eeg/results/unified/${EXP_TAG}_seed${PROBE_SEED}.json"
PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/unified/${EXP_TAG}_seed${PROBE_SEED}"
mkdir -p "$(dirname "$OUT_JSON")" "$PRED_DIR"

echo "Cfg: ext_dir=${EXT_DIR}  tag=${EXP_TAG}  probe_seed=${PROBE_SEED}  n_passes=${N_PASSES}"

PYTHONPATH=. uv run --group eeg python scripts/unified_probe_eval.py \
    --feature_source=external_npz \
    --external_npz_dir="${EXT_DIR}" \
    --n_windows="${NW}" --window_size_seconds="${WS}" \
    --norm_mode=per_recording \
    --n_passes="${N_PASSES}" \
    --seed="${PROBE_SEED}" \
    --output_json="${OUT_JSON}" \
    --save_predictions_dir="${PRED_DIR}"

echo "Tier 3 canonical Ridge ${EXP_TAG} seed=${PROBE_SEED} done → ${OUT_JSON}"
