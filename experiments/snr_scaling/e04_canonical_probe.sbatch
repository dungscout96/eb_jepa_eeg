#!/bin/bash
#SBATCH --job-name=e04_canon
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e04_canon_%j.out
#SBATCH --error=logs/e04_canon_%j.err
# ---------------------------------------------------------------
# The CANONICAL probe protocol (docs/evaluation_guide.md) on one E0.4 cell:
# 4 stim features x {reg r, reg R2, cls AUC, cls bal_acc}, subject age + sex
# (an ANTI-target -- a stimulus encoder should NOT gain here), and 20-bin
# movie-ID top-1/top-5. Followed by the mandatory recording-level bootstrap.
#
# This is a different contract from the snr_scaling readouts, which use a
# 12-feature CV probe. Both are needed: the 12-feature one places E0.4 on
# E0.3's scaling curve, this one says whether the encoder is actually better.
#
# NOTE: config/clip_pretrain.yaml sets eval.auto_run=false because "the
# probe-eval pipeline is JEPA-shaped; flip to true if you've validated that
# path against a CLIP checkpoint." That validation has not been done, so treat
# the first run of this script as the validation.
#
# corrca_filters is deliberately EMPTY: the E0.4 encoder is 129-channel with
# corrca_filters=null, and probe_eval pools with keep_channels=False, giving a
# 512-d readout rather than the guide's 320-d CorrCA-5 default.
#
# N_PASSES defaults to the protocol's 20. Set it lower ONLY to validate the path
# or measure cost: evaluation_guide fixes n_passes=20 for any reported number.
#
# Required env vars:
#   SLUG, CKPT
# Optional:
#   N_PASSES, TAG (output suffix, so a validation run cannot overwrite a real one)
# ---------------------------------------------------------------

set -euo pipefail

if [ -z "${SLUG:-}" ]; then
    echo "ERROR: SLUG env var must be set" >&2
    exit 1
fi
if [ -z "${CKPT:-}" ]; then
    echo "ERROR: CKPT env var must be set" >&2
    exit 1
fi

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/work/hdd/bbnv/dtyoung/hbn_preprocessed}"
export HBN_TRAIN_RELEASES="${HBN_TRAIN_RELEASES:-R1,R2,R3,R4,R7,R8,R9,R10}"
export UV_LINK_MODE=copy

if [ ! -d eb_jepa/training ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_DISABLED=true

CKPT_ROOT="${CKPT_ROOT:-/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling}"
CONFIG="${CKPT_ROOT}/${SLUG}/config_probe.yaml"
OUT=experiments/snr_scaling/raw_results
TAG="${TAG:-}"
# Defaults beside the checkpoint, but must be overridable: a reader running
# against someone else's group-readable CKPT_ROOT cannot write into it.
PRED="${PRED:-${CKPT_ROOT}/${SLUG}/canonical_preds${TAG}}"

echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "cfg: slug=${SLUG} ckpt=${CKPT}"

if [ ! -f "$CKPT" ]; then echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; fi
if [ ! -f "$CONFIG" ]; then echo "ERROR: no probe config at $CONFIG" >&2; exit 1; fi

mkdir -p "$OUT" "$PRED" logs

# n_windows/window_size must match how the cell was TRAINED (1 x 2s), not
# probe_eval's JEPA-era 2 x 4s defaults.
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.probe_eval \
    --checkpoint="$CKPT" \
    --fname="$CONFIG" \
    --n_windows=1 \
    --window_size_seconds=2 \
    --norm_mode=per_recording \
    --corrca_filters= \
    --n_passes="${N_PASSES:-20}" \
    --probe_seed=42 \
    --save_predictions_dir="$PRED" \
    --out_json="${OUT}/e04_canonical_${SLUG}${TAG}.json"

# Bootstrap is mandatory (evaluation_guide.md 3): a bare L1 is diagnostic only.
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.bootstrap \
    --predictions_npz="${PRED}/preds_seed42.npz" \
    --out_json="${OUT}/e04_canonical_boot_${SLUG}${TAG}.json" \
    --n_bootstrap=2000

for f in "${OUT}/e04_canonical_${SLUG}${TAG}.json" "${OUT}/e04_canonical_boot_${SLUG}${TAG}.json"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing $f" >&2; exit 1; fi
done

echo "Canonical probe complete for ${SLUG}"
