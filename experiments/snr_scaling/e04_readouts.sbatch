#!/bin/bash
#SBATCH --job-name=e04_readout
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e04_readout_%j.out
#SBATCH --error=logs/e04_readout_%j.err
# ---------------------------------------------------------------
# The three tracked readouts for ONE E0.4 cell, on its selected checkpoint.
#
# Run select_and_probe_e03.py --ckpt-root <e04 root> FIRST: the checkpoint must
# be the epoch chosen by the smoothed val diagnostic, not latest.pth.tar.
# Probing the endpoint instead is the bias RESULTS.md 2.10 documents.
#
# Time limit 04:00:00 (this account's A40 cap, delta-deploy SKILL.md 3). Measured:
# ~45 min per stage, and BOTH probe_traintest calls re-encode the full 1863-
# recording train pool to fit the head, so the four stages run ~2h45 -- a 2h
# limit times out after stage 2 with no retrieval number.
#
# Required env vars:
#   SLUG (e.g. e04_s701_a101_nd_d11)
#   CKPT (absolute path to the selected epoch_N.pth.tar)
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

CKPT_ROOT=/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling
CONFIG="${CKPT_ROOT}/${SLUG}/config_probe.yaml"
OUT=experiments/snr_scaling/raw_results

echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "cfg: slug=${SLUG} ckpt=${CKPT}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: no checkpoint at $CKPT" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: no probe config at $CONFIG" >&2
    exit 1
fi

mkdir -p "$OUT" logs

# Each stage is skipped when its output already exists, so a job killed at the
# walltime resumes instead of recomputing ~1.5 h of finished readouts. Set
# FORCE=1 to recompute regardless.
have() { [ -z "${FORCE:-}" ] && [ -s "$1" ]; }

# (1) 12-feature CV probe on val -- the Δr² readout.
if have "${OUT}/e04_probe_val_${SLUG}.json"; then
    echo "skip probe_val (exists)"
else
    PYTHONPATH=. uv run --group eeg python eb_jepa/evaluation/clip_probe/probe.py \
        --checkpoint "$CKPT" --config "$CONFIG" \
        --split val --cv-splits 5 \
        --output "${OUT}/e04_probe_val_${SLUG}.json"
fi

# (2) train->test ridge probe, both splits, with bootstrap CIs.
for SPLIT in val test; do
    if have "${OUT}/e04_tt_${SPLIT}_${SLUG}.json"; then
        echo "skip tt_${SPLIT} (exists)"
        continue
    fi
    PYTHONPATH=. uv run --group eeg python \
        eb_jepa/evaluation/clip_probe/probe_traintest.py \
        --checkpoint "$CKPT" --config "$CONFIG" \
        --eval-split "$SPLIT" --bootstrap 2000 \
        --output "${OUT}/e04_tt_${SPLIT}_${SLUG}.json"
done

# (3) Scene-level retrieval, TEST only: a random encoder scores 3.79x chance at
# scene level on val while sitting below chance on test, so val invites a
# wrong read.
if have "${OUT}/e04_retr_test_${SLUG}.json"; then
    echo "skip retr_test (exists)"
else
    PYTHONPATH=. uv run --group eeg python \
        eb_jepa/evaluation/clip_probe/retrieval.py \
        --checkpoint "$CKPT" --config "$CONFIG" \
        --split test --topks 1 5 10 \
        --output "${OUT}/e04_retr_test_${SLUG}.json"
fi

for f in "${OUT}/e04_probe_val_${SLUG}.json" "${OUT}/e04_tt_val_${SLUG}.json" \
         "${OUT}/e04_tt_test_${SLUG}.json" "${OUT}/e04_retr_test_${SLUG}.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing readout $f" >&2
        exit 1
    fi
done

echo "Cell ${SLUG} readouts complete"
