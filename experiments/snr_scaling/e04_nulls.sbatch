#!/bin/bash
#SBATCH --job-name=e04_nulls
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e04_nulls_%j.out
#SBATCH --error=logs/e04_nulls_%j.err
# ---------------------------------------------------------------
# The four E0.4 random-init nulls, at depth 22 / patch 200 / overlap 20.
#
# Every Δr² and Δr in the E0.4 tables is measured against a random encoder of
# IDENTICAL shape. E0.3's nulls are depth 12 and are not interchangeable, so
# reusing them would silently shift the whole surface.
#
# The null is an UNTRAINED encoder (probe.py --random-baseline), not a
# from-scratch training run: passing a trained subtrahend redefines the metric.
# No --checkpoint is given, so the probe builds the encoder from CONFIG alone —
# which is why CONFIG must be an E0.4 config_probe.yaml (knobs cleared) and not
# an E0.3 one.
#
# Optional env var:
#   CONFIG (path to an e04 cell's config_probe.yaml)
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/work/hdd/bbnv/dtyoung/hbn_preprocessed}"
export HBN_TRAIN_RELEASES="${HBN_TRAIN_RELEASES:-R1,R2,R3,R4,R7,R8,R9,R10}"
export UV_LINK_MODE=copy
export PATH="$HOME/.conda/envs/eb_jepa/bin:$HOME/.local/bin:$PATH"

CKPT_ROOT="${CKPT_ROOT:-/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling}"
CONFIG="${CONFIG:-${CKPT_ROOT}/e04_s1863_a101_nd/config_probe.yaml}"
OUT=experiments/snr_scaling/raw_results
# Result prefix follows the ARM. Every encoder SHAPE needs its own null:
# reusing the 512-wide null (r2=0.016572) for a 768/1216-wide readout would
# silently inflate every Delta-r2, since probe capacity grows with dim.
OUT_PREFIX="${OUT_PREFIX:-e04}"
# STAGE=probe_val runs only the CV-probe null, which is all the E0.6 width
# comparison reads. all = the original four.
STAGE="${STAGE:-all}"
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST | config=$CONFIG"

if [ ! -d eb_jepa/training ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: no config at $CONFIG — run a cell first, or pass CONFIG=" >&2
    exit 1
fi

mkdir -p "$OUT" logs

if want probe_val; then
PYTHONPATH=. uv run --group eeg python eb_jepa/evaluation/clip_probe/probe.py \
    --random-baseline --config "$CONFIG" \
    --split val --cv-splits 5 \
    --output "${OUT}/${OUT_PREFIX}_probe_val_random.json"
fi

for SPLIT in val test; do
    want tt_${SPLIT} || continue
    PYTHONPATH=. uv run --group eeg python \
        eb_jepa/evaluation/clip_probe/probe_traintest.py \
        --random-baseline --config "$CONFIG" \
        --eval-split "$SPLIT" --bootstrap 2000 \
        --output "${OUT}/${OUT_PREFIX}_tt_${SPLIT}_random.json"
done

# TEST only: a random encoder scores 3.79x chance at scene level on val while
# sitting below chance on test, so a val retrieval null invites a wrong read.
if want retr_test; then
PYTHONPATH=. uv run --group eeg python \
    eb_jepa/evaluation/clip_probe/retrieval.py \
    --random-baseline --config "$CONFIG" \
    --split test --topks 1 5 10 \
    --output "${OUT}/${OUT_PREFIX}_retr_test_random.json"
fi

echo "=== E0.4 nulls written to ${OUT} ==="
