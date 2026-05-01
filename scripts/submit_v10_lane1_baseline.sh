#!/bin/bash
# v10 lane #1 — minimal scope: baseline only, context_enc, keep_channels.
# 25 jobs: 1 config × 5 enc seeds × 5 probe seeds × 1 stage.
#
# Decision criterion:
#   If narr_corr rises from ~-0.01 (default pool) toward the per-subject
#   ceiling (~0.07) → channel mean-pool was the kill mechanism, eval-side
#   fix sufficient. If narr_corr stays at ~0 → encoder isn't preserving
#   the channel-1 narrative direction, pivot to lane #2 / lane #3.
#
# Re-uses the Phase D nw4_ws2_baseline checkpoints. EXP_TAG carries
# `_perch` suffix to keep predictions separate from the default-pool
# sweep.

set -euo pipefail

CKPT_BASE="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8"
ENC_SEEDS=(42 123 456 789 2025)
PROBE_SEEDS=(7 13 42 1234 2025)
STAGE="context_enc"
NW=4
WS=2
BS=64
TAG_PREFIX="nw4ws2_baseline"

mkdir -p logs

echo "=== checkpoint presence check ==="
MISSING=0
for SEED in "${ENC_SEEDS[@]}"; do
  EXP_TAG="phaseD_${TAG_PREFIX}_s${SEED}"
  CKPT="${CKPT_BASE}/${EXP_TAG}/latest.pth.tar"
  if [ -f "$CKPT" ]; then
    echo "  OK  ${EXP_TAG}"
  else
    echo "  MISSING  ${CKPT}"
    MISSING=$((MISSING+1))
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "Refusing to submit — ${MISSING} checkpoint(s) missing."
  exit 1
fi

N=0
TOTAL=$(( ${#ENC_SEEDS[@]} * ${#PROBE_SEEDS[@]} ))
for ENC_SEED in "${ENC_SEEDS[@]}"; do
  ENC_TAG="phaseD_${TAG_PREFIX}_s${ENC_SEED}"
  CKPT="${CKPT_BASE}/${ENC_TAG}/latest.pth.tar"
  for PROBE_SEED in "${PROBE_SEEDS[@]}"; do
    N=$((N+1))
    EXP_TAG="${TAG_PREFIX}_enc${ENC_SEED}_p${PROBE_SEED}_${STAGE}_perch"
    echo "[${N}/${TOTAL}] ${EXP_TAG}"
    CKPT="${CKPT}" NW="${NW}" WS="${WS}" BATCH_SIZE="${BS}" \
      PROBE_SEED="${PROBE_SEED}" EXP_TAG="${EXP_TAG}" STAGE="${STAGE}" \
      KEEP_CHANNELS=1 \
      sbatch scripts/probe_eval_v10_stage.sbatch
    sleep 1
  done
done

echo "Submitted ${N} v10 lane-1 baseline-only jobs."
