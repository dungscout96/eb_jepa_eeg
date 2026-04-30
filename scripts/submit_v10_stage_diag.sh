#!/bin/bash
# v10 stage-diagnostic — 150 probe-eval jobs.
#   2 configs × 5 enc seeds × 5 probe seeds × 3 stages = 150.
# Re-uses Phase D checkpoints from the issue8 worktree.
#
# Goal: localize where narr_corr signal dies in the JEPA stack.
# Compares pre-transformer patches → context_enc → target_enc.
#
# Quick sanity check before submitting: refuse to submit if any
# expected checkpoint is missing.

set -euo pipefail

CKPT_BASE="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8"
ENC_SEEDS=(42 123 456 789 2025)
PROBE_SEEDS=(7 13 42 1234 2025)
STAGES=(patches context_enc target_enc)

# (NW WS BATCH_SIZE TAG_PREFIX) — drop nw4_ws4 since it lost everywhere.
CONFIGS=(
  "4 2 64 nw4ws2_baseline"
  "2 4 64 nw2ws4"
)

mkdir -p logs

# 1. Verify all 10 checkpoints exist
echo "=== checkpoint presence check ==="
MISSING=0
for CFG in "${CONFIGS[@]}"; do
  read -r NW WS BS TAG_PREFIX <<<"${CFG}"
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
done
if [ "$MISSING" -ne 0 ]; then
  echo "Refusing to submit — ${MISSING} checkpoint(s) missing."
  exit 1
fi

# 2. Submit 150 jobs
N=0
TOTAL=$(( ${#CONFIGS[@]} * ${#ENC_SEEDS[@]} * ${#PROBE_SEEDS[@]} * ${#STAGES[@]} ))
for CFG in "${CONFIGS[@]}"; do
  read -r NW WS BS TAG_PREFIX <<<"${CFG}"
  for ENC_SEED in "${ENC_SEEDS[@]}"; do
    ENC_TAG="phaseD_${TAG_PREFIX}_s${ENC_SEED}"
    CKPT="${CKPT_BASE}/${ENC_TAG}/latest.pth.tar"
    for PROBE_SEED in "${PROBE_SEEDS[@]}"; do
      for STAGE in "${STAGES[@]}"; do
        N=$((N+1))
        EXP_TAG="${TAG_PREFIX}_enc${ENC_SEED}_p${PROBE_SEED}_${STAGE}"
        echo "[${N}/${TOTAL}] ${EXP_TAG}"
        CKPT="${CKPT}" NW="${NW}" WS="${WS}" BATCH_SIZE="${BS}" \
          PROBE_SEED="${PROBE_SEED}" EXP_TAG="${EXP_TAG}" STAGE="${STAGE}" \
          sbatch scripts/probe_eval_v10_stage.sbatch
        sleep 1
      done
    done
  done
done

echo "Submitted ${N} v10 stage-diagnostic jobs."
