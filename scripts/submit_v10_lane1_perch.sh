#!/bin/bash
# v10 lane #1 — per-channel probe (keep_channels=True).
# 100 jobs: 2 configs × 5 enc seeds × 5 probe seeds × 2 stages.
# Stages: context_enc and target_enc (skip patches — patches stage's
# narr signal already at +0.019 max from the prior sweep).
# Goal: confirm the channel-mean-pool diagnosis. If keep_channels
# raises narr_corr toward the per-subject ceiling (~0.07), pool is
# the cause; if not, the encoder destroys narrative independently.
#
# Re-uses the Phase D checkpoints from the issue8 worktree. EXP_TAG
# carries a `_perch` suffix to keep predictions separate from the
# default-pool sweep.

set -euo pipefail

CKPT_BASE="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8"
ENC_SEEDS=(42 123 456 789 2025)
PROBE_SEEDS=(7 13 42 1234 2025)
STAGES=(context_enc target_enc)

CONFIGS=(
  "4 2 64 nw4ws2_baseline"
  "2 4 64 nw2ws4"
)

mkdir -p logs

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
        EXP_TAG="${TAG_PREFIX}_enc${ENC_SEED}_p${PROBE_SEED}_${STAGE}_perch"
        echo "[${N}/${TOTAL}] ${EXP_TAG}"
        CKPT="${CKPT}" NW="${NW}" WS="${WS}" BATCH_SIZE="${BS}" \
          PROBE_SEED="${PROBE_SEED}" EXP_TAG="${EXP_TAG}" STAGE="${STAGE}" \
          KEEP_CHANNELS=1 \
          sbatch scripts/probe_eval_v10_stage.sbatch
        sleep 1
      done
    done
  done
done

echo "Submitted ${N} v10 lane-1 per-channel jobs."
