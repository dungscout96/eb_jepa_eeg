#!/bin/bash
# Resume Phase 2A submission after a partial failure (e.g. branch auto-switch
# mid-loop). Submits only the cells that were not submitted in the first run.
# Compares currently-running/completed p2a job names against the full cell list
# and submits the missing ones.
#
# Usage: bash scripts/submit_phase2a_resume.sh [--submit]

set -euo pipefail
SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then SUBMIT=1; fi

SEED=42
SBATCH=scripts/train_phase2a.sbatch

# (KNOB, VALUE_TAG, ENV_OVERRIDES) — same list as submit_phase2a.sh
CELLS=(
  "patch_size|25|PATCH_SIZE=25"
  "patch_size|100|PATCH_SIZE=100"
  "patch_overlap|0|PATCH_OVERLAP=0"
  "patch_overlap|40|PATCH_OVERLAP=40"
  "n_masks_long|1|N_PRED_MASKS_LONG=1"
  "n_masks_long|4|N_PRED_MASKS_LONG=4"
  "long_patch|lo|LONG_PATCH_SCALE=[0.3,0.7]"
  "long_patch|hi|LONG_PATCH_SCALE=[0.7,1.0]"
  "pred_depth|1|PREDICTOR_DEPTH=1"
  "pred_depth|3|PREDICTOR_DEPTH=3"
  "enc_depth|1|ENCODER_DEPTH=1"
  "enc_depth|3|ENCODER_DEPTH=3"
  "lr|2e-4|LR=2e-4"
  "lr|1e-3|LR=1e-3"
  "ema_end|09995|EMA_MOMENTUM_END=0.9995"
  "corrca|k3|CORRCA_FILE=corrca_filters_k3.npz"
  "corrca|k10|CORRCA_FILE=corrca_10.npz"
)

# Already-running/finished cells are inferred from existing EXP_DIRs.
EXISTING_DIR=/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/phase2a

N=0
SKIPPED=0
SUBMITTED=0
for cell in "${CELLS[@]}"; do
  KNOB="${cell%%|*}"
  rest="${cell#*|}"
  VALUE_TAG="${rest%%|*}"
  ENV_OVERRIDES="${rest#*|}"
  N=$((N+1))

  EXP_ID="p2a_${KNOB}_${VALUE_TAG}_s${SEED}"
  if [[ -d "${EXISTING_DIR}/${EXP_ID}" ]]; then
    printf "[%2d] SKIP (already has EXP_DIR) %s\n" "$N" "$EXP_ID"
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  CMD="KNOB=${KNOB} VALUE_TAG=${VALUE_TAG} SEED=${SEED} ${ENV_OVERRIDES} sbatch ${SBATCH}"
  printf "[%2d] %s\n" "$N" "$CMD"

  if [[ "$SUBMIT" -eq 1 ]]; then
    eval "$CMD"
    sleep 2
    SUBMITTED=$((SUBMITTED+1))
  fi
done

echo "---"
echo "Total cells: $N   Skipped (already have EXP_DIR): $SKIPPED   Submitted: $SUBMITTED"
if [[ "$SUBMIT" -eq 0 ]]; then
  echo "Dry-run. Pass --submit to actually submit."
fi
