#!/bin/bash
# Submit Exp 6 Phase 2A OFAT hyperparameter screening on Delta.
# One-factor-at-a-time from the Phase 1 best anchor (std=cov=0.25, pd=16).
# 18 jobs total: 9 knobs × 2 off-anchor values × 1 seed.
#
# Anchor seed = 42 (matches one of the Phase 1 seeds, so results of
# the "anchor cell" already exist in Phase 1 logs — we do not re-run it).
#
# Usage (on Delta, from repo root):
#   bash scripts/submit_phase2a.sh          # dry-run: prints what would submit
#   bash scripts/submit_phase2a.sh --submit # actually submit

set -euo pipefail

SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then SUBMIT=1; fi

SEED=42
SBATCH=scripts/train_phase2a.sbatch

# Each entry is "KNOB|VALUE_TAG|ENV_OVERRIDES"
# ENV_OVERRIDES is a space-separated list of VAR=value assignments passed to sbatch.
CELLS=(
  # patch_size (anchor = 50)
  "patch_size|25|PATCH_SIZE=25"
  "patch_size|100|PATCH_SIZE=100"

  # patch_overlap (anchor = 20)
  "patch_overlap|0|PATCH_OVERLAP=0"
  "patch_overlap|40|PATCH_OVERLAP=40"

  # n_pred_masks_long (anchor = 2)
  "n_masks_long|1|N_PRED_MASKS_LONG=1"
  "n_masks_long|4|N_PRED_MASKS_LONG=4"

  # long_patch_scale (anchor = [0.5,1.0])
  # value tag stripped of brackets/commas for filesystem safety
  "long_patch|lo|LONG_PATCH_SCALE=[0.3,0.7]"
  "long_patch|hi|LONG_PATCH_SCALE=[0.7,1.0]"

  # predictor_depth (anchor = 2)
  "pred_depth|1|PREDICTOR_DEPTH=1"
  "pred_depth|3|PREDICTOR_DEPTH=3"

  # encoder_depth (anchor = 2)
  "enc_depth|1|ENCODER_DEPTH=1"
  "enc_depth|3|ENCODER_DEPTH=3"

  # lr (anchor = 5e-4)
  "lr|2e-4|LR=2e-4"
  "lr|1e-3|LR=1e-3"

  # ema_momentum_end (anchor = 1.0)
  "ema_end|09995|EMA_MOMENTUM_END=0.9995"

  # corrca n_components (anchor = 5 → corrca_filters.npz)
  "corrca|k3|CORRCA_FILE=corrca_filters_k3.npz"
  "corrca|k10|CORRCA_FILE=corrca_10.npz"
)

N=0
for cell in "${CELLS[@]}"; do
  KNOB="${cell%%|*}"
  rest="${cell#*|}"
  VALUE_TAG="${rest%%|*}"
  ENV_OVERRIDES="${rest#*|}"
  N=$((N+1))

  CMD="KNOB=${KNOB} VALUE_TAG=${VALUE_TAG} SEED=${SEED} ${ENV_OVERRIDES} sbatch ${SBATCH}"
  printf "[%2d] %s\n" "$N" "$CMD"

  if [[ "$SUBMIT" -eq 1 ]]; then
    eval "$CMD"
    sleep 2  # stagger to avoid wandb/file-creation races
  fi
done

if [[ "$SUBMIT" -eq 1 ]]; then
  echo "Submitted $N jobs."
else
  echo "Dry-run: $N jobs would be submitted. Pass --submit to actually submit."
fi
