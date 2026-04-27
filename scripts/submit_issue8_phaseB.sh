#!/bin/bash
# Issue #8 Phase B — temporal context sweep, single seed (2025).
# Skips nw4_ws2 (already in Phase A or Exp 6). Adds nw1_ws1, nw2_ws1, nw2_ws4, nw4_ws4.
#
# Operator must set USE_CORRCA={0|1} based on the Phase A decision.
# Without CorrCA, n_chans=129; longer-context cells need bs=32 to fit on A40-40GB.

set -euo pipefail

: "${USE_CORRCA:?need USE_CORRCA (0|1) — set after Phase A decision}"

SEED=2025
mkdir -p logs

# (NW, WS, BATCH_SIZE_NO_CORRCA, BATCH_SIZE_CORRCA)
CELLS=(
  "1 1 64 64"
  "2 1 64 64"
  "2 4 32 64"
  "4 4 32 64"
)

CORRCA_TAG="corrca"
[ "${USE_CORRCA}" = "0" ] && CORRCA_TAG="nocorrca"

N=0
for CELL in "${CELLS[@]}"; do
  read -r NW WS BS_NOCC BS_CC <<<"${CELL}"
  if [ "${USE_CORRCA}" = "1" ]; then
    BS="${BS_CC}"
  else
    BS="${BS_NOCC}"
  fi
  N=$((N+1))
  EXP_TAG="sweep_${CORRCA_TAG}_nw${NW}ws${WS}_s${SEED}"
  echo "[${N}/${#CELLS[@]}] ${EXP_TAG}  bs=${BS}"
  NW="${NW}" WS="${WS}" SEED="${SEED}" BATCH_SIZE="${BS}" \
    USE_CORRCA="${USE_CORRCA}" EXP_TAG="${EXP_TAG}" \
    sbatch scripts/train_issue8.sbatch
  sleep 2
done

echo "Submitted ${N} Phase B jobs (USE_CORRCA=${USE_CORRCA})."
