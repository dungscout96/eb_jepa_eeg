#!/bin/bash
# Tier 3 canonical Ridge probe submission — chains afterok on each
# extraction job in the existing 18078626-35 batch.
#
# Each extraction → canonical Ridge job pair, dependency-chained so the
# probe only fires after extraction succeeds. 10 probe jobs total.
set -euo pipefail
cd "$(dirname "$0")/.."

# Map extraction job IDs (already submitted) to (model, seed) and chain probe.
declare -A EXTRACT_JID=(
  [biot_42]=18078626
  [biot_123]=18078627
  [biot_456]=18078628
  [biot_789]=18078629
  [biot_2025]=18078630
  [luna_42]=18078631
  [luna_123]=18078632
  [luna_456]=18078633
  [luna_789]=18078634
  [luna_2025]=18078635
)

PROBE_SEED=42

for KEY in "${!EXTRACT_JID[@]}"; do
    EXT_JID=${EXTRACT_JID[$KEY]}
    MODEL="${KEY%_*}"
    ENC_SEED="${KEY##*_}"
    EXT_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/tier3_canonical/${MODEL}_seed${ENC_SEED}"
    EXP_TAG="pB_t3_${MODEL}_canonical"

    PROBE_JID=$(sbatch --parsable \
        --dependency=afterok:${EXT_JID} \
        --export=ALL,EXT_DIR=${EXT_DIR},EXP_TAG=${EXP_TAG}_${ENC_SEED},PROBE_SEED=${ENC_SEED},NW=2,WS=4,N_PASSES=20 \
        scripts/probe_unified_external.sbatch)
    echo "  ${KEY}: extract=${EXT_JID} → probe=${PROBE_JID}"
done

echo
echo "=== queue (extract + probe) ==="
squeue -u kkokate -h | wc -l | xargs echo "total:"
