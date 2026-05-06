#!/bin/bash
# Tier 2 canonical Ridge — train Deep4 + extract per-clip features + Ridge.
# Runs in the tier4 worktree (Tier 2 supervised script lives there). Each
# extract→probe pair is dependency-chained.
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A EXT_JIDS=()
WORKTREE=/projects/bbnv/kkokate/eb_jepa_eeg_tier4
cd $WORKTREE

for SEED in 42 123 456 789 2025; do
    JID=$(sbatch --parsable \
        --export=ALL,MODEL=deep4,SEED=${SEED},NW=2,WS=4,N_PASSES=20,EPOCHS=30,EARLY_STOP=16 \
        scripts/extract_tier2_canonical.sbatch)
    EXT_JIDS[deep4_${SEED}]=$JID
    echo "deep4 seed=${SEED} extract → ${JID}"
done

cd /projects/bbnv/kkokate/eb_jepa_eeg
for SEED in 42 123 456 789 2025; do
    EXT_JID=${EXT_JIDS[deep4_${SEED}]}
    EXT_DIR=/projects/bbnv/kkokate/eb_jepa_eeg/predictions/tier2_canonical/deep4_seed${SEED}
    PROBE_JID=$(sbatch --parsable \
        --dependency=afterok:${EXT_JID} \
        --export=ALL,EXT_DIR=${EXT_DIR},EXP_TAG=pB_t2_deep4_canonical_${SEED},PROBE_SEED=${SEED},NW=2,WS=4,N_PASSES=20 \
        scripts/probe_unified_external.sbatch)
    echo "deep4 seed=${SEED} probe → ${PROBE_JID} (afterok ${EXT_JID})"
done
