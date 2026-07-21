#!/usr/bin/env bash
# Submit the Tier 2 supervised grid (4 models x 3 seeds = 12 jobs).
set -euo pipefail
MODELS="${MODELS:-shallow deep4 eegnet eegnex}"
SEEDS="${SEEDS:-42 123 456}"
for m in $MODELS; do
    for s in $SEEDS; do
        echo "Submitting tier2 ${m} seed=${s}"
        sbatch --export=ALL,MODEL="${m}",SEED="${s}" scripts/tier2_supervised.sbatch
    done
done
