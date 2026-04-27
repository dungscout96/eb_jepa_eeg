#!/usr/bin/env bash
# Submit the Tier 3 grid: 2 frozen FMs (BIOT, CBraMod) x 3 seeds = 6 jobs.
# LaBraM and EEGPT need additional adapter work (see tier3_foundation.py docstring).
set -euo pipefail
MODELS="${MODELS:-biot cbramod luna}"
SEEDS="${SEEDS:-42 123 456}"
for m in $MODELS; do
    for s in $SEEDS; do
        echo "Submitting tier3 ${m} seed=${s}"
        sbatch --export=ALL,MODEL="${m}",SEED="${s}" scripts/tier3_foundation.sbatch
    done
done
