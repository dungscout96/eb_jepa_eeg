#!/usr/bin/env bash
# Submit the Tier 4 grid: 3 FMs (BIOT, CBraMod, LUNA) × 3 seeds = 9 jobs.
# Each is full fine-tuning (encoder unfrozen) with neg-Pearson loss.
set -euo pipefail
MODELS="${MODELS:-biot cbramod luna}"
SEEDS="${SEEDS:-42 123 456}"
for m in $MODELS; do
    for s in $SEEDS; do
        echo "Submitting tier4 ${m} seed=${s}"
        sbatch --export=ALL,MODEL="${m}",SEED="${s}" scripts/tier4_full_ft.sbatch
    done
done
