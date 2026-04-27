#!/usr/bin/env bash
# Re-run Deep4Net with native preprocessing across 3 seeds.
set -euo pipefail
SEEDS="${SEEDS:-42 123 456}"
for s in $SEEDS; do
    echo "Submitting deep4_native seed=${s}"
    sbatch --export=ALL,SEED="${s}" scripts/tier2_deep4_native.sbatch
done
