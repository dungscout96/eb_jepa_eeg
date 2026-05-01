#!/bin/bash
# Submit ridge-probe trivial baselines × 5 probe seeds.
# Matches the draft's exact procedure (per-clip mean+std+log-bands × C,
# Ridge regression, n_passes=20). 2 baselines × 5 seeds = 10 jobs.
set -euo pipefail

BASELINES=(
    trivial_ridge_corrca35
    trivial_ridge_raw903
    trivial_ridge_chan1_only
    trivial_ridge_corrca_pooled35
    trivial_ridge_raw_pooled903
)
SEEDS=(7 13 42 1234 2025)

for B in "${BASELINES[@]}"; do
    for S in "${SEEDS[@]}"; do
        echo "Submitting $B seed=$S"
        BASELINE="$B" SEED="$S" \
        sbatch --job-name="ridge_${B}_s${S}" scripts/trivial_ridge.sbatch
    done
done
