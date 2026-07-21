#!/usr/bin/env bash
# Submit the full Tier 1 baseline grid (3 baselines x 3 seeds = 9 jobs).
#
# Run from the repo root on the Delta login node, with branch
# kkokate/tier1-baselines already checked out in the working tree.
#
#   cd ~/eb_jepa_eeg
#   git fetch origin
#   git checkout kkokate/tier1-baselines
#   git pull --ff-only origin kkokate/tier1-baselines
#   mkdir -p logs
#   bash scripts/submit_tier1.sh
#
# To submit only one baseline:
#   BASELINES="psd_band" bash scripts/submit_tier1.sh
# Fewer seeds:
#   SEEDS="42" bash scripts/submit_tier1.sh
set -euo pipefail

BASELINES="${BASELINES:-raw_corrca psd_band random_init}"
SEEDS="${SEEDS:-42 123 456}"

for baseline in $BASELINES; do
    for seed in $SEEDS; do
        echo "Submitting ${baseline} seed=${seed}"
        sbatch --export=ALL,BASELINE="${baseline}",SEED="${seed}" \
            scripts/tier1_baseline.sbatch
    done
done
