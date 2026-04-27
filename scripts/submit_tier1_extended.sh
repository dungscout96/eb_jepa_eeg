#!/usr/bin/env bash
# Submit tier1 + Exp-6 cells with embedding dumps for the extended analyses
# (bootstrap CIs, permutation tests, stacked probes, CKA, data-eff curves).
#
# Layout: 4 baselines x 3 seeds = 12 jobs.
#   tier1 baselines (raw_corrca, psd_band, random_init): seeds 42, 123, 456
#   exp6: seeds 42, 123, 2025  (only checkpoints we have for std=0.25 pd=24)
#
# Run on Delta with branch kkokate/tier1-baselines checked out.
set -euo pipefail

CKPT_ROOT="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/exp6_sweep"

# Tier 1 cells with original seeds
for baseline in raw_corrca psd_band random_init; do
    for seed in 42 123 456; do
        echo "Submitting ${baseline} seed=${seed}"
        sbatch --export=ALL,BASELINE="${baseline}",SEED="${seed}" \
            scripts/tier1_baseline.sbatch
    done
done

# Exp 6 cells (paired with checkpoint per seed)
for seed in 42 123 2025; do
    ckpt="${CKPT_ROOT}/std0.25_pd24_seed${seed}/best.pth.tar"
    if [ ! -f "$ckpt" ]; then
        echo "WARN: checkpoint not found, skipping: $ckpt"
        continue
    fi
    echo "Submitting exp6 seed=${seed} ckpt=${ckpt}"
    sbatch --export=ALL,BASELINE=exp6,SEED="${seed}",CKPT="${ckpt}" \
        scripts/tier1_baseline.sbatch
done
