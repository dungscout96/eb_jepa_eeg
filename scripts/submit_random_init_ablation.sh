#!/bin/bash
# Submit random_init ablation variants:
#   random_no_transformer / random_no_attn / random_no_pos
# 5 enc seeds × 3 variants = 15 jobs. Paired: probe_seed = enc_seed
# (same protocol as canonical pB_t1_random_init for direct comparison).
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(42 123 456 789 2025)
VARIANTS=(random_no_transformer random_no_attn random_no_pos)
TAGS=(pB_t1_random_no_transformer pB_t1_random_no_attn pB_t1_random_no_pos)

for i in "${!VARIANTS[@]}"; do
    VAR="${VARIANTS[$i]}"
    TAG="${TAGS[$i]}"
    for S in "${SEEDS[@]}"; do
        sbatch \
            --export=ALL,FEATURE_SOURCE="$VAR",ENC_SEED="$S",PROBE_SEED="$S",EXP_TAG="$TAG" \
            scripts/probe_unified_tier.sbatch
    done
done

echo "Submitted ${#SEEDS[@]} × ${#VARIANTS[@]} = $(( ${#SEEDS[@]} * ${#VARIANTS[@]} )) jobs"
