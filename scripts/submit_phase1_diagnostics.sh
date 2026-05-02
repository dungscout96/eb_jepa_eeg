#!/bin/bash
# ---------------------------------------------------------------
# Issue #10 Phase 1 — submit encoder-diagnostic probe sweep.
#
# Conditions (18 total):
#   Tier A: layer × tower × routing  (3 layers × 2 towers × 2 routings = 12)
#     - PROBE_LAYER ∈ {patch_embed, block0, final}
#     - USE_TEACHER ∈ {0, 1}
#     - KEEP_CHANNELS ∈ {0, 1}
#   Tier B: per-channel attribution at (final, student, keep_channels off)
#     - PROBE_CHANNEL ∈ {0, 1, 2, 3, 4}
#   Tier C: predictor bottleneck at (final, student, keep_channels off)
#     - PREPRED = 1
# Probe seeds: 5 (matches Phase D protocol).
# Total jobs: 18 × 5 = 90.
# ---------------------------------------------------------------

set -euo pipefail

# Anchor: nw4_ws2 Phase D enc seed 42 (matches PR #15 production baseline).
CKPT="${CKPT:-/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8/phaseD_nw4ws2_baseline_s42/latest.pth.tar}"
NW="${NW:-4}"
WS="${WS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: CKPT not found: $CKPT"
    exit 2
fi

PROBE_SEEDS=(42 123 456 789 2025)
LAYERS=(patch_embed block0 final)

mkdir -p logs

submit_one() {
    local tag=$1; shift
    for ps in "${PROBE_SEEDS[@]}"; do
        sbatch \
            --export=ALL,CKPT="$CKPT",NW="$NW",WS="$WS",BATCH_SIZE="$BATCH_SIZE",PROBE_SEED="$ps",EXP_TAG="$tag","$@" \
            scripts/probe_eval_phase1.sbatch
    done
}

echo "=== Phase 1 sweep: 18 conditions × 5 seeds = 90 jobs ==="

# Tier A: layer × tower × routing
for layer in "${LAYERS[@]}"; do
    for tower in 0 1; do            # 0=student, 1=teacher
        for keep in 0 1; do          # 0=mean-pool, 1=keep_channels
            tower_tag=$([ "$tower" = "1" ] && echo "tea" || echo "stu")
            keep_tag=$([ "$keep" = "1" ] && echo "kc" || echo "mp")
            tag="${layer}_${tower_tag}_${keep_tag}"
            submit_one "$tag" "PROBE_LAYER=$layer" "USE_TEACHER=$tower" "KEEP_CHANNELS=$keep"
        done
    done
done

# Tier B: per-channel attribution at final, student, keep_channels off
for ch in 0 1 2 3 4; do
    submit_one "final_stu_ch${ch}" "PROBE_LAYER=final" "USE_TEACHER=0" "KEEP_CHANNELS=0" "PROBE_CHANNEL=$ch"
done

# Tier C: predictor bottleneck at final, student
submit_one "final_stu_prepred" "PROBE_LAYER=final" "USE_TEACHER=0" "KEEP_CHANNELS=0" "PROBE_CHANNEL=-1" "PREPRED=1"

echo "=== All Phase 1 jobs submitted ==="
squeue -u "$USER" | tail -10
