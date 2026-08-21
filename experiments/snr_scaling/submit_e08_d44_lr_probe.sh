#!/bin/bash
# ---------------------------------------------------------------
# E0.8 -- learning-rate probe at depth 44.
#
# THE QUESTION. E0.7 puts depth 44 at or below depth 22, and the epoch
# explanation is refuted: the 800-epoch cells peak internally (best_ep 460-792,
# 1.3-4.9% decay) and a 1600-epoch cell gained +0.001 at d32 and LOST 0.010 at
# d44. But every arm in this project inherits lr=1e-4 from a jul1 sweep run at
# DEPTH 12 (iter4/17). Deeper transformers generally want a lower LR, so a
# mistuned d44 would underperform for optimisation reasons no amount of extra
# epochs can fix. That confound is untested, and it is the stronger version of
# the undertraining worry.
#
# Until it is closed, "depth past 22 does not help" is not a capacity claim --
# it is a claim about depth 44 AT lr=1e-4.
#
# DESIGN. Bracket 1e-4 on BOTH sides rather than only probing downward, so the
# result distinguishes "1e-4 is a local optimum" from "we are on a slope and
# picked the wrong end". lr=1e-4 already exists as E0.7's d44 cells at n=3, so
# only the other three LRs are submitted here.
#
#   3e-5   5e-5   [1e-4 = E0.7, already run]   2e-4
#
# Everything else is pinned to E0.7's d44 cells: depth 44, width 512, patch
# 200/20, init_depth_scaled=true, random init, 800 epochs, S=701, draws
# {11,22,33}. Same cohorts, so the comparison is paired.
#
# WHY n=3 AND NOT A 1-CELL SCREEN. A large effect would show at n=1, but the
# useful outcome here is likely NEGATIVE, and "no effect" needs power to state.
# Draw-to-draw sd on Delta-r2 runs ~0.001-0.003, which a single cell cannot
# resolve.
#
# READOUT. These cells share E0.7's d44 shape and init exactly, so they reuse
# e07d44_probe_val_random.json as the null. No new null is needed -- the null
# depends on architecture, not on the LR the trained model used.
#
# NOTE the same confound applies to E0.6: width 1216 also ran at lr=1e-4, and
# muP says a 2.4x wider model wants roughly 1/2.4 the LR. That is NOT closed by
# this script.
#
# DRY=1 prints without submitting.
# ---------------------------------------------------------------
set -euo pipefail

ANCHORS=101
S=701
DRAWS=(11 22 33)
EPOCHS="${EPOCHS:-800}"
CFG=experiments/snr_scaling/clip_pretrain_reve_d44.yaml
SBATCH=experiments/snr_scaling/train_e04_cell.sbatch
CKPT_ROOT=/work/hdd/bbnv/kkokate/eb_jepa/e07_depth_scaling

# lr : slug tag
LRS=("3e-5:lr3e5" "5e-5:lr5e5" "2e-4:lr2e4")

if [ ! -f "$SBATCH" ] || [ ! -f "$CFG" ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi
mkdir -p logs

N=0
for SPEC in "${LRS[@]}"; do
    LR_="${SPEC%%:*}"
    TAG="${SPEC##*:}"
    SUFFIX="_${TAG}"
    for D in "${DRAWS[@]}"; do
        EXP_ID="e07d44_s${S}_a${ANCHORS}${SUFFIX}_d${D}"
        N=$((N + 1))
        echo "[$N] depth=44 lr=$LR_ S=$S draw=$D epochs=$EPOCHS"
        if [ -z "${FORCE:-}" ] && [ -f "${CKPT_ROOT}/${EXP_ID}/latest.pth.tar" ]; then
            echo "     skip (latest.pth.tar exists)"
            continue
        fi
        [ -n "${DRY:-}" ] && continue
        # DRAW/LR via `env`: assignment prefixes are parsed before expansion.
        env SUBJECTS="$S" ANCHORS="$ANCHORS" DRAW="$D" LR="$LR_" \
            EPOCHS="$EPOCHS" SUFFIX="$SUFFIX" INIT_CKPT=none \
            CONFIG_SRC="$CFG" EXP_PREFIX="e07d44" CKPT_ROOT="$CKPT_ROOT" \
            sbatch --job-name="e08d44_${TAG}_d${D}" --time=12:00:00 "$SBATCH"
        sleep 2
    done
done

echo
echo "Submitted $N jobs."
echo
echo "READ IT AS: if any LR lifts d44 above E0.7's d22 (0.05122 at S=701),"
echo "depth was never the problem and the capacity conclusion needs revisiting."
echo "If none does, depth 44 is genuinely dead with the LR confound closed."
echo
echo "AUDIT min train/loss vs ln(64)=4.159 and vs siblings. lr=2e-4 at depth 44"
echo "is the cell most likely to diverge outright."
