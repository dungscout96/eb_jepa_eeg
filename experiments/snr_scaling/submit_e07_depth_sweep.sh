#!/bin/bash
# ---------------------------------------------------------------
# E0.7 -- depth scaling past 22, at a converged budget, with depth-scaled init.
#
# THE QUESTION. Depth is the only capacity axis with a positive signal.
# E0.6 tested WIDTH cleanly and it is negative (512 -> 768 = 0.999x at S=701;
# 512 -> 1216 = 0.79x at S=701, 0.66x at S=1863). Depth 12 -> 22 is positive at
# large S (+24% at S=1863, local exponent 0.265 vs 0.049), and BOTH arms of that
# comparison were converged -- E0.3 peaks at ep 182-380 with 0 of 24 cells still
# climbing at 400, E0.5 peaks at 53-698 of 800. Depth past 22 is untested.
#
# The prior null at depth 22 (jul1 iter 14, +0.00009) is NOT evidence against
# depth: it ran at 300 epochs, and E0.5 later showed a depth-22 model is still
# climbing at 400. It is an undertraining artifact.
#
# WHY d22 IS RE-RUN HERE. All three snapshots set init_depth_scaled=true.
# EEGEncoderTokens otherwise applies no explicit init and the residual stream's
# variance grows with depth, which is what caps trainable depth past ~22 (CaiT:
# 12 -> 36 layers costs ~10 points top-1 at fixed lr/wd). Enabling it changes
# the init at EVERY depth, so scaled-init d32/d44 cannot be compared against
# E0.5's unscaled-init d22 -- that would confound init with depth, exactly the
# mistake E0.4 made by moving depth, geometry and warm start together. The d22
# cells here ARE the baseline. E0.5's d22 additionally serves as the
# scaled-vs-unscaled contrast at matched depth.
#
# DESIGN. Only depth moves: width 512, patch 200/20, heads 8, head_dim 64,
# freqs 4, mlp_dim_ratio 2.66, proj_dim 512, random init, 800 epochs.
# S in {701, 1863} -- E0.5/E0.6 showed capacity only starts paying above ~700.
#
#   d22 ->  69.2M   d32 -> 100.6M   d44 -> 138.3M
#
# BUDGET. 800 epochs matches E0.5, where depth 22 converged (peaks at 53-698,
# then decays). Deeper may need more, so two cells run at 1600 as a one-sided
# convergence check -- the same test that caught E0.5 being undertrained at 400.
# They carry SUFFIX=_ep1600 so they stay out of the _d aggregation glob.
#
# NULLS. One per depth, mandatory: Delta-r2 is measured against a random
# encoder of the SAME shape and readout statistics change with depth. Submitted
# after the first cell of each depth writes its config_probe.yaml.
#
# WALL. gpuA40x4's real limit is 2 days (the "4 h cap" in delta-deploy SKILL.md
# is wrong; a 20 h job is accepted). Measured 69.2M at 800 ep = 1 h 09 - 2 h 25,
# so d32 gets 8 h and d44 gets 12 h. Do NOT reach for meta.resume_from to fit a
# shorter wall: clip_pretrain.py:500-503 applies a FRESH LR schedule on resume,
# so two legs re-warm and re-anneal instead of continuing one cosine.
#
# Run from the repo root ON DELTA, branch checked out there first.
# DRY=1 prints without submitting.
# ---------------------------------------------------------------
set -euo pipefail

ANCHORS=101
DRAWS=(11 22 33)
EPOCHS="${EPOCHS:-800}"
SUFFIX="${SUFFIX:-_d}"

SBATCH=experiments/snr_scaling/train_e04_cell.sbatch
CKPT_ROOT=/work/hdd/bbnv/kkokate/eb_jepa/e07_depth_scaling

if [ ! -f "$SBATCH" ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi
mkdir -p logs

# depth : wall limit
DEPTHS=("22:06:00:00" "32:08:00:00" "44:12:00:00")

N=0
for SPEC in "${DEPTHS[@]}"; do
    D_="${SPEC%%:*}"
    WALL="${SPEC#*:}"
    CFG="experiments/snr_scaling/clip_pretrain_reve_d${D_}.yaml"
    if [ ! -f "$CFG" ]; then
        echo "ERROR: missing frozen config $CFG" >&2
        exit 1
    fi
    for S in 701 1863; do
        if [ "$S" = "1863" ]; then
            CELL_DRAWS=("")
        else
            CELL_DRAWS=("${DRAWS[@]}")
        fi
        for DR in "${CELL_DRAWS[@]}"; do
            if [ -n "$DR" ]; then
                EXP_ID="e07d${D_}_s${S}_a${ANCHORS}${SUFFIX}_d${DR}"
            else
                EXP_ID="e07d${D_}_s${S}_a${ANCHORS}${SUFFIX}"
            fi
            N=$((N + 1))
            echo "[$N] depth=$D_ S=$S draw=${DR:-none} epochs=$EPOCHS wall=$WALL"
            if [ -z "${FORCE:-}" ] && [ -f "${CKPT_ROOT}/${EXP_ID}/latest.pth.tar" ]; then
                echo "     skip (latest.pth.tar exists)"
                continue
            fi
            [ -n "${DRY:-}" ] && continue
            # DRAW must go through `env`: assignment prefixes are parsed before
            # expansion, so a bare ${DR:+DRAW=$DR} becomes the command name.
            DRAW_ENV=()
            [ -n "$DR" ] && DRAW_ENV=(DRAW="$DR")
            env SUBJECTS="$S" ANCHORS="$ANCHORS" "${DRAW_ENV[@]+${DRAW_ENV[@]}}" \
                EPOCHS="$EPOCHS" SUFFIX="$SUFFIX" INIT_CKPT=none \
                CONFIG_SRC="$CFG" EXP_PREFIX="e07d${D_}" CKPT_ROOT="$CKPT_ROOT" \
                sbatch --job-name="e07d${D_}_s${S}${DR:+_d$DR}" --time="$WALL" "$SBATCH"
            sleep 2
        done
    done
done

# --- Convergence check ------------------------------------------------------
# One-sided, at the two deepest arms. If ep1600 BEATS its ep800 sibling on the
# selection metric, 800 was not enough and the whole arm reruns longer. If it
# ties or turns over earlier, 800 stands and we have measured it. This is the
# check that caught E0.5 being undertrained at 400.
if [ "$SUFFIX" = "_d" ]; then
  for D_ in 32 44; do
    N=$((N + 1))
    echo "[$N] depth=$D_ S=701 draw=11 epochs=1600 CONVERGENCE CHECK"
    if [ -z "${FORCE:-}" ] && [ -f "${CKPT_ROOT}/e07d${D_}_s701_a101_ep1600_d11/latest.pth.tar" ]; then
        echo "     skip (exists)"; continue
    fi
    [ -n "${DRY:-}" ] && continue
    env SUBJECTS=701 ANCHORS="$ANCHORS" DRAW=11 EPOCHS=1600 SUFFIX="_ep1600" \
        INIT_CKPT=none CONFIG_SRC="experiments/snr_scaling/clip_pretrain_reve_d${D_}.yaml" \
        EXP_PREFIX="e07d${D_}" CKPT_ROOT="$CKPT_ROOT" \
        sbatch --job-name="e07d${D_}_ep1600" --time=24:00:00 "$SBATCH"
    sleep 2
  done
fi

echo
echo "Per-depth nulls (submit once each depth has a config_probe.yaml):"
for D_ in 22 32 44; do
    echo "  CKPT_ROOT=$CKPT_ROOT OUT_PREFIX=e07d${D_} STAGE=probe_val \\"
    echo "    CONFIG=$CKPT_ROOT/e07d${D_}_s1863_a101${SUFFIX}/config_probe.yaml \\"
    echo "    sbatch --job-name=e07d${D_}_null --time=02:00:00 experiments/snr_scaling/e04_nulls.sbatch"
done
echo
echo "Submitted $N jobs."
echo
echo "AUDIT: min train/loss vs ln(64)=4.159 AND vs siblings. Depth 44 with a"
echo "flat init is the CaiT failure regime -- that is what init_depth_scaled is"
echo "for, but verify it worked rather than assuming it did."
