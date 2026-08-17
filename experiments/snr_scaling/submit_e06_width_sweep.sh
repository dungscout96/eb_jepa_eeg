#!/bin/bash
# ---------------------------------------------------------------
# E0.6 -- width scaling at fixed depth, random init, converged budget.
#
# THE QUESTION. E0.5 showed depth 12 -> 22 is a poor lever: it LOSES below
# S~700 and returns only +24% at S=1863, for 1.83x the parameters, 2x the
# tokens and 2x the steps. Width is the alternative, and it is what REVE
# themselves scaled (base 512 -> large 1216, depth held at 22).
#
# WHY THE OLD ANSWER DOESN'T COUNT. jul1 iter 13 tested width 512 -> 768 and
# got -0.00025, "width saturated". It ran at 300 epochs. E0.5 then showed a
# depth-22 model is still climbing at 400 epochs and only turns over near 800.
# Every jul1 capacity ablation (iter 13 width, 14 depth, 15 mlp_ratio) was
# therefore measured at a budget too short for the model under test, which
# biases all three toward "no effect". This arm retests width where the answer
# is trustworthy.
#
# DESIGN. Only width moves. Depth 22, patch 200/20, mlp_dim_ratio 2.66,
# proj_dim 512, lr, batch and n_windows are pinned by the frozen snapshots.
# Random init throughout, so this is a capacity result and not a warm-start
# result -- there is no pretrained checkpoint at 768 anyway.
#
#   width 512  = E0.5, already run (the reference; not resubmitted)
#   width 768  -> clip_pretrain_reve_w768.yaml   155.6M
#   width 1216 -> clip_pretrain_reve_w1216.yaml  390.0M  (reve-large's width)
#
# S is deliberately just {701, 1863}: E0.5 showed capacity only starts paying
# above ~700, so those two points bracket the effect. A full curve costs 5x
# more and answers nothing extra until we know the sign.
#
# WALL TIME. The "4 h account cap" in delta-deploy SKILL.md is WRONG --
# gpuA40x4's real limit is 2 days (verified: a 20 h job is accepted). Measured
# 512-wide at 800 epochs = 1 h 09 - 2 h 25, so 768 (~2.25x FLOPs) gets 10 h and
# 1216 (~5.6x) gets 24 h. This matters: chaining via meta.resume_from would NOT
# have been equivalent, because clip_pretrain.py:500-503 applies a FRESH LR
# schedule on resume -- two 400-epoch legs re-warm and re-anneal rather than
# continuing one 800-epoch cosine.
#
# NULLS ARE NOT OPTIONAL. Delta-r2 is measured against a random encoder of the
# SAME shape, and probe capacity grows with readout dimensionality. Reusing the
# 512-wide null (r2 = 0.016572) at width 1216 would inflate every number. One
# null job per width, submitted first.
#
# Run from the repo root ON DELTA, after checking the branch out there:
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && \
#              git fetch origin && git checkout <branch> && git pull --ff-only"
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && mkdir -p logs && \
#              bash experiments/snr_scaling/submit_e06_width_sweep.sh"
#
# DRY=1 prints without submitting.
# ---------------------------------------------------------------
set -euo pipefail

ANCHORS=101
DRAWS=(11 22 33)
EPOCHS="${EPOCHS:-800}"
SUFFIX="${SUFFIX:-_w}"

SBATCH=experiments/snr_scaling/train_e04_cell.sbatch
NULLS=experiments/snr_scaling/e04_nulls.sbatch
CKPT_ROOT=/work/hdd/bbnv/kkokate/eb_jepa/e06_width_scaling

if [ ! -f "$SBATCH" ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi
mkdir -p logs

# width : config snapshot : wall limit
WIDTHS=("768:clip_pretrain_reve_w768.yaml:10:00:00"
        "1216:clip_pretrain_reve_w1216.yaml:24:00:00")

N=0
for SPEC in "${WIDTHS[@]}"; do
    W="${SPEC%%:*}"
    REST="${SPEC#*:}"
    CFG="experiments/snr_scaling/${REST%%:*}"
    WALL="${REST#*:}"

    if [ ! -f "$CFG" ]; then
        echo "ERROR: missing frozen config $CFG" >&2
        exit 1
    fi

    for S in 701 1863; do
        # S=1863 is the whole pool: exactly one draw of it exists.
        if [ "$S" = "1863" ]; then
            CELL_DRAWS=("")
        else
            CELL_DRAWS=("${DRAWS[@]}")
        fi
        for D in "${CELL_DRAWS[@]}"; do
            if [ -n "$D" ]; then
                EXP_ID="e06w${W}_s${S}_a${ANCHORS}${SUFFIX}_d${D}"
            else
                EXP_ID="e06w${W}_s${S}_a${ANCHORS}${SUFFIX}"
            fi
            N=$((N + 1))
            echo "[$N] width=$W S=$S draw=${D:-none} epochs=$EPOCHS wall=$WALL"
            if [ -z "${FORCE:-}" ] && [ -f "${CKPT_ROOT}/${EXP_ID}/latest.pth.tar" ]; then
                echo "     skip (latest.pth.tar exists)"
                continue
            fi
            [ -n "${DRY:-}" ] && continue
            SUBJECTS="$S" ANCHORS="$ANCHORS" ${D:+DRAW="$D"} \
                EPOCHS="$EPOCHS" SUFFIX="$SUFFIX" INIT_CKPT=none \
                CONFIG_SRC="$CFG" EXP_PREFIX="e06w${W}" CKPT_ROOT="$CKPT_ROOT" \
                sbatch --job-name="e06w${W}_s${S}${D:+_d$D}" --time="$WALL" "$SBATCH"
            sleep 2
        done
    done
done

echo
echo "Submit the two width nulls AFTER the first cell of each width finishes"
echo "(they need that cell's config_probe.yaml):"
for W in 768 1216; do
    echo "  CKPT_ROOT=$CKPT_ROOT OUT_PREFIX=e06w${W} STAGE=probe_val \\"
    echo "    CONFIG=$CKPT_ROOT/e06w${W}_s1863_a101${SUFFIX}/config_probe.yaml \\"
    echo "    sbatch --job-name=e06w${W}_null --time=02:00:00 $NULLS"
done
echo
echo "Submitted $N training jobs."
echo
echo "AUDIT: check min train/loss against ln(64)=4.159 and against SIBLINGS"
echo "before trusting any cell. Wider models at random init have no LayerScale,"
echo "no stochastic depth and a flat 0.02 init -- the CaiT failure regime."
