#!/bin/bash
# ---------------------------------------------------------------
# Submit the E1.2 CBraMod subject-scaling sweep: two arms, one HBN-free init.
#
#   warm    e12cb_*   CBraMod geometry, warm-started from the TUEG-pretrained
#                     release (cbramod_eet_init.pth.tar).
#   random  e12cbr_*  Identical geometry, recipe, cohorts and seed; no warm
#                     start. The PAIRED control: cell e12cbr_s701_a101_nd_d11
#                     trains on the same 701 subjects as e12cb_s701_a101_nd_d11.
#
# WHY. E0.4/E0.5 measured that the pretrained init sets the LEVEL of the
# subject-scaling curve (~1.7x Delta-r^2 at every S) while subjects keep buying
# signal to the top of the axis. That warm start was reve-base, whose
# pretraining corpus contains HBN R1-R9 -- including R5 (val) and R6 (test).
# CBraMod has never seen HBN, so this sweep asks whether the same two facts
# hold for a warm start with zero subject exposure. See PLAN_cbramod.md.
#
# CELLS. S in {50, 200, 701, 1400} x draws {11, 22, 33}, plus the whole-pool
# S=1863 cell, all at A=101 -- a 5-point slice of the E0.4 axis spanning 1.6
# decades, with the 701->1863 step that tests "keeps climbing past 700". Draws
# nest within a seed exactly as in E0.4, so a curve measures ADDED subjects.
# 13 cells per arm, 26 in total.
#
# PROTOCOL (all inherited from train_e04_cell.sbatch, nothing new):
#   pool R1-R4 + R7-R10 (1863), val R5, test R6 -- E0.4's split, so the CV
#   probe on val stays the headline readout and the same nulls protocol
#   applies; 400 epochs at epoch_size=703 = 4400 steps in every cell;
#   soft_target_clip alpha 0.5 / tau 0.05; meta.seed 2026; save_every 25.
#
# Time limit: CBraMod is 4.9M parameters on 258 tokens, so a cell is bounded
# by data loading rather than compute. 02:00:00 is a first guess with margin;
# read the first cell's s/epoch from its log and tighten (or set TIME_LIMIT).
#
# Run from the repo root ON DELTA, after checking the branch out there
# (never git checkout inside an sbatch; concurrent jobs race on .git/index.lock):
#
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && \
#              git fetch origin && git checkout cbramod-experiments && git pull --ff-only"
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && mkdir -p logs && \
#              bash experiments/snr_scaling/submit_e12_cbramod_sweep.sh both"
#
# Usage: submit_e12_cbramod_sweep.sh [warm|random|both]     (default: both)
# Set DRY=1 to print the submissions without sbatching them. Cells that
# already have latest.pth.tar are skipped, so re-running is safe.
# ---------------------------------------------------------------
set -euo pipefail

ARM="${1:-both}"
case "$ARM" in warm|random|both) ;; *) echo "usage: $0 [warm|random|both]" >&2; exit 2 ;; esac

NESTED_S=(50 200 701 1400)
DRAWS=(11 22 33)
FULL_POOL_S=1863
ANCHORS=101

SBATCH=experiments/snr_scaling/train_e04_cell.sbatch
CONFIG_SRC=experiments/snr_scaling/clip_pretrain_cbramod.yaml
CKPT_ROOT="${CKPT_ROOT:-/work/hdd/bbnv/kkokate/eb_jepa/e12_cbramod_scaling}"
INIT_WARM="${INIT_WARM:-/work/hdd/bbnv/kkokate/eb_jepa/cbramod_eet_init.pth.tar}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"

if [ ! -f "$SBATCH" ] || [ ! -f "$CONFIG_SRC" ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi
if [ "$ARM" != "random" ] && [ ! -s "$INIT_WARM" ]; then
    echo "ERROR: no warm-start checkpoint at $INIT_WARM -- run" >&2
    echo "  prepare_cbramod_checkpoint.py --output $INIT_WARM  first" >&2
    exit 1
fi

mkdir -p logs

N=0
SKIP=0

submit_cell() {
    # $1 prefix, $2 init (path or "none"), $3 S, $4 draw ("" for whole pool)
    local prefix=$1 init=$2 s=$3 draw=$4
    local exp_id
    if [ -n "$draw" ]; then
        exp_id="${prefix}_s${s}_a${ANCHORS}_nd_d${draw}"
    else
        exp_id="${prefix}_s${s}_a${ANCHORS}_nd"
    fi
    if [ -s "${CKPT_ROOT}/${exp_id}/latest.pth.tar" ]; then
        SKIP=$((SKIP + 1))
        return
    fi
    N=$((N + 1))
    echo "[$N] ${exp_id}  init=${init}"
    if [ -z "${DRY:-}" ]; then
        # Per-cell job name: squeue stays readable and `scancel --name=<cell>`
        # targets one cell. DRAW is only exported when set -- the sbatch reads
        # an unset DRAW as "whole pool", which is the S=1863 convention.
        if [ -n "$draw" ]; then
            SUBJECTS="$s" ANCHORS="$ANCHORS" DRAW="$draw" SUFFIX=_nd \
                EXP_PREFIX="$prefix" INIT_CKPT="$init" CONFIG_SRC="$CONFIG_SRC" \
                CKPT_ROOT="$CKPT_ROOT" \
                sbatch --job-name="$exp_id" --time="$TIME_LIMIT" "$SBATCH"
        else
            SUBJECTS="$s" ANCHORS="$ANCHORS" SUFFIX=_nd \
                EXP_PREFIX="$prefix" INIT_CKPT="$init" CONFIG_SRC="$CONFIG_SRC" \
                CKPT_ROOT="$CKPT_ROOT" \
                sbatch --job-name="$exp_id" --time="$TIME_LIMIT" "$SBATCH"
        fi
        sleep 2   # stagger: avoids wandb-init and file-creation races
    fi
}

submit_arm() {
    local prefix=$1 init=$2
    for S in "${NESTED_S[@]}"; do
        for D in "${DRAWS[@]}"; do
            submit_cell "$prefix" "$init" "$S" "$D"
        done
    done
    submit_cell "$prefix" "$init" "$FULL_POOL_S" ""
}

if [ "$ARM" != "random" ]; then submit_arm e12cb "$INIT_WARM"; fi
if [ "$ARM" != "warm" ]; then submit_arm e12cbr none; fi

echo "Submitted $N job(s); skipped $SKIP already trained."
