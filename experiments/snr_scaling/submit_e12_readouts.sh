#!/bin/bash
# ---------------------------------------------------------------
# Fan out the E1.2 (CBraMod) readouts, one job per stage per cell, plus the
# shape-matched nulls.
#
# Every cell is read at FIXED epoch 325 -- the epoch the paper reports every
# arm at (RESULTS_epoch_curve.md: "Not a protocol change. The paper still
# reports at fixed epoch 325"). The whole 25-epoch grid is on disk, so a
# per-cell selection can be added later without retraining.
#
# Stages (e04_readouts.sbatch, unchanged):
#   probe_val   12-feature CV probe on val -> the Delta-r^2 headline
#   tt_test     train->test ridge probe with bootstrap -> r / ceiling on test
#   retr_test   scene retrieval on test (never val; RESULTS.md 2.12c)
# tt_val is deliberately NOT run for the preliminary pass: it is the tt
# protocol on the split probe_val already covers, at ~2 GPU-h per cell.
#
# NULLS. Delta-r^2 is measured against an UNTRAINED encoder of identical shape
# (e04_nulls.sbatch, --random-baseline). CBraMod is a new shape, so it needs
# its own four nulls; they are written under the warm prefix (e12cb_*_random)
# and COPIED to the random prefix (e12cbr_*_random) by `nulls-copy`, because
# aggregate_nested.py insists a null's filename prefix match the arm it is
# subtracted from. The two arms have byte-identical config_probe.yaml apart
# from the cell dir, so the copy is the same measurement, not a shortcut.
#
# Usage (repo root, ON DELTA):
#   bash experiments/snr_scaling/submit_e12_readouts.sh nulls        # 4 jobs
#   bash experiments/snr_scaling/submit_e12_readouts.sh warm         # 13 x 3
#   bash experiments/snr_scaling/submit_e12_readouts.sh random
#   bash experiments/snr_scaling/submit_e12_readouts.sh nulls-copy   # after nulls land
# DRY=1 prints without submitting. EPOCH=<n> overrides 325. Stages whose
# artifact exists, and cells whose checkpoint is missing, are skipped.
# ---------------------------------------------------------------
set -euo pipefail

WHAT="${1:-}"
case "$WHAT" in nulls|warm|random|both|nulls-copy) ;;
    *) echo "usage: $0 nulls|warm|random|both|nulls-copy" >&2; exit 2 ;; esac

NESTED_S=(50 200 701 1400)
DRAWS=(11 22 33)
FULL_POOL_S=1863
ANCHORS=101
EPOCH="${EPOCH:-325}"

READOUT_SBATCH=experiments/snr_scaling/e04_readouts.sbatch
NULLS_SBATCH=experiments/snr_scaling/e04_nulls.sbatch
CKPT_ROOT="${CKPT_ROOT:-/work/hdd/bbnv/kkokate/eb_jepa/e12_cbramod_scaling}"
OUT=experiments/snr_scaling/raw_results
STAGES=(probe_val tt_test retr_test)
declare -A TIME=( [probe_val]=01:00:00 [tt_val]=02:00:00
                  [tt_test]=02:00:00  [retr_test]=01:00:00 )

if [ ! -f "$READOUT_SBATCH" ] || [ ! -f "$NULLS_SBATCH" ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi
mkdir -p logs "$OUT"

cells() {  # $1 prefix -> slugs, one per line
    local prefix=$1
    for S in "${NESTED_S[@]}"; do
        for D in "${DRAWS[@]}"; do echo "${prefix}_s${S}_a${ANCHORS}_nd_d${D}"; done
    done
    echo "${prefix}_s${FULL_POOL_S}_a${ANCHORS}_nd"
}

N=0
SKIP=0

submit_readouts() {
    local prefix=$1
    while read -r SLUG; do
        local ckpt="${CKPT_ROOT}/${SLUG}/epoch_${EPOCH}.pth.tar"
        if [ ! -s "$ckpt" ]; then
            echo "  no checkpoint yet: $ckpt" >&2
            continue
        fi
        for STAGE in "${STAGES[@]}"; do
            if [ -s "${OUT}/${prefix}_${STAGE}_${SLUG}.json" ]; then
                SKIP=$((SKIP + 1)); continue
            fi
            N=$((N + 1))
            echo "[$N] ${SLUG} ${STAGE} (${TIME[$STAGE]})"
            if [ -z "${DRY:-}" ]; then
                SLUG="$SLUG" CKPT="$ckpt" STAGE="$STAGE" OUT_PREFIX="$prefix" \
                    CKPT_ROOT="$CKPT_ROOT" \
                    sbatch --job-name="ro_${STAGE}_${SLUG#e12}" \
                           --time="${TIME[$STAGE]}" "$READOUT_SBATCH" >/dev/null
                sleep 1
            fi
        done
    done < <(cells "$prefix")
}

case "$WHAT" in
    nulls)
        # Any cell's config_probe.yaml carries the shape; the whole-pool warm
        # cell is the canonical one (matches e04_nulls.sbatch's own default).
        CONFIG="${CKPT_ROOT}/e12cb_s${FULL_POOL_S}_a${ANCHORS}_nd/config_probe.yaml"
        if [ ! -f "$CONFIG" ]; then
            echo "ERROR: no $CONFIG -- train that cell first" >&2; exit 1
        fi
        for STAGE in probe_val tt_val tt_test retr_test; do
            if [ -s "${OUT}/e12cb_${STAGE}_random.json" ]; then SKIP=$((SKIP + 1)); continue; fi
            N=$((N + 1))
            echo "[$N] null ${STAGE} (${TIME[$STAGE]})"
            if [ -z "${DRY:-}" ]; then
                CONFIG="$CONFIG" OUT_PREFIX=e12cb STAGE="$STAGE" CKPT_ROOT="$CKPT_ROOT" \
                    sbatch --job-name="e12_null_${STAGE}" --time="${TIME[$STAGE]}" \
                           "$NULLS_SBATCH" >/dev/null
            fi
        done
        ;;
    nulls-copy)
        for STAGE in probe_val tt_val tt_test retr_test; do
            src="${OUT}/e12cb_${STAGE}_random.json"; dst="${OUT}/e12cbr_${STAGE}_random.json"
            if [ ! -s "$src" ]; then echo "  missing $src" >&2; continue; fi
            cp -v "$src" "$dst"
        done
        ;;
    warm)   submit_readouts e12cb ;;
    random) submit_readouts e12cbr ;;
    both)   submit_readouts e12cb; submit_readouts e12cbr ;;
esac

echo "Submitted $N job(s); skipped $SKIP already on disk."
