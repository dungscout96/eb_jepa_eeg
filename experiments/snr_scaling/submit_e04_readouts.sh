#!/bin/bash
# ---------------------------------------------------------------
# Fan the E0.4 readouts out as one job PER STAGE per cell.
#
# Why not one job per cell: the monolithic form asks for 4 h, and a 4 h request
# backfills badly. Nine of them sat unstarted for 6.5 h behind a ~1500-job
# queue while 23 nodes idled. Split, each stage asks for 1-2 h and slots into
# scheduler gaps. The stages are already idempotent (each skips when its output
# exists), so splitting costs nothing and a re-run is free.
#
# Reads the selection JSON so every cell is probed at its SELECTED epoch, not
# latest.pth.tar. Cells whose readout already exists are skipped.
#
# Run from the repo root ON DELTA:
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && \
#              bash experiments/snr_scaling/submit_e04_readouts.sh"
#
# DRY=1 prints without submitting.
# ---------------------------------------------------------------
set -euo pipefail

SEL=experiments/snr_scaling/raw_results/e04_selection.json
SBATCH=experiments/snr_scaling/e04_readouts.sbatch
OUT=experiments/snr_scaling/raw_results

if [ ! -f "$SEL" ]; then
    echo "ERROR: no selection at $SEL -- run select_and_probe_e03.py first" >&2
    exit 1
fi
if [ ! -f "$SBATCH" ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi

mkdir -p logs

# tt_* refit the head on the full 1863-recording train pool, so they need
# roughly 3x what the val-only stages need.
declare -A TIME=( [probe_val]=01:00:00 [tt_val]=02:00:00
                  [tt_test]=02:00:00  [retr_test]=01:00:00 )

N=0
SKIP=0
while read -r SLUG CKPT; do
    [ -z "$SLUG" ] && continue
    for STAGE in probe_val tt_val tt_test retr_test; do
        if [ -s "${OUT}/e04_${STAGE}_${SLUG}.json" ]; then
            SKIP=$((SKIP + 1))
            continue
        fi
        N=$((N + 1))
        echo "[$N] ${SLUG} ${STAGE} (${TIME[$STAGE]})"
        if [ -z "${DRY:-}" ]; then
            SLUG="$SLUG" CKPT="$CKPT" STAGE="$STAGE" \
                sbatch --job-name="ro_${STAGE}_${SLUG#e04_s}" \
                       --time="${TIME[$STAGE]}" "$SBATCH" >/dev/null
            sleep 1
        fi
    done
done < <(python3 -c "
import json
d = json.load(open('${SEL}'))
for slug, v in d.items():
    if isinstance(v, dict) and v.get('checkpoint'):
        print(slug, v['checkpoint'])
")

echo "Submitted $N stage job(s); skipped $SKIP already on disk."
