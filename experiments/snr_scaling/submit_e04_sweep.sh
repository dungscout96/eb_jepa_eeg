#!/bin/bash
# ---------------------------------------------------------------
# Submit the 13-cell E0.4 full-REVE scaling sweep.
#
# Nested + replicated S axis, matching E0.3 2.11 exactly so the surfaces are
# comparable: S in {400,701,1000,1400} x subsample_seed in {11,22,33}, plus
# S=1863 (the whole pool, so only one draw exists), all at A=101.
#
# Run from the repo root ON DELTA, after pre-checking out the branch there
# (delta-deploy SKILL.md 2.2 -- never git checkout inside an sbatch, concurrent
# jobs race on .git/index.lock):
#
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && \
#              git fetch origin && git checkout <branch> && git pull --ff-only"
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && mkdir -p logs && \
#              bash experiments/snr_scaling/submit_e04_sweep.sh"
#
# Set DRY=1 to print the submissions without sbatching them.
# ---------------------------------------------------------------
set -euo pipefail

NESTED_S=(400 701 1000 1400)
DRAWS=(11 22 33)
FULL_POOL_S=1863
ANCHORS=101

SBATCH=experiments/snr_scaling/train_e04_cell.sbatch

if [ ! -f "$SBATCH" ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi

mkdir -p logs

N=0
for S in "${NESTED_S[@]}"; do
  for D in "${DRAWS[@]}"; do
    N=$((N + 1))
    echo "[$N] S=$S A=$ANCHORS draw=$D"
    if [ -z "${DRY:-}" ]; then
        # Per-cell job name: squeue is unreadable when 13 rows share one name,
        # and it is what makes `scancel --name=<cell>` target a single cell.
        SUBJECTS="$S" ANCHORS="$ANCHORS" DRAW="$D" \
            sbatch --job-name="e04_s${S}_d${D}" "$SBATCH"
        sleep 2   # stagger: avoids wandb-init and file-creation races
    fi
  done
done

# The whole-pool cell takes no DRAW: there is only one possible draw of 1863.
N=$((N + 1))
echo "[$N] S=$FULL_POOL_S A=$ANCHORS draw=none (whole pool)"
if [ -z "${DRY:-}" ]; then
    SUBJECTS="$FULL_POOL_S" ANCHORS="$ANCHORS" \
        sbatch --job-name="e04_s${FULL_POOL_S}" "$SBATCH"
fi

echo "Submitted $N jobs."
