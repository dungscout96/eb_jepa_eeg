#!/bin/bash
# ---------------------------------------------------------------
# Submit the E0.5 random-init control arm: 28 cells + 2 convergence checks.
#
# WHY THIS ARM EXISTS. E0.4 moved three variables at once -- depth 12->22,
# patch geometry 400/0 -> 200/20, and init random -> reve-base -- so its 1.73x
# gain over E0.3 cannot be attributed to any one of them. The repo already has
# the depth-alone answer (jul1 iter 14: depth 12->22 from scratch at patch 400
# = +0.00009, a null) and the geometry-alone answer (jul1 iter 11: patch 200
# -> 400 = +0.0109, i.e. E0.4's geometry is the WORSE one from scratch). This
# arm supplies the third: same shape as E0.4, same recipe, random init.
#
# It is a PAIRED design. Every cell reuses E0.4's subsample_seed, so cell
# e05_s701_a101_nd_d11 trains on the identical 701-subject cohort as
# e04_s701_a101_nd_d11 and the two differ only in init. Do not renumber the
# draws -- the pairing is what buys the statistical power at n=3.
#
# EPOCHS=400 is NOT a guess and must not be "improved" upward. Two reasons:
#   1. Comparability. epoch_size=703 pins 11 steps/epoch, so 400 epochs is
#      exactly 4400 gradient steps -- identical to every E0.4 and E0.3 cell.
#      Change it and the arm no longer isolates init.
#   2. Longer actively hurts here. scene_clip_from_checkpoint/RESULTS.md 3.2:
#      the 1000-epoch run at lr=1e-4 sits at val Δr ~ +0.11 across ep500-999,
#      worse than its ep299 sibling at +0.198. jul1 saw the same past ep500.
# The open question -- whether a from-scratch encoder needs more steps than a
# warm-started one -- is answered by the two CONVERGENCE cells below rather
# than assumed in either direction.
#
# Early stopping is already in the protocol and needs nothing here:
# select_and_probe_e03.py takes each cell's SMOOTHED argmax of
# val/clip_scene_auc and probes that epoch, not the last one. SAVE_EVERY=25
# (the E0.4 default) gives it the checkpoints to snap to.
#
# The shape-matched null is UNCHANGED: this arm has E0.4's exact shape, so it
# reuses the depth-22 null (r^2 = 0.016572). Do not rerun e04_nulls.sbatch.
#
# Run from the repo root ON DELTA, after checking the branch out there
# (delta-deploy SKILL.md 2.2 -- never git checkout inside an sbatch, concurrent
# jobs race on .git/index.lock):
#
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && \
#              git fetch origin && git checkout <branch> && git pull --ff-only"
#   ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && mkdir -p logs && \
#              bash experiments/snr_scaling/submit_e05_random_sweep.sh"
#
# Set DRY=1 to print the submissions without sbatching them.
# ---------------------------------------------------------------
set -euo pipefail

# Matches the E0.4 grid cell-for-cell (28 cells: 9 scales x 3 draws + S=1863).
NESTED_S=(10 20 50 100 200 400 701 1000 1400)
DRAWS=(11 22 33)
FULL_POOL_S=1863
ANCHORS=101

SBATCH=experiments/snr_scaling/train_e04_cell.sbatch

export EXP_PREFIX=e05
export INIT_CKPT=none
export CKPT_ROOT=/work/hdd/bbnv/kkokate/eb_jepa/e05_random_scaling

# MEASURED 2026-08-13, not a guess: at 400 epochs 11 of 13 cells with S>=400
# peaked at ep399 on val/clip_scene_auc -- still climbing when the run ended.
# At 800 the same cells peak at ~690 and then decay ~4%, the converged shape.
# So the random-init arm needs ~2x the schedule a warm-started one does
# (E0.4: only 6 of 14 cells at S>=400 peaked late).
#
# EPOCHS applies to the WHOLE grid or none of it. Running small S at 400 and
# large S at 800 would confound the S axis with optimisation budget, and the
# slope is the entire contribution of this experiment.
EPOCHS="${EPOCHS:-400}"
SUFFIX="${SUFFIX:-_nd}"
export EPOCHS SUFFIX
WALL="${WALL:-03:00:00}"

# Idempotent: a cell that already has latest.pth.tar is not resubmitted, so
# this script doubles as the rescue/top-up path. Set FORCE=1 to override.
submitted_or_skipped() {
    local exp_id="$1"
    if [ -z "${FORCE:-}" ] && [ -f "${CKPT_ROOT}/${exp_id}/latest.pth.tar" ]; then
        echo "     skip (latest.pth.tar exists)"
        return 0
    fi
    return 1
}

if [ ! -f "$SBATCH" ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi

mkdir -p logs

N=0
for S in "${NESTED_S[@]}"; do
  for D in "${DRAWS[@]}"; do
    N=$((N + 1))
    echo "[$N] S=$S A=$ANCHORS draw=$D  (random init, ${EPOCHS} ep)"
    submitted_or_skipped "e05_s${S}_a${ANCHORS}${SUFFIX}_d${D}" && continue
    if [ -z "${DRY:-}" ]; then
        SUBJECTS="$S" ANCHORS="$ANCHORS" DRAW="$D" \
            sbatch --job-name="e05_s${S}_d${D}" --time="$WALL" "$SBATCH"
        sleep 2   # stagger: avoids wandb-init and file-creation races
    fi
  done
done

# The whole-pool cell takes no DRAW: there is only one possible draw of 1863.
N=$((N + 1))
echo "[$N] S=$FULL_POOL_S A=$ANCHORS draw=none (whole pool, random init, ${EPOCHS} ep)"
if ! submitted_or_skipped "e05_s${FULL_POOL_S}_a${ANCHORS}${SUFFIX}"; then
  if [ -z "${DRY:-}" ]; then
    SUBJECTS="$FULL_POOL_S" ANCHORS="$ANCHORS" \
        sbatch --job-name="e05_s${FULL_POOL_S}" --time="$WALL" "$SBATCH"
    sleep 2
  fi
fi

# --- Convergence check -------------------------------------------------------
# Two cells at double the schedule, at the two S where E0.3 and E0.4 diverge
# most. SUFFIX keeps them out of the `_nd` aggregation glob: they are a
# schedule-length probe, not extra cohort draws, and averaging them into the
# S=701 / S=1400 rows would silently mix two optimisation budgets.
#
# Read it as a one-sided test. If ep800 BEATS its ep400 sibling, 4400 steps was
# not enough for a from-scratch encoder and the whole arm must be rerun longer
# (at which point E0.4 needs the same treatment to stay comparable). If it ties
# or loses -- what RESULTS.md 3.2 predicts -- 400 stands, and we have measured
# that rather than assumed it.
# ALREADY RUN and already answered -- see the EPOCHS note above. Kept for the
# record, skipped unless CONVERGENCE_CHECK=1, and never fired when the grid is
# itself running at 800 (it would duplicate two cells of it).
if [ "${CONVERGENCE_CHECK:-0}" = "1" ] && [ "$SUFFIX" = "_nd" ]; then
  for S in 701 1400; do
    N=$((N + 1))
    echo "[$N] S=$S A=$ANCHORS draw=11 (random init, 800 ep, CONVERGENCE CHECK)"
    if [ -z "${DRY:-}" ]; then
        SUBJECTS="$S" ANCHORS="$ANCHORS" DRAW=11 EPOCHS=800 SUFFIX="_ep800" \
            sbatch --job-name="e05_s${S}_d11_ep800" --time=04:00:00 "$SBATCH"
        sleep 2
    fi
  done
fi

echo "Submitted $N jobs."
echo
echo "AUDIT BEFORE TRUSTING ANY CELL: a run can sit at exactly chance"
echo "(InfoNCE loss = ln 64 = 4.159) and still pass every artifact check."
echo "3 of 28 E0.4 cells did. From-scratch depth-22 is MORE exposed -- this"
echo "backbone has no LayerScale, no stochastic depth and a flat 0.02 init."
echo "Check min train/loss against siblings, not against an absolute threshold."
