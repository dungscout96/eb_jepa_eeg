#!/bin/bash
#SBATCH --job-name=ec_comp
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ec_%j.out
#SBATCH --error=logs/ec_%j.err
# ---------------------------------------------------------------
# Per-epoch held-out probe curve for ONE DRAW of a 2156-pool arm.
#
# WHY COMPLEMENT SELECTION. These arms fold R5 into the pretraining pool, so no
# release is held out and the in-loop val/clip_scene_auc is scored on data the
# model trained on -- the run log shows the val loader reading the same 293 R5
# recordings. The smoothed-AUC selector used on every 1863-pool arm is therefore
# unusable here. epoch_curve.py instead scores on the cell's cohort COMPLEMENT:
# subjects inside the 2156 pool that this particular draw never sampled.
#
# ONE DRAW PER JOB, ENFORCED BY THE CALLER. Cohorts nest within a draw, so the
# complement of S=1863 is held out from every smaller cell of the SAME draw --
# and trained on by other draws. Mixing draws in one job would score some cells
# on their own training data. _submit_epoch_curve.py raises on this; here the
# caller must respect it, so pass cells from a single draw only.
#
# Scan constants are main's (_submit_epoch_curve.py): --n-recordings 80
# --n-fit 60 --seed 42 --alpha 3162.2776601683795. Do not "improve" them -- they
# are what makes our curves comparable to e04/e03/e05rand/addval/fromscratch
# rather than a parallel standard.
#
# Required env vars:
#   ROOT     cell directory root
#   CELLS    space-separated cell names, ALL from one draw
#   CFG      probe config matching the arm's architecture. A cell's own
#            config_probe.yaml works: it carries the right depth/patch AND the
#            train_releases needed to reconstruct the cohort.
#   DRAW     the subsample_seed those cells share
#   OUTNAME  output filename under raw_results/
# ---------------------------------------------------------------

set -euo pipefail

for v in ROOT CELLS CFG DRAW OUTNAME; do
    if [ -z "${!v:-}" ]; then
        echo "ERROR: $v env var must be set" >&2
        exit 1
    fi
done

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/work/hdd/bbnv/dtyoung/hbn_preprocessed}"
export UV_LINK_MODE=copy

if [ ! -d eb_jepa/training ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

OUT=experiments/snr_scaling/raw_results
mkdir -p "$OUT" logs

if [ ! -f "$CFG" ]; then
    echo "ERROR: no probe config at $CFG" >&2
    exit 1
fi

echo "Job $SLURM_JOB_ID  draw=$DRAW  cells: $CELLS"

PYTHONPATH=. uv run --group eeg python eb_jepa/evaluation/clip_probe/epoch_curve.py \
    --ckpt-root "$ROOT" --cells $CELLS --config "$CFG" \
    --split train --holdout-complement-of 1863 --subsample-seed "$DRAW" \
    --device cuda --n-recordings 80 --n-fit 60 --seed 42 \
    --alpha 3162.2776601683795 \
    --output "${OUT}/${OUTNAME}"

# Honest exit code: the artifact, not the exit status of a trailing echo.
if [ ! -s "${OUT}/${OUTNAME}" ]; then
    echo "ERROR: no curve written to ${OUT}/${OUTNAME}" >&2
    exit 1
fi

echo "epoch curve complete: ${OUTNAME}"
