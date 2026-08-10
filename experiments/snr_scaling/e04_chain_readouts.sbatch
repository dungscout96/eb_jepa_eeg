#!/bin/bash
#SBATCH --job-name=e04_chain
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/e04_chain_%j.out
#SBATCH --error=logs/e04_chain_%j.err
# ---------------------------------------------------------------
# Select one cell's best epoch, then submit its four readout stages.
#
# Exists so a rescued cell needs no babysitting: chain it with
#   sbatch --dependency=afterok:<training_jobid> ...
# and the readouts appear the moment training succeeds. afterok, not afterany,
# so a cell that fails does NOT silently get probed at a garbage epoch.
#
# Needs no GPU -- it parses the wandb history and shells out to sbatch -- but
# this account has no CPU-partition allocation, so it takes a GPU slot anyway.
# Kept to 30 min so it backfills; a 4 h request on this partition has waited 24 h.
#
# Required env var:
#   SLUG (e.g. e04_s400_a101_nd_d33)
# ---------------------------------------------------------------

set -euo pipefail

if [ -z "${SLUG:-}" ]; then
    echo "ERROR: SLUG env var must be set" >&2
    exit 1
fi

export UV_LINK_MODE=copy
export PATH="$HOME/.conda/envs/eb_jepa/bin:$HOME/.local/bin:$PATH"

if [ ! -d eb_jepa/training ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi

CKPT_ROOT=/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling
SEL=experiments/snr_scaling/raw_results/e04_selection_${SLUG}.json
OUT=experiments/snr_scaling/raw_results

echo "Job $SLURM_JOB_ID  chaining readouts for ${SLUG}"

if [ ! -f "${CKPT_ROOT}/${SLUG}/latest.pth.tar" ]; then
    echo "ERROR: ${SLUG} has no latest.pth.tar -- training did not finish" >&2
    exit 1
fi

PYTHONPATH=. uv run --group eeg python \
    experiments/snr_scaling/select_and_probe_e03.py \
    --ckpt-root "$CKPT_ROOT" \
    --cells-glob "$SLUG" \
    --output "$SEL"

CKPT=$(PYTHONPATH=. uv run --group eeg python -c "
import json
d = json.load(open('${SEL}'))
v = d.get('${SLUG}') or {}
print(v.get('checkpoint', ''))
")

if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint selected for ${SLUG}" >&2
    exit 1
fi
echo "selected: $CKPT"

declare -A TIME=( [probe_val]=01:00:00 [tt_val]=02:00:00
                  [tt_test]=02:00:00  [retr_test]=01:00:00 )

for STAGE in probe_val tt_val tt_test retr_test; do
    if [ -s "${OUT}/e04_${STAGE}_${SLUG}.json" ]; then
        echo "skip ${STAGE} (exists)"
        continue
    fi
    SLUG="$SLUG" CKPT="$CKPT" STAGE="$STAGE" \
        sbatch --job-name="ro_${STAGE}_${SLUG#e04_s}" \
               --time="${TIME[$STAGE]}" \
               experiments/snr_scaling/e04_readouts.sbatch
    sleep 1
done

echo "Chained readouts for ${SLUG}"
