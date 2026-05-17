#!/bin/bash
#SBATCH --job-name=canonical_agg
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/canonical_agg_%j.out
#SBATCH --error=logs/canonical_agg_%j.err
# ---------------------------------------------------------------
# Aggregate canonical_L2.json across all 5 encoder seeds → L3 + t-test vs chance.
#
# Submit AFTER all canonical_replication.sbatch jobs complete:
#   sbatch --dependency=afterok:<id1>:<id2>:... eb_jepa/training/sbatch/canonical_aggregate.sbatch
# ---------------------------------------------------------------

set -euo pipefail

export UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"

REPL_ROOT="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints"
GLOB="${REPL_ROOT}/canonical_nw2ws4_s*/canonical_L2.json"
OUT_DIR="${REPL_ROOT}/canonical_nw2ws4_L3"
OUT_JSON="${OUT_DIR}/canonical_nw2ws4_L3.json"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "$OUT_DIR"

PYTHONPATH=. uv run --group eeg python scripts/aggregate_and_print.py \
    --name="canonical_nw2ws4_5seed" \
    --glob="${GLOB}" \
    --out="${OUT_JSON}"

echo "Aggregate complete: ${OUT_JSON}"
