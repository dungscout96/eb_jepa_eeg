#!/bin/bash
#SBATCH --job-name=hbn_preprocess
#SBATCH --account=csd403
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err
#SBATCH --array=0-2
# ---------------------------------------------------------------
# HBN EEG Preprocessing on Expanse (SDSC)
#
# Submits a SLURM array job that processes Release 1 for each of
# the three prioritized tasks in parallel.
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/preprocess_hbn.sbatch
#
# Override release:
#   sbatch --export=ALL,HBN_RELEASE=R2 scripts/preprocess_hbn.sbatch
# ---------------------------------------------------------------

set -euo pipefail

# --- Configuration ---
TASKS=("ThePresent" "DespicableMe" "RestingState")
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
RELEASE=${HBN_RELEASE:-R1}

# Paths are auto-detected; override with env vars if needed
export UV_LINK_MODE=copy

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Release:       $RELEASE"
echo "Task:          $TASK"
echo "========================================"

cd /expanse/projects/nemar/dtyoung/eb_jepa_eeg

export PATH=/home/dtyoung/.local/bin:$PATH

uv run --group eeg scripts/preprocess_hbn.py \
    release="$RELEASE" \
    task="$TASK"

echo "Done: release=$RELEASE task=$TASK"
