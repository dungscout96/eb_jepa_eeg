#!/bin/bash
#SBATCH --job-name=hbn_preprocess
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err
#SBATCH --array=0-3
# ---------------------------------------------------------------
# HBN EEG Preprocessing on Delta (NCSA)
#
# Submits a SLURM array job that processes Release 1 for each of
# the four prioritized tasks in parallel.
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
TASKS=("ThePresent" "DespicableMe" "RestingState" "ContrastChangeDetection")
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
RELEASE=${HBN_RELEASE:-R1}

# Save preprocessed data to projects filesystem
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Release:       $RELEASE"
echo "Task:          $TASK"
echo "Output dir:    $HBN_PREPROCESS_DIR"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"

# Load conda and activate environment
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

uv run --group eeg scripts/preprocess_hbn.py \
    release="$RELEASE" \
    task="$TASK" \
    output_dir="$HBN_PREPROCESS_DIR"

echo "Done: release=$RELEASE task=$TASK"
