#!/bin/bash
#SBATCH --job-name=v10diag
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/v10diag_%j.out
#SBATCH --error=logs/v10diag_%j.err
# CPU-only diagnostic, but bbnv-delta-gpu only has GPU partitions.
# Reads issue8E saved predictions to test the doc-stated mechanism
# behind nw2_ws4 lum_corr's bootstrap-t-test failure.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ISSUE8_VENV_PY="/projects/bbnv/kkokate/eb_jepa_eeg/.venv/bin/python"
"$ISSUE8_VENV_PY" scripts/diagnose_lum_seed_agreement.py
