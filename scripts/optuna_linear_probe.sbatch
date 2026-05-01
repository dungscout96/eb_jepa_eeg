#!/bin/bash
#SBATCH --job-name=optuna_lin
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/optuna_linear_%j.out
#SBATCH --error=logs/optuna_linear_%j.err
# Optuna TPE sweep over linear-movie-probe hyperparameters.
# 30 trials × ~5 min each ≈ 2.5 hr of sequential trials on one GPU.

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"

CKPT="${CKPT:-/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8/phaseD_nw4ws2_baseline_s42/latest.pth.tar}"
N_TRIALS="${N_TRIALS:-30}"
PROBE_SEED="${PROBE_SEED:-42}"
OUT_DIR="${OUT_DIR:-/projects/bbnv/kkokate/eb_jepa_eeg/predictions/optuna_linear}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

CONDA_PY="/u/kkokate/.conda/envs/eb_jepa/bin/python"

mkdir -p "$OUT_DIR"
PYTHONPATH=. "$CONDA_PY" scripts/optuna_linear_probe.py \
    --ckpt "$CKPT" \
    --probe_seed "$PROBE_SEED" \
    --n_trials "$N_TRIALS" \
    --out_dir "$OUT_DIR"

echo "optuna_linear sweep done"
