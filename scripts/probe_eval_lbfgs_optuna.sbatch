#!/bin/bash
#SBATCH --job-name=probe_lbfgs
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lbfgs_optuna_%j.out
#SBATCH --error=logs/lbfgs_optuna_%j.err
# probe_eval.py with linear-movie-probe + L-BFGS + BatchNorm + small WD,
# using the Optuna-picked best config from trial #26.

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"

: "${CKPT:?need CKPT}"; : "${NW:?need NW}"; : "${WS:?need WS}"
: "${BATCH_SIZE:?need BATCH_SIZE}"; : "${PROBE_SEED:?need PROBE_SEED}"
: "${EXP_TAG:?need EXP_TAG}"

if [ ! -f "$CKPT" ]; then echo "ERROR: CKPT not found"; exit 2; fi

cd "$SLURM_SUBMIT_DIR"
module reset; module load miniforge3-python
eval "$(conda shell.bash hook)"; conda activate eb_jepa

CONDA_PY="/u/kkokate/.conda/envs/eb_jepa/bin/python"
PRED_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/predictions/lbfgs_optuna/${EXP_TAG}"
mkdir -p "$PRED_DIR"

PYTHONPATH=. "$CONDA_PY" experiments/eeg_jepa/probe_eval.py \
    --checkpoint="${CKPT}" \
    --n_windows="${NW}" --window_size_seconds="${WS}" \
    --batch_size="${BATCH_SIZE}" --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording --corrca_filters=corrca_filters.npz \
    --keep_channels \
    --linear_movie_probe \
    --probe_method=lbfgs \
    --probe_lr=1.96e-4 \
    --probe_weight_decay=1.77e-4 \
    --input_batchnorm \
    --seed="${PROBE_SEED}" \
    --save_predictions_dir="${PRED_DIR}" \
    --wandb_group="lbfgs_optuna_keepch"

echo "lbfgs_optuna ${EXP_TAG} done"
