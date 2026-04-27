#!/bin/bash
#SBATCH --job-name=mlp_probe_exp6
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mlp_probe_%j.out
#SBATCH --error=logs/mlp_probe_%j.err
# ---------------------------------------------------------------
# Issue #8 follow-up — re-evaluate Exp 6 best checkpoint(s) with the
# new 2-layer MLP probe head, side-by-side with the linear baseline.
#
# Required env: CKPT, PROBE_TYPE (linear|mlp), HIDDEN_DIM, MOVIE_HIDDEN, EXP_TAG.
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${CKPT:?need CKPT (absolute path to best.pth.tar)}"
: "${PROBE_TYPE:?need PROBE_TYPE (linear|mlp)}"
: "${HIDDEN_DIM:?need HIDDEN_DIM (subject + movie-id probe hidden dim)}"
: "${MOVIE_HIDDEN:?need MOVIE_HIDDEN (MovieFeatureHead hidden dim; 0 = use cfg default 64)}"
: "${EXP_TAG:?need EXP_TAG}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Cfg: ckpt=${CKPT}  probe=${PROBE_TYPE}  hidden=${HIDDEN_DIM}  movie_hidden=${MOVIE_HIDDEN}  tag=${EXP_TAG}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="${CKPT}" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --batch_size=64 \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --probe_type="${PROBE_TYPE}" \
    --probe_hidden_dim="${HIDDEN_DIM}" \
    --movie_probe_hidden_dim="${MOVIE_HIDDEN}" \
    --wandb_group="issue8_mlpprobe_${EXP_TAG}"

echo "Probe eval ${EXP_TAG} complete!"
