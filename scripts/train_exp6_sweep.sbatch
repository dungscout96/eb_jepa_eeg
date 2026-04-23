#!/bin/bash
#SBATCH --job-name=exp6_sweep
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp6sw_%j.out
#SBATCH --error=logs/exp6sw_%j.err
# ---------------------------------------------------------------
# Exp 6 hyperparameter sweep: VCLoss coefficient x predictor
# bottleneck dim x seed. Each job trains a single cell and runs
# probe_eval on its best checkpoint. Expected ~30-45 min train +
# ~10-15 min eval = ~1 hr per job.
#
# Required env: STD_COEFF, PRED_DIM, SEED.
# Optional:     GIT_BRANCH (default kkokate/exp6-sweep).
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${STD_COEFF:?need STD_COEFF}"
: "${PRED_DIM:?need PRED_DIM}"
: "${SEED:?need SEED}"

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  cfg: std=${STD_COEFF} pd=${PRED_DIM} seed=${SEED}"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

GIT_BRANCH="${GIT_BRANCH:-kkokate/exp6-sweep}"
git checkout "$GIT_BRANCH"

# Unique per-cell output dir. Jobs with same (STD, SEED) but different
# PRED_DIM would otherwise collide since pred_dim is not in the run name.
EXP_ID="std${STD_COEFF}_pd${PRED_DIM}_seed${SEED}"
EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/exp6_sweep/${EXP_ID}"
mkdir -p "$EXP_DIR"

# ---- Train ----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs=100 \
    --optim.early_stopping_patience=20 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim="${PRED_DIM}" \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.num_workers=2 \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --loss.std_coeff="${STD_COEFF}" \
    --loss.cov_coeff="${STD_COEFF}" \
    --loss.pred_loss_type=smooth_l1 \
    --logging.wandb_group="exp6_sweep_${EXP_ID}"

CKPT="${EXP_DIR}/best.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no best.pth.tar at $CKPT"
    exit 1
fi

# ---- Eval ----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="$CKPT" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --num_workers=2 \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --wandb_group="exp6_sweep_eval_${EXP_ID}"

echo "Cell ${EXP_ID} complete!"
