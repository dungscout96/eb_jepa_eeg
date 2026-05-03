#!/bin/bash
#SBATCH --job-name=cellB
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/cellB_%j.out
#SBATCH --error=logs/cellB_%j.err
# ---------------------------------------------------------------
# Cell B — cross-subject masked latent prediction (Brain-JEPA / PopT mold).
# Pure JEPA: same predictor, same EMA target, same VCLoss; only the mask
# geometry changes. K=2 paired sampler is auto-enabled by xsub_coeff > 0.
# Predict EMA(A)[masked] from B's full token context (and vice versa).
#
# Required env: SEED, EXP_TAG.
# Optional env: NW (4), WS (2), BATCH_SIZE (64), LR (5e-4), WARMUP (5),
#               EPOCHS (100), XSUB_COEFF (0.5), XSUB_SYM (true),
#               STIM_BUCKET (0.025), TASKS (ThePresent), GROUP_TAG.
# ---------------------------------------------------------------

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${SEED:?need SEED}"
: "${EXP_TAG:?need EXP_TAG}"
NW="${NW:-4}"
WS="${WS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP:-5}"
EPOCHS="${EPOCHS:-100}"
XSUB_COEFF="${XSUB_COEFF:-0.5}"
XSUB_SYM="${XSUB_SYM:-true}"
STIM_BUCKET="${STIM_BUCKET:-0.025}"
TASKS="${TASKS:-ThePresent}"
TASKS="${TASKS//+/,}"
GROUP_TAG="${GROUP_TAG:-issue10_cellB_xsub}"

echo "Cfg: seed=${SEED} tag=${EXP_TAG} nw=${NW} ws=${WS} bs=${BATCH_SIZE}"
echo "     lr=${LR} epochs=${EPOCHS} xsub=${XSUB_COEFF} sym=${XSUB_SYM} bucket=${STIM_BUCKET}"

cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_ENTITY="${WANDB_ENTITY:-sccn}"
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue10/${EXP_TAG}"
mkdir -p "$EXP_DIR"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --optim.epochs="${EPOCHS}" \
    --optim.early_stopping_patience=0 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --optim.lr="${LR}" \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs="${WARMUP}" \
    --data.num_workers=2 \
    --data.batch_size="${BATCH_SIZE}" \
    --data.n_windows="${NW}" \
    --data.window_size_seconds="${WS}" \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --data.stim_position_bucket_size="${STIM_BUCKET}" \
    --data.tasks="${TASKS}" \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --loss.xsub_coeff="${XSUB_COEFF}" \
    --loss.xsub_symmetric="${XSUB_SYM}" \
    --logging.wandb_group="${GROUP_TAG}"

CKPT="${EXP_DIR}/latest.pth.tar"
[ -f "$CKPT" ] || { echo "ERROR: no latest.pth.tar at $CKPT"; exit 1; }
BEST_PROBE="${EXP_DIR}/best_by_online_probe.pth.tar"
if [ -f "$BEST_PROBE" ]; then
    echo "Cell B ${EXP_TAG} complete! latest=${CKPT}, best_probe=${BEST_PROBE}"
else
    echo "Cell B ${EXP_TAG} complete! latest=${CKPT}, best_probe=(none — never improved)"
fi
