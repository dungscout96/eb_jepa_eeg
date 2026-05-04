#!/bin/bash
#SBATCH --job-name=cellJK
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cellJK_%j.out
#SBATCH --error=logs/cellJK_%j.err
# ---------------------------------------------------------------
# Cell J (Cross-Time JEPA) and Cell K (PARS Δt regression) — both
# within-subject objectives that survive CorrCA preprocessing.
#
# Required env: SEED, EXP_TAG, CELL (j or k).
# Optional: NW (4), WS (2), BATCH_SIZE (64), LR (5e-4), WARMUP (5),
#           EPOCHS (100), TASKS (ThePresent), GROUP_TAG.
#   Cell J: MASK_FRAC (0.5), MASK_POS (tail|head|random)
#   Cell K: PARS_COEFF (0.3), PARS_MAX_DELTA_S (60)
# ---------------------------------------------------------------

set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${SEED:?need SEED}"
: "${EXP_TAG:?need EXP_TAG}"
: "${CELL:?need CELL (j or k)}"
NW="${NW:-4}"
WS="${WS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
WARMUP="${WARMUP:-5}"
EPOCHS="${EPOCHS:-100}"
TASKS="${TASKS:-ThePresent}"
TASKS="${TASKS//+/,}"
GROUP_TAG="${GROUP_TAG:-issue10_cell${CELL}}"

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

COMMON_ARGS=(
    --meta.seed="${SEED}"
    --meta.model_folder="${EXP_DIR}"
    --optim.epochs="${EPOCHS}"
    --optim.early_stopping_patience=0
    --model.encoder_depth=2
    --model.predictor_embed_dim=24
    --optim.lr="${LR}"
    --optim.lr_min=1e-6
    --optim.warmup_epochs="${WARMUP}"
    --data.num_workers=2
    --data.batch_size="${BATCH_SIZE}"
    --data.n_windows="${NW}"
    --data.window_size_seconds="${WS}"
    --data.norm_mode=per_recording
    --data.corrca_filters=corrca_filters.npz
    --data.tasks="${TASKS}"
    --loss.std_coeff=0.25
    --loss.cov_coeff=0.25
    --loss.pred_loss_type=smooth_l1
    --logging.wandb_group="${GROUP_TAG}"
)

if [ "${CELL}" = "j" ]; then
    MASK_FRAC="${MASK_FRAC:-0.5}"
    MASK_POS="${MASK_POS:-tail}"
    echo "Cfg: Cell J cross-time mask, frac=${MASK_FRAC} position=${MASK_POS}"
    PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
        "${COMMON_ARGS[@]}" \
        --masking.geometry=cross_time \
        --masking.mask_window_fraction="${MASK_FRAC}" \
        --masking.mask_position="${MASK_POS}"
elif [ "${CELL}" = "k" ]; then
    PARS_COEFF="${PARS_COEFF:-0.3}"
    PARS_MAX_DELTA_S="${PARS_MAX_DELTA_S:-60}"
    echo "Cfg: Cell K PARS Δt regression, coeff=${PARS_COEFF} max_delta_s=${PARS_MAX_DELTA_S}"
    PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
        "${COMMON_ARGS[@]}" \
        --loss.pars_coeff="${PARS_COEFF}" \
        --loss.pars_max_delta_seconds="${PARS_MAX_DELTA_S}"
else
    echo "ERROR: CELL must be 'j' or 'k', got '${CELL}'"
    exit 2
fi

CKPT="${EXP_DIR}/latest.pth.tar"
[ -f "$CKPT" ] || { echo "ERROR: no latest.pth.tar at $CKPT"; exit 1; }
echo "Cell ${CELL} ${EXP_TAG} complete!  ckpt=${CKPT}"
