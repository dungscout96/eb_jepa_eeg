#!/bin/bash
#SBATCH --job-name=e04_cell
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/e04_cell_%j.out
#SBATCH --error=logs/e04_cell_%j.err
# ---------------------------------------------------------------
# One E0.4 scaling cell: full REVE (depth 22 / patch 200 / overlap 20,
# 69.2M params, 258 tokens/sample) warm-started from reve-base, trained with
# the E0.3 soft_target_clip recipe so the two surfaces stay comparable.
#
# Required env vars (one per varying knob, per delta-deploy SKILL.md 3):
#   SUBJECTS (int)  — S, counted in SUBJECTS not recordings
#   ANCHORS  (int)  — A, movie anchors
# Optional:
#   DRAW (int)      — data.subsample_seed. Omit for the S=1863 whole-pool cell,
#                     which has only one possible draw.
#   EPOCHS, SAVE_EVERY, SUFFIX, INIT_CKPT, SEED
#
# Time limit: measured ~10 s/epoch steady state, ~80 min for 400 epochs.
# 03:00:00 clears the guide's 1.5x safety margin; 04:00:00 is this account's cap.
# ---------------------------------------------------------------

set -euo pipefail

if [ -z "${SUBJECTS:-}" ]; then
    echo "ERROR: SUBJECTS env var must be set" >&2
    exit 1
fi
if [ -z "${ANCHORS:-}" ]; then
    echo "ERROR: ANCHORS env var must be set" >&2
    exit 1
fi

EPOCHS="${EPOCHS:-400}"
# meta.seed is 2026 for every cell by protocol: only the COHORT varies, via
# data.subsample_seed, so model init and batch order are held fixed. Override
# only to rescue a cell whose optimisation failed to leave chance -- a dead run
# is not a sample of that cohort, and the override must be recorded.
SEED="${SEED:-2026}"
SAVE_EVERY="${SAVE_EVERY:-25}"
SUFFIX="${SUFFIX:-_nd}"
# INIT_CKPT=none runs the E0.5 random-init control: same shape, same recipe, no
# warm start. It is the only arm that varies init alone -- E0.4 moved depth,
# patch geometry AND init together, so nothing in it isolates the warm start.
INIT_CKPT="${INIT_CKPT:-/work/hdd/bbnv/kkokate/eb_jepa/reve_base_eet_init.pth.tar}"
# Frozen config snapshot to copy. E0.6 swaps in a width variant; the shape
# fields live in the snapshot, never on the CLI, so a cell dir always records
# exactly what it was trained with.
CONFIG_SRC="${CONFIG_SRC:-experiments/snr_scaling/clip_pretrain_reve.yaml}"
EXP_PREFIX="${EXP_PREFIX:-e04}"
CKPT_ROOT="${CKPT_ROOT:-/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling}"

# The extended pool (R7-R10) lives in a collaborator's tree; group-readable.
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/work/hdd/bbnv/dtyoung/hbn_preprocessed}"
export HBN_TRAIN_RELEASES="${HBN_TRAIN_RELEASES:-R1,R2,R3,R4,R7,R8,R9,R10}"
export UV_LINK_MODE=copy  # hardlinks fail across Delta filesystems

# Caller is responsible for cwd. Don't trust SLURM_SUBMIT_DIR -- under neurolab
# it points at HOME, not the repo. Fail loudly if we're not in the right place.
if [ ! -d eb_jepa/training ]; then
    echo "ERROR: must be run with cwd=repo root (cwd=$(pwd))" >&2
    exit 1
fi

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# EXP_ID carries every varying knob or cells collide and overwrite each other.
if [ -n "${DRAW:-}" ]; then
    EXP_ID="${EXP_PREFIX}_s${SUBJECTS}_a${ANCHORS}${SUFFIX}_d${DRAW}"
else
    EXP_ID="${EXP_PREFIX}_s${SUBJECTS}_a${ANCHORS}${SUFFIX}"
fi
EXP_DIR="${CKPT_ROOT}/${EXP_ID}"

echo "Job $SLURM_JOB_ID  Node: $SLURM_NODELIST  Cell: $EXP_ID"
echo "cfg: S=${SUBJECTS} A=${ANCHORS} draw=${DRAW:-none} epochs=${EPOCHS} seed=${SEED} init=${INIT_CKPT}"

mkdir -p logs "$EXP_DIR"

# Copy the FROZEN snapshot, then patch the copy -- never the on-disk template,
# which concurrent cells share.
cp "$CONFIG_SRC" "$EXP_DIR/config.yaml"

DRAW_PATCH=""
if [ -n "${DRAW:-}" ]; then
    DRAW_PATCH="c.data.subsample_seed = ${DRAW}; "
fi

# epoch_size=703 pins 11 steps/epoch regardless of S, so data scale is never
# confounded with optimisation budget: every cell runs exactly 4400 steps.
PYTHONPATH=. uv run --group eeg python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('${EXP_DIR}/config.yaml')
c.loss.mode = 'soft_target_clip'
c.loss.soft_alpha = 0.5
c.loss.soft_tau_teacher = 0.05
c.data.task = 'ThePresent'
c.data.max_subjects = ${SUBJECTS}
c.data.max_anchors = ${ANCHORS}
c.data.epoch_size = 703
${DRAW_PATCH}OmegaConf.save(c, '${EXP_DIR}/config.yaml')
"

# The probe must see the FULL data, or retrieval chance levels shift per cell.
PYTHONPATH=. uv run --group eeg python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('${EXP_DIR}/config.yaml')
c.data.max_subjects = None
c.data.max_anchors = None
c.data.epoch_size = None
OmegaConf.save(c, '${EXP_DIR}/config_probe.yaml')
"

# Omit the flag entirely for the random arm -- the frozen config already carries
# meta.encoder_init_from: null, and passing the string "none" would be read as a
# path. The ${A[@]+"${A[@]}"} form is required: under `set -u`, expanding an
# empty array unquoted is an unbound-variable error on bash < 4.4.
INIT_ARG=(--meta.encoder_init_from="${INIT_CKPT}")
if [ "${INIT_CKPT}" = "none" ]; then
    INIT_ARG=()
fi

PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain \
    --fname="${EXP_DIR}/config.yaml" \
    --optim.epochs="${EPOCHS}" \
    --meta.seed="${SEED}" \
    ${INIT_ARG[@]+"${INIT_ARG[@]}"} \
    --folder="${EXP_DIR}" \
    --logging.save_every="${SAVE_EVERY}" \
    --logging.wandb_group="${EXP_PREFIX}_scaling_${EXP_ID}"

# Honest exit code. Under the neurolab path a job that died instantly still
# reported COMPLETED 0:0, because a trailing echo reset $?. Check the artifact.
CKPT="${EXP_DIR}/latest.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no latest.pth.tar at $CKPT" >&2
    exit 1
fi

echo "Cell ${EXP_ID} complete: $(ls -1 "$EXP_DIR"/epoch_*.pth.tar 2>/dev/null | wc -l) epoch checkpoints"
