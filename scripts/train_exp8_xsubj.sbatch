#!/bin/bash
#SBATCH --job-name=exp8_xsubj
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# ---------------------------------------------------------------
# Experiment 8: Cross-subject paired JEPA (Idea 2).
#
# Exp 6 base (CorrCA 5 + per-recording norm + pred_dim=24 + VCLoss
# 0.25/0.25 + smooth_l1) with ONE change: with probability P, each
# batch feeds the JEPA target encoder a batch-permuted EEG — i.e.,
# subject_j's signal at the same movie time — while the context
# encoder sees subject_i. Since subjects only share the stimulus
# signal at matched movie time, the JEPA predictor must learn
# stimulus-driven features to minimize the prediction loss.
#
# JEPA loss is unchanged. VCLoss unchanged. Only the data-pairing
# and target-encoder input change.
#
# Batches are constructed via TimeAlignedBatchSampler so all B
# samples in a batch share the same movie time bin.
# ---------------------------------------------------------------

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST"

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

GIT_BRANCH="${GIT_BRANCH:-kkokate/exp8-xsubj-jepa}"
git fetch origin && git checkout "$GIT_BRANCH" && git pull --ff-only origin "$GIT_BRANCH"

# Defaults: 50% of batches use cross-subject pairing; 40 time bins
# (5-second resolution for a 200s movie) — enough granularity that
# two subjects at the same bin are at nearly the same movie moment,
# coarse enough that any B recordings can be sampled to fill a batch.
XSUBJ_PROB="${XSUBJ_PROB:-0.5}"
N_TIME_BINS="${N_TIME_BINS:-40}"
SEED="${SEED:-2025}"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --optim.epochs=100 \
    --optim.early_stopping_patience=20 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.num_workers=2 \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --data.time_aligned_bins="${N_TIME_BINS}" \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1 \
    --loss.cross_subject.probability="${XSUBJ_PROB}" \
    --loss.cross_subject.n_time_bins="${N_TIME_BINS}" \
    --logging.wandb_group=exp8_xsubj \
    ${EXTRA_ARGS:-}

echo "Training complete!"
