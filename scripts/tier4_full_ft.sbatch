#!/bin/bash
#SBATCH --job-name=tier4_ft
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/tier4_%j.out
#SBATCH --error=logs/tier4_%j.err
# ---------------------------------------------------------------
# Tier 4 — full fine-tuning of pretrained EEG foundation models.
#   MODEL ∈ {biot, cbramod, luna}
#   SEED  (default 42)
#
# Same per-FM preprocessing as Tier 3 (post-bug-fix: norm_mode=none
# upstream; FM-spec normalization is the only one applied). Encoder is
# unfrozen, fresh Linear(D, 4) regression head, neg-Pearson loss,
# discriminative LRs (encoder 1e-5, head 1e-3), AdamW + cosine + 5%
# warmup, 30 epochs, early-stop on val mean reg corr (patience 8).
#
# Walltime 6h: full FT is ~3-4× heavier than Tier 3 frozen probe and
# LUNA has 7.1M params with full backprop.
# ---------------------------------------------------------------
set -euo pipefail
export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy

: "${MODEL:?need MODEL (biot|cbramod|luna)}"
SEED="${SEED:-42}"

cd "$SLURM_SUBMIT_DIR"
module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

OUT_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/tier4"
mkdir -p "$OUT_DIR"
OUT_JSON="${OUT_DIR}/${MODEL}_seed${SEED}.json"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/tier4_full_ft.py \
    --model="${MODEL}" \
    --seed="${SEED}" \
    --n_windows=4 \
    --window_size_seconds=2 \
    --batch_size=16 \
    --num_workers=8 \
    --epochs=30 \
    --early_stop_patience=8 \
    --encoder_lr=1e-5 \
    --head_lr=1e-3 \
    --weight_decay=1e-2 \
    --warmup_frac=0.05 \
    --subject_probe_epochs=100 \
    --subject_probe_lr=1e-3 \
    --splits=val,test \
    --output_json="${OUT_JSON}"

echo "tier4 ${MODEL} seed=${SEED} done: ${OUT_JSON}"
