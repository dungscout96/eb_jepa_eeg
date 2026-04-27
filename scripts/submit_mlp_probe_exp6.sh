#!/bin/bash
# Issue #8 follow-up: re-evaluate Exp 6 baseline (std=0.25, pd=24, +CorrCA, per-rec)
# checkpoints with linear vs 2-layer MLP probe heads.
#
# For each available seed: submit (linear, mlp@128, mlp@256) — three jobs per seed.
# Linear is the baseline; MLP variants use hidden_dim ∈ {128, 256} (≈ 2D and 4D).
# Movie-feature head also widened to 128 for the MLP variants (default = 64 = embed_dim).

set -euo pipefail

CKPT_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/exp6_sweep"
SEEDS=(42 123 2025)  # only these have std0.25_pd24 checkpoints on Delta

mkdir -p logs

submit_one() {
  local SEED="$1" PROBE="$2" HIDDEN="$3" MOVIE_HIDDEN="$4"
  local CKPT="${CKPT_DIR}/std0.25_pd24_seed${SEED}/best.pth.tar"
  if [ ! -f "$CKPT" ]; then
    echo "  SKIP: missing ${CKPT}"; return 0
  fi
  local TAG
  if [ "$PROBE" = "linear" ]; then
    TAG="exp6_s${SEED}_linear"
  else
    TAG="exp6_s${SEED}_mlp_h${HIDDEN}_mh${MOVIE_HIDDEN}"
  fi
  echo "  -> ${TAG}"
  CKPT="${CKPT}" PROBE_TYPE="${PROBE}" HIDDEN_DIM="${HIDDEN}" \
    MOVIE_HIDDEN="${MOVIE_HIDDEN}" EXP_TAG="${TAG}" \
    sbatch scripts/probe_mlp_exp6.sbatch
  sleep 2
}

N=0
for SEED in "${SEEDS[@]}"; do
  echo "[seed=${SEED}]"
  submit_one "${SEED}" linear 128 0     ; N=$((N+1))   # baseline
  submit_one "${SEED}" mlp    128 128   ; N=$((N+1))   # mlp 2D
  submit_one "${SEED}" mlp    256 128   ; N=$((N+1))   # mlp 4D
done

echo "Submitted ${N} probe-eval jobs."
