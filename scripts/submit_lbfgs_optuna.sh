#!/bin/bash
# 25-job final sweep with Optuna's best L-BFGS+BN config (trial #26).
set -euo pipefail
CKPT_BASE="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8"
ENC_SEEDS=(42 123 456 789 2025); PROBE_SEEDS=(7 13 42 1234 2025)
mkdir -p logs
N=0
for ENC in "${ENC_SEEDS[@]}"; do
    CKPT="${CKPT_BASE}/phaseD_nw4ws2_baseline_s${ENC}/latest.pth.tar"
    [ -f "$CKPT" ] || { echo "missing s${ENC}"; exit 1; }
    for PROBE in "${PROBE_SEEDS[@]}"; do
        N=$((N+1))
        EXP_TAG="nw4ws2_lbfgs_enc${ENC}_p${PROBE}"
        echo "[${N}/25] ${EXP_TAG}"
        CKPT="${CKPT}" NW=4 WS=2 BATCH_SIZE=64 PROBE_SEED="${PROBE}" EXP_TAG="${EXP_TAG}" \
            sbatch --job-name="lbfgs_e${ENC}_p${PROBE}" \
                   scripts/probe_eval_lbfgs_optuna.sbatch
    done
done
echo "Submitted ${N} L-BFGS+BN final-sweep jobs"
