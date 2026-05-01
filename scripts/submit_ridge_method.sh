#!/bin/bash
# Submit Ridge-method probe sweep: 5 enc seeds × 5 probe seeds = 25 jobs.
# (Probe seed only varies clip sampling; Ridge solve is deterministic given features.)
set -euo pipefail

CKPT_BASE="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8"
ENC_SEEDS=(42 123 456 789 2025)
PROBE_SEEDS=(7 13 42 1234 2025)

mkdir -p logs

for SEED in "${ENC_SEEDS[@]}"; do
    [ -f "${CKPT_BASE}/phaseD_nw4ws2_baseline_s${SEED}/latest.pth.tar" ] || { echo "missing s${SEED}"; exit 1; }
done

N=0
TOTAL=$(( ${#ENC_SEEDS[@]} * ${#PROBE_SEEDS[@]} ))
for ENC_SEED in "${ENC_SEEDS[@]}"; do
    CKPT="${CKPT_BASE}/phaseD_nw4ws2_baseline_s${ENC_SEED}/latest.pth.tar"
    for PROBE_SEED in "${PROBE_SEEDS[@]}"; do
        N=$((N+1))
        EXP_TAG="nw4ws2_ridge_enc${ENC_SEED}_p${PROBE_SEED}"
        echo "[${N}/${TOTAL}] ${EXP_TAG}"
        CKPT="${CKPT}" NW=4 WS=2 BATCH_SIZE=64 PROBE_SEED="${PROBE_SEED}" EXP_TAG="${EXP_TAG}" \
            sbatch --job-name="ridge_e${ENC_SEED}_p${PROBE_SEED}" \
                   scripts/probe_eval_ridge_method.sbatch
    done
done
echo "Submitted ${N} ridge-method jobs"
