#!/bin/bash
# ---------------------------------------------------------------
# Issue #10 Lever 1 — submit 5 enc-seed pretrainings + chain a kc+Ridge
# probe-eval after each training finishes (via SLURM dependency).
#
# Each enc seed gets 1 probe seed (single-pass kc+Ridge). Total: 10 jobs.
# ---------------------------------------------------------------

set -euo pipefail

ENC_SEEDS=(42 123 456 789 2025)
STIM_COEFF="${STIM_COEFF:-0.5}"
STIM_TAU="${STIM_TAU:-0.1}"
STIM_BUCKET="${STIM_BUCKET:-4.0}"

mkdir -p logs

submitted_train_ids=()
submitted_probe_ids=()

for seed in "${ENC_SEEDS[@]}"; do
    tag="lever1_nw4ws2_s${seed}"

    train_jobid=$(sbatch \
        --parsable \
        --export="ALL,SEED=${seed},EXP_TAG=${tag},STIM_COEFF=${STIM_COEFF},STIM_TAU=${STIM_TAU},STIM_BUCKET=${STIM_BUCKET}" \
        scripts/train_lever1_infonce.sbatch)

    ckpt="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue10/${tag}/latest.pth.tar"
    probe_jobid=$(sbatch \
        --parsable \
        --dependency=afterok:${train_jobid} \
        --export="ALL,CKPT=${ckpt},EXP_TAG=${tag},PROBE_SEED=${seed}" \
        scripts/probe_lever1_ridge.sbatch)

    submitted_train_ids+=("$train_jobid")
    submitted_probe_ids+=("$probe_jobid")
    echo "seed=${seed}  train=${train_jobid}  probe=${probe_jobid} (chained afterok)"
done

echo ""
echo "Submitted 5 trainings + 5 chained probe-evals."
echo "Train job IDs: ${submitted_train_ids[*]}"
echo "Probe job IDs: ${submitted_probe_ids[*]}"
squeue -u "$USER" -h | tail -15
