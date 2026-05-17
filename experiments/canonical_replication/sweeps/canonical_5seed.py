"""Spec-faithful 5-seed replication sweep on Delta (A40 GPUs).

Each seed runs pretrain → canonical probe_eval → bootstrap end-to-end in a single
SLURM job (6h budget per job). After all 5 complete, submit the aggregate job
with --dependency=afterok:<all five jobids> to collapse to L3 + t-test vs chance.

Spec (from refactor verification, 2026-05):
  - Architecture: nw=2 ws=4, encoder_depth=2, embed_dim=64, predictor_embed_dim=24
  - Training: lr=5e-4 cosine→1e-6, warmup=5, 100 ep, bs=64, patience=0
  - Data: per_recording norm, CorrCA 129→5 filters, task=ThePresent
  - Loss: VCLoss(0.25, 0.25), smooth_l1
  - Probes: sklearn Ridge(α=1) / LogisticRegression(C=1, lbfgs), multinomial LR (movie_id)
  - n_passes=20 augmentation, probe_seed=42 fixed across encoder seeds
  - Bootstrap B=2000 over 108 test recordings; L3 = mean±1σ of L2 means across seeds

Usage
-----
# Preview SLURM scripts that will be submitted:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python experiments/canonical_replication/sweeps/canonical_5seed.py

# Submit all 5 pretrain+probe+bootstrap jobs:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python experiments/canonical_replication/sweeps/canonical_5seed.py --submit

# After all 5 complete (or with --dependency=afterok):
sbatch --dependency=afterok:<id1>:<id2>:<id3>:<id4>:<id5> \\
    eb_jepa/training/sbatch/canonical_aggregate.sbatch
"""

import os
import sys

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# Sweep configuration (spec-faithful)
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 2025]
TIME_LIMIT = "06:00:00"   # 100 ep pretrain ≈ 2-3h + ≈ 30 min probe+bootstrap
MEM_GB = 64
CPUS = 16


def _make_command(seed: int) -> str:
    """Single-seed command: pretrain → canonical probe_eval → bootstrap.

    `|| exit $?` propagates the inner sbatch's exit code so a real failure
    surfaces as a SLURM FAILED state instead of being masked by the neurolab
    wrapper's trailing echo (which silently returns 0).
    """
    return (
        f"SEED={seed} bash eb_jepa/training/sbatch/canonical_replication.sbatch "
        f"|| exit $?"
    )


def build_jobs():
    jobs = []
    for seed in SEEDS:
        # NOTE: deliberately no per-job `git fetch / pull`.
        # 5 jobs launching simultaneously race on .git/refs/* locks and break the
        # && chain (observed 2026-05-16: jobs 18294774/18294775). The expectation
        # is that the user pulls the target branch on Delta once before --submit.
        job = Job(
            name=f"canonical_nw2ws4_s{seed}",
            cluster="delta",
            repo_path="/u/dtyoung/eb_jepa_eeg",
            command=_make_command(seed),
            venv="__none__",
            branch="",
            partition="gpuA40x4",
            time_limit=TIME_LIMIT,
            mem_gb=MEM_GB,
            gpus=1,
            env_vars={
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                "WANDB_PROJECT": "eb_jepa",
                "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
                "SEED": str(seed),
            },
        )
        jobs.append((seed, job))
    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    submit = "--submit" in sys.argv
    jobs = build_jobs()

    print(f"Canonical 5-seed replication sweep on Delta")
    print(f"  Seeds: {SEEDS}")
    print(f"  Time per job: {TIME_LIMIT}  (pretrain + canonical probe_eval + bootstrap)")
    print()

    for seed, job in jobs:
        script = job.submit(dry_run=True)
        print("=" * 72)
        print(f"Job: canonical_nw2ws4_s{seed}")
        print("=" * 72)
        print(script)
        print()

    if submit:
        print("\n" + "=" * 72)
        print("SUBMITTING")
        print("=" * 72)
        ids = []
        for seed, job in jobs:
            job_id = job.submit()
            ids.append(str(job_id))
            print(f"  s{seed}: {job_id}")
        dep_arg = ":".join(ids)
        print()
        print("All 5 pretrain+probe+bootstrap jobs submitted.")
        print("After they finish, run the aggregate job (depends on all 5):")
        print()
        print(f"  sbatch --dependency=afterok:{dep_arg} "
              "eb_jepa/training/sbatch/canonical_aggregate.sbatch")
    else:
        print(f"\n{len(jobs)} jobs ready. Run with --submit to send to Delta.")
