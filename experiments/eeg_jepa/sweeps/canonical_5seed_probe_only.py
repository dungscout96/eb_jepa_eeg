"""Re-run canonical probe_eval + bootstrap against existing pretrain checkpoints.

Use when the canonical_5seed.py sweep's pretraining stage succeeded but the
probe/bootstrap stage failed (e.g., the sklearn LogisticRegression(multi_class=)
API drop on 2026-05-16). 5 short jobs (2h budget each); the aggregate dep
sbatch can be submitted after all 5 finish.
"""

import os
import sys

from neurolab.jobs import Job

SEEDS = [42, 123, 456, 789, 2025]
TIME_LIMIT = "02:00:00"


def _make_command(seed: int) -> str:
    return (
        f"SEED={seed} bash experiments/eeg_jepa/sbatch/canonical_probe_only.sbatch "
        f"|| exit $?"
    )


def build_jobs():
    jobs = []
    for seed in SEEDS:
        job = Job(
            name=f"canonical_probe_s{seed}",
            cluster="delta",
            repo_path="/u/dtyoung/eb_jepa_eeg",
            command=_make_command(seed),
            venv="__none__",
            branch="",
            partition="gpuA40x4",
            time_limit=TIME_LIMIT,
            mem_gb=32,
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


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    jobs = build_jobs()

    print(f"Canonical probe-only sweep on Delta")
    print(f"  Seeds: {SEEDS}")
    print(f"  Time per job: {TIME_LIMIT}")
    print()

    for seed, job in jobs:
        script = job.submit(dry_run=True)
        print("=" * 72)
        print(f"Job: canonical_probe_s{seed}")
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
        print("After all 5 finish, submit aggregate:")
        print()
        print(f"  sbatch --dependency=afterok:{dep_arg} "
              "experiments/eeg_jepa/sbatch/canonical_aggregate.sbatch")
    else:
        print(f"\n{len(jobs)} jobs ready. Run with --submit to send to Delta.")
