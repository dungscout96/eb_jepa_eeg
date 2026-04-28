"""Trivial-feature baseline on raw 129-ch EEG (no CorrCA), per-rec normalized.

Companion to run_position_leakage_tests_delta.py, but without CorrCA — to
isolate the trivial baseline's decoding capability from CorrCA's spatial
filtering. Ridge probe on 129 channels × 7 stats = 903 features.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_trivial_baseline_raw_delta.py --submit
"""

import sys

from neurolab.jobs import Job


def build_job():
    cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/trivial_position_baseline.py"
        " --norm_mode=per_recording"
        " --n_windows=4 --window_size_seconds=2"
        " --n_passes=20"
        " --seed=2025"
    )
    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )
    return Job(
        name="trivial_raw",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + cmd,
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="01:30:00",
        mem_gb=64,
        gpus=1,
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    job = build_job()
    print(f"Job: {job.name}  [{job.time_limit}]")
    print()
    print(job.submit(dry_run=True))
    if submit:
        print("=" * 72)
        job_id = job.submit()
        print(f"  {job.name}: {job_id}")
    else:
        print("Dry run only. Re-run with --submit.")
