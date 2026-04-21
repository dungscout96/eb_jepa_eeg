"""Variance decomposition for the CorrCA-preprocessed VICReg checkpoint.

One-off submission for the exp6 (kkokate/exp6-corrca-preprocessing) checkpoint
where movie-feature probes reportedly surpassed chance. Uses K=32 clips/rec
to match the cleaned baseline in outputs/variance_decomp_k32/.

Notes
-----
- Checkpoint and CorrCA filters live under kkokate's projects dir; the job
  must have read access (it does — /projects/bbnv/kkokate is group-shared).
- CorrCA projects 129 channels → k stimulus-driven components before the
  encoder sees anything, so Var_within here should be meaningfully more
  stimulus-related than in the earlier K=32 runs.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_variance_decomp_corrca_delta.py --submit
"""

import os
import sys

from neurolab.jobs import Job

CHECKPOINT = (
    "/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/eeg_jepa/"
    "dev_2026-04-20_14-29/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed2025/"
    "best.pth.tar"
)
CORRCA = "/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"
OUT = "/u/dtyoung/eb_jepa_eeg/outputs/variance_decomp_corrca"


def build_job():
    cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/variance_decomposition.py"
        f" --checkpoint={CHECKPOINT}"
        f" --corrca_filters={CORRCA}"
        " --n_windows=4 --window_size_seconds=2"
        " --batch_size=64 --num_workers=4"
        " --n_clips_per_rec=32"
        " --split=val"
        f" --output_dir={OUT}"
    )

    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )

    return Job(
        name="vardecomp_corrca",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + cmd,
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="02:00:00",  # nw4_ws2 at K=32: ~60-70 min expected
        mem_gb=64,
        gpus=1,
        env_vars={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    job = build_job()
    print(f"Job: {job.name}  [{job.time_limit}]")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  CorrCA: {CORRCA}")
    print(f"  Output: {OUT}")
    print()
    print(job.submit(dry_run=True))
    if submit:
        print("=" * 72)
        job_id = job.submit()
        print(f"  {job.name}: {job_id}")
    else:
        print("Dry run only. Re-run with --submit.")
