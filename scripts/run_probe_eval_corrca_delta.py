"""Submit probe_eval for the CorrCA-preprocessed VICReg best checkpoint.

Pairs with scripts/run_variance_decomp_corrca_delta.py — we already have the
variance decomposition for this checkpoint (stim/within ≈ 0.007, same as
baselines). This job runs the full probe_eval so we can measure linear
decodability of movie features (the probe R²/corr numbers) to confirm
whether CorrCA's stimulus-relevant signal lives in low-variance but
well-conditioned directions.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_probe_eval_corrca_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CHECKPOINT = (
    "/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/eeg_jepa/"
    "dev_2026-04-20_14-29/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed2025/"
    "best.pth.tar"
)
CORRCA = "/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"


def build_job():
    cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python experiments/eeg_jepa/probe_eval.py"
        f" --checkpoint={CHECKPOINT}"
        f" --corrca_filters={CORRCA}"
        " --n_windows=4 --window_size_seconds=2"
        " --batch_size=64 --num_workers=4"
        " --probe_epochs=20"
        " --subject_probe_epochs=100"
        " --splits=val,test"
        " --wandb_group=probe_eval_corrca"
        " --seed=2025"
    )

    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )

    return Job(
        name="probeeval_corrca",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + cmd,
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="01:30:00",  # CorrCA cuts channels to k<10, encoder forward is cheap
        mem_gb=64,
        gpus=1,
        env_vars={
            # wandb auto-reads ~/.netrc on Delta; no need to pipe the key.
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    job = build_job()
    print(f"Job: {job.name}  [{job.time_limit}]")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  CorrCA:     {CORRCA}")
    print()
    print(job.submit(dry_run=True))
    if submit:
        print("=" * 72)
        job_id = job.submit()
        print(f"  {job.name}: {job_id}")
    else:
        print("Dry run only. Re-run with --submit.")
