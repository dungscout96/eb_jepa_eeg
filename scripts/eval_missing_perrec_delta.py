"""Resubmit probe_eval for the single per-rec checkpoint that timed out.

ev_probe_pr (17779595) hit the 5h wall during checkpoint 7 of 7
(vc0.1_noproj_nw4_ws4) — movie probes finished but subject-trait probe
training got cut off. This script targets just that checkpoint.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_missing_perrec_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CHECKPOINT = (
    "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/"
    "dev_2026-04-21_10-21/eeg_jepa_bs32_lr0.0005_std0.1_cov0.1_noproj_nw4_ws4s_seed2025/"
    "best.pth.tar"
)


def build_job():
    cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python experiments/eeg_jepa/probe_eval.py"
        f" --checkpoint={CHECKPOINT}"
        " --norm_mode=per_recording"
        " --n_windows=4 --window_size_seconds=4"
        " --batch_size=32 --num_workers=4"
        " --probe_epochs=20 --subject_probe_epochs=100"
        " --splits=val,test"
        " --wandb_group=probe_eval_retrained_perrec"
        " --seed=2025"
    )
    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )
    return Job(
        name="ev_probe_pr_miss",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + cmd,
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="02:00:00",  # one nw4_ws4 ~45-60 min with buffer
        mem_gb=64,
        gpus=1,
        env_vars={
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            "WANDB_PROJECT": "eb_jepa",
        },
    )


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    job = build_job()
    print(f"Job: {job.name}  [{job.time_limit}]")
    print(job.submit(dry_run=True))
    if submit:
        print("=" * 72)
        job_id = job.submit()
        print(f"  {job.name}: {job_id}")
    else:
        print("Dry run. Re-run with --submit.")
