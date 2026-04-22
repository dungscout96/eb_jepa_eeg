"""probe_eval + variance_decomp on kkokate's exp6_sigreg checkpoint.

Evaluates the single pre-existing SIGReg + per-rec + CorrCA checkpoint
(nw4_ws2, coeff=1.0) to fill in the data point for that combination while
the 3 new configs from train_sigreg_corrca_delta.py are training.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_exp6_sigreg_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CHECKPOINT = (
    "/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/eeg_jepa/"
    "dev_2026-04-15_21-58/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw4_ws2s_seed2025/"
    "best.pth.tar"
)
CORRCA = "/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"
VAR_OUT = "/u/dtyoung/eb_jepa_eeg/outputs/variance_decomp_exp6_sigreg"


def _git_cmd():
    return (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )


def build_jobs():
    probe_cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python experiments/eeg_jepa/probe_eval.py"
        f" --checkpoint={CHECKPOINT}"
        f" --corrca_filters={CORRCA}"
        " --norm_mode=per_recording"
        " --n_windows=4 --window_size_seconds=2"
        " --batch_size=64 --num_workers=4"
        " --probe_epochs=20 --subject_probe_epochs=100"
        " --splits=val,test"
        " --wandb_group=probe_eval_exp6_sigreg"
        " --seed=2025"
    )
    vardec_cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/variance_decomposition.py"
        f" --checkpoint={CHECKPOINT}"
        f" --corrca_filters={CORRCA}"
        " --norm_mode=per_recording"
        " --n_windows=4 --window_size_seconds=2"
        " --batch_size=64 --num_workers=4"
        " --n_clips_per_rec=32 --split=val"
        f" --output_dir={VAR_OUT}"
    )

    probe_job = Job(
        name="ev_probe_exp6sr",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=_git_cmd() + probe_cmd,
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="01:30:00",
        mem_gb=64,
        gpus=1,
        env_vars={
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            "WANDB_PROJECT": "eb_jepa",
        },
    )
    vardec_job = Job(
        name="ev_vardec_exp6sr",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=_git_cmd() + vardec_cmd,
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="01:00:00",  # CorrCA projects to few channels → fast
        mem_gb=64,
        gpus=1,
        env_vars={
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )
    return [("probe_eval_exp6sr", probe_job), ("vardec_exp6sr", vardec_job)]


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    jobs = build_jobs()
    print("Evaluating exp6_sigreg best.pth.tar (SIGReg 1.0 nw4_ws2 + per-rec + CorrCA)\n")
    for label, job in jobs:
        print("=" * 72)
        print(f"Job: {job.name} ({label})  [{job.time_limit}]")
        print("=" * 72)
        print(job.submit(dry_run=True))
        print()
    if submit:
        print("=" * 72)
        print("SUBMITTING")
        print("=" * 72)
        for label, job in jobs:
            job_id = job.submit()
            print(f"  {job.name}: {job_id}")
    else:
        print("Dry run. Re-run with --submit.")
