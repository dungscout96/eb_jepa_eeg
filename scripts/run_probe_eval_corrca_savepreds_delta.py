"""probe_eval on CorrCA-VICReg checkpoint with --save_predictions_dir.

Runs over 5 seeds (2025-2029), each saving per-recording predictions to a
separate .npz so we can both (a) compare seed-σ to colleague's, and (b)
bootstrap recording-level CIs locally for population uncertainty.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_probe_eval_corrca_savepreds_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CHECKPOINT = (
    "/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/eeg_jepa/"
    "dev_2026-04-20_14-29/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed2025/"
    "best.pth.tar"
)
CORRCA = "/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"
SAVE_DIR = "/u/dtyoung/eb_jepa_eeg/outputs/probe_eval_corrca_predictions"
SEEDS = [2025, 2026, 2027, 2028, 2029]


def _eval_cmd(seed):
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python experiments/eeg_jepa/probe_eval.py"
        f" --checkpoint={CHECKPOINT}"
        f" --corrca_filters={CORRCA}"
        " --norm_mode=per_recording"
        " --n_windows=4 --window_size_seconds=2"
        " --batch_size=64 --num_workers=4"
        " --probe_epochs=20"
        " --subject_probe_epochs=100"
        " --splits=val,test"
        f" --save_predictions_dir={SAVE_DIR}"
        " --wandb_group=probe_eval_corrca_savepreds"
        f" --seed={seed}"
    )


def build_job():
    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )
    cmds = [_eval_cmd(s) for s in SEEDS]
    return Job(
        name="pe_corrca_savepreds",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + " &&\n".join(cmds),
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
    print(f"  Save dir: {SAVE_DIR}")
    print(f"  Seeds: {SEEDS}")
    print()
    print(job.submit(dry_run=True))
    if submit:
        print("=" * 72)
        job_id = job.submit()
        print(f"  {job.name}: {job_id}")
    else:
        print("Dry run only. Re-run with --submit.")
