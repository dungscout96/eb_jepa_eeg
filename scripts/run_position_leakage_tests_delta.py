"""Position-leakage tests on the CorrCA-VICReg checkpoint.

1. probe_eval.py with --shuffle_position_within_rec: per-recording position
   labels are randomly permuted before probe training. If the encoder's
   position signal is genuinely clip-aligned (stimulus content), corr drops
   to ~0 here. If it's a within-recording temporal trend, corr survives.

2. trivial_position_baseline.py: linear probe on per-channel mean / std /
   bandpowers (no encoder) → all four movie features. If position is well-
   decoded by trivial features but luminance/contrast/narrative aren't, the
   encoder's position prediction is likely the same trivial leak.

Both tests use the same data preprocessing as the CorrCA-VICReg checkpoint
(per-rec norm + CorrCA filters) so the trivial baseline sees the same
pre-encoder EEG the encoder gets.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_position_leakage_tests_delta.py --submit
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
    shuffle_cmd = (
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
        " --shuffle_position_within_rec=True"
        " --wandb_group=probe_eval_corrca_shuffle_position"
        " --seed=2025"
    )

    trivial_cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/trivial_position_baseline.py"
        f" --corrca_filters={CORRCA}"
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
        name="pos_leak_tests",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + shuffle_cmd + " &&\n" + trivial_cmd,
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
