"""Trivial-baseline tests on CorrCA-projected EEG:
1. Shuffle luminance globally across movie frames (consistent across recordings).
2. Shuffle position_in_movie globally.
3. Time-aligned (K=32 evenly-spaced clips per recording, deterministic).
4. Time-aligned + shuffled luminance — gold-standard discriminator.

Compares against the unshuffled, random-clip baseline (already run as
job 17923581: pos val 0.183 / test 0.151; lum val 0.153 / test 0.144;
con val 0.143 / test 0.148).

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_trivial_shuffle_and_aligned_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CORRCA = "/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"


def _cmd(extra_flags):
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/trivial_position_baseline.py"
        f" --corrca_filters={CORRCA}"
        " --norm_mode=per_recording"
        " --n_windows=4 --window_size_seconds=2"
        " --seed=2025"
        f" {extra_flags}"
    )


def build_job():
    cmds = [
        # 1. Shuffle luminance globally
        _cmd("--n_passes=20 --shuffle_label=luminance_mean"),
        # 2. Shuffle position globally
        _cmd("--n_passes=20 --shuffle_label=position_in_movie"),
        # 3. Time-aligned, true labels
        _cmd("--time_aligned_K=32"),
        # 4. Time-aligned + shuffled luminance
        _cmd("--time_aligned_K=32 --shuffle_label=luminance_mean"),
    ]
    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )
    return Job(
        name="trivial_shuf_aligned",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + " &&\n".join(cmds),
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="02:00:00",
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
