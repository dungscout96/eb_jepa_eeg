"""Submit input-space variance decomposition across all 4 conditions.

No encoder forward passes — just reading each recording, applying the
preprocessing for the condition, computing per-channel RMS per clip, and
running the nested-ANOVA decomposition. Should be ~15-25 min per condition
(bottleneck is raw FIF I/O), ~1.5h for all 4 packed into one job.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_input_vardecomp_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CORRCA = "/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"
# Keep separate output dirs per feature so results don't overwrite.
OUT_BY_FEATURE = {
    "rms":         "/u/dtyoung/eb_jepa_eeg/outputs/input_variance_decomp",
    "bandpower":   "/u/dtyoung/eb_jepa_eeg/outputs/input_variance_decomp_bp",
    "alpha_phase": "/u/dtyoung/eb_jepa_eeg/outputs/input_variance_decomp_phase",
}
# Cleanest Littwin discrimination: alpha_phase has moderate static variance
# (phase distributes on the unit circle) but near-zero context→target
# predictability over multi-second gaps (phase wanders). If representation
# variance tracks predictability (not static), we expect representation to
# encode almost no alpha phase despite its non-trivial input variance.
FEATURE = "alpha_phase"
OUT = OUT_BY_FEATURE[FEATURE]


def build_job():
    cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/input_variance_decomposition.py"
        " --n_windows=4 --window_size_seconds=4"
        " --n_clips_per_rec=32"
        " --split=val"
        f" --feature={FEATURE}"
        " --conditions=all"
        f" --corrca_filters={CORRCA}"
        f" --output_dir={OUT}"
    )
    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )
    return Job(
        name="ivd_all",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + cmd,
        venv="__none__",
        branch="",
        partition="gpuA40x4",  # we don't need GPU, but partition is what we have access to
        time_limit="03:00:00",  # 4 conditions × ~30 min each
        mem_gb=64,
        gpus=1,
        env_vars={
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    job = build_job()
    print(f"Job: {job.name}  [{job.time_limit}]  (input decomp × 4 conditions)")
    print(job.submit(dry_run=True))
    if submit:
        print("=" * 72)
        job_id = job.submit()
        print(f"  {job.name}: {job_id}")
    else:
        print("Dry run. Re-run with --submit.")
