"""Phase 1 sweep: temporal vs static pretraining on Delta (A40 GPUs).

Sweeps n_windows × window_size_seconds with 3 seeds each.
Groups small experiments to run in parallel on the same GPU to maximize
utilization (A40 has 48GB VRAM; small configs use ~2GB each).

Usage
-----
# 1. Preview generated SLURM scripts:
uv run python scripts/sweep_phase1_delta.py

# 2. Submit all jobs:
uv run python scripts/sweep_phase1_delta.py --submit
"""

import os
import sys
from itertools import groupby

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

SEEDS = [2025, 42, 7]

# (n_windows, window_size_seconds, batch_size, size_class)
# size_class determines how many run in parallel per SLURM job
GRID = [
    # Static (single window)
    (1, 1, 64, "small"),
    (1, 2, 64, "small"),
    (1, 4, 64, "medium"),
    # Temporal (2 windows)
    (2, 1, 64, "small"),
    (2, 2, 64, "medium"),
    (2, 4, 64, "medium"),
    # Temporal (4 windows)
    (4, 1, 64, "medium"),
    (4, 2, 64, "medium"),   # current best config
    (4, 4, 32, "large"),    # bs=32 to fit in VRAM
    # Temporal (8 windows)
    (8, 1, 64, "medium"),
    (8, 2, 32, "large"),    # bs=32 to fit in VRAM
]

# How many experiments to run in parallel per size class
PARALLEL = {"small": 4, "medium": 3, "large": 2}

# SLURM time limits per size class (enough for longest experiment in group)
TIME_LIMITS = {"small": "02:00:00", "medium": "02:00:00", "large": "03:00:00"}

# Fixed hyperparameters (best config from exp26)
FIXED_ARGS = (
    " --model.encoder_depth=2"
    " --optim.lr=5e-4"
    " --optim.lr_min=1e-6"
    " --optim.warmup_epochs=5"
    " --loss.std_coeff=0.25"
    " --loss.cov_coeff=0.25"
    " --loss.pred_loss_type=smooth_l1"
    " --optim.epochs=100"
    " --logging.wandb_group=sweep_phase1"
)

# ---------------------------------------------------------------------------
# Build all (config, seed) pairs and group into SLURM jobs
# ---------------------------------------------------------------------------


def _make_run_cmd(nw, ws, bs, seed):
    """Build a single training command."""
    return (
        "PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py"
        f" --data.n_windows={nw}"
        f" --data.window_size_seconds={ws}"
        f" --data.batch_size={bs}"
        f" --data.num_workers=2"
        f" --meta.seed={seed}"
        f"{FIXED_ARGS}"
    )


def build_jobs():
    """Group experiments into SLURM jobs by size class and seed.

    For each size class, we pack `PARALLEL[size]` experiments into one SLURM
    job, launching them as background processes (&) and waiting for all.
    """
    # Expand grid × seeds
    all_runs = []
    for nw, ws, bs, size in GRID:
        for seed in SEEDS:
            all_runs.append((nw, ws, bs, seed, size))

    # Group by size class, then chunk into groups of PARALLEL[size]
    all_runs.sort(key=lambda x: x[4])  # sort by size class
    jobs = []

    for size, group_iter in groupby(all_runs, key=lambda x: x[4]):
        group = list(group_iter)
        n_parallel = PARALLEL[size]

        for i in range(0, len(group), n_parallel):
            chunk = group[i : i + n_parallel]
            cmds = [_make_run_cmd(nw, ws, bs, seed) for nw, ws, bs, seed, _ in chunk]

            # Launch all in parallel with & and wait
            parallel_cmd = " &\n".join(cmds) + " &\nwait"

            # Descriptive name
            desc = " + ".join(f"nw{nw}_ws{ws}s_s{seed}" for nw, ws, _, seed, _ in chunk)
            job_name = f"phase1_{size}_{i // n_parallel}"

            job = Job(
                name=job_name,
                cluster="delta",
                repo_path="/u/dtyoung/eb_jepa_eeg",
                command=(
                    "git fetch origin && git checkout main && git pull --ff-only &&\n"
                    + parallel_cmd
                ),
                venv="__none__",
                branch="",
                partition="gpuA40x4",
                time_limit=TIME_LIMITS[size],
                mem_gb=64,
                gpus=1,
                env_vars={
                    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                    "WANDB_PROJECT": "eb_jepa",
                    "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
                },
            )
            jobs.append((job_name, job, desc, chunk))

    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    submit = "--submit" in sys.argv
    jobs = build_jobs()

    print(f"Phase 1 sweep: {sum(len(c) for _, _, _, c in jobs)} experiments in {len(jobs)} SLURM jobs\n")

    for name, job, desc, chunk in jobs:
        script = job.submit(dry_run=True)
        print("=" * 72)
        print(f"Job: {name} ({len(chunk)} experiments in parallel)")
        print(f"  Runs: {desc}")
        print("=" * 72)
        print(script)
        print()

    if submit:
        print("\n" + "=" * 72)
        print("SUBMITTING ALL JOBS")
        print("=" * 72)
        for name, job, desc, chunk in jobs:
            job_id = job.submit()
            print(f"  {name}: {job_id} ({len(chunk)} runs: {desc})")
    else:
        print(
            f"\n{len(jobs)} SLURM jobs ready. Run with --submit to send to Delta."
        )
