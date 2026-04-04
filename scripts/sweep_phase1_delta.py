"""Phase 1 sweep: temporal vs static pretraining on Delta (A40 GPUs).

Sweeps n_windows × window_size_seconds with 3 seeds each.
Packs 2 experiments per SLURM job (parallel on same GPU) to maximize
utilization. Each job uses num_workers=4 per experiment (8 cores total
out of 16 available, ~50GB host RAM out of 62.5GB).

Usage
-----
# 1. Preview generated SLURM scripts:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab python scripts/sweep_phase1_delta.py

# 2. Submit all jobs:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab python scripts/sweep_phase1_delta.py --submit
"""

import os
import sys

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

SEEDS = [2025, 42, 7]

# (n_windows, window_size_seconds, batch_size)
GRID = [
    # Static (single window)
    (1, 1, 64),
    (1, 2, 64),
    (1, 4, 64),
    # Temporal (2 windows)
    (2, 1, 64),
    (2, 2, 64),
    (2, 4, 64),
    # Temporal (4 windows)
    (4, 1, 64),
    (4, 2, 64),   # current best config
    (4, 4, 32),   # bs=32 to fit in VRAM
    # Temporal (8 windows)
    (8, 1, 64),
    (8, 2, 32),   # bs=32 to fit in VRAM
]

N_PARALLEL = 2  # 2 experiments per SLURM job
TIME_LIMIT = "02:30:00"  # ~111 min per experiment, 2.5h gives margin

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
        f" --data.num_workers=4"
        f" --meta.seed={seed}"
        f"{FIXED_ARGS}"
    )


def build_jobs():
    """Group experiments into SLURM jobs, 2 per job running in parallel."""
    # Expand grid × seeds
    all_runs = []
    for nw, ws, bs in GRID:
        for seed in SEEDS:
            all_runs.append((nw, ws, bs, seed))

    # Chunk into pairs
    jobs = []
    for i in range(0, len(all_runs), N_PARALLEL):
        chunk = all_runs[i : i + N_PARALLEL]
        cmds = [_make_run_cmd(nw, ws, bs, seed) for nw, ws, bs, seed in chunk]

        # Launch all in parallel with & and wait
        parallel_cmd = " &\n".join(cmds) + " &\nwait"

        # Descriptive name
        desc = " + ".join(f"nw{nw}_ws{ws}s_s{seed}" for nw, ws, _, seed in chunk)
        job_name = f"phase1_{i // N_PARALLEL:02d}"

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
            time_limit=TIME_LIMIT,
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

    total_exps = sum(len(c) for _, _, _, c in jobs)
    print(f"Phase 1 sweep: {total_exps} experiments in {len(jobs)} SLURM jobs "
          f"({N_PARALLEL} parallel per job, {TIME_LIMIT} each)\n")

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
