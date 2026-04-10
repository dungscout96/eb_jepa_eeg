"""Sweep SIGReg regularizer on the top 3 temporal configs from Phase 1.

Compares SIGReg (isotropic Gaussian regularizer) vs the Phase 1 VCLoss
baseline to test whether enforcing isotropic Gaussian embeddings degrades
downstream probe performance for EEG.

Sweep grid:
  - Temporal configs: nw1_ws1, nw2_ws1, nw4_ws4 (Phase 1 winners)
  - SIGReg coeff: [0.01, 0.1, 1.0]
  - Seeds: [2025, 42, 7]
  = 3 configs × 3 coeffs × 3 seeds = 27 experiments

VCLoss baselines (std=0.25, cov=0.25) already completed in Phase 1 sweep.

Usage
-----
# Preview:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/sweep_sigreg_delta.py

# Submit:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/sweep_sigreg_delta.py --submit
"""

import os
import sys

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

TEMPORAL_CONFIGS = [
    # (n_windows, window_size, batch_size)
    (1, 1, 64),   # nw1_ws1 — best subject-trait AUC
    (2, 1, 64),   # nw2_ws1 — best movie-feature regression
    (4, 4, 32),   # nw4_ws4 — best all-around (needs bs=32 for memory)
]

SIGREG_COEFFS = [0.01, 0.1, 1.0]
SEEDS = [2025, 42, 7]

# Fixed hyperparameters (same as Phase 1 except regularizer)
FIXED_ARGS = (
    " --model.encoder_depth=2"
    " --optim.lr=5e-4"
    " --optim.lr_min=1e-6"
    " --optim.warmup_epochs=5"
    " --loss.pred_loss_type=smooth_l1"
    " --optim.epochs=100"
    " --loss.regularizer=sigreg"
    " --logging.wandb_group=sweep_sigreg"
)

# n_windows × window_size ≤ this → safe to run 2 in parallel
PARALLEL_THRESHOLD = 4

# Time estimates (from Phase 1 observations, per experiment solo):
#   nw1_ws1: ~45 min, nw2_ws1: ~1h, nw4_ws4: ~6.5h
TIME_LIMITS = {
    (1, 1): "01:30:00",   # nw1_ws1: 45 min × 1.5 safety
    (2, 1): "02:00:00",   # nw2_ws1: 1h × 1.5 safety
    (4, 4): "09:00:00",   # nw4_ws4: 6.5h × 1.4 safety
}


# ---------------------------------------------------------------------------
# Build experiment list
# ---------------------------------------------------------------------------

def _build_runs():
    """Generate all (nw, ws, bs, seed, coeff, time_limit) tuples."""
    runs = []
    for nw, ws, bs in TEMPORAL_CONFIGS:
        for coeff in SIGREG_COEFFS:
            for seed in SEEDS:
                tl = TIME_LIMITS[(nw, ws)]
                runs.append((nw, ws, bs, seed, coeff, tl))
    return runs


RUNS = _build_runs()


def _make_run_cmd(nw, ws, bs, seed, coeff):
    return (
        "PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py"
        f" --data.n_windows={nw}"
        f" --data.window_size_seconds={ws}"
        f" --data.batch_size={bs}"
        f" --data.num_workers=4"
        f" --meta.seed={seed}"
        f" --loss.sigreg.coeff={coeff}"
        f"{FIXED_ARGS}"
    )


def _minutes(t):
    h, m, _ = map(int, t.split(":"))
    return h * 60 + m


def _add_times(t1, t2):
    total = _minutes(t1) + _minutes(t2)
    return f"{total // 60:02d}:{total % 60:02d}:00"


def _can_parallelize(nw, ws):
    return nw * ws <= PARALLEL_THRESHOLD


def build_jobs():
    jobs = []
    for idx in range(0, len(RUNS), 2):
        chunk = RUNS[idx: idx + 2]
        cmds = [_make_run_cmd(r[0], r[1], r[2], r[3], r[4]) for r in chunk]

        if len(chunk) == 2 and all(_can_parallelize(r[0], r[1]) for r in chunk):
            time_limit = max(chunk, key=lambda r: _minutes(r[5]))[5]
            combined_cmd = " &\n".join(cmds) + " &\nwait"
            mode = "parallel"
        elif len(chunk) == 2:
            time_limit = _add_times(chunk[0][5], chunk[1][5])
            combined_cmd = " &&\n".join(cmds)
            mode = "sequential"
        else:
            time_limit = chunk[0][5]
            combined_cmd = cmds[0]
            mode = "solo"

        desc = " + ".join(
            f"nw{nw}_ws{ws}_c{coeff}_s{seed}"
            for nw, ws, bs, seed, coeff, tl in chunk
        )
        job_name = f"sig_{idx // 2:02d}"

        git_cmd = (
            "sleep $((RANDOM % 60)) &&"
            " (git fetch origin && git checkout main && git pull --ff-only"
            " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
        )

        job = Job(
            name=job_name,
            cluster="delta",
            repo_path="/u/dtyoung/eb_jepa_eeg",
            command=git_cmd + combined_cmd,
            venv="__none__",
            branch="",
            partition="gpuA40x4",
            time_limit=time_limit,
            mem_gb=64,
            gpus=1,
            env_vars={
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                "WANDB_PROJECT": "eb_jepa",
                "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            },
        )
        jobs.append((job_name, job, desc, mode, chunk))

    return jobs


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    jobs = build_jobs()

    total_exps = sum(len(c) for _, _, _, _, c in jobs)
    print(
        f"SIGReg sweep: {total_exps} experiments "
        f"({len(TEMPORAL_CONFIGS)} configs × {len(SIGREG_COEFFS)} coeffs × {len(SEEDS)} seeds) "
        f"in {len(jobs)} SLURM jobs\n"
    )

    for name, job, desc, mode, chunk in jobs:
        script = job.submit(dry_run=True)
        print("=" * 72)
        print(f"Job: {name}  [{job.time_limit}]  {mode}  ({len(chunk)} exp(s))")
        print(f"  Runs: {desc}")
        print("=" * 72)
        print(script)
        print()

    if submit:
        print("\n" + "=" * 72)
        print("SUBMITTING ALL JOBS")
        print("=" * 72)
        for name, job, desc, mode, chunk in jobs:
            job_id = job.submit()
            print(f"  {name} [{mode}]: {job_id}  ({desc})")
    else:
        print(
            f"\n{len(jobs)} SLURM jobs ready."
            " Run with --submit to send to Delta."
        )
