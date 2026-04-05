"""Resubmit incomplete Phase 1 sweep experiments on Delta.

Two categories:
  1. Fresh restarts — 4 experiments that crashed or never saved a checkpoint
  2. Resumes — 13 experiments interrupted by the 2.5 h time limit; picks up
     from the saved latest.pth.tar via --meta.load_model=true

GPU-hour efficiency strategy
-----------------------------
Experiments are paired 2-per-SLURM-job.  Within each job they run either
in parallel or sequentially depending on config size (n_windows × window_size):

  n_windows × window_size ≤ PARALLEL_THRESHOLD
      → background parallel (&+wait): wall = max(T1,T2), GPU-hrs = max(T1,T2)
  n_windows × window_size > PARALLEL_THRESHOLD
      → sequential (&&): wall = T1+T2, GPU-hrs = T1+T2

Sequential pairing uses the *same* GPU-hours as 2 solo submissions but
halves queue overhead.  Parallel pairing halves both wall time and GPU-hours.

From observations: nw×ws=4 (nw4_ws1) ran fine in parallel; nw×ws=8
(nw4_ws2, nw2_ws4) caused GPU contention → 5–15× slowdown.
PARALLEL_THRESHOLD is therefore set to 4.

Time limits are per-experiment solo estimates × 1.5× safety margin, summed
for sequential pairs.  A short random sleep before git pull reduces the
chance of index.lock collisions when many jobs start at the same time.

Usage
-----
# Preview generated SLURM scripts:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab python scripts/sweep_phase1_resume_delta.py

# Submit all jobs:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab python scripts/sweep_phase1_resume_delta.py --submit
"""

import os
import sys

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# Checkpoint base path on Delta
# ---------------------------------------------------------------------------

CKPT_BASE = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa"

# Fixed hyperparameters (same as original sweep)
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
# Runs to submit
# Each entry: (nw, ws, bs, seed, ckpt_path_or_None, time_limit)
#   ckpt_path=None  → fresh start
#   ckpt_path=str   → resume from that checkpoint
# ---------------------------------------------------------------------------

RUNS = [
    # ── Fresh restarts ─────────────────────────────────────────────────────
    # Crashed silently on gpub032 (node CUDA issue); no checkpoint saved
    (2, 1, 64, 42,   None, "01:00:00"),
    (2, 2, 64, 2025, None, "01:00:00"),
    # Never ran: job 12 timed-out before seed2025 saved anything
    (4, 4, 32, 2025, None, "06:00:00"),
    # Never ran: job 16 failed immediately due to git index.lock
    (8, 2, 32, 7,    None, "14:00:00"),

    # ── Resumes from checkpoint ────────────────────────────────────────────
    # nw=2, ws=4s  (ep 64 / 45 / 45 saved)
    (2, 4, 64, 2025,
     f"{CKPT_BASE}/dev_2026-04-04_18-03"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws4s_seed2025/latest.pth.tar",
     "01:30:00"),
    (2, 4, 64, 42,
     f"{CKPT_BASE}/dev_2026-04-04_18-03"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws4s_seed42/latest.pth.tar",
     "01:30:00"),
    (2, 4, 64, 7,
     f"{CKPT_BASE}/dev_2026-04-04_18-03"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws4s_seed7/latest.pth.tar",
     "01:30:00"),

    # nw=4, ws=2s  (ep 73 / 51 / 51 saved)
    (4, 2, 64, 2025,
     f"{CKPT_BASE}/dev_2026-04-04_18-04"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed2025/latest.pth.tar",
     "01:30:00"),
    (4, 2, 64, 42,
     f"{CKPT_BASE}/dev_2026-04-04_18-09"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed42/latest.pth.tar",
     "02:00:00"),
    (4, 2, 64, 7,
     f"{CKPT_BASE}/dev_2026-04-04_18-09"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed7/latest.pth.tar",
     "02:00:00"),

    # nw=4, ws=4s  (ep 29 / 10 saved; seed2025 is a fresh start above)
    (4, 4, 32, 42,
     f"{CKPT_BASE}/dev_2026-04-04_18-09"
     "/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw4_ws4s_seed42/latest.pth.tar",
     "04:00:00"),
    (4, 4, 32, 7,
     f"{CKPT_BASE}/dev_2026-04-04_18-13"
     "/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw4_ws4s_seed7/latest.pth.tar",
     "05:00:00"),

    # nw=8, ws=1s  (ep 48 / 51 / 51 saved)
    (8, 1, 64, 2025,
     f"{CKPT_BASE}/dev_2026-04-04_18-13"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw8_ws1s_seed2025/latest.pth.tar",
     "02:00:00"),
    (8, 1, 64, 42,
     f"{CKPT_BASE}/dev_2026-04-04_18-19"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw8_ws1s_seed42/latest.pth.tar",
     "02:00:00"),
    (8, 1, 64, 7,
     f"{CKPT_BASE}/dev_2026-04-04_18-19"
     "/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw8_ws1s_seed7/latest.pth.tar",
     "02:00:00"),

    # nw=8, ws=2s  (ep 11 / 11 saved; seed7 is a fresh start above)
    (8, 2, 32, 2025,
     f"{CKPT_BASE}/dev_2026-04-04_18-19"
     "/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw8_ws2s_seed2025/latest.pth.tar",
     "12:00:00"),
    (8, 2, 32, 42,
     f"{CKPT_BASE}/dev_2026-04-04_18-19"
     "/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw8_ws2s_seed42/latest.pth.tar",
     "12:00:00"),
]

# ---------------------------------------------------------------------------
# n_windows × window_size ≤ this → safe to run 2 experiments in parallel
# (verified: nw4_ws1=4 completed fine; nw4_ws2=8 caused 5-15x slowdown)
# ---------------------------------------------------------------------------
PARALLEL_THRESHOLD = 4


def _minutes(t):
    h, m, _ = map(int, t.split(":"))
    return h * 60 + m


def _add_times(t1, t2):
    """Return HH:MM:SS string that is the sum of two HH:MM:SS strings."""
    total = _minutes(t1) + _minutes(t2)
    return f"{total // 60:02d}:{total % 60:02d}:00"


def _can_parallelize(nw, ws):
    return nw * ws <= PARALLEL_THRESHOLD


def _make_run_cmd(nw, ws, bs, seed, ckpt_path):
    base = (
        "PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py"
        f" --data.n_windows={nw}"
        f" --data.window_size_seconds={ws}"
        f" --data.batch_size={bs}"
        f" --data.num_workers=4"
        f" --meta.seed={seed}"
        f"{FIXED_ARGS}"
    )
    if ckpt_path:
        base += f" --meta.load_model=true --meta.load_checkpoint={ckpt_path}"
    return base


def build_jobs():
    jobs = []
    for idx in range(0, len(RUNS), 2):
        chunk = RUNS[idx : idx + 2]
        cmds = [_make_run_cmd(*r[:5]) for r in chunk]

        if len(chunk) == 2 and all(_can_parallelize(r[0], r[1]) for r in chunk):
            # Both small enough: run in parallel — wall = max, GPU-hrs = max
            time_limit = max(chunk, key=lambda r: _minutes(r[5]))[5]
            combined_cmd = " &\n".join(cmds) + " &\nwait"
            mode = "parallel"
        elif len(chunk) == 2:
            # One or both too large: run sequentially — wall = sum, GPU-hrs = sum
            # (same GPU-hrs as 2 solo jobs, but only 1 queue slot)
            time_limit = _add_times(chunk[0][5], chunk[1][5])
            combined_cmd = " &&\n".join(cmds)
            mode = "sequential"
        else:
            time_limit = chunk[0][5]
            combined_cmd = cmds[0]
            mode = "solo"

        desc = " + ".join(
            f"nw{nw}_ws{ws}s_s{seed}{'(resume)' if ck else '(fresh)'}[{tl}]"
            for nw, ws, bs, seed, ck, tl in chunk
        )
        job_name = f"p1r_{idx // 2:02d}"

        git_cmd = (
            "sleep $((RANDOM % 15)) &&"
            " git fetch origin && git checkout main && git pull --ff-only &&\n"
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
    fresh = sum(1 for r in RUNS if r[4] is None)
    resume = sum(1 for r in RUNS if r[4] is not None)
    print(
        f"Phase 1 resume: {total_exps} experiments "
        f"({fresh} fresh, {resume} resumes) in {len(jobs)} SLURM jobs\n"
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
        print(f"\n{len(jobs)} SLURM jobs ready. Run with --submit to send to Delta.")
