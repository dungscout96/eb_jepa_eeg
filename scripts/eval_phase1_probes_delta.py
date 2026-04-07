"""Run post-hoc linear probe evaluation on all Phase 1 sweep checkpoints.

For each completed Phase 1 experiment, loads the frozen encoder from
latest.pth.tar, trains a fresh linear probe on the train set, then evaluates
on val and test.  Results are logged to W&B under group=probe_eval_phase1.

Packing strategy: probe eval is CPU-light and GPU-light per job.  Two
experiments run sequentially in one SLURM job (same queue slot, same
GPU-hours as solo — but halves scheduler overhead).

Usage
-----
# Preview:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_phase1_probes_delta.py

# Submit:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_phase1_probes_delta.py --submit
"""

import os
import sys

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# All Phase 1 checkpoint dirs on Delta (one entry per completed experiment)
# Format: (n_windows, window_size_seconds, batch_size, seed, checkpoint_path)
# ---------------------------------------------------------------------------

CKPT_APR5_2250 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-05_22-50"
CKPT_APR5_2316 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-05_23-16"
CKPT_APR5_2317 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-05_23-17"
CKPT_APR6_0657 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_06-57"
CKPT_APR6_0700 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_07-00"
CKPT_APR6_0701 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_07-01"
CKPT_APR6_0702 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_07-02"
CKPT_APR6_0705 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_07-05"
CKPT_APR6_0706 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_07-06"
CKPT_APR6_0740 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_07-40"
CKPT_APR6_0804 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_08-04"
CKPT_APR6_0811 = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-06_08-11"

def ckpt(base, nw, ws, bs, seed):
    bs_str = f"bs{bs}"
    return (
        f"{base}/eeg_jepa_{bs_str}_lr0.0005_std0.25_cov0.25"
        f"_nw{nw}_ws{ws}s_seed{seed}/latest.pth.tar"
    )

# (nw, ws, bs, seed, checkpoint_path)
CHECKPOINTS = [
    # nw1_ws1 — from original April 5 sweep
    (1, 1, 64, 2025, ckpt(CKPT_APR5_2250, 1, 1, 64, 2025)),
    (1, 1, 64, 42,   ckpt(CKPT_APR5_2250, 1, 1, 64, 42)),
    (1, 1, 64, 7,    ckpt(CKPT_APR5_2250, 1, 1, 64, 7)),

    # nw1_ws2
    (1, 2, 64, 2025, ckpt(CKPT_APR5_2250, 1, 2, 64, 2025)),
    (1, 2, 64, 42,   ckpt(CKPT_APR5_2250, 1, 2, 64, 42)),
    (1, 2, 64, 7,    ckpt(CKPT_APR5_2250, 1, 2, 64, 7)),

    # nw1_ws4 — seed42 from April 5, seeds 2025+7 from April 6 resume
    (1, 4, 64, 42,   ckpt(CKPT_APR5_2250, 1, 4, 64, 42)),
    (1, 4, 64, 2025, ckpt(CKPT_APR6_0657, 1, 4, 64, 2025)),
    (1, 4, 64, 7,    ckpt(CKPT_APR6_0657, 1, 4, 64, 7)),

    # nw2_ws1
    (2, 1, 64, 2025, ckpt(CKPT_APR5_2250, 2, 1, 64, 2025)),
    (2, 1, 64, 42,   ckpt(CKPT_APR5_2250, 2, 1, 64, 42)),
    (2, 1, 64, 7,    ckpt(CKPT_APR5_2250, 2, 1, 64, 7)),

    # nw2_ws2
    (2, 2, 64, 2025, ckpt(CKPT_APR5_2250, 2, 2, 64, 2025)),
    (2, 2, 64, 42,   ckpt(CKPT_APR5_2250, 2, 2, 64, 42)),
    (2, 2, 64, 7,    ckpt(CKPT_APR5_2250, 2, 2, 64, 7)),

    # nw2_ws4
    (2, 4, 64, 2025, ckpt(CKPT_APR6_0701, 2, 4, 64, 2025)),
    (2, 4, 64, 42,   ckpt(CKPT_APR6_0740, 2, 4, 64, 42)),
    (2, 4, 64, 7,    ckpt(CKPT_APR6_0702, 2, 4, 64, 7)),

    # nw4_ws1
    (4, 1, 64, 2025, ckpt(CKPT_APR5_2250, 4, 1, 64, 2025)),
    (4, 1, 64, 42,   ckpt(CKPT_APR5_2250, 4, 1, 64, 42)),
    (4, 1, 64, 7,    ckpt(CKPT_APR5_2250, 4, 1, 64, 7)),

    # nw4_ws2
    (4, 2, 64, 2025, ckpt(CKPT_APR5_2250, 4, 2, 64, 2025)),
    (4, 2, 64, 42,   ckpt(CKPT_APR6_0740, 4, 2, 64, 42)),
    (4, 2, 64, 7,    ckpt(CKPT_APR5_2250, 4, 2, 64, 7)),

    # nw4_ws4
    (4, 4, 32, 2025, ckpt(CKPT_APR6_0700, 4, 4, 32, 2025)),
    (4, 4, 32, 42,   ckpt(CKPT_APR6_0811, 4, 4, 32, 42)),
    (4, 4, 32, 7,    ckpt(CKPT_APR6_0811, 4, 4, 32, 7)),

    # nw8_ws1
    (8, 1, 64, 2025, ckpt(CKPT_APR5_2316, 8, 1, 64, 2025)),
    (8, 1, 64, 42,   ckpt(CKPT_APR6_0705, 8, 1, 64, 42)),
    (8, 1, 64, 7,    ckpt(CKPT_APR6_0804, 8, 1, 64, 7)),

    # nw8_ws2
    (8, 2, 32, 2025, ckpt(CKPT_APR6_0706, 8, 2, 32, 2025)),
    (8, 2, 32, 42,   ckpt(CKPT_APR6_0811, 8, 2, 32, 42)),
    (8, 2, 32, 7,    ckpt(CKPT_APR6_0706, 8, 2, 32, 7)),
]


def _make_eval_cmd(nw, ws, bs, seed, ckpt_path):
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python experiments/eeg_jepa/probe_eval.py"
        f" --checkpoint={ckpt_path}"
        f" --n_windows={nw}"
        f" --window_size_seconds={ws}"
        f" --batch_size={bs}"
        f" --num_workers=4"
        f" --probe_epochs=20"
        f" --splits=val,test"
        f" --wandb_group=probe_eval_phase1"
        f" --seed={seed}"
    )


def build_jobs():
    jobs = []
    for idx in range(0, len(CHECKPOINTS), 2):
        chunk = CHECKPOINTS[idx: idx + 2]
        cmds = [_make_eval_cmd(*c) for c in chunk]

        # All probe evals are sequential (encoder frozen = low GPU use,
        # but running 2 in parallel risks memory pressure for large configs)
        combined_cmd = " &&\n".join(cmds)

        # Time estimate: ~5-10 min per probe eval (20 epochs on frozen encoder)
        # Largest configs (nw8_ws2) take longer due to data loading
        # 2 per job × 15 min × 1.5 safety = ~45 min → 1h per job
        # nw8_ws2 is slower: 2 × 30 min × 1.5 = 1.5h
        max_nw_ws = max(c[0] * c[1] for c in chunk)
        time_limit = "01:30:00" if max_nw_ws > 8 else "01:00:00"

        desc = " + ".join(f"nw{nw}_ws{ws}s_s{seed}" for nw, ws, bs, seed, _ in chunk)
        job_name = f"pe1_{idx // 2:02d}"

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
        jobs.append((job_name, job, desc, chunk))

    return jobs


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    jobs = build_jobs()

    print(
        f"Phase 1 probe eval: {len(CHECKPOINTS)} checkpoints "
        f"in {len(jobs)} SLURM jobs\n"
    )

    for name, job, desc, chunk in jobs:
        script = job.submit(dry_run=True)
        print("=" * 72)
        print(f"Job: {name}  [{job.time_limit}]  ({len(chunk)} checkpoints)")
        print(f"  Evals: {desc}")
        print("=" * 72)
        print(script)
        print()

    if submit:
        print("\n" + "=" * 72)
        print("SUBMITTING ALL JOBS")
        print("=" * 72)
        for name, job, desc, chunk in jobs:
            job_id = job.submit()
            print(f"  {name}: {job_id}  ({desc})")
    else:
        print(
            f"\n{len(jobs)} SLURM jobs ready."
            " Run with --submit to send to Delta."
        )
