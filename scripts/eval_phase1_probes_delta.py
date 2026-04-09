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

BASE = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa"

# (nw, ws, bs, seed, checkpoint_path) — paths verified from Delta filesystem
CHECKPOINTS = [
    # nw1_ws1
    (1, 1, 64, 2025, f"{BASE}/dev_2026-04-05_22-44/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws1s_seed2025/latest.pth.tar"),
    (1, 1, 64, 42,   f"{BASE}/dev_2026-04-05_22-44/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws1s_seed42/latest.pth.tar"),
    (1, 1, 64, 7,    f"{BASE}/dev_2026-04-05_22-45/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws1s_seed7/latest.pth.tar"),

    # nw1_ws2
    (1, 2, 64, 2025, f"{BASE}/dev_2026-04-05_22-45/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws2s_seed2025/latest.pth.tar"),
    (1, 2, 64, 42,   f"{BASE}/dev_2026-04-05_22-46/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws2s_seed42/latest.pth.tar"),
    (1, 2, 64, 7,    f"{BASE}/dev_2026-04-05_22-46/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws2s_seed7/latest.pth.tar"),

    # nw1_ws4 — seed42 ep99, seeds 2025+7 at ep72-73 (best available)
    (1, 4, 64, 42,   f"{BASE}/dev_2026-04-05_22-45/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws4s_seed42/latest.pth.tar"),
    (1, 4, 64, 2025, f"{BASE}/dev_2026-04-07_07-55/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws4s_seed2025/latest.pth.tar"),
    (1, 4, 64, 7,    f"{BASE}/dev_2026-04-07_07-55/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw1_ws4s_seed7/latest.pth.tar"),

    # nw2_ws1
    (2, 1, 64, 2025, f"{BASE}/dev_2026-04-05_22-46/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, 42,   f"{BASE}/dev_2026-04-05_22-46/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws1s_seed42/latest.pth.tar"),
    (2, 1, 64, 7,    f"{BASE}/dev_2026-04-05_22-46/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws1s_seed7/latest.pth.tar"),

    # nw2_ws2
    (2, 2, 64, 2025, f"{BASE}/dev_2026-04-05_22-48/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws2s_seed2025/latest.pth.tar"),
    (2, 2, 64, 42,   f"{BASE}/dev_2026-04-05_22-48/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws2s_seed42/latest.pth.tar"),
    (2, 2, 64, 7,    f"{BASE}/dev_2026-04-05_22-49/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws2s_seed7/latest.pth.tar"),

    # nw2_ws4
    (2, 4, 64, 2025, f"{BASE}/dev_2026-04-05_22-49/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws4s_seed2025/latest.pth.tar"),
    (2, 4, 64, 42,   f"{BASE}/dev_2026-04-06_09-04/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws4s_seed42/latest.pth.tar"),
    (2, 4, 64, 7,    f"{BASE}/dev_2026-04-06_07-02/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws4s_seed7/latest.pth.tar"),

    # nw4_ws1
    (4, 1, 64, 2025, f"{BASE}/dev_2026-04-05_23-16/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws1s_seed2025/latest.pth.tar"),
    (4, 1, 64, 42,   f"{BASE}/dev_2026-04-05_23-16/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws1s_seed42/latest.pth.tar"),
    (4, 1, 64, 7,    f"{BASE}/dev_2026-04-06_06-58/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws1s_seed7/latest.pth.tar"),

    # nw4_ws2
    (4, 2, 64, 2025, f"{BASE}/dev_2026-04-05_23-16/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed2025/latest.pth.tar"),
    (4, 2, 64, 42,   f"{BASE}/dev_2026-04-06_07-40/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed42/latest.pth.tar"),
    (4, 2, 64, 7,    f"{BASE}/dev_2026-04-05_23-16/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed7/latest.pth.tar"),

    # nw4_ws4
    (4, 4, 32, 2025, f"{BASE}/dev_2026-04-06_07-00/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, 42,   f"{BASE}/dev_2026-04-06_08-11/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw4_ws4s_seed42/latest.pth.tar"),
    (4, 4, 32, 7,    f"{BASE}/dev_2026-04-06_15-06/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw4_ws4s_seed7/latest.pth.tar"),

    # nw8_ws1
    (8, 1, 64, 2025, f"{BASE}/dev_2026-04-05_23-16/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw8_ws1s_seed2025/latest.pth.tar"),
    (8, 1, 64, 42,   f"{BASE}/dev_2026-04-06_07-05/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw8_ws1s_seed42/latest.pth.tar"),
    (8, 1, 64, 7,    f"{BASE}/dev_2026-04-06_08-04/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw8_ws1s_seed7/latest.pth.tar"),

    # nw8_ws2
    (8, 2, 32, 2025, f"{BASE}/dev_2026-04-05_23-17/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw8_ws2s_seed2025/latest.pth.tar"),
    (8, 2, 32, 42,   f"{BASE}/dev_2026-04-06_13-36/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw8_ws2s_seed42/latest.pth.tar"),
    (8, 2, 32, 7,    f"{BASE}/dev_2026-04-05_23-17/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw8_ws2s_seed7/latest.pth.tar"),
]


def _make_eval_cmd(nw, ws, bs, seed, ckpt_path, subject_only=False):
    cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python experiments/eeg_jepa/probe_eval.py"
        f" --checkpoint={ckpt_path}"
        f" --n_windows={nw}"
        f" --window_size_seconds={ws}"
        f" --batch_size={bs}"
        f" --num_workers=4"
        f" --probe_epochs=20"
        f" --splits=val,test"
        f" --seed={seed}"
    )
    if subject_only:
        cmd += " --subject_only"
        cmd += " --wandb_group=probe_eval_phase1_subject"
    else:
        cmd += " --wandb_group=probe_eval_phase1"
    return cmd


def build_jobs(subject_only=False):
    jobs = []
    for idx in range(0, len(CHECKPOINTS), 2):
        chunk = CHECKPOINTS[idx: idx + 2]
        cmds = [_make_eval_cmd(*c, subject_only=subject_only) for c in chunk]

        # All probe evals are sequential (encoder frozen = low GPU use,
        # but running 2 in parallel risks memory pressure for large configs)
        combined_cmd = " &&\n".join(cmds)

        if subject_only:
            # Subject-only: no movie probe training, just embed + linear probe
            # ~15 min per eval (embed all recordings + 100 epoch probe)
            # 2 per job × 15 min + git jitter → 45 min safe
            max_nw_ws = max(c[0] * c[1] for c in chunk)
            time_limit = "01:30:00" if max_nw_ws > 8 else "01:00:00"
        else:
            # Full eval: ~40 min for 1st (20 min norm stats + 20 min probe)
            # + ~20 min for 2nd (norm stats cached) + ~1 min git jitter
            max_nw_ws = max(c[0] * c[1] for c in chunk)
            time_limit = "02:30:00" if max_nw_ws > 8 else "01:30:00"

        desc = " + ".join(f"nw{nw}_ws{ws}s_s{seed}" for nw, ws, bs, seed, _ in chunk)
        prefix = "ps1" if subject_only else "pe1"
        job_name = f"{prefix}_{idx // 2:02d}"

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
        jobs.append((job_name, job, desc, chunk))

    return jobs


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    subject_only = "--subject-only" in sys.argv
    jobs = build_jobs(subject_only=subject_only)

    mode = "subject-only" if subject_only else "full"
    print(
        f"Phase 1 probe eval ({mode}): {len(CHECKPOINTS)} checkpoints "
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
