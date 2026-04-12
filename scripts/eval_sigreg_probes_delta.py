"""Run post-hoc probe evaluation on all SIGReg sweep checkpoints.

Runs movie-feature probes, movie identity probe, and subject-trait probes
on all 27 SIGReg experiments (3 configs x 3 coeffs x 3 seeds).

Usage
-----
# Preview:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_sigreg_probes_delta.py

# Submit:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_sigreg_probes_delta.py --submit
"""

import os
import sys

from neurolab.jobs import Job

BASE = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa"

# (nw, ws, bs, seed, coeff, checkpoint_path)
CHECKPOINTS = [
    # nw1_ws1 — coeff=0.01
    (1, 1, 64, 2025, 0.01, f"{BASE}/dev_2026-04-10_15-55/eeg_jepa_bs64_lr0.0005_sigreg0.01_nw1_ws1s_seed2025/latest.pth.tar"),
    (1, 1, 64, 42,   0.01, f"{BASE}/dev_2026-04-10_15-55/eeg_jepa_bs64_lr0.0005_sigreg0.01_nw1_ws1s_seed42/latest.pth.tar"),
    (1, 1, 64, 7,    0.01, f"{BASE}/dev_2026-04-10_15-58/eeg_jepa_bs64_lr0.0005_sigreg0.01_nw1_ws1s_seed7/latest.pth.tar"),
    # nw1_ws1 — coeff=0.1
    (1, 1, 64, 2025, 0.1,  f"{BASE}/dev_2026-04-10_15-57/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw1_ws1s_seed2025/latest.pth.tar"),
    (1, 1, 64, 42,   0.1,  f"{BASE}/dev_2026-04-10_15-58/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw1_ws1s_seed42/latest.pth.tar"),
    (1, 1, 64, 7,    0.1,  f"{BASE}/dev_2026-04-10_15-57/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw1_ws1s_seed7/latest.pth.tar"),
    # nw1_ws1 — coeff=1.0
    (1, 1, 64, 2025, 1.0,  f"{BASE}/dev_2026-04-10_16-00/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw1_ws1s_seed2025/latest.pth.tar"),
    (1, 1, 64, 42,   1.0,  f"{BASE}/dev_2026-04-10_15-59/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw1_ws1s_seed42/latest.pth.tar"),
    (1, 1, 64, 7,    1.0,  f"{BASE}/dev_2026-04-10_16-01/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw1_ws1s_seed7/latest.pth.tar"),

    # nw2_ws1 — coeff=0.01
    (2, 1, 64, 2025, 0.01, f"{BASE}/dev_2026-04-10_16-00/eeg_jepa_bs64_lr0.0005_sigreg0.01_nw2_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, 42,   0.01, f"{BASE}/dev_2026-04-10_16-04/eeg_jepa_bs64_lr0.0005_sigreg0.01_nw2_ws1s_seed42/latest.pth.tar"),
    (2, 1, 64, 7,    0.01, f"{BASE}/dev_2026-04-10_16-03/eeg_jepa_bs64_lr0.0005_sigreg0.01_nw2_ws1s_seed7/latest.pth.tar"),
    # nw2_ws1 — coeff=0.1
    (2, 1, 64, 2025, 0.1,  f"{BASE}/dev_2026-04-10_16-04/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw2_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, 42,   0.1,  f"{BASE}/dev_2026-04-10_16-04/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw2_ws1s_seed42/latest.pth.tar"),
    (2, 1, 64, 7,    0.1,  f"{BASE}/dev_2026-04-10_16-05/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw2_ws1s_seed7/latest.pth.tar"),
    # nw2_ws1 — coeff=1.0
    (2, 1, 64, 2025, 1.0,  f"{BASE}/dev_2026-04-10_16-04/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw2_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, 42,   1.0,  f"{BASE}/dev_2026-04-10_16-06/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw2_ws1s_seed42/latest.pth.tar"),
    (2, 1, 64, 7,    1.0,  f"{BASE}/dev_2026-04-10_16-05/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw2_ws1s_seed7/latest.pth.tar"),

    # nw4_ws4 — coeff=0.01
    (4, 4, 32, 2025, 0.01, f"{BASE}/dev_2026-04-10_16-09/eeg_jepa_bs32_lr0.0005_sigreg0.01_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, 42,   0.01, f"{BASE}/dev_2026-04-11_00-09/eeg_jepa_bs32_lr0.0005_sigreg0.01_nw4_ws4s_seed42/latest.pth.tar"),
    (4, 4, 32, 7,    0.01, f"{BASE}/dev_2026-04-10_16-09/eeg_jepa_bs32_lr0.0005_sigreg0.01_nw4_ws4s_seed7/latest.pth.tar"),
    # nw4_ws4 — coeff=0.1
    (4, 4, 32, 2025, 0.1,  f"{BASE}/dev_2026-04-11_00-15/eeg_jepa_bs32_lr0.0005_sigreg0.1_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, 42,   0.1,  f"{BASE}/dev_2026-04-10_16-17/eeg_jepa_bs32_lr0.0005_sigreg0.1_nw4_ws4s_seed42/latest.pth.tar"),
    (4, 4, 32, 7,    0.1,  f"{BASE}/dev_2026-04-11_00-17/eeg_jepa_bs32_lr0.0005_sigreg0.1_nw4_ws4s_seed7/latest.pth.tar"),
    # nw4_ws4 — coeff=1.0
    (4, 4, 32, 2025, 1.0,  f"{BASE}/dev_2026-04-10_16-17/eeg_jepa_bs32_lr0.0005_sigreg1.0_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, 42,   1.0,  f"{BASE}/dev_2026-04-11_00-22/eeg_jepa_bs32_lr0.0005_sigreg1.0_nw4_ws4s_seed42/latest.pth.tar"),
    (4, 4, 32, 7,    1.0,  f"{BASE}/dev_2026-04-10_16-17/eeg_jepa_bs32_lr0.0005_sigreg1.0_nw4_ws4s_seed7/latest.pth.tar"),
]


def _make_eval_cmd(nw, ws, bs, seed, coeff, ckpt_path):
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
        f" --wandb_group=probe_eval_sigreg"
        f" --seed={seed}"
    )


def build_jobs():
    jobs = []
    for idx in range(0, len(CHECKPOINTS), 2):
        chunk = CHECKPOINTS[idx: idx + 2]
        cmds = [_make_eval_cmd(*c) for c in chunk]
        combined_cmd = " &&\n".join(cmds)

        # Time: ~20 min per eval (norm stats cached) + movie_id + subject probes
        # Small configs: 2 evals ~ 40 min → 1.5h safe
        # Large (nw4_ws4): 2 evals ~ 90 min → 2.5h safe
        max_nw_ws = max(c[0] * c[1] for c in chunk)
        time_limit = "02:30:00" if max_nw_ws > 8 else "01:30:00"

        desc = " + ".join(
            f"nw{nw}_ws{ws}_c{coeff}_s{seed}"
            for nw, ws, bs, seed, coeff, _ in chunk
        )
        job_name = f"esr_{idx // 2:02d}"

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
    jobs = build_jobs()

    print(
        f"SIGReg probe eval: {len(CHECKPOINTS)} checkpoints "
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
