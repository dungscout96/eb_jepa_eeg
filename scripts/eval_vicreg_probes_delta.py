"""Run probe evaluation on VICReg projector ablation checkpoints.

Evaluates all valid VICReg sweep checkpoints (proj + noproj conditions).

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_vicreg_probes_delta.py --submit
"""

import os
import sys

from neurolab.jobs import Job

BASE = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa"

# (nw, ws, bs, seed, label, checkpoint_path)
# label is for display only (coeff + proj/noproj)
CHECKPOINTS = [
    # === PROJ (trained projector) ===
    # nw1_ws1 — c=0.1 and c=0.25 corrupted, only c=1.0 valid
    (1, 1, 64, 2025, "vc1.0_proj",
     f"{BASE}/dev_2026-04-12_12-41/eeg_jepa_bs64_lr0.0005_std1.0_cov1.0_nw1_ws1s_seed2025/latest.pth.tar"),
    # nw2_ws1
    (2, 1, 64, 2025, "vc0.1_proj",
     f"{BASE}/dev_2026-04-12_12-41/eeg_jepa_bs64_lr0.0005_std0.1_cov0.1_nw2_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, 2025, "vc0.25_proj",
     f"{BASE}/dev_2026-04-12_12-41/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw2_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, 2025, "vc1.0_proj",
     f"{BASE}/dev_2026-04-12_12-41/eeg_jepa_bs64_lr0.0005_std1.0_cov1.0_nw2_ws1s_seed2025/latest.pth.tar"),
    # nw4_ws4 — use sequential job checkpoints (dev_2026-04-12_20-*)
    (4, 4, 32, 2025, "vc0.1_proj",
     f"{BASE}/dev_2026-04-12_20-47/eeg_jepa_bs32_lr0.0005_std0.1_cov0.1_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, 2025, "vc0.25_proj",
     f"{BASE}/dev_2026-04-12_20-43/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, 2025, "vc1.0_proj",
     f"{BASE}/dev_2026-04-12_20-42/eeg_jepa_bs32_lr0.0005_std1.0_cov1.0_nw4_ws4s_seed2025/latest.pth.tar"),

    # === NOPROJ (no projector, Identity) ===
    # nw1_ws1
    (1, 1, 64, 2025, "vc0.1_noproj",
     f"{BASE}/dev_2026-04-13_14-57/eeg_jepa_bs64_lr0.0005_std0.1_cov0.1_noproj_nw1_ws1s_seed2025/latest.pth.tar"),
    (1, 1, 64, 2025, "vc0.25_noproj",
     f"{BASE}/dev_2026-04-13_14-57/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_noproj_nw1_ws1s_seed2025/latest.pth.tar"),
    (1, 1, 64, 2025, "vc1.0_noproj",
     f"{BASE}/dev_2026-04-13_14-58/eeg_jepa_bs64_lr0.0005_std1.0_cov1.0_noproj_nw1_ws1s_seed2025/latest.pth.tar"),
    # nw2_ws1
    (2, 1, 64, 2025, "vc0.1_noproj",
     f"{BASE}/dev_2026-04-13_14-58/eeg_jepa_bs64_lr0.0005_std0.1_cov0.1_noproj_nw2_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, 2025, "vc0.25_noproj",
     f"{BASE}/dev_2026-04-13_14-58/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_noproj_nw2_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, 2025, "vc1.0_noproj",
     f"{BASE}/dev_2026-04-13_14-58/eeg_jepa_bs64_lr0.0005_std1.0_cov1.0_noproj_nw2_ws1s_seed2025/latest.pth.tar"),
    # nw4_ws4
    (4, 4, 32, 2025, "vc0.1_noproj",
     f"{BASE}/dev_2026-04-13_14-58/eeg_jepa_bs32_lr0.0005_std0.1_cov0.1_noproj_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, 2025, "vc0.25_noproj",
     f"{BASE}/dev_2026-04-13_23-00/eeg_jepa_bs32_lr0.0005_std0.25_cov0.25_noproj_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, 2025, "vc1.0_noproj",
     f"{BASE}/dev_2026-04-13_15-04/eeg_jepa_bs32_lr0.0005_std1.0_cov1.0_noproj_nw4_ws4s_seed2025/latest.pth.tar"),
]


def _make_eval_cmd(nw, ws, bs, seed, label, ckpt_path):
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
        f" --wandb_group=probe_eval_vicreg"
        f" --seed={seed}"
    )


def build_jobs():
    jobs = []
    for idx in range(0, len(CHECKPOINTS), 2):
        chunk = CHECKPOINTS[idx: idx + 2]
        cmds = [_make_eval_cmd(*c) for c in chunk]
        combined_cmd = " &&\n".join(cmds)

        max_nw_ws = max(c[0] * c[1] for c in chunk)
        time_limit = "02:30:00" if max_nw_ws > 8 else "01:30:00"

        desc = " + ".join(f"nw{nw}_ws{ws}_{label}" for nw, ws, bs, seed, label, _ in chunk)
        job_name = f"evr_{idx // 2:02d}"

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

    print(f"VICReg probe eval: {len(CHECKPOINTS)} checkpoints in {len(jobs)} SLURM jobs\n")

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
        print(f"\n{len(jobs)} SLURM jobs ready. Run with --submit to send to Delta.")
