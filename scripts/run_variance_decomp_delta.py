"""Submit variance decomposition analysis for the best subject-trait checkpoints.

Runs scripts/variance_decomposition.py on the checkpoints documented in
docs/best_subject_trait_checkpoints.md, packing all of them into a single
SLURM job (embedding extraction is cheap — ~5 min per checkpoint).

Usage
-----
# Preview:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_variance_decomp_delta.py

# Submit:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/run_variance_decomp_delta.py --submit
"""

import os
import sys

from neurolab.jobs import Job

BASE = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa"
OUT = "/u/dtyoung/eb_jepa_eeg/outputs/variance_decomp_k32"

# (nw, ws, bs, label, checkpoint_path)
# Drawn from docs/best_subject_trait_checkpoints.md (seed=2025, 100 ep).
CHECKPOINTS = [
    # --- SIGReg (top-3 per metric) ---
    (1, 1, 64, "sigreg1.0_nw1_ws1_best_sex_auc",
     f"{BASE}/dev_2026-04-10_16-00/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw1_ws1s_seed2025/latest.pth.tar"),
    (2, 1, 64, "sigreg0.1_nw2_ws1_best_balanced",
     f"{BASE}/dev_2026-04-10_16-04/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw2_ws1s_seed2025/latest.pth.tar"),
    (4, 4, 32, "sigreg0.1_nw4_ws4_best_overall",
     f"{BASE}/dev_2026-04-11_00-15/eeg_jepa_bs32_lr0.0005_sigreg0.1_nw4_ws4s_seed2025/latest.pth.tar"),
    # --- VICReg + trained projector (top per metric) ---
    (2, 1, 64, "vicreg1.0proj_nw2_ws1_best_age_cls",
     f"{BASE}/dev_2026-04-12_12-41/eeg_jepa_bs64_lr0.0005_std1.0_cov1.0_nw2_ws1s_seed2025/latest.pth.tar"),
    (4, 4, 32, "vicreg1.0proj_nw4_ws4_best_sex_auc",
     f"{BASE}/dev_2026-04-12_20-42/eeg_jepa_bs32_lr0.0005_std1.0_cov1.0_nw4_ws4s_seed2025/latest.pth.tar"),
    (4, 4, 32, "vicreg0.1proj_nw4_ws4_best_age_corr",
     f"{BASE}/dev_2026-04-12_20-47/eeg_jepa_bs32_lr0.0005_std0.1_cov0.1_nw4_ws4s_seed2025/latest.pth.tar"),
]


def _make_cmd(nw, ws, bs, ckpt_path):
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/variance_decomposition.py"
        f" --checkpoint={ckpt_path}"
        f" --n_windows={nw}"
        f" --window_size_seconds={ws}"
        f" --batch_size={bs}"
        f" --num_workers=4"
        f" --n_clips_per_rec=32"
        f" --split=val"
        f" --output_dir={OUT}"
    )


def _make_aggregate_cmd():
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/variance_decomposition.py"
        f" --aggregate_dir={OUT}"
    )


def build_job():
    per_ckpt = [_make_cmd(nw, ws, bs, ckpt) for nw, ws, bs, _, ckpt in CHECKPOINTS]
    all_cmds = per_ckpt + [_make_aggregate_cmd()]
    combined = " &&\n".join(all_cmds)

    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )

    return Job(
        name="vardecomp",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=git_cmd + combined,
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="08:00:00",  # K=32: ~90 min/ckpt for nw4_ws4 × 3 + smaller configs
        mem_gb=64,
        gpus=1,
        env_vars={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    job = build_job()

    print(f"Variance decomposition: {len(CHECKPOINTS)} checkpoints + aggregation\n")
    print("=" * 72)
    print(f"Job: {job.name}  [{job.time_limit}]")
    for nw, ws, bs, label, _ in CHECKPOINTS:
        print(f"  - {label}  (nw={nw}, ws={ws}, bs={bs})")
    print(f"  - aggregate → {OUT}/summary.md + figures/")
    print("=" * 72)
    print(job.submit(dry_run=True))
    print()

    if submit:
        print("=" * 72)
        print("SUBMITTING")
        print("=" * 72)
        job_id = job.submit()
        print(f"  {job.name}: {job_id}")
    else:
        print("Dry run only. Re-run with --submit to send to Delta.")
