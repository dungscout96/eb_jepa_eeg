"""probe_eval + variance_decomp on the 7 per-rec retrained checkpoints.

Pairs with scripts/retrain_best_configs_perrec_delta.py. Each checkpoint is
the val-peak best.pth.tar from an early-stopped training run that used
--data.norm_mode=per_recording; eval must match, so all commands pass
--norm_mode=per_recording.

Emits 2 SLURM jobs (one for probe_eval, one for variance_decomp at K=32);
each iterates over all 7 checkpoints sequentially.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_retrained_perrec_delta.py            # dry run
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_retrained_perrec_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CKPT_BASE = "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa"
VAR_OUT = "/u/dtyoung/eb_jepa_eeg/outputs/variance_decomp_retrained_perrec"

# (nw, ws, bs, rel_path, label)
CHECKPOINTS = [
    (1, 1, 64,
     "dev_2026-04-21_10-19/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw1_ws1s_seed2025/best.pth.tar",
     "sigreg1.0_nw1_ws1"),
    (2, 1, 64,
     "dev_2026-04-21_10-20/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw2_ws1s_seed2025/best.pth.tar",
     "sigreg0.1_nw2_ws1"),
    (4, 4, 32,
     "dev_2026-04-21_10-20/eeg_jepa_bs32_lr0.0005_sigreg0.1_nw4_ws4s_seed2025/best.pth.tar",
     "sigreg0.1_nw4_ws4"),
    (2, 1, 64,
     "dev_2026-04-21_10-20/eeg_jepa_bs64_lr0.0005_std1.0_cov1.0_nw2_ws1s_seed2025/best.pth.tar",
     "vc1.0_proj_nw2_ws1"),
    (4, 4, 32,
     "dev_2026-04-21_10-21/eeg_jepa_bs32_lr0.0005_std1.0_cov1.0_nw4_ws4s_seed2025/best.pth.tar",
     "vc1.0_proj_nw4_ws4"),
    (4, 4, 32,
     "dev_2026-04-21_10-20/eeg_jepa_bs32_lr0.0005_std0.1_cov0.1_nw4_ws4s_seed2025/best.pth.tar",
     "vc0.1_proj_nw4_ws4"),
    (4, 4, 32,
     "dev_2026-04-21_10-21/eeg_jepa_bs32_lr0.0005_std0.1_cov0.1_noproj_nw4_ws4s_seed2025/best.pth.tar",
     "vc0.1_noproj_nw4_ws4"),
]


def _probe_cmd(nw, ws, bs, ckpt_rel):
    ckpt = f"{CKPT_BASE}/{ckpt_rel}"
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python experiments/eeg_jepa/probe_eval.py"
        f" --checkpoint={ckpt}"
        " --norm_mode=per_recording"
        f" --n_windows={nw} --window_size_seconds={ws}"
        f" --batch_size={bs} --num_workers=4"
        " --probe_epochs=20 --subject_probe_epochs=100"
        " --splits=val,test"
        " --wandb_group=probe_eval_retrained_perrec"
        " --seed=2025"
    )


def _vardecomp_cmd(nw, ws, bs, ckpt_rel):
    ckpt = f"{CKPT_BASE}/{ckpt_rel}"
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/variance_decomposition.py"
        f" --checkpoint={ckpt}"
        " --norm_mode=per_recording"
        f" --n_windows={nw} --window_size_seconds={ws}"
        f" --batch_size={bs} --num_workers=4"
        " --n_clips_per_rec=32 --split=val"
        f" --output_dir={VAR_OUT}"
    )


def _git_cmd():
    return (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )


def build_jobs():
    jobs = []

    # Job A: probe_eval × 7 (~30 min/ckpt → ~3.5h total, budget 5h)
    probe_cmds = [_probe_cmd(nw, ws, bs, rel) for nw, ws, bs, rel, _ in CHECKPOINTS]
    jobs.append(("probe_eval_perrec", Job(
        name="ev_probe_pr",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=_git_cmd() + " &&\n".join(probe_cmds),
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="05:00:00",
        mem_gb=64,
        gpus=1,
        env_vars={
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            "WANDB_PROJECT": "eb_jepa",
        },
    )))

    # Job B: variance_decomposition × 7 at K=32 + aggregate (~7h total, budget 9h)
    # Cost dominated by 4 nw4_ws4 × ~90 min each.
    var_cmds = [_vardecomp_cmd(nw, ws, bs, rel) for nw, ws, bs, rel, _ in CHECKPOINTS]
    agg_cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/variance_decomposition.py"
        f" --aggregate_dir={VAR_OUT}"
    )
    jobs.append(("vardecomp_perrec", Job(
        name="ev_vardec_pr",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=_git_cmd() + " &&\n".join(var_cmds + [agg_cmd]),
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="09:00:00",
        mem_gb=64,
        gpus=1,
        env_vars={
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )))
    return jobs


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    jobs = build_jobs()
    print(f"Per-rec eval: {len(CHECKPOINTS)} checkpoints × 2 eval types\n")
    for label, job in jobs:
        print("=" * 72)
        print(f"Job: {job.name} ({label})  [{job.time_limit}]")
        print("=" * 72)
        print(job.submit(dry_run=True))
        print()
    if submit:
        print("=" * 72)
        print("SUBMITTING")
        print("=" * 72)
        for label, job in jobs:
            job_id = job.submit()
            print(f"  {job.name}: {job_id}")
    else:
        print(f"\n{len(jobs)} jobs ready. Re-run with --submit.")
