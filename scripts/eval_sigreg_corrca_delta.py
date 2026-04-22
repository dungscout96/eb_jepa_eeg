"""probe_eval + variance_decomp on the 3 SIGReg + per-rec + CorrCA checkpoints.

Companion to scripts/train_sigreg_corrca_delta.py. All three were trained
with --data.norm_mode=per_recording and --data.corrca_filters=corrca_filters.npz,
so eval must match. Packs probe_eval (3 ckpts) and variance_decomp (3 ckpts
+ aggregate) into 2 SLURM jobs.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/eval_sigreg_corrca_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CORRCA = "/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"
VAR_OUT = "/u/dtyoung/eb_jepa_eeg/outputs/variance_decomp_sigreg_corrca"

CHECKPOINTS = [
    (1, 1, 64,
     "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-21_22-44/eeg_jepa_bs64_lr0.0005_sigreg1.0_nw1_ws1s_seed2025/best.pth.tar",
     "sigreg1.0_nw1_ws1_corrca"),
    (2, 1, 64,
     "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-21_22-44/eeg_jepa_bs64_lr0.0005_sigreg0.1_nw2_ws1s_seed2025/best.pth.tar",
     "sigreg0.1_nw2_ws1_corrca"),
    (4, 4, 32,
     "/u/dtyoung/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-21_22-44/eeg_jepa_bs32_lr0.0005_sigreg0.1_nw4_ws4s_seed2025/best.pth.tar",
     "sigreg0.1_nw4_ws4_corrca"),
]


def _probe_cmd(nw, ws, bs, ckpt):
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python experiments/eeg_jepa/probe_eval.py"
        f" --checkpoint={ckpt}"
        f" --corrca_filters={CORRCA}"
        " --norm_mode=per_recording"
        f" --n_windows={nw} --window_size_seconds={ws}"
        f" --batch_size={bs} --num_workers=4"
        " --probe_epochs=20 --subject_probe_epochs=100"
        " --splits=val,test"
        " --wandb_group=probe_eval_sigreg_corrca"
        " --seed=2025"
    )


def _vardecomp_cmd(nw, ws, bs, ckpt):
    return (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/variance_decomposition.py"
        f" --checkpoint={ckpt}"
        f" --corrca_filters={CORRCA}"
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

    # Job A: probe_eval × 3 (CorrCA makes forward passes cheap; ~5-10 min each)
    probe_cmds = [_probe_cmd(nw, ws, bs, ckpt) for nw, ws, bs, ckpt, _ in CHECKPOINTS]
    jobs.append(("probe_sigreg_corrca", Job(
        name="ev_probe_sgcc",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=_git_cmd() + " &&\n".join(probe_cmds),
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="01:30:00",
        mem_gb=64,
        gpus=1,
        env_vars={
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            "WANDB_PROJECT": "eb_jepa",
        },
    )))

    # Job B: variance_decomposition × 3 + aggregate (~5 min each with CorrCA)
    var_cmds = [_vardecomp_cmd(nw, ws, bs, ckpt) for nw, ws, bs, ckpt, _ in CHECKPOINTS]
    agg_cmd = (
        "PYTHONPATH=. uv run --group eeg"
        " python scripts/variance_decomposition.py"
        f" --aggregate_dir={VAR_OUT}"
    )
    jobs.append(("vardecomp_sigreg_corrca", Job(
        name="ev_vardec_sgcc",
        cluster="delta",
        repo_path="/u/dtyoung/eb_jepa_eeg",
        command=_git_cmd() + " &&\n".join(var_cmds + [agg_cmd]),
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="01:00:00",
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
    print(f"SIGReg + per-rec + CorrCA eval: {len(CHECKPOINTS)} checkpoints × 2 eval types\n")
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
