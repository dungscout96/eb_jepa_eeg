"""Probe-only follow-up for the timed-out 400-ep runs.

Submits a short (15 min) probe job on Delta that runs val probes on both
ThePresent and DespicableMe against the checkpoint's snapshotted per-movie
configs. Use when the train+probe job in `_submit_ab.py` hit its wall clock
limit before the probe step ran.

Usage:
    uv run --group eeg python _probe_only.py <arm> <run_tag> [--seed=N] [submit]

    arm       : "clip" | "scene_clip" | "soft_target_clip"
    run_tag   : same run_tag as the original submit (e.g. "jul7-e400")
    --seed=N  : match the seed used by the training submit; controls the
                exp_dir path (<run_tag>_<arm>_seed<N>) and output filenames.
                Default 2025. Omit if the training submit didn't pass --seed.
    submit    : append to actually sbatch. Otherwise dry-run.

Uses latest.pth.tar from the training run's checkpoint dir. Reads the
per-movie config snapshots (config_TP.yaml / config_DM.yaml) that _submit_ab.py
wrote there at train-job start.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/clip_pretraining/soft_target_clip"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip"


def build_job(
    arm: str, run_tag: str, ckpt_name: str = "latest.pth.tar", seed: int | None = None
) -> Job:
    if arm not in {"clip", "scene_clip", "soft_target_clip"}:
        raise ValueError(f"unknown arm {arm!r}")
    slug = f"{run_tag}_{arm}" if seed is None else f"{run_tag}_{arm}_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"
    ckpt = f"{exp_dir}/{ckpt_name}"
    out_TP = f"{EXP_DIR}/probe_val_{slug}_TP.json"
    out_DM = f"{EXP_DIR}/probe_val_{slug}_DM.json"
    return Job(
        name=f"soft_clip_{slug}_probe",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:20:00",
        command=(
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {ckpt}"
            f" --config {exp_dir}/config_TP.yaml"
            " --split val --cv-splits 5"
            f" --output {out_TP}"
            " && "
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {ckpt}"
            f" --config {exp_dir}/config_DM.yaml"
            " --split val --cv-splits 5"
            f" --output {out_DM}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: _probe_only.py <arm> <run_tag> [--seed=N] [ckpt_name] [submit]")
        sys.exit(1)
    arm = sys.argv[1]
    run_tag = sys.argv[2]
    ckpt_name = "latest.pth.tar"
    seed: int | None = None
    submit = False
    for arg in sys.argv[3:]:
        if arg == "submit":
            submit = True
        elif arg.startswith("--seed="):
            seed = int(arg.split("=", 1)[1])
        else:
            ckpt_name = arg
    job = build_job(arm, run_tag, ckpt_name, seed=seed)
    if submit:
        print(f"Submitting {job.name} (ckpt={ckpt_name}, seed={seed})")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name} (arm={arm}, run_tag={run_tag}, ckpt={ckpt_name}, seed={seed})")
        print(job.submit(dry_run=True))
