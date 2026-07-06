"""Probe-only follow-up when the wall kills a train+probe job before probe runs.

Usage:
    uv run --group eeg python _probe_only.py <iter_num> [ckpt_name] [submit]

Defaults to `latest.pth.tar` in checkpoints/autoresearch/jul1/iter<N>/.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/autoresearch/jul1"


def build_job(iter_num: int, ckpt_name: str = "latest.pth.tar") -> Job:
    ckpt = f"{CKPT_ROOT}/iter{iter_num}/{ckpt_name}"
    output = f"{AUTORESEARCH_DIR}/probe_val_iter{iter_num}.json"
    return Job(
        name=f"auto_jul1_iter{iter_num}_probe",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:15:00",
        command=(
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {ckpt}"
            " --config config/clip_pretrain.yaml"
            " --split val --cv-splits 5"
            f" --output {output}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _probe_only.py <iter_num> [ckpt_name] [submit]")
        sys.exit(1)
    iter_num = int(sys.argv[1])
    ckpt_name = "latest.pth.tar"
    submit = False
    for arg in sys.argv[2:]:
        if arg == "submit":
            submit = True
        else:
            ckpt_name = arg
    job = build_job(iter_num, ckpt_name)
    if submit:
        print(f"Submitting {job.name} (ckpt={ckpt_name})...")
        print(f"job_id: {job.submit()}")
    else:
        print(job.submit(dry_run=True))
