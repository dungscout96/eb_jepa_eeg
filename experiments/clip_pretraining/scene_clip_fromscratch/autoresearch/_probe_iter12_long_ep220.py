"""Probe the ep220 checkpoint from iter12 long-training crash (job 19865238).
Training crashed at ep220 during the epoch_220.pth.tar write, but latest.pth.tar
was already saved successfully at ep220 per the training log.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
CKPT_DIR = "/work/hdd/bbnv/dtyoung/eb_jepa/autoresearch/jul1/iter12_long_ep500"

job = Job(
    name="auto_jul1_iter12_long_probe_ep220",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:15:00",
    command=(
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe.py"
        f" --checkpoint {CKPT_DIR}/latest.pth.tar"
        f" --config {CKPT_DIR}/config.yaml"
        " --split val --cv-splits 5"
        f" --output {AUTORESEARCH_DIR}/probe_val_iter12_long_ep220.json"
    ),
    venv="__none__",
    branch="",
    env_vars={
        "WANDB_PROJECT": "eb_jepa",
        "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
    },
)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        print(f"Submitting {job.name}")
        print(f"job_id: {job.submit()}")
    else:
        print(job.submit(dry_run=True))
