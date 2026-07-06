"""Follow-up probe for iter 0 (training reached ep128 but SLURM SIGKILL'd
mid-write of latest.pth.tar). Probe the clean epoch_120 checkpoint instead."""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
CKPT = f"{REPO}/checkpoints/autoresearch/jul1/iter0_baseline/epoch_120.pth.tar"

job = Job(
    name="auto_jul1_iter0_probe_ep120",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:15:00",
    command=(
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe.py"
        f" --checkpoint {CKPT}"
        " --config config/clip_pretrain.yaml"
        " --split val --cv-splits 5"
        f" --output {AUTORESEARCH_DIR}/probe_val_iter0.json"
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
        print(f"Submitting {job.name}...")
        print(f"job_id: {job.submit()}")
    else:
        print(job.submit(dry_run=True))
