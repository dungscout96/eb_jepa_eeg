"""probe_traintest on iter12 long-training ep500 checkpoint (val split, B=2000)."""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
CKPT_DIR = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/jul1/iter12_long_ep500_v4"

job = Job(
    name="auto_jul1_iter12_ep500_probe_tt",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:30:00",
    command=(
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe_traintest.py"
        f" --checkpoint {CKPT_DIR}/latest.pth.tar"
        f" --config {CKPT_DIR}/config.yaml"
        " --eval-split val"
        " --bootstrap 2000 --seed 42"
        f" --output {AUTORESEARCH_DIR}/probe_tt_iter12_ep500_val.json"
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
