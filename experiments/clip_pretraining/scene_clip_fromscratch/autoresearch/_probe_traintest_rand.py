"""probe_traintest random baseline at iter12's exact shape (val split, B=2000).

Uses the iter12 config snapshot so the encoder architecture matches ep500
exactly. Then swaps in `--random-baseline` so no checkpoint is loaded - the
encoder stays at its fresh initialization, giving the "no CLIP training"
null result for our specific architecture.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
CKPT_DIR = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/jul1/iter12_long_ep500_v4"

job = Job(
    name="auto_jul1_iter12_rand_probe_tt",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:20:00",
    command=(
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe_traintest.py"
        " --random-baseline"
        f" --config {CKPT_DIR}/config.yaml"
        " --eval-split val"
        " --bootstrap 2000 --seed 42"
        f" --output {AUTORESEARCH_DIR}/probe_tt_iter12_rand_val.json"
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
