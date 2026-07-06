"""iter 8 probe follow-up.

Iter 8 (continuation from iter3) trained cleanly but its probe step
crashed because I sync'd the Delta repo mid-flight to iter 9's config
(patch_size=100), so the probe couldn't build a matching encoder for
iter 8's patch_size=50 checkpoint. Rerun probe with iter 8's original
config (extracted from git commit d97096b onto Delta at /tmp/config_iter8.yaml).
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
CKPT = "/work/hdd/bbnv/dtyoung/eb_jepa/autoresearch/jul1/iter8/latest.pth.tar"
OUTPUT = f"{AUTORESEARCH_DIR}/probe_val_iter8.json"

job = Job(
    name="auto_jul1_iter8_probe",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:15:00",
    command=(
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe.py"
        f" --checkpoint {CKPT}"
        " --config /work/hdd/bbnv/dtyoung/eb_jepa/autoresearch/jul1/iter8/config.yaml"
        " --split val --cv-splits 5"
        f" --output {OUTPUT}"
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
