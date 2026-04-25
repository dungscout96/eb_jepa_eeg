"""Submit the TRF prototype baseline to the jamming workstation.

Runs experiments/trf_baseline/run_trf.py on a 100-subject subset of R1-R4
and evaluates on full R5/R6. CPU-only, ~3-4 GB RAM peak.

Usage
-----
# Always preview first:
uv run python scripts/submit_trf_prototype_jamming.py

# To submit, set DRY_RUN = False below and rerun.
"""

import os

from neurolab.jobs import Job

job = Job(
    name="trf_baseline_prototype",
    cluster="jamming",
    repo_path="/home/dung/Documents/eb_jepa_eeg",
    command=(
        "git fetch origin && git checkout main && git pull --ff-only &&"
        " PYTHONPATH=. uv run --group eeg python experiments/trf_baseline/run_trf.py"
        " --input=raw"
        " --max_train_recs=100"
        " --n_lags_ms=1000"
        " --fs_target=50"
        " --output_dir=outputs/trf_prototype_raw"
    ),
    venv="__none__",
    branch="",
    env_vars={
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "WANDB_PROJECT": "eb_jepa",
        "HBN_PREPROCESS_DIR": "/mnt/v1/dtyoung/data/eb_jepa_eeg/hbn_preprocessed",
    },
)

DRY_RUN = True

if __name__ == "__main__":
    script = job.submit(dry_run=True)
    print("=" * 72)
    print("DRY RUN — shell script that would be piped to jamming:")
    print("=" * 72)
    print(script)
    print("=" * 72)

    if not DRY_RUN:
        job_id = job.submit()
        print(f"Submitted job: {job_id}")
    else:
        print("\nSet DRY_RUN = False to actually submit.")
