"""Submit a training run to the Delta cluster (SLURM, A40 GPUs).

Usage
-----
# 1. Preview the generated SLURM script:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab python scripts/submit_job_delta.py

# 2. To actually submit, set DRY_RUN = False and rerun.
"""

import os

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------

job = Job(
    name="eeg_jepa_timing_test",
    cluster="delta",
    repo_path="/u/dtyoung/eb_jepa_eeg",
    partition="gpuA40x4",
    time_limit="02:00:00",
    command=(
        # Fetch + checkout explicitly so new sweep branches are found on jamming
        "git fetch origin && git checkout main && git pull --ff-only &&"
        # === timing test: baseline config with num_workers=2 ===
        " PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py"
        " --optim.epochs=100 --model.encoder_depth=2 --optim.lr=5e-4"
        " --optim.lr_min=1e-6 --optim.warmup_epochs=5"
        " --data.num_workers=2"
        " --loss.std_coeff=0.25 --loss.cov_coeff=0.25"
        " --loss.pred_loss_type=smooth_l1"
        " --logging.wandb_group=delta_test"
        # timing test: baseline (nw=4, ws=2s) with num_workers=2 on A40
    ),
    venv="__none__",
    branch="",
    env_vars={
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "WANDB_PROJECT": "eb_jepa",
        "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
    },
)

# ---------------------------------------------------------------------------
# Dry run — inspect the generated script before committing to submit
# ---------------------------------------------------------------------------

DRY_RUN = False

if __name__ == "__main__":
    script = job.submit(dry_run=True)
    print("=" * 72)
    print("DRY RUN — shell script that would be piped to delta:")
    print("=" * 72)
    print(script)
    print("=" * 72)

    if not DRY_RUN:
        # Uncomment to actually submit:
        job_id = job.submit()
        print(f"Submitted job: {job_id}")
        pass
    else:
        print(
            "\nSet DRY_RUN = False (and uncomment job.submit()) to actually submit."
        )
