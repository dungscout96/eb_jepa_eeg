"""Submit a sanity-check training run to the jamming workstation.

The jamming workstation is a direct-execution (non-SLURM) GPU box.
Jobs are launched via SSH + nohup; no scheduler is involved.

Usage
-----
# 1. Preview the generated shell script (always do this first):
uv run python scripts/submit_job_jamming.py

# 2. To actually submit, set DRY_RUN = False at the bottom and rerun.

Environment
-----------
Set WANDB_API_KEY in your shell (or add it to env_vars below) before submitting.
The run will log to the "eb_jepa" W&B project under the group "sanity_checks".
"""

import os

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# Job definition
# ---------------------------------------------------------------------------

job = Job(
    name="eeg_jepa_sanity_checks",
    cluster="jamming",
    # Absolute path to the repository on the jamming workstation.
    # Update this if your remote checkout lives elsewhere.
    repo_path="/home/dtyoung/eb_jepa_eeg",
    command=(
        "uv run python experiments/eeg_jepa/main.py"
        " sanity_checks.enabled=true"
        " logging.log_wandb=true"
        " logging.wandb_group=sanity_checks"
        " data.batch_size=32"          # smaller batch for quick iteration
        " optim.epochs=20"             # short run to verify sanity metrics
    ),
    # Full path to the virtual environment (conda or venv) on the remote.
    # Adjust to match the actual environment name on jamming.
    venv="/home/dtyoung/miniconda3/envs/eb_jepa",
    branch="feature/sanity-checks",
    env_vars={
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "WANDB_PROJECT": "eb_jepa",
        # Use the preprocessed data cache if available on jamming.
        # Remove or update this if the cache path differs on the workstation.
        "EB_JEPA_PREPROCESSED_DIR": "/mnt/v1/dtyoung/data/eb_jepa_eeg/hbn_preprocessed",
    },
)

# ---------------------------------------------------------------------------
# Dry run — inspect the generated script before committing to submit
# ---------------------------------------------------------------------------

DRY_RUN = True

if __name__ == "__main__":
    script = job.submit(dry_run=True)
    print("=" * 72)
    print("DRY RUN — shell script that would be piped to jamming:")
    print("=" * 72)
    print(script)
    print("=" * 72)

    if not DRY_RUN:
        # Uncomment to actually submit:
        # job_id = job.submit()
        # print(f"Submitted job: {job_id}")
        pass
    else:
        print(
            "\nSet DRY_RUN = False (and uncomment job.submit()) to actually submit."
        )
