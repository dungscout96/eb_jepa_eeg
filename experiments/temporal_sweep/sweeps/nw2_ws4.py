"""Single-config pretraining sweep: nw2_ws4 on ThePresent (jamming).

Trains MaskedJEPA at n_windows=2, window_size_seconds=4 -- a temporal
config that didn't make the Phase 1 spotlight but is worth a closer
look in isolation. Same FIXED_ARGS as the Phase 1 sweep (best config
from exp26: encoder_depth=2, lr=5e-4, VCLoss(0.25, 0.25), Huber, 100ep)
so results are directly comparable to the phase1 grid.

Targets the jamming workstation: direct execution (no SLURM), single
GPU. With multiple seeds in SEEDS, runs them sequentially in one job.

Usage
-----
# Preview the shell script that would run on jamming:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python experiments/eeg_jepa/sweeps/nw2_ws4.py

# Submit:
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python experiments/eeg_jepa/sweeps/nw2_ws4.py --submit
"""

import os
import sys

from neurolab.jobs import Job

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

SEEDS = [7]
N_WINDOWS = 2
WINDOW_SIZE = 4
BATCH_SIZE = 64
TASK = "ThePresent"

# Fixed hyperparameters (best config from exp26, same as phase1 sweep)
FIXED_ARGS = (
    " --model.encoder_depth=2"
    " --optim.lr=5e-4"
    " --optim.lr_min=1e-6"
    " --optim.warmup_epochs=5"
    " --loss.vicreg.std_coeff=0.25"
    " --loss.vicreg.cov_coeff=0.25"
    " --loss.pred_loss_type=smooth_l1"
    " --optim.epochs=100"
    " --logging.wandb_group=nw2_ws4"
)


def _make_run_cmd(seed):
    return (
        "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.jepa_pretrain"
        f" --data.task={TASK}"
        f" --data.n_windows={N_WINDOWS}"
        f" --data.window_size_seconds={WINDOW_SIZE}"
        f" --data.batch_size={BATCH_SIZE}"
        f" --data.num_workers=4"
        f" --meta.seed={seed}"
        f"{FIXED_ARGS}"
    )


def build_job():
    # Run all seeds sequentially in one jamming job (single GPU workstation).
    sequential_cmd = " &&\n".join(_make_run_cmd(seed) for seed in SEEDS)

    return Job(
        name=f"nw2_ws4_jamming",
        cluster="jamming",
        repo_path="/home/dung/Documents/eb_jepa_eeg",
        command=(
            "git fetch origin && git checkout refactor/eeg-only-library &&"
            " git pull --ff-only origin refactor/eeg-only-library &&\n"
            + sequential_cmd
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/mnt/v1/dtyoung/data/eb_jepa_eeg/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    job = build_job()

    desc = ", ".join(f"nw{N_WINDOWS}_ws{WINDOW_SIZE}s_s{seed}" for seed in SEEDS)
    print(f"nw2_ws4 sweep on jamming: {len(SEEDS)} experiment(s) sequential\n"
          f"  Runs: {desc}\n")

    script = job.submit(dry_run=True)
    print("=" * 72)
    print("DRY RUN -- shell script that would be piped to jamming:")
    print("=" * 72)
    print(script)
    print("=" * 72)

    if submit:
        job_id = job.submit()
        print(f"\nSubmitted: {job_id}")
    else:
        print("\nRun with --submit to send to jamming.")
