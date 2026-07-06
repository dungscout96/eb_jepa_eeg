"""Preview / submit the iter-0 baseline + random-probe jobs for autoresearch jul1.

Two jobs run in parallel on Delta A40:

1. Baseline training + val probe. Trains scene_clip from scratch with the current
   config (unmodified), for 130 epochs (~30 min at 14 s/epoch), then runs the
   5-fold CV probe on R5.

2. Random-baseline probe. Same probe, `--random-baseline` on the matched
   encoder architecture. Its `mean_r2` is cached as `rand_r2`, subtracted from
   every iteration's `val_mean_r2` to give `val_delta_r2`.

Set DRY_RUN=False to submit.
"""

import os
import sys

from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = (
    "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
)
EXP_DIR = f"{REPO}/checkpoints/autoresearch/jul1/iter0_baseline"

ENV = {
    "WANDB_PROJECT": "eb_jepa",
    "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
}

# git pre-sync is done on the login node (neurolab skill guidance: avoid the
# concurrent-git race with an inline `git fetch && checkout && pull`).
train_and_probe = Job(
    name="auto_jul1_iter0_baseline",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:45:00",
    command=(
        f"mkdir -p {EXP_DIR} && "
        # 1. Train (~30 min, 130 epochs).
        "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
        " --fname=config/clip_pretrain.yaml"
        " --optim.epochs=130"
        f" --folder={EXP_DIR}"
        " --logging.wandb_group=auto_jul1_iter0_baseline"
        " && "
        # 2. Probe val (5-fold CV, no bootstrap).
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe.py"
        f" --checkpoint {EXP_DIR}/latest.pth.tar"
        " --config config/clip_pretrain.yaml"
        " --split val --cv-splits 5"
        f" --output {AUTORESEARCH_DIR}/probe_val_iter0.json"
    ),
    venv="__none__",
    branch="",
    env_vars=ENV,
)

rand_probe = Job(
    name="auto_jul1_rand_baseline",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:15:00",
    command=(
        # Only the probe — random-init encoder, matched architecture.
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe.py"
        " --random-baseline"
        " --config config/clip_pretrain.yaml"
        " --split val --cv-splits 5"
        f" --output {AUTORESEARCH_DIR}/probe_val_rand.json"
    ),
    venv="__none__",
    branch="",
    env_vars=ENV,
)


DRY_RUN = True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        DRY_RUN = False

    for j in (train_and_probe, rand_probe):
        print("=" * 72)
        print(f"{'DRY RUN' if DRY_RUN else 'SUBMITTING'} — {j.name}")
        print("=" * 72)
        print(j.submit(dry_run=True))
        print()

    if not DRY_RUN:
        train_id = train_and_probe.submit()
        rand_id = rand_probe.submit()
        print(f"Submitted train+probe: {train_id}")
        print(f"Submitted rand probe:  {rand_id}")
