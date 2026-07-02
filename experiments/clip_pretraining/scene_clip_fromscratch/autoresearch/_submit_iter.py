"""Submit an autoresearch iteration to Delta.

Usage:
    uv run --group eeg python _submit_iter.py <iter_num> [submit]

Dry-run by default; append `submit` to actually sbatch.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
# Write checkpoints to /work/hdd (HDD-backed Lustre, team quota 2 TB soft, more
# reliable than /u/dtyoung which has hit "Cannot send after transport endpoint
# shutdown" RPC errors during iter 0/3/4 write windows).
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/autoresearch/jul1"

# iter12 actual: depth=12 at patch=400 ran ~3.6 s/ep (faster than 7s estimate).
# Iter13 widens embed 512 to 768 (FFN cost scales as D^2, ~2.25x). Expect
# ~8 s/ep. 250 ep at 8 s = 33 min train + ~10 min probe (probe also scales
# with width) = 43 min. Fits with 2-min margin.
EPOCHS = 250


def build_job(iter_num: int) -> Job:
    exp_dir = f"{CKPT_ROOT}/iter{iter_num}"
    output_json = f"{AUTORESEARCH_DIR}/probe_val_iter{iter_num}.json"
    return Job(
        name=f"auto_jul1_iter{iter_num}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:45:00",
        command=(
            f"mkdir -p {exp_dir} && "
            # Snapshot the config at job start so a mid-flight sync of
            # config/clip_pretrain.yaml can't break the probe step (iter8
            # got hit by exactly this problem).
            f"cp config/clip_pretrain.yaml {exp_dir}/config.yaml && "
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={EPOCHS}"
            f" --folder={exp_dir}"
            f" --logging.wandb_group=auto_jul1_iter{iter_num}"
            " && "
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            f" --config {exp_dir}/config.yaml"
            " --split val --cv-splits 5"
            f" --output {output_json}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _submit_iter.py <iter_num> [submit]")
        sys.exit(1)
    iter_num = int(sys.argv[1])
    submit = len(sys.argv) > 2 and sys.argv[2] == "submit"
    job = build_job(iter_num)
    if submit:
        print(f"Submitting iter {iter_num}: {job.name}")
        print(f"job_id: {job.submit()}")
    else:
        print(job.submit(dry_run=True))
