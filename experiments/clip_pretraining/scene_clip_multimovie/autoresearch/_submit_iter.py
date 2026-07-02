"""Submit an autoresearch iteration for the multi-movie loop.

Usage:
    uv run --group eeg python _submit_iter.py <iter_num> [submit]

Dry-run by default; append `submit` to actually sbatch.

Sbatch flow per iteration:
  1. mkdir CKPT_DIR
  2. cp config/clip_pretrain.yaml CKPT_DIR/config.yaml  (multi-movie)
  3. python -c 'OmegaConf: task=ThePresent' > CKPT_DIR/config_TP.yaml
  4. python -c 'OmegaConf: task=DespicableMe' > CKPT_DIR/config_DM.yaml
  5. Train (reads config.yaml, multi-movie)
  6. Probe val on ThePresent (reads config_TP.yaml)
  7. Probe val on DespicableMe (reads config_DM.yaml)

Both probe JSONs go under AUTORESEARCH_DIR for post-processing.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_multimovie/autoresearch"
# /u/dtyoung home Lustre: has room; /work/hdd/bbnv is currently over team soft quota
CKPT_ROOT = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/multimovie_jul2"

# iter1: 400 ep to test whether multi-movie has more training headroom than
# single-movie (jul1 iter12 saturated at ep500 on ~700 recordings; multi-movie
# has ~2x data so may keep gaining). iter0 measured 11.3 s/ep so 400 ep = 75
# min train + 10 min probes = 85 min. 105-min wall (01:45:00) has margin.
EPOCHS = 400


def build_job(iter_num: int) -> Job:
    exp_dir = f"{CKPT_ROOT}/iter{iter_num}"
    output_TP = f"{AUTORESEARCH_DIR}/probe_val_iter{iter_num}_TP.json"
    output_DM = f"{AUTORESEARCH_DIR}/probe_val_iter{iter_num}_DM.json"
    return Job(
        name=f"auto_mm_iter{iter_num}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="01:45:00",
        command=(
            f"mkdir -p {exp_dir} && "
            # Snapshot multi-movie config (used for training)
            f"cp config/clip_pretrain.yaml {exp_dir}/config.yaml && "
            # Also snapshot single-task variants for per-movie probing
            "PYTHONPATH=. uv run --group eeg python -c \""
            "from omegaconf import OmegaConf; "
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            "c.data.task = 'ThePresent'; "
            f"OmegaConf.save(c, '{exp_dir}/config_TP.yaml'); "
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            "c.data.task = 'DespicableMe'; "
            f"OmegaConf.save(c, '{exp_dir}/config_DM.yaml')\" && "
            # Train (multi-movie)
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={EPOCHS}"
            f" --folder={exp_dir}"
            f" --logging.wandb_group=auto_mm_iter{iter_num}"
            " && "
            # Probe val on ThePresent
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            f" --config {exp_dir}/config_TP.yaml"
            " --split val --cv-splits 5"
            f" --output {output_TP}"
            " && "
            # Probe val on DespicableMe
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            f" --config {exp_dir}/config_DM.yaml"
            " --split val --cv-splits 5"
            f" --output {output_DM}"
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
