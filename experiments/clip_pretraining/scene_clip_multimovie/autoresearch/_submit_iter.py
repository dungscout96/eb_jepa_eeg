"""Submit an autoresearch iteration for the multi-movie loop.

Usage:
    uv run --group eeg python _submit_iter.py <iter_num> [--wider] [submit]

Dry-run by default; append `submit` to actually sbatch.

Options:
    --wider   Override encoder to embed=768 heads=12 freqs=5 (jul1 iter13
              shape) via Fire CLI. Compute-matched: EPOCHS_WIDER (85) fits
              the same ~30-min training budget as EPOCHS_BASE (160) at
              baseline width.

Sbatch flow per iteration:
  1. mkdir CKPT_DIR
  2. cp config/clip_pretrain.yaml CKPT_DIR/config.yaml
  3. If --wider, dump modified config with wider fields to CKPT_DIR/config.yaml
  4. python -c 'OmegaConf: task=ThePresent' > CKPT_DIR/config_TP.yaml
  5. python -c 'OmegaConf: task=DespicableMe' > CKPT_DIR/config_DM.yaml
  6. Train (reads config.yaml)
  7. Probe val on ThePresent + DespicableMe (single-task configs)
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_multimovie/autoresearch"
CKPT_ROOT = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/multimovie_jul2"

# 30-min-training compute budget. Multi-movie ~11 s/ep at baseline width.
EPOCHS_BASE = 160     # embed=512 baseline: 160 ep at ~11 s = 29 min train
EPOCHS_WIDER = 85     # embed=768: ~20.6 s/ep (1.875x FFN+attn cost) so 85 ep = 29 min train


def build_job(iter_num: int, wider: bool = False) -> Job:
    epochs = EPOCHS_WIDER if wider else EPOCHS_BASE
    exp_dir = f"{CKPT_ROOT}/iter{iter_num}"
    output_TP = f"{AUTORESEARCH_DIR}/probe_val_iter{iter_num}_TP.json"
    output_DM = f"{AUTORESEARCH_DIR}/probe_val_iter{iter_num}_DM.json"
    # If wider, patch the snapshotted config in-place with the wider fields
    # (embed 768, heads 12, freqs 5). This runs INSIDE the sbatch script so
    # it's isolated from the on-disk config the next job might read.
    wider_patch = (
        "PYTHONPATH=. uv run --group eeg python -c \""
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        "c.model.encoder_embed_dim = 768; "
        "c.model.encoder_heads = 12; "
        "c.model.freqs = 5; "
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')\" && "
    ) if wider else ""
    return Job(
        name=f"auto_mm_iter{iter_num}" + ("_wider" if wider else ""),
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="01:00:00",
        command=(
            f"mkdir -p {exp_dir} && "
            # Snapshot base config
            f"cp config/clip_pretrain.yaml {exp_dir}/config.yaml && "
            # If wider, patch the snapshot with widening fields
            f"{wider_patch}"
            # Snapshot single-task variants for per-movie probing (reads the
            # possibly-patched config.yaml so probes match the trained model)
            "PYTHONPATH=. uv run --group eeg python -c \""
            "from omegaconf import OmegaConf; "
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            "c.data.task = 'ThePresent'; "
            f"OmegaConf.save(c, '{exp_dir}/config_TP.yaml'); "
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            "c.data.task = 'DespicableMe'; "
            f"OmegaConf.save(c, '{exp_dir}/config_DM.yaml')\" && "
            # Train (multi-movie, whatever config.yaml now specifies)
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={epochs}"
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
        print("Usage: _submit_iter.py <iter_num> [--wider] [submit]")
        sys.exit(1)
    iter_num = int(sys.argv[1])
    args = sys.argv[2:]
    wider = "--wider" in args
    submit = "submit" in args
    job = build_job(iter_num, wider=wider)
    if submit:
        print(f"Submitting iter {iter_num}: {job.name}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run for iter {iter_num} (wider={wider}). Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
