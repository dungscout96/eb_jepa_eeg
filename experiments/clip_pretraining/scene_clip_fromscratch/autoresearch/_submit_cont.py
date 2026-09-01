"""Continuation runs from iter12_long_ep500_v4 checkpoint.

Two schedule options tested in parallel:
  - lr5e5: safe continuation - fresh cosine at half original peak LR (5e-5)
           matches iter 8 continuation recipe, program section 2 finding 2
  - lr1e4: SGDR-style warm restart - fresh cosine at same peak LR (1e-4)
           more aggressive re-exploration

Both:
  - Resume from iter12_long_ep500_v4 latest.pth.tar (val delta_r2 = +0.047)
  - Same architecture (patch=400 depth=12 embed=512 heads=8 head_dim=64 proj=512)
  - Train 500 more epochs with fresh cosine to lr_min=1e-6, warmup=5
  - Write to /u/dtyoung with save_every=50
  - Legacy torch serialization (via training_utils.py patch)
  - Config snapshotting

Usage:
    uv run --group eeg python _submit_cont.py lr5e5 [submit]
    uv run --group eeg python _submit_cont.py lr1e4 [submit]
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
BASE_CKPT = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/jul1/iter12_long_ep500_v4/latest.pth.tar"
CKPT_ROOT = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/jul1"
EPOCHS = 500

# Option → (peak_lr, tag_suffix)
OPTIONS = {
    "lr5e5": ("5e-5", "cont_lr5e5"),
    "lr1e4": ("1e-4", "cont_lr1e4"),
}


def build_job(option: str) -> Job:
    if option not in OPTIONS:
        raise ValueError(f"Unknown option: {option}. Choose one of {list(OPTIONS)}")
    peak_lr, tag = OPTIONS[option]
    exp_dir = f"{CKPT_ROOT}/iter12_{tag}"
    output_json = f"{AUTORESEARCH_DIR}/probe_val_iter12_{tag}.json"
    return Job(
        name=f"auto_jul1_iter12_{tag}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="01:00:00",
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp experiments/clip_pretraining/scene_clip_fromscratch/autoresearch/clip_pretrain.yaml {exp_dir}/config.yaml && "
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --meta.resume_from={BASE_CKPT}"
            f" --optim.epochs={EPOCHS}"
            f" --optim.lr={peak_lr}"
            f" --folder={exp_dir}"
            " --logging.save_every=50"
            f" --logging.wandb_group=auto_jul1_iter12_{tag}"
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
            "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _submit_cont.py {lr5e5,lr1e4} [submit]")
        sys.exit(1)
    option = sys.argv[1]
    submit = len(sys.argv) > 2 and sys.argv[2] == "submit"
    job = build_job(option)
    if submit:
        print(f"Submitting {job.name} (peak_lr={OPTIONS[option][0]})")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run for option={option}. Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
