"""iter 8: continuation from iter 3's ep99 checkpoint.

Program.md section 2 finding 2: "Resuming an ep99 checkpoint with a fresh
lr=5e-5 cosine over 100 more epochs hits delta = +0.0149." This tests
whether that continuation pattern generalizes to the REVE-shape encoder.

We resume from iter 3's latest.pth.tar (ep99, val_delta_r2=+0.01093)
and train 70 more epochs at lr=5e-5 fresh cosine. Effective total ~170
epochs of REVE-shape training. Everything else identical to iter 3
(depth=8, embed=512, heads=8, head_dim=64, proj=512, per_window).
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/autoresearch/jul1"
ITER = 8
RESUME_FROM = f"{REPO}/checkpoints/autoresearch/jul1/iter3/latest.pth.tar"
EXP_DIR = f"{CKPT_ROOT}/iter{ITER}"
OUTPUT_JSON = f"{AUTORESEARCH_DIR}/probe_val_iter{ITER}.json"
EPOCHS = 70

job = Job(
    name=f"auto_jul1_iter{ITER}",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:45:00",
    command=(
        f"mkdir -p {EXP_DIR} && "
        "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
        " --fname=config/clip_pretrain.yaml"
        f" --meta.resume_from={RESUME_FROM}"
        f" --optim.epochs={EPOCHS}"
        " --optim.lr=5e-5"
        f" --folder={EXP_DIR}"
        f" --logging.wandb_group=auto_jul1_iter{ITER}"
        " && "
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe.py"
        f" --checkpoint {EXP_DIR}/latest.pth.tar"
        " --config config/clip_pretrain.yaml"
        " --split val --cv-splits 5"
        f" --output {OUTPUT_JSON}"
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
        print(f"Submitting iter {ITER}: {job.name} (resume from iter3 ep99)")
        print(f"job_id: {job.submit()}")
    else:
        print(job.submit(dry_run=True))
