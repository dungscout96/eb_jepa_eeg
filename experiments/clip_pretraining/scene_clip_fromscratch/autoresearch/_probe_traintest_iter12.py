"""probe_traintest on iter12's ep300 checkpoint (val split, B=2000).

Same SSL-literature protocol fresh500_ep499 was measured with:
  - Fit RidgeCV on R1-R4 train encoder embeddings
  - Evaluate on R5 val embeddings, Pearson r per feature
  - Bootstrap 2000 iterations by val-recording for CI

Iter 12 config summary (search-loop winner, val CV Delta_r2 = +0.03902):
  patch_size=400 patch_overlap=0 freqs=4
  encoder depth=12 embed=512 heads=8 head_dim=64 mlp_dim_ratio=2.66
  projection proj_dim=512 n_residual_blocks=1 temperature=0.07
  optim adam lr=1e-4 warmup=5 epochs=300 bs=64
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
CKPT_DIR = "/work/hdd/bbnv/dtyoung/eb_jepa/autoresearch/jul1/iter12"

job = Job(
    name="auto_jul1_iter12_probe_tt_val",
    cluster="delta",
    repo_path=REPO,
    partition="gpuA40x4",
    time_limit="00:30:00",
    command=(
        "PYTHONPATH=. uv run --group eeg python"
        " eb_jepa/evaluation/clip_probe/probe_traintest.py"
        f" --checkpoint {CKPT_DIR}/latest.pth.tar"
        f" --config {CKPT_DIR}/config.yaml"
        " --eval-split val"
        " --bootstrap 2000 --seed 42"
        f" --output {AUTORESEARCH_DIR}/probe_tt_iter12_val.json"
    ),
    venv="__none__",
    branch="",
    env_vars={
        "WANDB_PROJECT": "eb_jepa",
        "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/kkokate/hbn_preprocessed",
    },
)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        print(f"Submitting {job.name}")
        print(f"job_id: {job.submit()}")
    else:
        print(job.submit(dry_run=True))
