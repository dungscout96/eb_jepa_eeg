"""iter 12 scale-up: long-training run of the search-winning config.

Config summary (val_delta_r2 = +0.03902 at 300 ep in the 45-min autoresearch loop):
  patch_size=400 patch_overlap=0 freqs=4
  encoder depth=12 embed=512 heads=8 head_dim=64 mlp_dim_ratio=2.66
  projection proj_dim=512 n_residual_blocks=1 drop_proj=0.5 temperature=0.07
  optim adam lr=1e-4 lr_min=1e-6 warmup_epochs=5
  data batch_size=64 window_size_seconds=2 per_window mean_center

Under the users framing, the search loop found this config; this script
takes it to fresh500-length training (500 epochs, matching the prior
best-published from-scratch run).

Fresh500 (depth=4 config) trajectory: ep99 +0.010, ep299 +0.021,
ep399 +0.027, ep499 +0.027 (saturated). Iter 12 trajectory is expected
to be similar shape but shifted upward - starting +0.039 at ep300
suggests ep500 could land in ~+0.045 to +0.055 range if it follows
the same shape.

Usage:
    uv run --group eeg python _submit_long.py [--epochs 500] [submit]

Notes:
- 60-min wall (extends past the 45-min autoresearch cap since scale-up
  runs longer)
- Snapshot config to /work/hdd/... so mid-flight edits do not affect probe
- Automatically probes val at the end
"""

import argparse
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"
# Long-training runs are checkpoint-heavy (469 MB per save); /work/hdd/bbnv
# has repeatedly hit torch.save "zipfile pos mismatch" on this branch. Route
# scale-up runs to /u/dtyoung (home; stabilized) with save_every=50 to reduce
# write frequency.
CKPT_ROOT = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/jul1"


def build_job(epochs: int, tag: str) -> Job:
    exp_dir = f"{CKPT_ROOT}/iter12_long_{tag}"
    output = f"{AUTORESEARCH_DIR}/probe_val_iter12_long_{tag}.json"
    return Job(
        name=f"auto_jul1_iter12_long_{tag}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="01:00:00",  # 60 min - iter12 was 25 min at 300 ep; 500 ep ~ 42 min + probe
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp experiments/clip_pretraining/scene_clip_fromscratch/autoresearch/clip_pretrain.yaml {exp_dir}/config.yaml && "
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={epochs}"
            f" --folder={exp_dir}"
            " --logging.save_every=50"
            f" --logging.wandb_group=auto_jul1_iter12_long_{tag}"
            " && "
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            f" --config {exp_dir}/config.yaml"
            " --split val --cv-splits 5"
            f" --output {output}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500, help="Training epochs")
    ap.add_argument("--tag", type=str, default="ep500",
                    help="Short tag for the run (e.g. ep500, ep1000)")
    ap.add_argument("--submit", action="store_true", help="Actually submit (default is dry-run)")
    args = ap.parse_args()

    job = build_job(args.epochs, args.tag)
    if args.submit:
        print(f"Submitting {job.name} ({args.epochs} epochs)")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run for {args.epochs} epochs, tag={args.tag}. Add --submit to actually run.")
        print(job.submit(dry_run=True))
