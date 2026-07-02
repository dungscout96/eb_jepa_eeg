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

# iter9+: patch_size 50 to 100 halves tokens per window from 12 to 6.
# Attention cost is O(N^2), so ~4x cheaper on attention; FFN cost O(N) so ~2x.
# Expect ~15-18 s/ep vs iter3's 27 s/ep. Bump to 130 ep to actually fill the
# 45-min wall: 130 * 18 = 39 min train + 9 min probe = 48 min tight. Use 120
# for safety: 120 * 18 = 36 min + 9 = 45 min. If per-epoch beats 18 s, bump up.
EPOCHS = 120


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
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            " --fname=config/clip_pretrain.yaml"
            f" --optim.epochs={EPOCHS}"
            f" --folder={exp_dir}"
            f" --logging.wandb_group=auto_jul1_iter{iter_num}"
            " && "
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            " --config config/clip_pretrain.yaml"
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
