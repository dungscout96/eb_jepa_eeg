"""Submit an autoresearch iteration to Delta.

Usage:
    uv run --group eeg python _submit_iter.py <iter_num> [submit]

Dry-run by default; append `submit` to actually sbatch.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_fromscratch/autoresearch"

# iter2 changes encoder_depth 4 to 8. Transformer cost is ~linear in depth;
# other overhead is fixed. Empirically iter1 ran ~22 s/ep at depth=4; expect
# ~37 s/ep at depth=8 (1.7x factor). 60 ep ≈ 37 min train, matching iter1's
# ~37 min of training compute (wall-clock-matched per program §Experimentation).
EPOCHS = 60


def build_job(iter_num: int) -> Job:
    exp_dir = f"{REPO}/checkpoints/autoresearch/jul1/iter{iter_num}"
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
