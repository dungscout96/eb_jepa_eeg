"""Random-init baseline probes for the multi-movie loop (per movie).

Runs probe.py --random-baseline with the iter12-shape encoder at
each movie's val split separately. Produces rand_r2_TP and rand_r2_DM
that we subtract from every trained iteration's mean_r2 to compute
val_delta_r2 per movie.

Two jobs run in parallel (both ~5 min actual work; 15 min wall each).

Usage:
    uv run --group eeg python _submit_rand.py [submit]
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_multimovie/autoresearch"
RAND_DIR = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/multimovie_jul2/rand_baseline"


def build_prep_and_prob(task_tag: str) -> Job:
    """One job = prep single-task config from the current on-disk config, then probe."""
    task_full = "ThePresent" if task_tag == "TP" else "DespicableMe"
    output = f"{AUTORESEARCH_DIR}/probe_val_rand_{task_tag}.json"
    return Job(
        name=f"auto_mm_rand_{task_tag}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:15:00",
        command=(
            f"mkdir -p {RAND_DIR} && "
            # Snapshot config + override task
            f"cp experiments/clip_pretraining/scene_clip_multimovie/autoresearch/clip_pretrain.yaml {RAND_DIR}/config_{task_tag}.yaml && "
            "PYTHONPATH=. uv run --group eeg python -c \""
            "from omegaconf import OmegaConf; "
            f"c = OmegaConf.load('{RAND_DIR}/config_{task_tag}.yaml'); "
            f"c.data.task = '{task_full}'; "
            f"OmegaConf.save(c, '{RAND_DIR}/config_{task_tag}.yaml')\" && "
            # Probe with random-init encoder
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            " --random-baseline"
            f" --config {RAND_DIR}/config_{task_tag}.yaml"
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
    submit = len(sys.argv) > 1 and sys.argv[1] == "submit"
    for task_tag in ("TP", "DM"):
        job = build_prep_and_prob(task_tag)
        print("=" * 60)
        print(f"{'SUBMITTING' if submit else 'DRY RUN'} — {job.name}")
        print("=" * 60)
        if submit:
            print(f"job_id: {job.submit()}")
        else:
            print(job.submit(dry_run=True))
