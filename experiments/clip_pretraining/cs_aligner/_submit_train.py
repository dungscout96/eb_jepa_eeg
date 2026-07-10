"""Submit a cs_aligner training run to Delta (TP-only by default).

Mirrors ``soft_target_clip/_submit_ab.py`` but targets the CS-Aligner loss
mode. Snapshots the shared config yaml into the experiment directory,
patches ``loss.mode=cs_aligner`` + the CS-specific knobs, trains, then runs
the val probe on the trained-on movie (skipped for short smoke runs).

Usage:
    uv run --group eeg python _submit_train.py <run_tag> \\
        [--cs-weight=1.0] [--kernel-bw=null] [--epochs=60] \\
        [--seed=2026] [--task=ThePresent] [--skip-probe] [submit]

run_tag flows into the exp_dir slug and the probe JSON filename so sweep
replicates don't collide.

Sweep example (Phase 3, cs_weight calibration on TP-only 60 ep):
    for w in 0.0 0.1 1.0 10.0; do
        uv run --group eeg python _submit_train.py sweep-jul9 \\
            --cs-weight=$w --epochs=60 --seed=2026 --task=ThePresent submit
    done
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/clip_pretraining/cs_aligner"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner"


def build_job(
    run_tag: str,
    cs_weight: float,
    kernel_bw: str,
    epochs: int,
    seed: int,
    task: str | None,
    skip_probe: bool,
) -> Job:
    single_movie = task is not None
    probe_movies = [task] if single_movie else ["ThePresent", "DespicableMe"]
    movie_to_suffix = {"ThePresent": "TP", "DespicableMe": "DM"}
    # Encode cs_weight into slug so sweep arms don't collide.
    w_slug = f"w{cs_weight:g}".replace(".", "p")
    bw_slug = "bwauto" if kernel_bw.lower() in {"null", "none", "auto"} else f"bw{kernel_bw}"
    slug = f"{run_tag}_{w_slug}_{bw_slug}_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"

    # Patch config in-place inside sbatch — never touches the shared template.
    patch_lines = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        "c.loss.mode = 'cs_aligner'; "
        f"c.loss.cs_weight = {cs_weight}; "
    )
    if kernel_bw.lower() in {"null", "none", "auto"}:
        patch_lines += "c.loss.kernel_bandwidth = None; "
    else:
        patch_lines += f"c.loss.kernel_bandwidth = {float(kernel_bw)}; "
    if single_movie:
        patch_lines += f"c.data.task = '{task}'; "
    patch_lines += f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    patch_cmd = f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "

    # Rough sizing: single-movie ~8 s/ep at embed=512. Probe ~8 min per movie.
    sec_per_ep = 8 if single_movie else 15
    probe_min = 0 if skip_probe else 8 * len(probe_movies)
    est_min = int(((epochs * sec_per_ep) / 60 + probe_min) * 1.25)
    hours, mins = divmod(max(30, est_min), 60)
    time_limit = f"{hours:02d}:{mins:02d}:00"

    snapshot_lines = "from omegaconf import OmegaConf; "
    for m in probe_movies:
        snapshot_lines += (
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            f"c.data.task = '{m}'; "
            f"OmegaConf.save(c, '{exp_dir}/config_{movie_to_suffix[m]}.yaml'); "
        )
    snapshot_cmd = f"PYTHONPATH=. uv run --group eeg python -c \"{snapshot_lines}\" && "

    probe_block = ""
    if not skip_probe:
        parts = []
        for m in probe_movies:
            suf = movie_to_suffix[m]
            parts.append(
                "PYTHONPATH=. uv run --group eeg python"
                " eb_jepa/evaluation/clip_probe/probe.py"
                f" --checkpoint {exp_dir}/latest.pth.tar"
                f" --config {exp_dir}/config_{suf}.yaml"
                " --split val --cv-splits 5"
                f" --output {EXP_DIR}/probe_val_{slug}_{suf}.json"
            )
        probe_block = " && " + " && ".join(parts)

    return Job(
        name=f"cs_aligner_{slug}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit=time_limit,
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp config/clip_pretrain.yaml {exp_dir}/config.yaml && "
            f"{patch_cmd}"
            f"{snapshot_cmd}"
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={epochs}"
            f" --meta.seed={seed}"
            f" --folder={exp_dir}"
            " --logging.save_every=99999"
            f" --logging.wandb_group=cs_aligner_{run_tag}"
            f"{probe_block}"
        ),
        venv="__none__",
        branch="feature/cs-aligner",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


def _parse_kv(args, key, default):
    for a in args:
        if a.startswith(f"--{key}="):
            return a.split("=", 1)[1]
    return default


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _submit_train.py <run_tag> [--cs-weight=1.0] "
              "[--kernel-bw=null] [--epochs=60] [--seed=2026] "
              "[--task=ThePresent] [--skip-probe] [submit]")
        sys.exit(1)
    run_tag = sys.argv[1]
    args = sys.argv[2:]
    cs_weight = float(_parse_kv(args, "cs-weight", "1.0"))
    kernel_bw = _parse_kv(args, "kernel-bw", "null")
    epochs = int(_parse_kv(args, "epochs", "60"))
    seed = int(_parse_kv(args, "seed", "2026"))
    task = _parse_kv(args, "task", None)
    skip_probe = "--skip-probe" in args
    submit = "submit" in args
    if task is not None and task not in {"ThePresent", "DespicableMe"}:
        print(f"--task must be ThePresent or DespicableMe, got {task!r}")
        sys.exit(1)
    job = build_job(run_tag, cs_weight, kernel_bw, epochs, seed, task, skip_probe)
    if submit:
        print(f"Submitting {job.name} "
              f"(cs_weight={cs_weight}, kernel_bw={kernel_bw}, "
              f"epochs={epochs}, seed={seed}, task={task or 'multi'}, "
              f"skip_probe={skip_probe})")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name}")
        print(job.submit(dry_run=True))
