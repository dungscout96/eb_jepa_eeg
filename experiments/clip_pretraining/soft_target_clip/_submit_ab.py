"""Submit a soft_target_clip / scene_clip / vanilla-clip run to Delta (multi-movie).

Mirrors the flow of `scene_clip_multimovie/autoresearch/_submit_iter.py`:
train once on the multi-movie config (task=[ThePresent, DespicableMe]), then
probe val on each movie separately using per-movie config snapshots.

Usage:
    uv run --group eeg python _submit_ab.py <arm> <run_tag> [submit]

    arm      : "clip" (vanilla InfoNCE, diagonal positives, mean-centered
               per-window targets — the pre-scene_clip baseline extended to
               matched training conditions), "scene_clip" (scene-ID
               multi-positive baseline), or "soft_target_clip" (test).
    run_tag  : short slug to disambiguate this pair of runs (e.g. "jul7").
    submit   : append to actually sbatch. Otherwise dry-run.

Optional overrides (positional --kw=):
    --alpha=0.5              soft_alpha (blend weight on teacher); test arm only
    --tau=0.1                soft_tau_teacher (teacher softmax temperature); test arm only
    --epochs=160             training epochs. Default 160 (~45 min). Use 400 for
                             the long-budget schedule that jul2 iter 1 established
                             as best for multi-movie scene_clip (~2h 25m job).
    --seed=2025              training seed (cfg.meta.seed). Include in exp_dir
                             and probe output filenames so seed replicates don't
                             collide. Probe fold seed is independent (cfg.eval.probe_seed).
    --task=<movie>           single-movie training on ThePresent or DespicableMe
                             (patches data.task in the snapshotted config). If
                             omitted, trains on the config default (multi-movie).
                             Only probes the trained-on movie.

Job structure per arm:
  1. mkdir checkpoint dir
  2. snapshot config/clip_pretrain.yaml → <ckpt>/config.yaml (multi-movie)
  3. patch loss.mode and (for soft_target_clip) soft_alpha / soft_tau_teacher
  4. snapshot per-movie configs (config_TP.yaml, config_DM.yaml)
  5. train (multi-movie, reads config.yaml)
  6. probe val on ThePresent (reads config_TP.yaml)
  7. probe val on DespicableMe (reads config_DM.yaml)
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/clip_pretraining/soft_target_clip"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip"

# Default matches scene_clip_multimovie jul2 iter 4: ~11 s/ep at embed=512
# baseline, 160 ep = 29 min train + ~8 min per-movie probes = ~45 min total.
# 400 ep = jul2 iter 1 long-budget best, ~1h 45m total (needs longer time_limit).
DEFAULT_EPOCHS = 160


def build_job(
    arm: str,
    run_tag: str,
    alpha: float,
    tau: float,
    epochs: int,
    seed: int,
    task: str | None = None,
) -> Job:
    if arm not in {"clip", "scene_clip", "soft_target_clip"}:
        raise ValueError(
            f"arm must be clip, scene_clip, or soft_target_clip, got {arm!r}"
        )
    # Encode seed into the exp_dir and output filenames so seed replicates
    # don't overwrite each other. Naming: <run_tag>_<arm>_seed<seed>.
    # When task is passed (single-movie), we skip the other movie's probe.
    single_movie = task is not None
    probe_movies = (
        [task] if single_movie else ["ThePresent", "DespicableMe"]
    )
    movie_to_suffix = {"ThePresent": "TP", "DespicableMe": "DM"}
    slug = f"{run_tag}_{arm}_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"

    # Patch loss.mode (+ soft-target knobs) and — for single-movie runs — also
    # override data.task in the shared config before the per-movie snapshots
    # are written. Runs inside sbatch so the on-disk template is untouched by
    # concurrent runs.
    patch_lines = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.loss.mode = '{arm}'; "
    )
    if arm == "soft_target_clip":
        patch_lines += (
            f"c.loss.soft_alpha = {alpha}; "
            f"c.loss.soft_tau_teacher = {tau}; "
        )
    if single_movie:
        patch_lines += f"c.data.task = '{task}'; "
    patch_lines += f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    patch_cmd = (
        f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "
    )

    # Multi-movie: ~15 s/ep at embed=512 (measured). Single-movie: ~half the
    # recordings → estimate ~8 s/ep. Each per-movie probe takes ~8 min; single-
    # movie run does just one. 25% headroom for startup / I/O.
    sec_per_ep = 8 if single_movie else 15
    est_min = int(
        ((epochs * sec_per_ep + len(probe_movies) * 8 * 60) / 60) * 1.25
    )
    hours, mins = divmod(max(60, est_min), 60)
    time_limit = f"{hours:02d}:{mins:02d}:00"
    # Build the per-movie config-snapshot patch inline: one OmegaConf.save per
    # movie we plan to probe. Under single-movie the loop degenerates to one.
    snapshot_lines = "from omegaconf import OmegaConf; "
    for m in probe_movies:
        snapshot_lines += (
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            f"c.data.task = '{m}'; "
            f"OmegaConf.save(c, '{exp_dir}/config_{movie_to_suffix[m]}.yaml'); "
        )
    snapshot_cmd = (
        f"PYTHONPATH=. uv run --group eeg python -c \"{snapshot_lines}\" && "
    )

    # Build the probe commands, one per movie.
    probe_cmds = []
    for m in probe_movies:
        suf = movie_to_suffix[m]
        probe_cmds.append(
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            f" --config {exp_dir}/config_{suf}.yaml"
            " --split val --cv-splits 5"
            f" --output {EXP_DIR}/probe_val_{slug}_{suf}.json"
        )
    probe_block = " && ".join(probe_cmds)

    return Job(
        name=f"soft_clip_{slug}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit=time_limit,
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp config/clip_pretrain.yaml {exp_dir}/config.yaml && "
            f"{patch_cmd}"
            f"{snapshot_cmd}"
            # Train (multi-movie or single-movie per patched config.yaml).
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={epochs}"
            f" --meta.seed={seed}"
            f" --folder={exp_dir}"
            f" --logging.wandb_group=soft_target_clip_{run_tag}"
            " && "
            f"{probe_block}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


def _parse_kv(args: list[str], key: str, default: float) -> float:
    for a in args:
        if a.startswith(f"--{key}="):
            return float(a.split("=", 1)[1])
    return default


def _parse_str_kv(args: list[str], key: str, default: str | None) -> str | None:
    for a in args:
        if a.startswith(f"--{key}="):
            return a.split("=", 1)[1]
    return default


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: _submit_ab.py <arm> <run_tag> [--alpha=0.5] [--tau=0.1] [--epochs=160] [--seed=2025] [--task=<movie>] [submit]")
        sys.exit(1)
    arm = sys.argv[1]
    run_tag = sys.argv[2]
    args = sys.argv[3:]
    alpha = _parse_kv(args, "alpha", 0.5)
    tau = _parse_kv(args, "tau", 0.1)
    epochs = int(_parse_kv(args, "epochs", DEFAULT_EPOCHS))
    seed = int(_parse_kv(args, "seed", 2025))
    task = _parse_str_kv(args, "task", None)
    if task is not None and task not in {"ThePresent", "DespicableMe"}:
        print(f"--task must be ThePresent or DespicableMe, got {task!r}")
        sys.exit(1)
    submit = "submit" in args
    job = build_job(arm, run_tag, alpha, tau, epochs, seed, task=task)
    if submit:
        print(f"Submitting {job.name} (alpha={alpha}, tau={tau}, epochs={epochs}, seed={seed}, task={task or 'multi'})")
        print(f"job_id: {job.submit()}")
    else:
        print(
            f"Dry-run {job.name} (arm={arm}, run_tag={run_tag}, "
            f"alpha={alpha}, tau={tau}, epochs={epochs}, seed={seed}, task={task or 'multi'}). "
            "Add 'submit' to actually run."
        )
        print(job.submit(dry_run=True))
