"""Submit a soft_target_clip vs scene_clip A/B run to Delta (multi-movie).

Mirrors the flow of `scene_clip_multimovie/autoresearch/_submit_iter.py`:
train once on the multi-movie config (task=[ThePresent, DespicableMe]), then
probe val on each movie separately using per-movie config snapshots.

Usage:
    uv run --group eeg python _submit_ab.py <arm> <run_tag> [submit]

    arm      : "scene_clip" (baseline) or "soft_target_clip" (test).
    run_tag  : short slug to disambiguate this pair of runs (e.g. "jul7").
    submit   : append to actually sbatch. Otherwise dry-run.

Optional overrides for the test arm (positional --kw=):
    --alpha=0.5              soft_alpha (blend weight on teacher)
    --tau=0.1                soft_tau_teacher (teacher softmax temperature)

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

# Match scene_clip_multimovie jul2 budget: ~11 s/ep at embed=512 baseline,
# 160 ep = 29 min train + ~8 min per-movie probes = ~45 min total.
EPOCHS = 160


def build_job(arm: str, run_tag: str, alpha: float, tau: float) -> Job:
    if arm not in {"scene_clip", "soft_target_clip"}:
        raise ValueError(f"arm must be scene_clip or soft_target_clip, got {arm!r}")
    exp_dir = f"{CKPT_ROOT}/{run_tag}_{arm}"
    output_TP = f"{EXP_DIR}/probe_val_{run_tag}_{arm}_TP.json"
    output_DM = f"{EXP_DIR}/probe_val_{run_tag}_{arm}_DM.json"

    # Patch loss.mode; add soft-target knobs if applicable. Runs inside sbatch
    # so the on-disk config/clip_pretrain.yaml is untouched by concurrent runs.
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
    patch_lines += f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    patch_cmd = (
        f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "
    )

    return Job(
        name=f"soft_clip_{run_tag}_{arm}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="01:00:00",
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp config/clip_pretrain.yaml {exp_dir}/config.yaml && "
            # Patch multi-movie config with loss knobs for this arm.
            f"{patch_cmd}"
            # Snapshot single-task variants for per-movie probing (reads the
            # patched config.yaml so probes match the trained model).
            "PYTHONPATH=. uv run --group eeg python -c \""
            "from omegaconf import OmegaConf; "
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            "c.data.task = 'ThePresent'; "
            f"OmegaConf.save(c, '{exp_dir}/config_TP.yaml'); "
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            "c.data.task = 'DespicableMe'; "
            f"OmegaConf.save(c, '{exp_dir}/config_DM.yaml')\" && "
            # Train (multi-movie).
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={EPOCHS}"
            f" --folder={exp_dir}"
            f" --logging.wandb_group=soft_target_clip_{run_tag}"
            " && "
            # Probe val on ThePresent.
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            f" --config {exp_dir}/config_TP.yaml"
            " --split val --cv-splits 5"
            f" --output {output_TP}"
            " && "
            # Probe val on DespicableMe.
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


def _parse_kv(args: list[str], key: str, default: float) -> float:
    for a in args:
        if a.startswith(f"--{key}="):
            return float(a.split("=", 1)[1])
    return default


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: _submit_ab.py <arm> <run_tag> [--alpha=0.5] [--tau=0.1] [submit]")
        sys.exit(1)
    arm = sys.argv[1]
    run_tag = sys.argv[2]
    args = sys.argv[3:]
    alpha = _parse_kv(args, "alpha", 0.5)
    tau = _parse_kv(args, "tau", 0.1)
    submit = "submit" in args
    job = build_job(arm, run_tag, alpha, tau)
    if submit:
        print(f"Submitting {job.name} (alpha={alpha}, tau={tau})")
        print(f"job_id: {job.submit()}")
    else:
        print(
            f"Dry-run {job.name} (arm={arm}, run_tag={run_tag}, "
            f"alpha={alpha}, tau={tau}). Add 'submit' to actually run."
        )
        print(job.submit(dry_run=True))
