"""Submit soft_target_clip fine-tunes warm-started from a JEPA checkpoint.

Two model variants (both from the jul10 H200 bs=64 seed=2025 runs):

  --model=lejepa  → LeJEPA (convex SIGReg λ=0.05, 12L/512d/8h, patch=50)
  --model=laya    → Laya (additive SIGReg λ=0.02, 12L/384d/6h, patch=25)

Each variant reads its own on-disk config template so encoder shape stays
locked to the JEPA checkpoint's — see config_{lejepa,laya}.yaml headers for
why exact shape match matters (strict=False silently drops mismatches).

Usage:
    uv run --group eeg python _submit.py <model> <run_tag> [submit]

    model     : lejepa | laya
    run_tag   : short slug to disambiguate (e.g. "jul10").
    submit    : append to actually sbatch.

Optional --key=value:
    --epochs=100         training epochs
    --seed=2025          training seed
    --alpha=0.5          soft_alpha
    --tau=0.05           soft_tau_teacher (jul7 winner = 0.05)
    --lr=1e-5            optimizer LR (warm-start default = 1/10 of from-scratch)
    --warmup=10          warmup_epochs
    --batch-size=64      data.batch_size
    --task=ThePresent|DespicableMe|multi
    --auto-eval=false    logging.save_every fixed at 20
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/clip_pretraining/soft_target_clip_from_jepa_checkpoint"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip_from_jepa"

# JEPA checkpoints to warm-start from (jul10 H200 bs=64 seed=2025 runs).
JEPA_CKPTS = {
    "lejepa": "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-h200-bs64_lejepa_reve_seed2025/latest.pth.tar",
    "laya":   "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-h200-bs64_laya_seed2025/latest.pth.tar",
}


def build_job(
    model: str,
    run_tag: str,
    epochs: int,
    seed: int,
    alpha: float,
    tau: float,
    lr: float,
    warmup: int,
    batch_size: int,
    task: str,
    auto_eval: bool,
) -> Job:
    if model not in JEPA_CKPTS:
        raise ValueError(f"--model must be one of {list(JEPA_CKPTS)}, got {model!r}")
    jepa_ckpt = JEPA_CKPTS[model]

    slug = f"{run_tag}_soft_from_{model}_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"
    config_src = f"{EXP_DIR}/config_{model}.yaml"

    task_expr = (
        "['ThePresent', 'DespicableMe']"
        if task == "multi"
        else f"'{task}'"
    )

    patch_lines = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.meta.seed = {seed}; "
        f"c.meta.encoder_init_from = '{jepa_ckpt}'; "
        f"c.optim.epochs = {epochs}; "
        f"c.optim.lr = {lr}; "
        f"c.optim.warmup_epochs = {warmup}; "
        f"c.data.batch_size = {batch_size}; "
        f"c.data.task = {task_expr}; "
        f"c.loss.soft_alpha = {alpha}; "
        f"c.loss.soft_tau_teacher = {tau}; "
        f"c.logging.wandb_group = 'soft_from_jepa_{run_tag}'; "
        f"c.eval.auto_run = {auto_eval}; "
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    )
    patch_cmd = f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "

    # Timing estimate: CLIP is lighter than JEPA per epoch (no predictor / no
    # target encoder pass / no SIGReg). At bs=64 H200 for embed=512, expect
    # ~20 s/ep on ThePresent, ~40 s/ep multi-movie.
    sec_per_ep = 40 if task == "multi" else 20
    eval_overhead_min = 15 if auto_eval else 0
    est_min = int((epochs * sec_per_ep / 60 + eval_overhead_min) * 1.25)
    hours, mins = divmod(max(60, est_min), 60)
    time_limit = f"{hours:02d}:{mins:02d}:00"

    return Job(
        name=slug,  # already encodes model + run_tag + seed
        cluster="delta",
        repo_path=REPO,
        partition="gpuH200x8",   # match JEPA runs — safe headroom
        time_limit=time_limit,
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp {config_src} {exp_dir}/config.yaml && "
            f"{patch_cmd}"
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --folder={exp_dir}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            "MNE_DATA": "/u/dtyoung/mne_data",
        },
    )


def _parse_kv(args: list[str], key: str, cast, default):
    for a in args:
        if a.startswith(f"--{key}="):
            return cast(a.split("=", 1)[1])
    return default


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: _submit.py <model> <run_tag> "
            "[--epochs=100] [--seed=2025] [--alpha=0.5] [--tau=0.05] "
            "[--lr=1e-5] [--warmup=10] [--batch-size=64] "
            "[--task=ThePresent|DespicableMe|multi] "
            "[--auto-eval=false] [submit]\n"
            "  model : lejepa | laya"
        )
        sys.exit(1)
    model = sys.argv[1]
    run_tag = sys.argv[2]
    args = sys.argv[3:]
    epochs = _parse_kv(args, "epochs", int, 100)
    seed = _parse_kv(args, "seed", int, 2025)
    alpha = _parse_kv(args, "alpha", float, 0.5)
    tau = _parse_kv(args, "tau", float, 0.05)
    lr = _parse_kv(args, "lr", float, 1e-5)
    warmup = _parse_kv(args, "warmup", int, 10)
    batch_size = _parse_kv(args, "batch-size", int, 64)
    task = _parse_kv(args, "task", str, "ThePresent")
    auto_eval = _parse_kv(args, "auto-eval", lambda s: s.lower() != "false", False)
    if task not in {"ThePresent", "DespicableMe", "multi"}:
        print(f"--task must be ThePresent, DespicableMe, or multi, got {task!r}")
        sys.exit(1)
    submit = "submit" in args

    job = build_job(
        model=model,
        run_tag=run_tag,
        epochs=epochs,
        seed=seed,
        alpha=alpha,
        tau=tau,
        lr=lr,
        warmup=warmup,
        batch_size=batch_size,
        task=task,
        auto_eval=auto_eval,
    )
    banner = (
        f"(model={model}, epochs={epochs}, seed={seed}, alpha={alpha}, tau={tau}, "
        f"lr={lr}, warmup={warmup}, bs={batch_size}, task={task}, auto_eval={auto_eval})"
    )
    if submit:
        print(f"Submitting {job.name} {banner}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name} {banner}. Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
