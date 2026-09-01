"""Submit cs_aligner fine-tunes warm-started from a JEPA checkpoint.

Two model variants (both from the jul10 H200 bs=64 seed=2025 runs):

  --model=lejepa  → LeJEPA (convex SIGReg λ=0.05, 12L/512d/8h, patch=50)
  --model=laya    → Laya (additive SIGReg λ=0.02, 12L/384d/6h, patch=25)

Each variant reads its own on-disk config template so encoder shape stays
locked to the JEPA checkpoint's — see config_{lejepa,laya}.yaml headers.

Loss: cs_aligner (Yin et al. 2025, arXiv:2502.17028), L = L_InfoNCE + λ·D_CS
on batch marginals. cs_weight=1.0 default (paper); calibrate per-experiment
via --cs-weight.

Usage:
    uv run --group eeg python _submit.py <model> <run_tag> [submit]

    model     : lejepa | laya
    run_tag   : short slug to disambiguate (e.g. "jul10").
    submit    : append to actually sbatch.

Optional --key=value:
    --epochs=100         training epochs
    --seed=2025          training seed
    --cs-weight=1.0      λ in L = InfoNCE + λ·D_CS  (paper default 1.0;
                         jul9 sweep-winner on TP-only from-scratch was 0)
    --kernel-bw=median   'median' → per-batch heuristic (recommended),
                         or a float for fixed bandwidth.
    --lr=1e-5            optimizer LR (warm-start default = 1/10 of from-scratch)
    --warmup=10          warmup_epochs
    --batch-size=64      data.batch_size
    --task=ThePresent|DespicableMe|multi
    --auto-eval=false    logging.save_every fixed at 20
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/clip_pretraining/cs_aligner_from_checkpoint"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa"

# JEPA checkpoints to warm-start from (jul10 H200 bs=64 seed=2025 runs).
# --jepa-source picks between the single-movie and multi-movie JEPA runs.
JEPA_CKPTS = {
    "single": {
        "lejepa": "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-h200-bs64_lejepa_reve_seed2025/latest.pth.tar",
        "laya":   "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-h200-bs64_laya_seed2025/latest.pth.tar",
    },
    "multi": {
        "lejepa": "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-multi_lejepa_reve_seed2025/latest.pth.tar",
        "laya":   "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-multi_laya_seed2025/latest.pth.tar",
    },
}


def build_job(
    model: str,
    run_tag: str,
    epochs: int,
    seed: int,
    cs_weight: float,
    kernel_bw: str,
    lr: float,
    warmup: int,
    batch_size: int,
    task: str,
    auto_eval: bool,
    jepa_source: str,
) -> Job:
    if jepa_source not in JEPA_CKPTS:
        raise ValueError(f"--jepa-source must be one of {list(JEPA_CKPTS)}, got {jepa_source!r}")
    if model not in JEPA_CKPTS[jepa_source]:
        raise ValueError(f"--model must be one of {list(JEPA_CKPTS[jepa_source])}, got {model!r}")
    jepa_ckpt = JEPA_CKPTS[jepa_source][model]

    slug = f"{run_tag}_cs_from_{model}_{jepa_source}_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"
    config_src = f"{EXP_DIR}/config_{model}.yaml"

    task_expr = (
        "['ThePresent', 'DespicableMe']"
        if task == "multi"
        else f"'{task}'"
    )

    kb_expr = "None" if kernel_bw.lower() in ("median", "none", "null", "") else str(float(kernel_bw))

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
        f"c.loss.cs_weight = {cs_weight}; "
        f"c.loss.kernel_bandwidth = {kb_expr}; "
        f"c.logging.wandb_group = 'cs_from_jepa_{run_tag}'; "
        f"c.eval.auto_run = {auto_eval}; "
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    )
    patch_cmd = f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "

    # cs_aligner adds an extra kernel-similarity computation per batch on top
    # of InfoNCE — still lighter than JEPA. At bs=64 H200 for embed=512, expect
    # ~22 s/ep on ThePresent, ~44 s/ep multi-movie.
    sec_per_ep = 44 if task == "multi" else 22
    eval_overhead_min = 15 if auto_eval else 0
    est_min = int((epochs * sec_per_ep / 60 + eval_overhead_min) * 1.25)
    hours, mins = divmod(max(60, est_min), 60)
    time_limit = f"{hours:02d}:{mins:02d}:00"

    return Job(
        name=slug,
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
            "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/kkokate/hbn_preprocessed",
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
            "[--epochs=100] [--seed=2025] [--cs-weight=1.0] [--kernel-bw=median] "
            "[--lr=1e-5] [--warmup=10] [--batch-size=64] "
            "[--task=ThePresent|DespicableMe|multi] "
            "[--jepa-source=single|multi] "
            "[--auto-eval=false] [submit]\n"
            "  model : lejepa | laya"
        )
        sys.exit(1)
    model = sys.argv[1]
    run_tag = sys.argv[2]
    args = sys.argv[3:]
    epochs = _parse_kv(args, "epochs", int, 100)
    seed = _parse_kv(args, "seed", int, 2025)
    cs_weight = _parse_kv(args, "cs-weight", float, 1.0)
    kernel_bw = _parse_kv(args, "kernel-bw", str, "median")
    lr = _parse_kv(args, "lr", float, 1e-5)
    warmup = _parse_kv(args, "warmup", int, 10)
    batch_size = _parse_kv(args, "batch-size", int, 64)
    task = _parse_kv(args, "task", str, "ThePresent")
    jepa_source = _parse_kv(args, "jepa-source", str, "single")
    auto_eval = _parse_kv(args, "auto-eval", lambda s: s.lower() != "false", False)
    if task not in {"ThePresent", "DespicableMe", "multi"}:
        print(f"--task must be ThePresent, DespicableMe, or multi, got {task!r}")
        sys.exit(1)
    if jepa_source not in {"single", "multi"}:
        print(f"--jepa-source must be single or multi, got {jepa_source!r}")
        sys.exit(1)
    submit = "submit" in args

    job = build_job(
        model=model,
        run_tag=run_tag,
        epochs=epochs,
        seed=seed,
        cs_weight=cs_weight,
        kernel_bw=kernel_bw,
        lr=lr,
        warmup=warmup,
        batch_size=batch_size,
        task=task,
        auto_eval=auto_eval,
        jepa_source=jepa_source,
    )
    banner = (
        f"(model={model}, epochs={epochs}, seed={seed}, cs_weight={cs_weight}, "
        f"kernel_bw={kernel_bw}, lr={lr}, warmup={warmup}, bs={batch_size}, "
        f"task={task}, jepa_source={jepa_source}, auto_eval={auto_eval})"
    )
    if submit:
        print(f"Submitting {job.name} {banner}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name} {banner}. Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
