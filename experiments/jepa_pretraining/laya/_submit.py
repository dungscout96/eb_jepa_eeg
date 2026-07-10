"""Submit Laya-style JEPA pretraining (SIGReg additive) to Delta.

Uses the canonical eb_jepa.training.jepa_pretrain entry point with the
Laya config (see ./config.yaml). Laya-specific pieces wired up:

  loss.sigreg.combine_mode='additive_weighted'  → total = pred + λ·sigreg
  optim.optimizer='adamw', optim.weight_decay=0.05
  loss.pred_loss_type='mse'
  masking.long_channel_scale=[1.0, 1.0]         → temporal-only block masks
  REVE encoder 12L/384d/6h + predictor 4L/128d + patch=25 (no overlap)

Usage:
    uv run --group eeg python _submit.py <run_tag> [submit]

    run_tag  : short slug to disambiguate runs (e.g. "jul10").
    submit   : append to actually sbatch. Otherwise dry-run.

Optional --key=value overrides (positional):
    --epochs=100         training epochs
    --seed=2025          training seed
    --coeff=0.02         SIGReg λ (Laya-B=0.02, Laya-S=0.05)
    --num-slices=1024    SIGReg random projection count
    --lr=1e-4            optimizer LR
    --weight-decay=0.05  AdamW weight decay
    --batch-size=64      data.batch_size
    --task=ThePresent    data.task (ThePresent or DespicableMe)
    --save-every=20      logging.save_every (99999 = only keep latest.pth.tar)
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/jepa_pretraining/laya"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/laya"

DEFAULT_EPOCHS = 100


def build_job(
    run_tag: str,
    epochs: int,
    seed: int,
    coeff: float,
    num_slices: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    task: str,
    save_every: int,
) -> Job:
    slug = f"{run_tag}_laya_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"

    patch_lines = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.meta.seed = {seed}; "
        f"c.optim.epochs = {epochs}; "
        f"c.optim.lr = {lr}; "
        f"c.optim.weight_decay = {weight_decay}; "
        f"c.data.batch_size = {batch_size}; "
        f"c.data.task = '{task}'; "
        f"c.loss.sigreg.coeff = {coeff}; "
        f"c.loss.sigreg.num_slices = {num_slices}; "
        f"c.logging.save_every = {save_every}; "
        f"c.logging.wandb_group = 'laya_{run_tag}'; "
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    )
    patch_cmd = f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "

    # ~2064 tokens/window at patch=25, embed=384/depth=12: rough estimate
    # ~35 s/ep on A40 at bs=64. Auto-eval adds ~15 min.
    est_min = int((epochs * 35 / 60 + 15) * 1.25)
    hours, mins = divmod(max(60, est_min), 60)
    time_limit = f"{hours:02d}:{mins:02d}:00"

    return Job(
        name=f"laya_{slug}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit=time_limit,
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp {EXP_DIR}/config.yaml {exp_dir}/config.yaml && "
            f"{patch_cmd}"
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.jepa_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --folder={exp_dir}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            # Delta's neurolab profile defaults MNE_DATA=/projects/bcfj/dtyoung/mne_data
            # which is not writable for this user → RevePositionBank fails on
            # cache_dir stat with PermissionError. Point at $HOME instead.
            "MNE_DATA": "/u/dtyoung/mne_data",
        },
    )


def _parse_kv(args: list[str], key: str, cast, default):
    for a in args:
        if a.startswith(f"--{key}="):
            return cast(a.split("=", 1)[1])
    return default


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: _submit.py <run_tag> "
            "[--epochs=100] [--seed=2025] [--coeff=0.02] [--num-slices=1024] "
            "[--lr=1e-4] [--weight-decay=0.05] [--batch-size=64] "
            "[--task=ThePresent] [--save-every=20] [submit]"
        )
        sys.exit(1)
    run_tag = sys.argv[1]
    args = sys.argv[2:]
    epochs = _parse_kv(args, "epochs", int, DEFAULT_EPOCHS)
    seed = _parse_kv(args, "seed", int, 2025)
    coeff = _parse_kv(args, "coeff", float, 0.02)
    num_slices = _parse_kv(args, "num-slices", int, 1024)
    lr = _parse_kv(args, "lr", float, 1e-4)
    weight_decay = _parse_kv(args, "weight-decay", float, 0.05)
    batch_size = _parse_kv(args, "batch-size", int, 64)
    task = _parse_kv(args, "task", str, "ThePresent")
    save_every = _parse_kv(args, "save-every", int, 20)
    if task not in {"ThePresent", "DespicableMe"}:
        print(f"--task must be ThePresent or DespicableMe, got {task!r}")
        sys.exit(1)
    submit = "submit" in args

    job = build_job(
        run_tag=run_tag,
        epochs=epochs,
        seed=seed,
        coeff=coeff,
        num_slices=num_slices,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        task=task,
        save_every=save_every,
    )
    banner = (
        f"(epochs={epochs}, seed={seed}, coeff={coeff}, num_slices={num_slices}, "
        f"lr={lr}, wd={weight_decay}, bs={batch_size}, task={task}, save_every={save_every})"
    )
    if submit:
        print(f"Submitting {job.name} {banner}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name} {banner}. Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
