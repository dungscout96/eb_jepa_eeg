"""Submit LeJEPA pretraining of a random-init REVE encoder to Delta.

Snapshots config.yaml into the per-run checkpoint dir (so concurrent runs
don't race on the shared template), then invokes the canonical entry
point ``eb_jepa.training.jepa_pretrain`` with ``--fname=<snapshot>``.
The training loop already supports LeJEPA via ``loss.anti_collapse=sigreg``
(see eb_jepa/anti_collapse.py :: SIGRegAntiCollapse) — no per-experiment
training code needed.

Usage:
    uv run --group eeg python _submit.py <run_tag> [submit]

    run_tag  : short slug to disambiguate runs (e.g. "jul9").
    submit   : append to actually sbatch. Otherwise dry-run.

Optional --key=value overrides (positional):
    --epochs=100         training epochs
    --seed=2025          training seed (cfg.meta.seed)
    --coeff=0.05         SIGReg λ (convex weight on sigreg vs pred_loss)
    --num-slices=1024    SIGReg random projection count
    --lr=5e-4            optimizer LR
    --batch-size=128     data.batch_size
    --task=ThePresent    data.task (ThePresent or DespicableMe)
    --save-every=20      logging.save_every (99999 = only keep latest.pth.tar)
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/jepa_pretraining/lejepa_reve"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve"

DEFAULT_EPOCHS = 100


def build_job(
    run_tag: str,
    epochs: int,
    seed: int,
    coeff: float,
    num_slices: int,
    lr: float,
    batch_size: int,
    task: str,
    save_every: int,
    auto_eval: bool,
) -> Job:
    slug = f"{run_tag}_lejepa_reve_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"

    # --task=multi is the sentinel for multi-movie training. Patch data.task
    # as a Python list literal rather than a quoted string in that case.
    task_expr = (
        "['ThePresent', 'DespicableMe']"
        if task == "multi"
        else f"'{task}'"
    )

    # Patch snapshotted config before training so on-disk template stays clean
    # even under concurrent runs. We only patch fields that vary per submission;
    # architecture stays whatever config.yaml declares.
    patch_lines = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.meta.seed = {seed}; "
        f"c.optim.epochs = {epochs}; "
        f"c.optim.lr = {lr}; "
        f"c.data.batch_size = {batch_size}; "
        f"c.data.task = {task_expr}; "
        f"c.loss.sigreg.coeff = {coeff}; "
        f"c.loss.sigreg.num_slices = {num_slices}; "
        f"c.logging.save_every = {save_every}; "
        f"c.logging.wandb_group = 'lejepa_reve_{run_tag}'; "
        f"c.eval.auto_run = {auto_eval}; "
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    )
    patch_cmd = f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "

    # Timing estimate: depth=12, embed=512, patch=50 (~12 tokens/win/chan × 129
    # chans = ~1500 tokens/sample). H200 bs=64: ~37 s/ep on ThePresent; multi-movie
    # doubles the dataset → ~74 s/ep. Add ~15 min for auto-eval if enabled.
    sec_per_ep = 74 if task == "multi" else 37
    eval_overhead_min = 15 if auto_eval else 0
    est_min = int((epochs * sec_per_ep / 60 + eval_overhead_min) * 1.25)
    hours, mins = divmod(max(60, est_min), 60)
    time_limit = f"{hours:02d}:{mins:02d}:00"

    return Job(
        name=f"lejepa_{slug}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuH200x8",  # Delta A40=44GB and A100=40GB both OOM at bs=32; H200=141GB fits comfortably
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
            "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/kkokate/hbn_preprocessed",
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
            "[--epochs=100] [--seed=2025] [--coeff=0.05] [--num-slices=1024] "
            "[--lr=5e-4] [--batch-size=128] [--task=ThePresent|DespicableMe|multi] "
            "[--save-every=20] [--auto-eval=true] [submit]"
        )
        sys.exit(1)
    run_tag = sys.argv[1]
    args = sys.argv[2:]
    epochs = _parse_kv(args, "epochs", int, DEFAULT_EPOCHS)
    seed = _parse_kv(args, "seed", int, 2025)
    coeff = _parse_kv(args, "coeff", float, 0.05)
    num_slices = _parse_kv(args, "num-slices", int, 1024)
    lr = _parse_kv(args, "lr", float, 5e-4)
    batch_size = _parse_kv(args, "batch-size", int, 128)
    task = _parse_kv(args, "task", str, "ThePresent")
    save_every = _parse_kv(args, "save-every", int, 20)
    auto_eval = _parse_kv(args, "auto-eval", lambda s: s.lower() != "false", True)
    if task not in {"ThePresent", "DespicableMe", "multi"}:
        print(f"--task must be ThePresent, DespicableMe, or multi (both movies), got {task!r}")
        sys.exit(1)
    submit = "submit" in args

    job = build_job(
        run_tag=run_tag,
        epochs=epochs,
        seed=seed,
        coeff=coeff,
        num_slices=num_slices,
        lr=lr,
        batch_size=batch_size,
        task=task,
        save_every=save_every,
        auto_eval=auto_eval,
    )
    banner = (
        f"(epochs={epochs}, seed={seed}, coeff={coeff}, num_slices={num_slices}, "
        f"lr={lr}, bs={batch_size}, task={task}, save_every={save_every}, "
        f"auto_eval={auto_eval})"
    )
    if submit:
        print(f"Submitting {job.name} {banner}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name} {banner}. Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
