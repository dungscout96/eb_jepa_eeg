"""Submit cross-subject predictive JEPA pretraining to Delta.

Predicts subject B's target tokens from subject A's context at the same movie
time, so the subject fingerprint (~96% of variance, and what masked prediction
rationally learns instead) is marginalized out. See config.yaml's header for the
motivation, the jul10 baseline numbers, and the collapse-detection protocol.

Snapshots config.yaml into the per-run checkpoint dir (so concurrent runs don't
race on the shared template), then invokes the canonical entry point
``eb_jepa.training.jepa_pretrain`` with ``--fname=<snapshot>``. The training loop
supports this via ``loss.objective=cross_subject``, which selects both
``PairedSubjectJEPADataset`` and ``CrossSubjectJEPA`` — no per-experiment
training code.

Usage:
    uv run --group eeg python _submit.py <run_tag> [submit]

    run_tag  : short slug to disambiguate runs (e.g. "xsubj-run1").
    submit   : append to actually sbatch. Otherwise dry-run.

Optional --key=value overrides (positional):
    --epochs=100          training epochs
    --seed=2025           training seed (cfg.meta.seed)
    --coeff=0.05          SIGReg λ (convex weight on sigreg vs pred_loss)
    --num-slices=1024     SIGReg random projection count
    --lr=5e-4             optimizer LR
    --batch-size=64       data.batch_size (-> 2x this many flattened rows)
    --task=ThePresent     data.task (ThePresent, DespicableMe, or multi)
    --save-every=20       logging.save_every (99999 = only keep latest.pth.tar)
    --auto-eval=true      run the post-training probe eval
    --pairs-per-rec=1     data.pairs_per_recording — the main step-budget lever
    --context-mask-mode=masked   masked | full
    --pred-target-mode=masked    masked | all
    --within-weight=0.0   loss.cross_subject.within_subject_weight (rescue lever)
    --symmetric=true      train both A->B and B->A (free)
    --partition=gpuH200x8 SLURM partition
    --sec-per-ep=N        override the per-epoch time estimate (seconds)

MEMORY / PARTITION. Cross-subject at bs=B pushes 2B rows through the encoder, so it
costs about the same as masked JEPA at bs=2B. Only H200 (141GB) fits the intended
bs=64. Measured on A40 (44.4GB) 2026-07-29, this exact geometry, fwd+bwd+step:

    bs= 4  rows= 8  peak=11.33 GiB (25.5%)
    bs= 8  rows=16  peak=22.75 GiB (51.2%)   <- safe operating point
    bs=12  rows=24  peak=35.95 GiB (80.9%)   <- hard ceiling
    bs=16  rows=32  OOM

Linear at ~2.84 GiB per unit batch_size. bs=12 fits a single probe draw but is NOT
safe in practice: MultiBlockMaskCollator redraws every step and n_ctx ranges from
~15% to ~100% of the 1548 tokens, so the context encoder pass varies and 80.9% can
spike over the limit across thousands of steps. Hence --batch-size<=8 off H200
(enforced below).

CAVEAT when you do that: SIGReg's Cramer-Wold test acts on `pooled`, whose sample
count is 2*batch_size*n_windows. bs=64/nw=1 gives N=128 (matching lejepa_reve, itself
already a compromise vs the LeJEPA paper's 256+). bs=8/nw=1 gives N=16 -- 8x below
that compromise. A small-batch run is therefore useful for EARLY COLLAPSE SIGNAL and
for buying optimizer steps (701/8 = 88 steps/epoch vs 11 at bs=64), but it is NOT a
clean parity comparison against jepa_lejepa_single, and a collapse observed at N=16
may be a SIGReg artifact rather than a property of the objective.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/jepa_pretraining/cross_subject"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/cross_subject"

DEFAULT_EPOCHS = 100

# Measured memory model (job 20571303, A40, this exact geometry: 129 chans,
# n_windows=1, depth=12, embed=512, 1548 tokens, SIGReg 1024 slices, fwd+bwd+step):
#   bs= 4 -> 11.33 GiB    bs= 8 -> 22.75 GiB
#   bs=12 -> 35.95 GiB    bs=16 -> OOM
# Linear at ~3.08 GiB per unit batch_size. Peak scales with the total row count
# 2*batch_size*n_windows, so n_windows multiplies the cost identically.
GIB_PER_BATCH = 3.08
GIB_INTERCEPT = -0.98
GPU_CAPACITY_GIB = {
    "gpuA40x4": 44.4,     # measured via nvidia-smi
    "gpuA100x4": 40.0,    # NOTE: SMALLER than the A40
    "gpuA100x8": 40.0,
    "gpuH200x8": 139.81,  # measured: usable capacity reported by torch
}
# MultiBlockMaskCollator redraws every step and n_ctx ranges from ~15% to ~100% of
# the 1548 tokens, so the context encoder pass varies run-to-run. Leave 25% headroom
# rather than fitting to a single observed draw.
SAFE_FRACTION = 0.75


def estimate_peak_gib(batch_size: int, n_windows: int = 1) -> float:
    """Predicted peak GiB for a cross-subject step at this batch size."""
    return GIB_INTERCEPT + GIB_PER_BATCH * batch_size * n_windows


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
    pairs_per_rec: int,
    context_mask_mode: str,
    pred_target_mode: str,
    within_weight: float,
    symmetric: bool,
    partition: str = "gpuH200x8",
    sec_per_ep_override: int | None = None,
) -> Job:
    slug = f"{run_tag}_cross_subject_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"

    # --task=multi is the sentinel for multi-movie training. Patch data.task
    # as a Python list literal rather than a quoted string in that case.
    task_expr = (
        "['ThePresent', 'DespicableMe']"
        if task == "multi"
        else f"'{task}'"
    )

    # Patch snapshotted config before training so the on-disk template stays
    # clean even under concurrent runs. Only fields that vary per submission are
    # patched; model.* and masking.* stay whatever config.yaml declares (they are
    # deliberately bit-identical to lejepa_reve — see config.yaml's header).
    patch_lines = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.meta.seed = {seed}; "
        f"c.optim.epochs = {epochs}; "
        f"c.optim.lr = {lr}; "
        f"c.data.batch_size = {batch_size}; "
        f"c.data.task = {task_expr}; "
        f"c.data.pairs_per_recording = {pairs_per_rec}; "
        f"c.loss.sigreg.coeff = {coeff}; "
        f"c.loss.sigreg.num_slices = {num_slices}; "
        f"c.loss.cross_subject.context_mask_mode = '{context_mask_mode}'; "
        f"c.loss.cross_subject.pred_target_mode = '{pred_target_mode}'; "
        f"c.loss.cross_subject.within_subject_weight = {within_weight}; "
        f"c.loss.cross_subject.symmetric = {symmetric}; "
        f"c.logging.save_every = {save_every}; "
        f"c.logging.wandb_group = 'cross_subject_{run_tag}'; "
        f"c.eval.auto_run = {auto_eval}; "
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    )
    patch_cmd = f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "

    # Timing, calibrated against the 2026-07-29 smoke run (job 20566881):
    #   - 1 epoch took 2:39 (159 s) on gpuA40x4 at bs=4 / num_workers=2.
    #     Per-epoch data volume is fixed (701 items x 2 clips = 1402 FIF reads)
    #     regardless of batch size, so the real run's 4x workers + H200 should
    #     land well under that; 120 s/ep is the conservative side of 40-160 s.
    #   - auto-eval ran 36+ min WITHOUT finishing before the 40 min wall. The
    #     old 15 min allowance here was simply wrong; 45 is the measured floor.
    # A TIMEOUT is recoverable (latest.pth.tar is written every epoch), but on a
    # heavily-queued H200 partition a requeue is expensive, so budget generously.
    # Per-epoch cost is dominated by the fixed 1402 FIF reads, NOT by batch size,
    # so small-batch runs are not proportionally cheaper -- override explicitly.
    sec_per_ep = sec_per_ep_override or (240 if task == "multi" else 120) * pairs_per_rec
    eval_overhead_min = 45 if auto_eval else 0
    est_min = int((epochs * sec_per_ep / 60 + eval_overhead_min) * 1.25)
    hours, mins = divmod(max(60, est_min), 60)
    time_limit = f"{hours:02d}:{mins:02d}:00"

    return Job(
        name=f"xsubj_{slug}",
        cluster="delta",
        repo_path=REPO,
        # H200=141GB is the only partition that fits bs=64 (=128 flattened rows).
        # Delta A40=44GB and A100=40GB both OOM at *masked* bs=32, and cross-subject
        # at bs=B costs about the same as masked at bs=2B -- so an A40 run needs
        # bs<=8. See the --partition/--batch-size note in the module docstring.
        partition=partition,
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
            "[--epochs=100] [--seed=2025] [--coeff=0.05] [--num-slices=1024] "
            "[--lr=5e-4] [--batch-size=64] [--task=ThePresent|DespicableMe|multi] "
            "[--save-every=20] [--auto-eval=true] [--pairs-per-rec=1] "
            "[--context-mask-mode=masked|full] [--pred-target-mode=masked|all] "
            "[--within-weight=0.0] [--symmetric=true] [submit]"
        )
        sys.exit(1)
    run_tag = sys.argv[1]
    args = sys.argv[2:]
    _bool = lambda s: s.lower() != "false"
    epochs = _parse_kv(args, "epochs", int, DEFAULT_EPOCHS)
    seed = _parse_kv(args, "seed", int, 2025)
    coeff = _parse_kv(args, "coeff", float, 0.05)
    num_slices = _parse_kv(args, "num-slices", int, 1024)
    lr = _parse_kv(args, "lr", float, 5e-4)
    batch_size = _parse_kv(args, "batch-size", int, 64)
    task = _parse_kv(args, "task", str, "ThePresent")
    save_every = _parse_kv(args, "save-every", int, 20)
    auto_eval = _parse_kv(args, "auto-eval", _bool, True)
    pairs_per_rec = _parse_kv(args, "pairs-per-rec", int, 1)
    context_mask_mode = _parse_kv(args, "context-mask-mode", str, "masked")
    pred_target_mode = _parse_kv(args, "pred-target-mode", str, "masked")
    within_weight = _parse_kv(args, "within-weight", float, 0.0)
    symmetric = _parse_kv(args, "symmetric", _bool, True)
    partition = _parse_kv(args, "partition", str, "gpuH200x8")
    sec_per_ep_override = _parse_kv(args, "sec-per-ep", int, None)

    est = estimate_peak_gib(batch_size, n_windows=1)
    cap = GPU_CAPACITY_GIB.get(partition)
    if cap is None:
        print(f"WARNING: no measured capacity for partition={partition!r}; skipping OOM guard.")
    elif est > SAFE_FRACTION * cap:
        biggest = max(
            (b for b in range(1, 129) if estimate_peak_gib(b, 1) <= SAFE_FRACTION * cap),
            default=0,
        )
        print(
            f"--batch-size={batch_size} on {partition} is predicted to need {est:.0f} GiB of "
            f"{cap:.0f} GiB ({100 * est / cap:.0f}%), above the {100 * SAFE_FRACTION:.0f}% "
            f"safety line.\n"
            f"  Cross-subject bs=B pushes 2B rows through the encoder; measured cost is "
            f"{GIB_PER_BATCH:.2f} GiB per unit batch_size (job 20571303, A40).\n"
            f"  Largest safe batch_size on {partition} is {biggest} "
            f"(pooled SIGReg N = {2 * biggest}).\n"
            f"  NOTE: pooled N = 2*batch_size*n_windows and memory is proportional to the same "
            f"product, so trading batch_size for n_windows does NOT buy more SIGReg samples."
        )
        sys.exit(1)

    if task not in {"ThePresent", "DespicableMe", "multi"}:
        print(f"--task must be ThePresent, DespicableMe, or multi (both movies), got {task!r}")
        sys.exit(1)
    if context_mask_mode not in {"masked", "full"}:
        print(f"--context-mask-mode must be masked or full, got {context_mask_mode!r}")
        sys.exit(1)
    if pred_target_mode not in {"masked", "all"}:
        print(f"--pred-target-mode must be masked or all, got {pred_target_mode!r}")
        sys.exit(1)
    if pairs_per_rec < 1:
        print(f"--pairs-per-rec must be >= 1, got {pairs_per_rec}")
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
        pairs_per_rec=pairs_per_rec,
        context_mask_mode=context_mask_mode,
        pred_target_mode=pred_target_mode,
        within_weight=within_weight,
        symmetric=symmetric,
        partition=partition,
        sec_per_ep_override=sec_per_ep_override,
    )
    banner = (
        f"(epochs={epochs}, seed={seed}, coeff={coeff}, num_slices={num_slices}, "
        f"lr={lr}, bs={batch_size}, task={task}, save_every={save_every}, "
        f"auto_eval={auto_eval}, pairs_per_rec={pairs_per_rec}, "
        f"ctx={context_mask_mode}, tgt={pred_target_mode}, "
        f"within={within_weight}, symmetric={symmetric}, partition={partition}, "
        f"pooled_sigreg_N={2 * batch_size})"
    )
    if submit:
        print(f"Submitting {job.name} {banner}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name} {banner}. Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
