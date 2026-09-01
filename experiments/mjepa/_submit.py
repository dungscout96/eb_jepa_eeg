"""Submit MJEPA-adapted pretraining (masked EEG + cross-modal L1) to Delta.

Tests whether an L1 regression to a continuous 1408-d V-JEPA-2 target escapes
the ~101-class ceiling that every contrastive objective in this repo hits (see
experiments/snr_scaling/PLAN.md and the config header).

Snapshots config.yaml into the per-run checkpoint dir, patches the snapshot,
then invokes the canonical entry point. The training loop supports this via
`loss.objective=mjepa`, which selects both recipe_mode on the dataset and the
MJEPA model -- no per-experiment training code.

Usage:
    uv run --group eeg python _submit.py <run_tag> [submit]

Optional --key=value overrides (positional):
    --lam=0.0             loss.mjepa.lambda_intra -- THE SWEEP {0, 0.5, 1.0}
    --init=scratch        scratch | reve   (reve sets meta.encoder_init_from)
    --epochs=             default 300 for reve, 400 for scratch
    --lr=                 default 1e-4 for reve, 5e-4 for scratch
    --seed=2025
    --batch-size=64       pooled SIGReg N = batch_size * n_windows
    --task=ThePresent     ThePresent | DespicableMe | multi
    --coeff=0.05          SIGReg lambda (only used when --lam > 0)
    --cross-loss=l1       l1 | smooth_l1
    --ve-enabled=true     include L_v2e (a collapse detector; no encoder grad)
    --save-every=50
    --partition=gpuH200x8
    --sec-per-ep=         override the per-epoch time estimate (seconds)
    --auto-eval=false     keep OFF; probe separately on A40 (see config header)

MEMORY. The cross-subject constant (3.08 GiB per unit batch) does NOT transfer:
that model pushes 2 rows per item through a depth-12 / 1548-token encoder, while
this one is single-stream at depth-22 / 258 tokens (patch_size=200 -> P=2). The
constants below are a PLACEHOLDER extrapolation and are deliberately
conservative. Replace GIB_PER_BATCH with torch.cuda.max_memory_allocated() from
the smoke run before submitting the full matrix -- that is the sequence that
produced the trustworthy cross-subject numbers.

ANTI-COLLAPSE. lambda=0 requires anti_collapse=none (enforced in build_jepa):
L_e2v regresses to a fixed external target, so there is no collapse attractor.
lambda>0 uses sigreg to match lejepa_reve / cross_subject, making lambda the
only variable.
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/mjepa"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/mjepa"

# REVE->EET-converted checkpoint, produced by
# experiments/clip_pretraining/scene_clip_from_checkpoint/prepare_reve_checkpoint.py.
# Verified 2026-07-30: 139 encoder tensors, depth=22, embed=512, patch_size=200 --
# matches this config's model section exactly (a mismatch would now hard-fail in
# jepa_pretrain.py rather than silently training from scratch).
REVE_CKPT = "/work/hdd/bbnv/dtyoung/reve_init/reve_base_eet_init.pth.tar"

# MEASURED 2026-07-30 on an A40 at this exact geometry (129 ch, nw=1, depth 22,
# embed 512, patch 200), full fwd+bwd+AdamW step:
#
#   lambda=0 (1 unmasked pass, no predictor, no anti-collapse)
#     bs=  8 ->  2.93 GiB    bs= 64 -> 15.73 GiB    bs=128 -> 30.38 GiB (68% of A40)
#   lambda>0 (masked ctx pass + full target pass + predictor + SIGReg)
#     bs=  8 ->  4.22 GiB    bs= 64 -> 26.59 GiB    bs=128 -> OOM on A40
#
# lambda>0 costs ~1.8x lambda=0, so the model is keyed on it. NOTE the earlier
# placeholder (1.00) was 4.4x too conservative at lambda=0 -- these numbers are
# what make the A40 viable, and the A40 queue is far healthier than H200's.
GIB_PER_BATCH = {0.0: 0.229, 1.0: 0.404}   # per unit batch_size, by lambda regime
GIB_INTERCEPT = {0.0: 1.10, 1.0: 0.99}
GPU_CAPACITY_GIB = {
    "gpuA40x4": 44.4,
    "gpuA100x4": 40.0,      # NOTE: SMALLER than the A40
    "gpuA100x8": 40.0,
    "gpuH200x8": 139.81,
}
# MultiBlockMaskCollator redraws every step and n_ctx varies, so leave headroom
# rather than fitting to a single observed draw.
SAFE_FRACTION = 0.75


def estimate_peak_gib(batch_size: int, lam: float = 0.0, n_windows: int = 1) -> float:
    """Predicted peak GiB. Keyed on whether the masked branch runs at all."""
    regime = 0.0 if lam == 0.0 else 1.0
    return GIB_INTERCEPT[regime] + GIB_PER_BATCH[regime] * batch_size * n_windows


def build_job(run_tag, lam, init, epochs, lr, seed, batch_size, task, coeff,
              cross_loss, ve_enabled, save_every, partition, auto_eval,
              sec_per_ep_override):
    slug = f"{run_tag}_mjepa_lam{lam}_{init}_seed{seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"

    task_expr = ("['ThePresent', 'DespicableMe']" if task == "multi" else f"'{task}'")
    # lambda=0 must pair with anti_collapse=none; build_jepa enforces it too.
    ac = "none" if lam == 0.0 else "sigreg"
    init_expr = f"'{REVE_CKPT}'" if init == "reve" else "None"

    patch_lines = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.meta.seed = {seed}; "
        f"c.meta.encoder_init_from = {init_expr}; "
        f"c.optim.epochs = {epochs}; "
        f"c.optim.lr = {lr}; "
        f"c.data.batch_size = {batch_size}; "
        f"c.data.task = {task_expr}; "
        f"c.loss.anti_collapse = '{ac}'; "
        f"c.loss.sigreg.coeff = {coeff}; "
        f"c.loss.mjepa.lambda_intra = {lam}; "
        f"c.loss.mjepa.cross_loss_type = '{cross_loss}'; "
        f"c.loss.mjepa.ve_enabled = {ve_enabled}; "
        f"c.logging.save_every = {save_every}; "
        f"c.logging.wandb_group = 'mjepa_{run_tag}'; "
        f"c.eval.auto_run = {auto_eval}; "
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    )
    patch_cmd = f"PYTHONPATH=. uv run --group eeg python -c \"{patch_lines}\" && "

    # MEASURED 2026-07-30: 27 s/epoch at lambda=0, bs=64, num_workers=8 on an A40
    # (11 steps of 701 recordings). lambda>0 costs ~1.8x the GPU work per step.
    # 40 / 70 leaves headroom for IO variance on Lustre.
    base_sec = 40 if lam == 0.0 else 70
    sec_per_ep = sec_per_ep_override or (base_sec * (2 if task == "multi" else 1))
    est_min = int((epochs * sec_per_ep / 60 + (45 if auto_eval else 0)) * 1.25)
    hours, mins = divmod(max(60, est_min), 60)

    return Job(
        name=f"mjepa_{slug}",
        cluster="delta",
        repo_path=REPO,
        partition=partition,
        time_limit=f"{hours:02d}:{mins:02d}:00",
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp {EXP_DIR}/config.yaml {exp_dir}/config.yaml && "
            f"{patch_cmd}"
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.jepa_pretrain"
            f" --fname={exp_dir}/config.yaml --folder={exp_dir}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/kkokate/hbn_preprocessed",
            # Delta's default MNE_DATA is not writable for this user ->
            # RevePositionBank fails on cache_dir stat with PermissionError.
            "MNE_DATA": "/u/dtyoung/mne_data",
        },
    )


def _parse_kv(args, key, cast, default):
    for a in args:
        if a.startswith(f"--{key}="):
            return cast(a.split("=", 1)[1])
    return default


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: _submit.py <run_tag> [--lam=0.0] [--init=scratch|reve] "
            "[--epochs=] [--lr=] [--seed=2025] [--batch-size=64] "
            "[--task=ThePresent] [--coeff=0.05] [--cross-loss=l1] "
            "[--ve-enabled=true] [--save-every=50] [--partition=gpuH200x8] "
            "[--sec-per-ep=] [--auto-eval=false] [submit]"
        )
        sys.exit(1)
    run_tag, args = sys.argv[1], sys.argv[2:]
    _bool = lambda s: s.lower() != "false"

    lam = _parse_kv(args, "lam", float, 0.0)
    init = _parse_kv(args, "init", str, "scratch")
    seed = _parse_kv(args, "seed", int, 2025)
    batch_size = _parse_kv(args, "batch-size", int, 64)
    task = _parse_kv(args, "task", str, "ThePresent")
    coeff = _parse_kv(args, "coeff", float, 0.05)
    cross_loss = _parse_kv(args, "cross-loss", str, "l1")
    ve_enabled = _parse_kv(args, "ve-enabled", _bool, True)
    save_every = _parse_kv(args, "save-every", int, 50)
    # A40 default: measured memory says the whole matrix fits (lambda=0 bs=64 is
    # 15.7 GiB of 44.4), and the A40 queue is far healthier than H200's
    # (~255 running/335 pending vs ~26/404). Use --partition=gpuH200x8 only when
    # a batch size above the A40 ceiling is actually needed.
    partition = _parse_kv(args, "partition", str, "gpuA40x4")
    auto_eval = _parse_kv(args, "auto-eval", _bool, False)
    sec_per_ep_override = _parse_kv(args, "sec-per-ep", int, None)
    # Defaults differ by init: warm-start fine-tunes shorter and gentler.
    epochs = _parse_kv(args, "epochs", int, 300 if init == "reve" else 400)
    lr = _parse_kv(args, "lr", float, 1.0e-4 if init == "reve" else 5.0e-4)

    if init not in {"scratch", "reve"}:
        print(f"--init must be scratch or reve, got {init!r}"); sys.exit(1)
    if task not in {"ThePresent", "DespicableMe", "multi"}:
        print(f"--task must be ThePresent, DespicableMe or multi, got {task!r}"); sys.exit(1)
    if cross_loss not in {"l1", "smooth_l1"}:
        print(f"--cross-loss must be l1 or smooth_l1, got {cross_loss!r}"); sys.exit(1)
    if lr >= 1e-3:
        print(f"--lr={lr} looks too high; the record recipe uses 1e-4/3e-4 "
              "(warm-start) or 5e-4 (from scratch)."); sys.exit(1)

    est = estimate_peak_gib(batch_size, lam=lam, n_windows=1)
    cap = GPU_CAPACITY_GIB.get(partition)
    if cap is None:
        print(f"WARNING: no measured capacity for partition={partition!r}; skipping OOM guard.")
    elif est > SAFE_FRACTION * cap:
        biggest = max((b for b in range(1, 513)
                       if estimate_peak_gib(b, lam=lam) <= SAFE_FRACTION * cap), default=0)
        print(
            f"--batch-size={batch_size} at lambda={lam} on {partition} is predicted to need "
            f"{est:.1f} GiB of {cap:.0f} GiB ({100*est/cap:.0f}%), above the "
            f"{100*SAFE_FRACTION:.0f}% safety line.\n"
            f"  Largest safe batch_size here is {biggest} (pooled SIGReg N = {biggest}).\n"
            f"  Measured 2026-07-30: lambda=0 costs {GIB_PER_BATCH[0.0]} GiB/unit-bs, "
            f"lambda>0 costs {GIB_PER_BATCH[1.0]}."
        )
        sys.exit(1)

    submit = "submit" in args
    job = build_job(run_tag, lam, init, epochs, lr, seed, batch_size, task, coeff,
                    cross_loss, ve_enabled, save_every, partition, auto_eval,
                    sec_per_ep_override)
    banner = (
        f"(lam={lam}, init={init}, epochs={epochs}, lr={lr}, seed={seed}, "
        f"bs={batch_size}, task={task}, anti_collapse={'none' if lam == 0.0 else 'sigreg'}, "
        f"cross_loss={cross_loss}, ve={ve_enabled}, partition={partition}, "
        f"pooled_sigreg_N={batch_size}, est_peak={est:.0f}GiB, "
        f"time={job.time_limit}, auto_eval={auto_eval})"
    )
    if submit:
        print(f"Submitting {job.name} {banner}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name} {banner}. Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
