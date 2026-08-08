"""Submit the E0.4 (full-REVE subject-scaling) sweep on Delta.

E0.3 re-run at the full REVE geometry. Only the encoder shape and the init
change; everything else is pinned to E0.3 so the rows drop straight into the
existing tables.

    E0.3   embed 512, depth 12, patch 400 / overlap  0, random init
           37.9 M params, 129 tokens/sample
    E0.4   embed 512, depth 22, patch 200 / overlap 20, warm-started from
           reve-base, 69.2 M params, 258 tokens/sample

Patch 200 is FORCED, not chosen: ``to_patch_embedding.weight`` in the REVE
checkpoint is ``(512, 200)``, so loading it into a patch-400 model raises
``RuntimeError``. Taking REVE's weights means taking REVE's geometry. The rest
-- soft_target_clip at alpha=0.5 / tau=0.05, lr 1e-4, adam, warmup 5, batch 64,
n_windows=1, 400 epochs, ``meta.seed=2026`` -- is E0.3's, unchanged.

``--init=random`` drops the warm start and trains the same shape from scratch.
That is the from-scratch ARM -- it answers "how much of the gain is the
pretrained init?" -- and it is NOT the Delta-r^2 null. The null in this repo is
an UNTRAINED encoder: ``raw_results/e03_probe_val_random.json`` carries
``"random_baseline": true``, ``analyse_e03.py`` calls it "an untrained encoder,
so it has no stopping point", and it comes from ``probe.py --random-baseline``,
not from a training job. E0.3's null is depth 12 and cannot be reused, so E0.4
needs its own at this geometry -- four probe-only jobs, no training:

    PYTHONPATH=. uv run --group eeg python eb_jepa/evaluation/clip_probe/probe.py \
        --random-baseline --config <any e04 cell>/config_probe.yaml \
        --split val --cv-splits 5 \
        --output experiments/snr_scaling/raw_results/e04_probe_val_random.json

plus the ``probe_traintest.py`` (val, test) and ``retrieval.py`` (test)
equivalents, which are what ``aggregate_nested.py --null-*`` expects. Passing a
TRAINED from-scratch run as the subtrahend instead would redefine the metric,
and the E0.4 rows would no longer sit on E0.3's scale.

Two protocol requirements from PLAN.md E0.3, carried over unchanged:

**Hold gradient steps constant, not epochs.** ``data.epoch_size=703`` on every
cell, so ``len(dataset)`` -- and therefore steps/epoch -- is 703/64 -> 11
regardless of how many subjects survive subsampling. Every cell runs exactly
4400 steps under an identical LR schedule.

**Always evaluate on the full anchor set.** The knobs are applied to the train
dataset only (see ``clip_pretrain.py``); the val loader and the probe build
their own datasets with no subsampling, so retrieval chance levels are
identical across cells.

Usage:
    uv run --group eeg python experiments/snr_scaling/_submit_e04.py    # dry run
    uv run --group eeg python experiments/snr_scaling/_submit_e04.py \
        --nested-draws --suffix=_nd --save-every=25 --skip-probe submit
    uv run --group eeg python experiments/snr_scaling/_submit_e04.py \
        --init=random --nested-draws --suffix=_nd --save-every=25 \
        --skip-probe submit
    uv run --group eeg python experiments/snr_scaling/_submit_e04.py \
        --only=1863x101 --extended --suffix=_nd submit

``--skip-probe`` belongs on BOTH sweeps. Without it a cell is probed on
``latest.pth.tar`` (epoch 399) while every cell that went through
``select_and_probe`` is probed on its smoothed-val-AUC epoch, so the two sides
of a comparison would be read off checkpoints chosen by different protocols --
and both files would then claim the same (slug, readout) in
``aggregate_nested.discover``, which exits rather than guess.

A single cell above S=701 -- the R1-R4 pool holds 701 subjects in 703
recordings -- needs --extended explicitly: --nested-draws implies it, --only
does not. Asking for more without it is now refused rather than run against the
R1-R4 pool under a slug that misreports S.
"""

import argparse

from neurolab.jobs import Job

# E0.3 submits from /u/dtyoung/eb_jepa_eeg, which is permission-denied from this
# account -- the single blocker that stops _submit_e03.py running here.
REPO = "/projects/bbnv/kkokate/eb_jepa_eeg"
EXP_DIR = "experiments/snr_scaling"
CKPT_ROOT = "/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling"

# REVE-base rekeyed to EET ``encoder.*`` names by
# experiments/clip_pretraining/scene_clip_from_checkpoint/prepare_reve_checkpoint.py.
REVE_CKPT = "/work/hdd/bbnv/kkokate/eb_jepa/reve_base_eet_init.pth.tar"

# On this account uv lives inside the conda env rather than ~/.local/bin, and the
# generated batch script sources no shell profile, so the compute node starts with
# a bare PATH. Without this every stage dies instantly with "uv: command not found"
# and the job still exits 0, which reads as a 7-second success in sacct.
PATH_EXPORT = 'export PATH="$HOME/.conda/envs/eb_jepa/bin:$HOME/.local/bin:$PATH"'

# jul7 best from-scratch (RESULTS_jul7.md 3.4): TP-only soft target, tau=0.05.
ARM = "soft_target_clip"
ALPHA = 0.5
TAU = 0.05
SEED = 2026
TASK = "ThePresent"
EPOCHS = 400
# Measured, not extrapolated: 11.5 s/epoch at depth 22 / patch 200 (258 tokens),
# commit 080339b, Delta job 20400451 -- ~77 min of training at 400 epochs plus
# ~8 min for the probe. E0.3's 02:00:00 was sized for the 129-token model and
# does not transfer. The remaining margin is deliberate: the S=1863 cells ran
# 1.2x the S=701 cells in E0.3, three jul7-e400 jobs already timed out at 1h39,
# and a timeout wastes the entire cell.
TIME_LIMIT = "03:00:00"
# Measured on Delta, not assumed: the TP train split is 703 recordings from 701
# distinct subjects (two subjects contribute two recordings each). The S axis is
# in SUBJECTS, so its top point is 701; epoch_size is in ITEMS and tracks the
# recording count, 703, which is what sets steps/epoch.
FULL_SUBJECTS = 701
FULL_RECORDINGS = 703
FULL_ANCHORS = 101  # 2 s windows spanning ThePresent (202.5 s)

SUFFIX = ""

# Extended-cohort S axis (R7-R10 preprocessed 2026-08-02, train pool 703 ->
# 1863 recordings). S=400 and S=701 are re-run FROM THE EXTENDED POOL as
# calibration: the original cells at those S drew from R1-R4 only, so without
# an overlap point, stitching the old and new curves would silently mix two
# different sampling pools. If the calibration points reproduce the originals,
# the axis stitches honestly; if they do not, that is itself the finding.
EXTENDED_CELLS = [
    (400, 101),  # calibration overlap with the R1-R4 curve
    (701, 101),  # calibration overlap with the R1-R4 curve
    (1000, 101),
    (1400, 101),
    (1863, 101),
]

# Nested + replicated S axis (RESULTS.md 2.11). Subject draws nest by
# construction, so a curve measures ADDED subjects; repeating each S under
# several data.subsample_seed values measures the between-draw variance that
# made the first extension unreadable. S=1863 is the whole pool -- there is only
# one possible draw -- so it gets a single cell rather than three identical ones.
NESTED_S = [400, 701, 1000, 1400]
NESTED_DRAW_SEEDS = [11, 22, 33]
FULL_POOL_S = 1863

# L-shape + diagonal, 11 cells. NOT the full 20-cell grid. Kept so --only and
# the non-extended default still resolve; E0.4 itself runs --nested-draws.
CELLS = [
    # S axis at full A (includes the shared corner)
    (701, 101),
    (400, 101),
    (200, 101),
    (100, 101),
    (50, 101),
    # A axis at full S
    (701, 50),
    (701, 25),
    (701, 13),
    # diagonal, to test separability of the two axes
    (400, 50),
    (200, 25),
    (100, 13),
]


def build_job(
    subjects: int,
    anchors: int,
    partition: str,
    time_limit: str,
    epochs: int,
    save_every: int = 99999,
    skip_probe: bool = False,
    extended: bool = False,
    draw_seed: int | None = None,
    init: str = "reve",
    init_checkpoint: str = REVE_CKPT,
) -> Job:
    # The init tag sits BEFORE the suffix so the sweep glob 'e04_s*_a101_nd*'
    # selects the warm-start cells only; the null answers to 'e04_s*_a101_random*'
    # and cannot clobber the cell it is the null for.
    slug = f"e04_s{subjects}_a{anchors}"
    if init != "reve":
        slug += f"_{init}"
    slug += SUFFIX
    if draw_seed is not None:
        slug += f"_d{draw_seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"

    # Patch the frozen template inside the job, so the on-disk config is never
    # mutated by concurrent runs.
    patch = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.loss.mode = '{ARM}'; "
        f"c.loss.soft_alpha = {ALPHA}; "
        f"c.loss.soft_tau_teacher = {TAU}; "
        f"c.data.task = '{TASK}'; "
        f"c.data.max_subjects = {subjects}; "
        f"c.data.max_anchors = {anchors}; "
        f"c.data.epoch_size = {FULL_RECORDINGS}; "
        + (f"c.data.subsample_seed = {draw_seed}; " if draw_seed is not None else "")
        + f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    )
    # The probe must see the FULL data, so its config snapshot clears the knobs.
    probe_patch = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        "c.data.max_subjects = None; "
        "c.data.max_anchors = None; "
        "c.data.epoch_size = None; "
        f"OmegaConf.save(c, '{exp_dir}/config_probe.yaml')"
    )
    # The null passes no path at all rather than an empty one: the copied config
    # already carries meta.encoder_init_from: null, and the trainer's warm-start
    # branch keys off truthiness.
    init_flag = (
        "" if init == "random" else f" --meta.encoder_init_from={init_checkpoint}"
    )
    return Job(
        name=slug,
        cluster="delta",
        repo_path=REPO,
        partition=partition,
        time_limit=time_limit,
        command=(
            f"{PATH_EXPORT} && "
            f"mkdir -p {exp_dir} && "
            f"cp experiments/snr_scaling/clip_pretrain_reve.yaml {exp_dir}/config.yaml && "
            f'PYTHONPATH=. uv run --group eeg python -c "{patch}" && '
            f'PYTHONPATH=. uv run --group eeg python -c "{probe_patch}" && '
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={epochs}"
            f" --meta.seed={SEED}"
            f"{init_flag}"
            f" --folder={exp_dir}"
            f" --logging.save_every={save_every}"
            " --logging.wandb_group=e04_reve_scaling"
            + (
                ""
                if skip_probe
                # This JSON lands in the repo working tree, not in raw_results/,
                # where aggregate_nested.py looks by default (it has no fallback
                # to this directory the way analyse_e03.py does). Move it before
                # aggregating. The --skip-probe sweeps never reach this branch.
                else (
                    " && "
                    "PYTHONPATH=. uv run --group eeg python"
                    " eb_jepa/evaluation/clip_probe/probe.py"
                    f" --checkpoint {exp_dir}/latest.pth.tar"
                    f" --config {exp_dir}/config_probe.yaml"
                    " --split val --cv-splits 5"
                    f" --output {EXP_DIR}/e04_probe_val_{slug}.json"
                )
            )
        ),
        venv="__none__",
        # branch="" makes neurolab SKIP the compute-node git checkout/pull,
        # because concurrent jobs race on .git/refs locks. PULL THE REPO ON
        # DELTA BY HAND BEFORE SUBMITTING -- the jobs run whatever is checked
        # out at REPO, including clip_pretrain_reve.yaml.
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            # uv hardlinks fail across Delta's filesystems, and the cells share one
            # .venv: run `uv sync --group eeg` once before submitting so concurrent
            # jobs read a warm venv instead of each rebuilding eb-jepa into it.
            "UV_LINK_MODE": "copy",
            # The unified root (R1-R6 symlinked, R7-R10 real) only when
            # extended; otherwise the original root, so existing cells stay
            # bit-comparable. The extended tree belongs to another user but is
            # group-readable -- verified.
            "HBN_PREPROCESS_DIR": (
                "/work/hdd/bbnv/dtyoung/hbn_preprocessed"
                if extended
                else "/projects/bbnv/kkokate/hbn_preprocessed"
            ),
            **({"HBN_TRAIN_RELEASES": "R1,R2,R3,R4,R7,R8,R9,R10"} if extended else {}),
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", default="gpuA40x4")
    p.add_argument("--time-limit", default=TIME_LIMIT)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--only", default=None, help="Run one cell, e.g. --only=703x101.")
    p.add_argument(
        "--init",
        default="reve",
        choices=["reve", "random"],
        help="reve warm-starts from --init-checkpoint. random trains "
        "the same depth-22 shape from scratch: the from-scratch ARM, "
        "NOT the Delta-r^2 null. The null is an UNTRAINED encoder from "
        "probe.py --random-baseline at this geometry (see the module "
        "docstring); subtracting a trained run redefines the metric.",
    )
    p.add_argument(
        "--init-checkpoint",
        default=REVE_CKPT,
        help="REVE->EET converted checkpoint. Ignored by --init=random.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=99999,
        help="Periodic epoch_N.pth.tar checkpoints. 99999 keeps only "
        "latest.pth.tar. Set ~25 for the early-stopped protocol "
        "-- without intermediate checkpoints there is nothing to "
        "early-stop TO (see RESULTS.md 2.10).",
    )
    p.add_argument(
        "--skip-probe",
        action="store_true",
        help="Train only. Under the early-stopped protocol the probe "
        "runs later, on the selected checkpoint, not on latest.",
    )
    p.add_argument(
        "--nested-draws",
        action="store_true",
        help="Nested + replicated S axis: NESTED_S x NESTED_DRAW_SEEDS "
        "plus a single full-pool cell. Implies --extended.",
    )
    p.add_argument(
        "--extended",
        action="store_true",
        help="Run the extended-cohort S axis: enables R7-R10 via "
        "HBN_TRAIN_RELEASES, points HBN_PREPROCESS_DIR at the "
        "unified root, and uses EXTENDED_CELLS.",
    )
    p.add_argument(
        "--suffix",
        default="",
        help="Appended to the cell slug, so a re-run does not "
        "overwrite the endpoint-protocol results.",
    )
    p.add_argument("action", nargs="?", default="dry", choices=["dry", "submit"])
    args = p.parse_args()

    global SUFFIX
    SUFFIX = args.suffix
    if args.nested_draws:
        args.extended = True
        cells = [(s_, 101, d) for s_ in NESTED_S for d in NESTED_DRAW_SEEDS]
        cells.append((FULL_POOL_S, 101, None))  # whole pool: one draw only
    else:
        cells = [
            (s_, a_, None) for s_, a_ in (EXTENDED_CELLS if args.extended else CELLS)
        ]
    if args.only:
        s_, a_ = (int(v) for v in args.only.lower().split("x"))
        cells = [(s_, a_, None)]
    # S counts SUBJECTS and the R1-R4 pool holds 701. Asking for more without
    # --extended fails nowhere downstream: max_subjects merely exceeds the pool,
    # the cell trains on all 703 recordings, and the slug still claims the
    # larger S -- so `--only=1863x101 --suffix=_nd` would overwrite the real
    # full-pool cell with an R1-R4 run under its own name. Refuse instead.
    if not args.extended and any(s_ > FULL_SUBJECTS for s_, _, _ in cells):
        p.error(
            f"S > {FULL_SUBJECTS} needs --extended; without it the job draws "
            "from the 703-recording R1-R4 pool while the slug reports the "
            "larger S, and the result silently replaces the extended cell."
        )

    print(
        f"{len(cells)} cell(s), {args.epochs} ep each on {args.partition} "
        f"[{args.time_limit}], init={args.init}"
    )
    print("depth 22 / patch 200 / overlap 20, 69.2M params, 258 tokens/sample")
    print(
        "init from "
        + (args.init_checkpoint if args.init == "reve" else "scratch (depth-22 null)")
        + "\n"
    )
    print(f"  {'cell':<20}{'subjects':>10}{'anchors':>9}{'draw':>7}  steps")
    for s_, a_, d in cells:
        tag = f"s{s_}xa{a_}" + (f"_d{d}" if d is not None else "")
        print(
            f"  {tag:<20}{s_:>10}{a_:>9}{str(d or '-'):>7}  "
            f"{args.epochs * -(-FULL_RECORDINGS // 64)}"
        )
    print()

    if args.action != "submit":
        s_, a_, d = cells[0]
        print(
            build_job(
                s_,
                a_,
                args.partition,
                args.time_limit,
                args.epochs,
                save_every=args.save_every,
                skip_probe=args.skip_probe,
                extended=args.extended,
                draw_seed=d,
                init=args.init,
                init_checkpoint=args.init_checkpoint,
            ).command
        )
        print("\nDry run. Re-run with 'submit' to sbatch.")
        return

    for s_, a_, d in cells:
        job = build_job(
            s_,
            a_,
            args.partition,
            args.time_limit,
            args.epochs,
            save_every=args.save_every,
            skip_probe=args.skip_probe,
            extended=args.extended,
            draw_seed=d,
            init=args.init,
            init_checkpoint=args.init_checkpoint,
        )
        print(f"submitted {job.name}: {job.submit()}")


if __name__ == "__main__":
    main()
