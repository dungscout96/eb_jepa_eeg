"""Submit the E0.3 (anchors x subjects) data-scaling surface on Delta.

Retrains the jul7 best from-scratch recipe -- TP-only soft_target_clip,
alpha=0.5, tau=0.05, seed 2026 -- while subsampling the two data axes
independently, then probes each cell on the FULL val anchor set.

Verified against the jul7 checkpoint rather than assumed:
``latest.pth.tar`` reports ``{'epoch': 399, 'step': 4400}``, i.e. 400 epochs at
11 steps/epoch = ceil(703 / 64), with ``n_windows: 1``.

Two protocol requirements from PLAN.md E0.3, and how each is met:

**Hold gradient steps constant, not epochs.** ``data.epoch_size=703`` on every
cell, so ``len(dataset)`` -- and therefore steps/epoch -- is 703/64 -> 11
regardless of how many subjects survive subsampling. Every cell runs exactly
4400 steps under an identical LR schedule. The naive alternative (scale epochs
as 703/S) would need 5624 epochs at S=50 and per-epoch overhead would dominate.

**Always evaluate on the full anchor set.** The knobs are applied to the train
dataset only (see ``clip_pretrain.py``); the val loader and the probe build
their own datasets with no subsampling, so retrieval chance levels are
identical across cells.

One design note worth keeping: because ``n_windows == 1``, an item is a single
window, so restricted-anchor sampling draws 1 window uniformly from the allowed
set exactly as the unrestricted path draws 1 uniformly from all windows. The
A=101 corner is therefore statistically identical to no restriction -- there is
no sampling-mode confound between the corner cell and the rest of the surface.
Passing ``max_anchors`` on every cell keeps that explicit.

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_e03.py          # dry run
    uv run --group eeg python experiments/snr_scaling/submit/_submit_e03.py submit
    uv run --group eeg python experiments/snr_scaling/submit/_submit_e03.py --only=703x101 submit
"""
import argparse

from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/snr_scaling/raw_results"   # jobs write measured JSONs straight here
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling"

# jul7 best from-scratch (RESULTS_jul7.md 3.4): TP-only soft target, tau=0.05.
ARM = "soft_target_clip"
ALPHA = 0.5
TAU = 0.05
SEED = 2026
TASK = "ThePresent"          # default; override with --task
EPOCHS = 400
# Measured on Delta, not assumed: the TP train split is 703 recordings from 701
# distinct subjects (two subjects contribute two recordings each). The S axis is
# in SUBJECTS, so its top point is 701; epoch_size is in ITEMS and tracks the
# recording count, 703, which is what sets steps/epoch.
FULL_SUBJECTS = 701
FULL_RECORDINGS = 703
FULL_ANCHORS = 101           # 2 s windows spanning ThePresent (202.5 s)

# L-shape + diagonal, 11 cells. NOT the full 20-cell grid.
#   S axis at full A ..... 5 cells (includes the shared corner)
#   A axis at full S ..... 3 cells
#   diagonal ............. 3 cells, to test separability of the two axes
SUFFIX = ""

# Extended-cohort S axis (R7-R10 preprocessed 2026-08-02, train pool 703 ->
# 1863 recordings). S=400 and S=701 are re-run FROM THE EXTENDED POOL as
# calibration: the original cells at those S drew from R1-R4 only, so without
# an overlap point, stitching the old and new curves would silently mix two
# different sampling pools. If the calibration points reproduce the originals,
# the axis stitches honestly; if they do not, that is itself the finding.
EXTENDED_CELLS = [
    (400, 101), (701, 101),          # calibration overlap with the R1-R4 curve
    (1000, 101), (1400, 101), (1863, 101),
]

# Nested + replicated S axis (RESULTS.md 2.11). Subject draws now nest by
# construction, so a curve measures ADDED subjects; repeating each S under
# several data.subsample_seed values measures the between-draw variance that
# made the first extension unreadable. S=1863 is the whole pool -- there is only
# one possible draw -- so it gets a single cell rather than three identical ones.
NESTED_S = [400, 701, 1000, 1400]

# Low-S arm: where does the model start learning at all? Below the nested axis's
# floor of 400. Three draws each matters MORE here than at 400+: with 10-20
# subjects, which particular subjects you drew is a large fraction of the
# signal, so a single draw would be uninterpretable. Nesting still holds
# (10 subset 20 subset 50 ... subset 1400 within a draw seed).
LOW_S = [10, 20, 50, 100, 200]

# DM-native reference cells (cross-task study). Trained on DespicableMe with the
# identical recipe so that TP->DM transfer can be reported as a FRACTION of
# DM-native performance rather than only against random -- random cannot
# distinguish "transfers well" from "DM is easy".
#
# max_anchors=85, not 101: DM is 170.6 s, so its 2 s grid holds ~85 anchors
# (RESULTS.md 2.9 measured exactly 85 on R5 val). 101 would be a silent no-op;
# 85 states the anchor count explicitly, as the TP cells do.
# epoch_size stays 703 so these run the same 4400 steps as every TP cell -- the
# comparison is about data CONTENT, not budget.
DM_CELLS = [(701, 85), (1841, 85)]
NESTED_DRAW_SEEDS = [11, 22, 33]
FULL_POOL_S = 1863

# --- the 2156-pool axis (--av-axis) ------------------------------------------
# Folding R5 into the pool takes it from 1863 to 2156 recordings, which buys two
# things the 1863-pool axis cannot have: S=1863 becomes a real subsample and so
# gets three draws instead of being the single unreplicable full-pool cell, and
# a new top point appears at S=2156. It also puts this arm on the SAME pool as
# the depth-22 arms, so a depth comparison against them is no longer confounded
# by which subjects each pool contains.
#
# The budget splits at S=701 because that is where it stops being sufficient.
# Below it the measured cost of the shorter budget is 0.0000 and those cells are
# already past their optimum by epoch 300; at and above it the optimum sits at
# or beyond the last checkpoint. Run the two halves as two submissions, with the
# suffix naming the budget so neither can overwrite the other.
AV_S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]
AV_FULL_POOL_S = 2156
AV_BUDGET_SPLIT_S = 701
POOL_2156 = False          # set by main() from --av-axis

CELLS = [
    (701, 101), (400, 101), (200, 101), (100, 101), (50, 101),   # vary S
    (701, 50), (701, 25), (701, 13),                             # vary A
    (400, 50), (200, 25), (100, 13),                             # diagonal
]


def build_job(subjects: int, anchors: int, partition: str,
              time_limit: str, epochs: int, save_every: int = 99999,
              skip_probe: bool = False, extended: bool = False,
              draw_seed: int | None = None, task: str = TASK) -> Job:
    slug = f"e03_s{subjects}_a{anchors}{SUFFIX}"
    if draw_seed is not None:
        slug += f"_d{draw_seed}"
    exp_dir = f"{CKPT_ROOT}/{slug}"

    # Patch the shared template inside the job, so the on-disk config is never
    # mutated by concurrent runs.
    patch = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.loss.mode = '{ARM}'; "
        f"c.loss.soft_alpha = {ALPHA}; "
        f"c.loss.soft_tau_teacher = {TAU}; "
        f"c.data.task = '{task}'; "
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
    return Job(
        name=slug,
        cluster="delta",
        repo_path=REPO,
        partition=partition,
        time_limit=time_limit,
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp experiments/snr_scaling/config/clip_pretrain.yaml {exp_dir}/config.yaml && "
            f"PYTHONPATH=. uv run --group eeg python -c \"{patch}\" && "
            f"PYTHONPATH=. uv run --group eeg python -c \"{probe_patch}\" && "
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={epochs}"
            f" --meta.seed={SEED}"
            f" --folder={exp_dir}"
            f" --logging.save_every={save_every}"
            " --logging.wandb_group=e03_scaling"
            + ("" if skip_probe else (
                " && "
                "PYTHONPATH=. uv run --group eeg python"
                " eb_jepa/evaluation/clip_probe/probe.py"
                f" --checkpoint {exp_dir}/latest.pth.tar"
                f" --config {exp_dir}/config_probe.yaml"
                " --split val --cv-splits 5"
                f" --output {EXP_DIR}/e03_probe_val_{slug}.json"))
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            # The unified root (R1-R6 symlinked, R7-R10 real) only when
            # extended; otherwise the original root, so existing cells stay
            # bit-comparable.
            "HBN_PREPROCESS_DIR": (
                "/work/hdd/bbnv/dtyoung/hbn_preprocessed" if extended
                else "/projects/bbnv/kkokate/hbn_preprocessed"),
            # R5 is in the list only for the 2156-pool axis. Getting this wrong
            # does not fail -- it trains a perfectly healthy model on the wrong
            # pool, and only the "Scaling subsample: N -> S" line in the log
            # would ever show it.
            **({"HBN_TRAIN_RELEASES":
                "R1,R2,R3,R4,R5,R7,R8,R9,R10" if POOL_2156
                else "R1,R2,R3,R4,R7,R8,R9,R10"}
               if extended else {}),
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", default="gpuA40x4")
    p.add_argument("--time-limit", default="02:00:00")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--only", default=None,
                   help="Run one cell, e.g. --only=703x101.")
    p.add_argument("--save-every", type=int, default=99999,
                   help="Periodic epoch_N.pth.tar checkpoints. 99999 keeps only "
                        "latest.pth.tar. Set ~25 for the early-stopped protocol "
                        "-- without intermediate checkpoints there is nothing to "
                        "early-stop TO (see RESULTS.md 2.10).")
    p.add_argument("--skip-probe", action="store_true",
                   help="Train only. Under the early-stopped protocol the probe "
                        "runs later, on the selected checkpoint, not on latest.")
    p.add_argument("--task", default=TASK,
                   help="Training movie. Default ThePresent.")
    p.add_argument("--dm-reference", action="store_true",
                   help="Train the DM-native reference cells (implies "
                        "--task=DespicableMe --extended).")
    p.add_argument("--low-s", action="store_true",
                   help="Low-S arm: LOW_S x NESTED_DRAW_SEEDS. Implies "
                        "--extended so draws come from the same 1863 pool and "
                        "nest with the existing cells.")
    p.add_argument("--av-axis", default=None, choices=["low", "high", "all"],
                   help="The 2156-pool axis (R5 folded in), which gives S=1863 "
                        "three draws and adds S=2156. Implies --extended. "
                        "'low' = S<=400, the half that converges inside 400 "
                        "epochs; 'high' = S>=701 plus the whole-pool cell, the "
                        "half that needs 800. Run the two halves separately and "
                        "give each its own --suffix, or the longer run "
                        "overwrites the shorter one's checkpoints.")
    p.add_argument("--nested-draws", action="store_true",
                   help="Nested + replicated S axis: NESTED_S x NESTED_DRAW_SEEDS "
                        "plus a single full-pool cell. Implies --extended.")
    p.add_argument("--extended", action="store_true",
                   help="Run the extended-cohort S axis: enables R7-R10 via "
                        "HBN_TRAIN_RELEASES, points HBN_PREPROCESS_DIR at the "
                        "unified root, and uses EXTENDED_CELLS.")
    p.add_argument("--suffix", default="",
                   help="Appended to the cell slug, so a re-run does not "
                        "overwrite the endpoint-protocol results.")
    p.add_argument("action", nargs="?", default="dry", choices=["dry", "submit"])
    args = p.parse_args()

    global SUFFIX, POOL_2156
    SUFFIX = args.suffix
    POOL_2156 = args.av_axis is not None
    if args.av_axis:
        args.extended = True
        lo = [s_ for s_ in AV_S_AXIS if s_ < AV_BUDGET_SPLIT_S]
        hi = [s_ for s_ in AV_S_AXIS if s_ >= AV_BUDGET_SPLIT_S]
        pick = {"low": lo, "high": hi, "all": AV_S_AXIS}[args.av_axis]
        cells = [(s_, 101, d) for s_ in pick for d in NESTED_DRAW_SEEDS]
        if args.av_axis in ("high", "all"):
            cells.append((AV_FULL_POOL_S, 101, None))  # whole pool: one draw
        if not SUFFIX:
            raise SystemExit(
                "--av-axis needs an explicit --suffix. Without one these cells "
                "take the same slugs as the 1863-pool cells and overwrite them.")
    elif args.dm_reference:
        args.extended = True
        args.task = "DespicableMe"
        cells = [(s_, a_, None) for s_, a_ in DM_CELLS]
    elif args.low_s:
        args.extended = True
        cells = [(s_, 101, d) for s_ in LOW_S for d in NESTED_DRAW_SEEDS]
    elif args.nested_draws:
        args.extended = True
        cells = [(s_, 101, d) for s_ in NESTED_S for d in NESTED_DRAW_SEEDS]
        cells.append((FULL_POOL_S, 101, None))     # whole pool: one draw only
    else:
        cells = [(s_, a_, None)
                 for s_, a_ in (EXTENDED_CELLS if args.extended else CELLS)]
    if args.only:
        s_, a_ = (int(v) for v in args.only.lower().split("x"))
        cells = [(s_, a_, None)]

    print(f"{len(cells)} cell(s), {args.epochs} ep each on {args.partition} "
          f"[{args.time_limit}]\n")
    print(f"  {'cell':<20}{'subjects':>10}{'anchors':>9}{'draw':>7}  steps")
    for s_, a_, d in cells:
        tag = f"s{s_}xa{a_}" + (f"_d{d}" if d is not None else "")
        print(f"  {tag:<20}{s_:>10}{a_:>9}{str(d or '-'):>7}  "
              f"{args.epochs * -(-FULL_RECORDINGS // 64)}")
    print()

    if args.action != "submit":
        s_, a_, d = cells[0]
        print(build_job(s_, a_, args.partition, args.time_limit,
                        args.epochs, args.save_every, args.skip_probe,
                        args.extended, d, args.task).command)
        print("\nDry run. Re-run with 'submit' to sbatch.")
        return

    for s_, a_, d in cells:
        job = build_job(s_, a_, args.partition, args.time_limit, args.epochs,
                        args.save_every, args.skip_probe, args.extended, d,
                        args.task)
        print(f"submitted {job.name}: {job.submit()}")


if __name__ == "__main__":
    main()
