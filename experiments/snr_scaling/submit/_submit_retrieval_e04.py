"""Submit Top-K retrieval evaluation (time/shot/scene, e->v and v->e in one
pass) over e04_reve_scaling (depth-22) cells on Delta.

Depth-22 counterpart of _submit_retrieval.py. See _submit_traintest_e04.py's
docstring for why this reads a per-cell epoch from
``experiments/snr_scaling/e04_selection.json`` instead of a fixed EPOCH, and
why the DM config is a repo-relative path rather than a file under kkokate's
checkpoint root.

The evaluator itself is ``eb_jepa/evaluation/clip_probe/retrieval.py``; this
only chooses checkpoints, config and split. It computes e->v and v->e at three
pool granularities (time / shot / scene) in one pass, for topk in {1, 5, 10}.

Presets:

  within    All 28 cells, val AND test, on ThePresent.
  cross     The same 28 checkpoints evaluated on DespicableMe, TEST ONLY
            (matching e03's cross retrieval preset), via config_probe_DM_e04.yaml.
            Zero-shot: nothing is refit, so this is the strictest transfer test.

Pass ``--epoch 325`` to evaluate every cell's fixed epoch 325 checkpoint
instead of its per-cell selected one, matching e03's own convention -- needed
for a depth-12 vs depth-22 delta uncomplicated by the two arms using
different selection protocols. Fixed-epoch outputs get an ``_ep<N>`` filename
suffix so they never collide with the per-cell-selected ones.

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_retrieval_e04.py within
    uv run --group eeg python experiments/snr_scaling/submit/_submit_retrieval_e04.py cross submit --chunks 4
    uv run --group eeg python experiments/snr_scaling/submit/_submit_retrieval_e04.py within submit --epoch 325 --chunks 6
"""
import argparse
import json
from pathlib import Path

from neurolab.jobs import Job

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

REPO = "/u/dtyoung/eb_jepa_eeg"
CKPT_ROOT = "/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling"
OUT_DIR = "experiments/snr_scaling/raw_results"
SELECTION = ROOT / "e04_selection.json"
DM_CONFIG = "experiments/snr_scaling/config/config_probe_DM_e04.yaml"
TOPKS = "1 5 10"
RANDOM_BASELINE_CELL = "e04_s1863_a101_nd"

# --- arms -------------------------------------------------------------------
# Every arm is the SAME depth-22 architecture, evaluated by the same code
# against the same two configs. Only three things vary: where the checkpoints
# live, which cells exist, and how the epoch is chosen. One table rather than a
# per-arm branch, because there are now three of them.
#
#   e04         kkokate's e04_reve_scaling. Pool 1863, REVE warm start, 28
#               cells, per-cell smoothed selection from e04_selection.json.
#   addval      e05_addval. Pool 2156, REVE warm start, 31 cells.
#   fromscratch e05_fromscratch. Pool 2156, NO warm start, 31 cells. Isolates
#               what the initialisation buys -- at S=1400 it is worth +0.111 on
#               the within-task probe, ~15x the depth-12/depth-22 difference.
#
# Both e05 arms report TEST ONLY: their encoders trained on R5, so a val number
# would be measured on data they saw. Neither can use smoothed val selection
# for the same reason, so both are pinned to a fixed epoch. The suffix
# convention is uniform across arms -- unsuffixed means E05_DEFAULT_EPOCH (375),
# `_ep<N>` means a fixed N -- so pass `--epoch 325` for the official
# cross-arm comparison, where e04's own 28-cell ep325 sweep also exists.
#
# EXPECTED_TRAIN_RECORDINGS is deliberately NOT per-arm: the ridge head is fit
# on [R1..R4, R7..R10] for every cell in every arm, so 1863/1832 must hold
# throughout. A 2156 there means a probe read a pretraining config.
E05_CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e05_addval"
E05FS_CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e05_fromscratch"
E03AV_CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling"
E05_DEFAULT_EPOCH = 375
E05_S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]


def _e05_cells(prefix: str) -> list[str]:
    """Must stay in step with _submit_e05_addval.py's CELLS + FULL_AXIS_CELLS;
    tests/unit/test_addval_arm.py pins that."""
    return ([f"{prefix}_s{s}_a101_av_d{d}" for s in E05_S_AXIS for d in (11, 22, 33)]
            + [f"{prefix}_s2156_a101_av"])


E05_CELLS = _e05_cells("e05")
E05FS_CELLS = _e05_cells("e05fs")

ARMS = {
    "e04": dict(root=None, cells=None, splits=None),
    "addval": dict(root=E05_CKPT_ROOT, cells=E05_CELLS, splits=["test"]),
    "fromscratch": dict(root=E05FS_CKPT_ROOT, cells=E05FS_CELLS, splits=["test"]),
    # Depth 12 on the 2156 pool. Same pool and seeds as the depth-22 arms, so
    # its draws are PAIRED with theirs -- d11 of each trains on the identical
    # cohort -- which is what makes a depth contrast between them a paired
    # comparison rather than two independent samples. Cells come from the
    # selection file, since the ladder's two budget halves carry different
    # slugs. TEST ONLY: this arm trained on R5.
    # Its own configs, and this is not cosmetic: evaluating a depth-12
    # checkpoint against the depth-22 config builds the wrong encoder, loads
    # almost no weights under strict=False, and probes a random model while
    # reporting normal-looking numbers.
    "d12av": dict(root=E03AV_CKPT_ROOT, cells=None, splits=["test"],
                  configs={"within": "experiments/snr_scaling/config/config_probe_TP_e03.yaml",
                           "cross": "experiments/snr_scaling/config/config_probe_DM_e03.yaml"}),
}
ARM_SPLITS = {a: v["splits"] for a, v in ARMS.items()}


def load_selection() -> dict[str, int]:
    sel = json.loads(SELECTION.read_text())
    epochs, bad = {}, []
    for slug, info in sel.items():
        if "selected_epoch" in info:
            epochs[slug] = info["selected_epoch"]
        else:
            bad.append(slug)
    if bad:
        print(f"WARNING: {len(bad)} cell(s) had no selected epoch, skipped: {bad}")
    return epochs


def load_probe_selection(path: str) -> dict[str, int]:
    """{slug: epoch} from an epoch-curve probe selection.

    Written by src/analyse_epoch_curve.py --write-selection. Covers only the
    cells the curve could reach: a full-pool cell has no cohort complement and
    so no held-out data to select on, and is deliberately ABSENT here rather
    than defaulted to something. Evaluating the cells present leaves that cell's
    existing fixed-epoch artifacts untouched, which is the honest handling --
    the paper already excludes it from every fit.
    """
    sel = json.loads(Path(path).read_text())
    return {slug: info["selected_epoch"] for slug, info in sel.items()
            if "selected_epoch" in info}


def resolve_arm(arm: str, epoch: int | None,
                selection: str | None = None) -> tuple[str, dict[str, int], str, str]:
    """-> (ckpt_root, {slug: epoch}, filename_suffix, human description)."""
    root = CKPT_ROOT if arm == "e04" else ARMS[arm]["root"]
    if selection is not None:
        # Per-cell epochs from the probe curve. The _epprb suffix keeps these
        # artifacts from colliding with the _ep325/_ep375 sets every published
        # number comes from -- the two protocols must remain separately
        # readable, or a later reader cannot tell which produced a given file.
        eps = load_probe_selection(selection)
        return (root, eps, "_epprb",
                f"per-cell probe-selected epoch ({len(eps)} cells, "
                f"from {Path(selection).name})")
    if arm == "e04":
        if epoch is not None:
            return (CKPT_ROOT, {s: epoch for s in load_selection()},
                    f"_ep{epoch}", f"fixed epoch {epoch}")
        return CKPT_ROOT, load_selection(), "", "per-cell selected epoch"
    spec = ARMS[arm]
    ep = E05_DEFAULT_EPOCH if epoch is None else epoch
    return (spec["root"], {s: ep for s in spec["cells"]},
            "" if ep == E05_DEFAULT_EPOCH else f"_ep{ep}",
            f"fixed epoch {ep} (no held-out val; see _submit_e05_addval.py)")


PRESETS = {
    "within": dict(splits=["val", "test"], config=None, prefix="e04_retr"),
    "cross": dict(splits=["test"], config=DM_CONFIG, prefix="xtask_retr_DM"),
}


def build_steps(preset: str, epochs: dict[str, int], epoch_suffix: str = "",
                ckpt_root: str = CKPT_ROOT, arm: str = "e04") -> list[str]:
    """One shell step per (split, checkpoint). Each step is independently
    idempotent (guarded by its own skip-if-exists test), so callers must NOT
    split these strings on " && " -- each step already contains that token
    internally as part of its skip guard.
    """
    p = PRESETS[preset]
    # An arm may override the preset's config -- a different architecture
    # needs its own, and the mismatch is silent rather than fatal.
    cfg = (ARMS[arm].get("configs") or {}).get(preset) or p["config"]
    allowed = ARM_SPLITS[arm]
    splits = p["splits"] if allowed is None else [
        s for s in p["splits"] if s in allowed]
    steps = []
    for split in splits:
        tag = f"{p['prefix']}{split}" if p["prefix"].endswith("DM") else f"{p['prefix']}_{split}"
        for slug, epoch in epochs.items():
            cfg = p["config"] or f"{ckpt_root}/{slug}/config_probe.yaml"
            out = f"{OUT_DIR}/{tag}_{slug}{epoch_suffix}.json"
            steps.append(
                f"([ -f {out} ] && echo 'SKIP {slug}') || "
                "PYTHONPATH=. uv run --group eeg python "
                "eb_jepa/evaluation/clip_probe/retrieval.py "
                f"--checkpoint {ckpt_root}/{slug}/epoch_{epoch}.pth.tar "
                f"--config {cfg} --split {split} --device cuda "
                f"--topks {TOPKS} --output {out}"
            )
        # The random baseline loads no checkpoint, so it is identical across
        # arms AND epochs -- the addval arm reuses the e04 artifact rather than
        # recomputing a bit-identical one.
        if arm != "e04":
            continue
        if preset == "within":
            out = f"{OUT_DIR}/{tag}_random.json"
            cfg = f"{CKPT_ROOT}/{RANDOM_BASELINE_CELL}/config_probe.yaml"
        else:
            out = f"{OUT_DIR}/{tag}_random_e04.json"
            cfg = p["config"]
        steps.append(
            f"([ -f {out} ] && echo 'SKIP random') || "
            "PYTHONPATH=. uv run --group eeg python "
            "eb_jepa/evaluation/clip_probe/retrieval.py "
            f"--random-baseline --config {cfg} --split {split} "
            f"--device cuda --topks {TOPKS} --output {out}"
        )
    return steps


def chunk(steps: list[str], n: int) -> list[list[str]]:
    n = max(1, min(n, len(steps)))
    size, extra = divmod(len(steps), n)
    out, i = [], 0
    for c in range(n):
        take = size + (1 if c < extra else 0)
        out.append(steps[i:i + take])
        i += take
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("action", nargs="?", default="dry", choices=["dry", "submit"])
    ap.add_argument("--partition", default="gpuA40x4")
    ap.add_argument("--time-limit", default="02:00:00")
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--epoch", type=int, default=None,
                    help="Use this FIXED epoch for every cell instead of "
                         "per-cell selection (e.g. 325, matching e03's own "
                         "convention). Outputs get an _ep<N> filename suffix.")
    ap.add_argument("--selection", default=None,
                    help="Probe-selected epochs from an epoch curve, e.g. "
                         "experiments/snr_scaling/raw_results/"
                         "addval_selection_probe.json. Overrides --epoch and "
                         "evaluates each cell at its own optimum; artifacts get "
                         "an _epprb suffix. Cells absent from the file (full-pool "
                         "cells, which have no held-out data to select on) are "
                         "skipped rather than defaulted.")
    ap.add_argument("--arm", default="e04", choices=sorted(ARMS),
                    help="e04 = kkokate's e04_reve_scaling (default). "
                         "addval = the e05_addval cells, trained with R5 in "
                         "the pool; TEST SPLIT ONLY, fixed epoch "
                         f"{E05_DEFAULT_EPOCH}.")
    args = ap.parse_args()

    ckpt_root, epochs, epoch_suffix, epoch_desc = resolve_arm(args.arm, args.epoch, args.selection)

    p = PRESETS[args.preset]
    steps = build_steps(args.preset, epochs, epoch_suffix, ckpt_root, args.arm)
    groups = chunk(steps, args.chunks)
    splits = ARM_SPLITS[args.arm] or p["splits"]
    print(f"arm={args.arm}  preset={args.preset}  splits={splits}  "
          f"cells={len(epochs)}  "
          f"-> {len(steps)} retrieval run(s), {epoch_desc}, in {len(groups)} job(s)")
    # Report the config the RUNS will use, not the preset default -- an arm may
    # override it, and printing the default would hide exactly the mismatch that
    # matters (an architecture config that does not describe the checkpoints).
    _cfg = (ARMS[args.arm].get("configs") or {}).get(args.preset) or p["config"]
    print(f"  config: {_cfg or 'per-cell config_probe.yaml (ThePresent)'}\n")

    for i, group in enumerate(groups):
        job = Job(
            name=f"retr_{args.arm}_{args.preset}" + (f"_ep{args.epoch}" if args.epoch else "") + (f"_{i}" if len(groups) > 1 else ""),
            cluster="delta",
            repo_path=REPO,
            partition=args.partition,
            time_limit=args.time_limit,
            command=f"mkdir -p {OUT_DIR} && " + " && ".join(group),
            venv="__none__",
            branch="",
            env_vars={
                "WANDB_MODE": "disabled",
                "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/dtyoung/hbn_preprocessed",
            },
        )
        if args.action != "submit":
            print(f"--- {job.name} ---")
            print(job.command.replace(" && ", " &&\n\n"), "\n")
        else:
            print(f"submitted {job.name}: {job.submit()}")

    if args.action != "submit":
        print("Dry run. Re-run with 'submit' to sbatch.")


if __name__ == "__main__":
    main()
