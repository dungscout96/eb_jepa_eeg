"""Submit the per-epoch probe+retrieval curve over e04_reve_scaling cells on Delta.

WHAT THIS IS FOR. Every e04 cell currently picks its checkpoint by smoothed
argmax of `val/clip_scene_auc`, a 29-window AUC whose adjacent-epoch swings
(~0.09) are the same size as its whole trend. This submits
``eb_jepa/evaluation/clip_probe/epoch_curve.py``, which scores every saved
checkpoint of every cell with a small held-out probe AND with retrieval, so the
selection can be redone on a low-variance signal and compared against both the
AUC selection and the paper's fixed epoch 325.

It is a ROBUSTNESS CHECK, not a protocol change: the paper reports at fixed 325
and the already-measured protocol spread (mean |delta| 0.0019 r, max 0.0064 over
40 probe cells) bounds what any selector could be worth.

WHY IT IS AFFORDABLE. probe_traintest.py re-reads 1971 FIF files and re-encodes
~199k windows per checkpoint (~40 min), so 28 cells x 15 epochs would be ~280
GPU-hours. epoch_curve.py reads a fixed 80-recording set ONCE per job and
re-encodes only that (~8k windows) per checkpoint, a few seconds each. Chunking
costs one cache build per job, so prefer FEW, LONG jobs here -- the opposite of
the probe submitters, where every step re-read the data anyway.

WHICH ARMS, AND WHY NOT ALL OF THEM. The selector reads the val split (R5), so
it works for any arm that holds R5 out and for no arm that does not. Whether an
arm qualifies is a fact about its pretraining pool, NOT about its name -- two
different from-scratch arms sit on opposite sides of this line:

  --arm e04         e04_reve_scaling, pool 1863. Warm start, depth 22. Default.
  --arm e03         e03_scaling, pool 1863. From scratch, depth 12, patch 400/0.
  --arm e05rand     e05_random_scaling, pool 1863. From scratch, depth 22,
                    400 epochs. Architecturally matched to e04, so e04 vs this
                    isolates INITIALISATION with nothing else moving.
  --arm e05rand800  the same arm's 800-epoch cells: 8800 steps against 4400,
                    31 checkpoints out to epoch 775.

  NOT SUPPORTED, and cannot be:
  e05_addval        pool 2156 -- R5 IS IN TRAIN.
  e05_fromscratch   pool 2156 -- R5 IS IN TRAIN. Note this is a DIFFERENT
                    from-scratch arm from e05_random_scaling above; same
                    initialisation, different pool, opposite verdict.

For that pair the only unseen split is R6, the reported test split, so they have
no held-out selection data at all and stay pinned to a fixed epoch -- the same
constraint config_probe_TP_headfit_R5.yaml documents for the zero-overlap head
fit.

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py dry
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py submit --arm e05rand --chunks 7
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py smoke submit --arm e03
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py merge --arm e05rand
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py verify --arm e05rand
"""
import argparse
import json
from pathlib import Path

from neurolab.jobs import Job

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # experiments/snr_scaling
RAW_DIR = ROOT / "raw_results"

REPO = "/u/dtyoung/eb_jepa_eeg"
OUT_DIR = "experiments/snr_scaling/raw_results"

# Every supported arm holds R5 out, draws from the same 1863 pool, and uses
# identical windowing (task, n_windows, window_size_seconds, temporal_stride,
# norm_mode). So ONE selection set serves all of them and the curves are
# directly comparable cell-for-cell -- only the checkpoint tree, the
# architecture config, and the incumbent selection differ.
#
# Together they close the initialisation x depth square at matched pool, which
# no single arm can:
#
#            depth 22                       depth 12
#   warm     e04 (REVE init)                --  not trained
#   scratch  e05rand / e05rand800           e03
#
# and e05rand800 is the same recipe as e05rand at DOUBLE the step budget (800
# epochs x 703 = 8800 steps vs 4400), which turns "is the budget binding?" from
# an inference about where an argmax lands into a direct measurement.
#
# `cells` is the authority for `merge`/`verify`, which must run locally without
# the checkpoint tree mounted; the jobs themselves take an explicit --cells list
# built from it. For e04 that list comes from e04_selection.json (28 cells).
# Every other arm carries an explicit list, because their incumbent selection
# JSONs either cover a subset (e03: S>=400 only, 13 of 28) or do not exist
# (e05rand*) -- deriving cells from those would silently scan part of the arm
# and look complete.
S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400]


def _cells(prefix: str, suffix: str) -> list[str]:
    """The standard 28: 9 S values x 3 draws, plus the single full-pool cell."""
    return ([f"{prefix}_s{s}_a101_{suffix}_d{d}" for s in S_AXIS for d in (11, 22, 33)]
            + [f"{prefix}_s1863_a101_{suffix}"])


E03_CELLS = _cells("e03", "nd")
E05RAND_CELLS = _cells("e05", "nd")          # 400 epochs, 15 checkpoints
E05RAND800_CELLS = _cells("e05", "ep800")    # 800 epochs, 31 checkpoints

ARMS = {
    "e04": dict(
        ckpt_root="/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling",
        config="experiments/snr_scaling/config/config_probe_TP_e04.yaml",
        selection=ROOT / "e04_selection.json",
        cells=None,          # taken from `selection`, which covers all 28
        merged=RAW_DIR / "e04_epoch_curve.json",
        desc="warm start from REVE, depth 22, patch 200/20",
    ),
    "e03": dict(
        ckpt_root="/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling",
        config="experiments/snr_scaling/config/config_probe_TP_e03.yaml",
        selection=RAW_DIR / "e03_selection_nd_smooth.json",
        cells=E03_CELLS,     # 28 on disk; `selection` only covers 13 of them
        merged=RAW_DIR / "e03_epoch_curve.json",
        desc="from scratch, depth 12, patch 400/0",
    ),
    # kkokate's e05_random_scaling. NOT the same thing as dtyoung's
    # e05_fromscratch: that one pretrains on the 2156 pool (R5 included) and is
    # therefore unscannable, while this one holds R5 out exactly like e04.
    # Verified 2026-08-21 from the run logs -- "rejected 23/1886" for train and
    # "293 recordings" for val -- and from wandb-metadata args, which carry no
    # encoder_init_from, i.e. random init. Architecturally matched to e04
    # (depth 22, patch 200/20, epoch_size 703), so e04-vs-e05rand isolates
    # initialisation with nothing else moving.
    "e05rand": dict(
        ckpt_root="/work/hdd/bbnv/kkokate/eb_jepa/e05_random_scaling",
        config="experiments/snr_scaling/config/config_probe_TP_e04.yaml",
        selection=None,      # no incumbent AUC selection JSON exists for this arm
        cells=E05RAND_CELLS,
        merged=RAW_DIR / "e05rand_epoch_curve.json",
        desc="from scratch, depth 22, 400 epochs (4400 steps), 15 ckpts",
    ),
    "e05rand800": dict(
        ckpt_root="/work/hdd/bbnv/kkokate/eb_jepa/e05_random_scaling",
        config="experiments/snr_scaling/config/config_probe_TP_e04.yaml",
        selection=None,
        cells=E05RAND800_CELLS,
        merged=RAW_DIR / "e05rand800_epoch_curve.json",
        desc="from scratch, depth 22, 800 epochs (8800 steps), 31 ckpts",
    ),
}

# Selection set. 80 of R5's 293 recordings, 60 fitting the ridge and 20 scoring
# it, split BY RECORDING so the scored half is unseen subjects. 80 keeps the
# cache near 1.7 GB and one checkpoint near 3 s; the zero-overlap head-fit check
# already showed a 293-recording pool preserves the S-curve shape (r = 0.9989),
# and a selector needs the ranking, not the absolute r.
N_RECORDINGS = 80
N_FIT = 60
SEED = 42
# Held fixed across every cell and epoch. Re-tuning alpha per checkpoint would
# let the curve move because the head's regularisation moved. Measured
# 2026-08-20 by `epoch_curve.py --calibrate-alpha` on e04_s701_a101_nd_d22 at
# epochs 100/200/375: RidgeCV put all 36 feature-draws in 1e3--1e4, median 10^3.5.
ALPHA = 3162.2776601683795

ENV = {
    "WANDB_MODE": "disabled",
    "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/dtyoung/hbn_preprocessed",
    # HBN_TRAIN_RELEASES is deliberately ABSENT. It only rebinds split='train'
    # (hbn.py raises if a train_releases override is applied to another split),
    # and this selector reads split='val' -- R5, fixed. Setting it here would
    # imply the val split could move, which is exactly the confusion that would
    # make someone "fix" the selector onto pretraining data.
}


def arm_cells(arm: str) -> list[str]:
    """Cell list for an arm, resolvable locally without the checkpoint tree.

    e04 reads its selection JSON, which covers all 28 cells. e03's covers only
    S>=400, so that arm carries an explicit list instead -- taking its cells from
    the selection file would silently scan 13 of 28 and look complete.
    """
    spec = ARMS[arm]
    names = spec["cells"]
    if names is None:
        names = list(json.loads(spec["selection"].read_text()))
    return sorted(names, key=lambda n: (int(n.split("_")[1][1:]), n))


def chunk(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items)))
    size, extra = divmod(len(items), n)
    out, i = [], 0
    for c in range(n):
        take = size + (1 if c < extra else 0)
        out.append(items[i:i + take])
        i += take
    return out


def build_command(arm: str, cells: list[str], out_name: str,
                  epochs: list[int] | None) -> str:
    spec = ARMS[arm]
    ep = f" --epochs {' '.join(str(e) for e in epochs)}" if epochs else ""
    return (
        f"mkdir -p {OUT_DIR} && "
        "PYTHONPATH=. uv run --group eeg python "
        "eb_jepa/evaluation/clip_probe/epoch_curve.py "
        f"--ckpt-root {spec['ckpt_root']} --cells {' '.join(cells)} "
        f"--config {spec['config']} --split val --device cuda "
        f"--n-recordings {N_RECORDINGS} --n-fit {N_FIT} --seed {SEED} "
        f"--alpha {ALPHA}{ep} "
        f"--output {OUT_DIR}/{out_name}"
    )


def merge(arm: str) -> int:
    """Combine per-chunk curve files into one artifact."""
    merged_path = ARMS[arm]["merged"]
    parts = sorted(RAW_DIR.glob(f"{arm}_epoch_curve_[0-9]*.json"))
    if not parts:
        print(f"No {arm}_epoch_curve_[0-9]*.json under {RAW_DIR}")
        return 1
    merged, meta = {}, None
    for p in parts:
        d = json.loads(p.read_text())
        if meta is None:
            meta = d["_meta"]
        else:
            # A chunk run with a different selection set is not mergeable: the
            # cells would be scored on different recordings and the argmaxes
            # would not be comparable across chunks.
            for k in ("split", "seed", "n_recordings", "n_fit_recordings", "alpha"):
                if d["_meta"][k] != meta[k]:
                    print(f"  INCOMPATIBLE {p.name}: {k}={d['_meta'][k]!r} "
                          f"expected {meta[k]!r}")
                    return 1
        overlap = set(d["cells"]) & set(merged)
        if overlap:
            print(f"  DUPLICATE cells in {p.name}: {sorted(overlap)}")
            return 1
        merged.update(d["cells"])
    merged_path.write_text(json.dumps({"_meta": meta, "cells": merged}, indent=2))
    n_ck = sum(len(v) for v in merged.values())
    print(f"Merged {len(parts)} part(s) -> {merged_path.name}: "
          f"{len(merged)} cells, {n_ck} checkpoints")
    return 0


def verify(arm: str) -> int:
    """Check the merged curve covers every cell at every checkpoint, cleanly.

    A job that died mid-cell still leaves a well-formed JSON -- epoch_curve.py
    writes after each cell precisely so partial work survives -- so file
    existence proves nothing. Coverage and NaN are what distinguish a finished
    sweep from a truncated one.
    """
    merged_path = ARMS[arm]["merged"]
    if not merged_path.exists():
        print(f"MISSING {merged_path} -- run `merge` first")
        return 1
    d = json.loads(merged_path.read_text())
    meta, cells = d["_meta"], d["cells"]
    bad = 0

    if meta["split"] != "val":
        print(f"  WRONG SPLIT: selected on {meta['split']!r}, must be 'val' (R5)")
        bad += 1
    if meta["n_recordings"] != N_RECORDINGS or meta["n_fit_recordings"] != N_FIT:
        print(f"  WRONG SELECTION SET: {meta['n_recordings']}/{meta['n_fit_recordings']} "
              f"expected {N_RECORDINGS}/{N_FIT}")
        bad += 1

    expected = arm_cells(arm)
    missing = [c for c in expected if c not in cells]
    extra = [c for c in cells if c not in expected]
    if missing:
        print(f"  MISSING {len(missing)} cell(s): {missing}")
        bad += len(missing)
    if extra:
        print(f"  UNEXPECTED cell(s) not in the {arm} cell list: {extra}")
        bad += len(extra)

    n_ck = {len(v) for v in cells.values()}
    if len(n_ck) > 1:
        # Ragged means some cell was cut short: every cell of an arm saves on
        # the same schedule, so a mixed count is a truncated job, not a
        # difference between arms (15 for the 400-epoch arms, 31 for ep800).
        print(f"  RAGGED coverage: cells have {sorted(n_ck)} checkpoints")
        bad += 1
    for cell, curve in cells.items():
        for epoch, rec in curve.items():
            vals = [rec["probe_mean_r"], rec["e2v1_time"], rec["e2v1_scene"]]
            if any(v != v for v in vals):          # NaN
                print(f"  NaN  {cell} ep{epoch}: {vals}")
                bad += 1
    print(f"verify: {len(cells)} cells x {sorted(n_ck)} checkpoints -- "
          f"{'all clean' if not bad else f'{bad} problem(s)'}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="dry",
                    choices=["dry", "submit", "smoke", "merge", "verify"])
    ap.add_argument("sub_action", nargs="?", default="dry",
                    choices=["dry", "submit"],
                    help="For `smoke`: whether to actually sbatch it.")
    ap.add_argument("--arm", default="e04", choices=sorted(ARMS),
                    help="e04 = warm-started depth-22 (default). e03 = "
                         "from-scratch depth-12. The e05 arms are absent and "
                         "cannot be added: R5 is in their pretraining pool, so "
                         "they have no held-out split to select on.")
    ap.add_argument("--partition", default="gpuA40x4")
    ap.add_argument("--time-limit", default="04:00:00")
    ap.add_argument("--chunks", type=int, default=4,
                    help="Parallel jobs. Each rebuilds the recording cache, so "
                         "more chunks is not free. Short chunks backfill better "
                         "on a busy queue: at ~30 s/checkpoint, 4 cells is ~31 min.")
    args = ap.parse_args()

    if args.action == "merge":
        raise SystemExit(merge(args.arm))
    if args.action == "verify":
        raise SystemExit(1 if verify(args.arm) else 0)

    spec = ARMS[args.arm]
    cells = arm_cells(args.arm)
    if args.action == "smoke":
        # One cell, every checkpoint: confirms the per-checkpoint cost and that
        # the curve is smoother than the raw AUC history before 28 cells run.
        groups = [cells[len(cells) // 2:len(cells) // 2 + 1]]
        epochs, action = None, args.sub_action
    else:
        groups = chunk(cells, args.chunks)
        epochs, action = None, args.action

    print(f"arm={args.arm} ({spec['desc']})  cells={len(cells)}  "
          f"chunks={len(groups)}")
    print(f"  selection set = {N_RECORDINGS} R5 recordings "
          f"({N_FIT} fit / {N_RECORDINGS - N_FIT} score), alpha={ALPHA:.0f}")
    print(f"  config: {spec['config']}")
    print(f"  checkpoints: every epoch_*.pth.tar per cell\n")

    for i, group in enumerate(groups):
        tag = "smoke" if args.action == "smoke" else str(i)
        job = Job(
            name=f"epochcurve_{args.arm}_{tag}",
            cluster="delta",
            repo_path=REPO,
            partition=args.partition,
            time_limit=args.time_limit,
            command=build_command(args.arm, group,
                                  f"{args.arm}_epoch_curve_{tag}.json", epochs),
            venv="__none__",
            branch="",
            env_vars=ENV,
        )
        if action != "submit":
            print(f"--- {job.name}  ({len(group)} cell(s)) ---")
            print(job.command, "\n")
        else:
            print(f"submitted {job.name} ({len(group)} cells): {job.submit()}")

    if action != "submit":
        print("Dry run. Re-run with 'submit' to sbatch.")


if __name__ == "__main__":
    main()
