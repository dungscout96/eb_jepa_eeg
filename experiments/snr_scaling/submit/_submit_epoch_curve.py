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

E04 ONLY, AND THE REASON IS NOT CONVENIENCE. The selector reads the val split
(R5), which is disjoint from every e04 cohort because e04 pretrains on
[R1..R4, R7..R10]. The e05_addval and e05_fromscratch arms have R5 IN their
pretraining pool, so for them this measures training-set fit, not selection.
Those arms have no held-out selection data at all and stay pinned to a fixed
epoch -- the same constraint config_probe_TP_headfit_R5.yaml documents for the
zero-overlap head fit. There is deliberately no --arm flag.

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py dry
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py submit --chunks 4
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py smoke submit
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py merge
    uv run --group eeg python experiments/snr_scaling/submit/_submit_epoch_curve.py verify
"""
import argparse
import json
from pathlib import Path

from neurolab.jobs import Job

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # experiments/snr_scaling
RAW_DIR = ROOT / "raw_results"

REPO = "/u/dtyoung/eb_jepa_eeg"
CKPT_ROOT = "/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling"
OUT_DIR = "experiments/snr_scaling/raw_results"
CONFIG = "experiments/snr_scaling/config/config_probe_TP_e04.yaml"
SELECTION = ROOT / "e04_selection.json"
MERGED = RAW_DIR / "e04_epoch_curve.json"

# The 28-cell sweep, same glob and same exclusions as src/select_e04.py -- the
# two selectors must be computed over the same cells or the comparison is
# between different populations, not between different metrics.
CELLS_GLOB = "e04_s*_a101_nd*"

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


def cells_from_selection() -> list[str]:
    """Cell list from e04_selection.json, so `verify` and `merge` can run
    locally without the checkpoint tree mounted. The jobs themselves glob
    CKPT_ROOT on Delta; these must agree, and `verify` checks that they do."""
    sel = json.loads(SELECTION.read_text())
    return sorted(sel, key=lambda n: (int(n.split("_")[1][1:]), n))


def chunk(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items)))
    size, extra = divmod(len(items), n)
    out, i = [], 0
    for c in range(n):
        take = size + (1 if c < extra else 0)
        out.append(items[i:i + take])
        i += take
    return out


def build_command(cells: list[str], out_name: str, epochs: list[int] | None) -> str:
    ep = f" --epochs {' '.join(str(e) for e in epochs)}" if epochs else ""
    return (
        f"mkdir -p {OUT_DIR} && "
        "PYTHONPATH=. uv run --group eeg python "
        "eb_jepa/evaluation/clip_probe/epoch_curve.py "
        f"--ckpt-root {CKPT_ROOT} --cells {' '.join(cells)} "
        f"--config {CONFIG} --split val --device cuda "
        f"--n-recordings {N_RECORDINGS} --n-fit {N_FIT} --seed {SEED} "
        f"--alpha {ALPHA}{ep} "
        f"--output {OUT_DIR}/{out_name}"
    )


def merge() -> int:
    """Combine per-chunk curve files into one artifact."""
    parts = sorted(RAW_DIR.glob("e04_epoch_curve_*.json"))
    if not parts:
        print(f"No e04_epoch_curve_*.json under {RAW_DIR}")
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
    MERGED.write_text(json.dumps({"_meta": meta, "cells": merged}, indent=2))
    n_ck = sum(len(v) for v in merged.values())
    print(f"Merged {len(parts)} part(s) -> {MERGED.name}: "
          f"{len(merged)} cells, {n_ck} checkpoints")
    return 0


def verify() -> int:
    """Check the merged curve covers every cell at every checkpoint, cleanly.

    A job that died mid-cell still leaves a well-formed JSON -- epoch_curve.py
    writes after each cell precisely so partial work survives -- so file
    existence proves nothing. Coverage and NaN are what distinguish a finished
    sweep from a truncated one.
    """
    if not MERGED.exists():
        print(f"MISSING {MERGED} -- run `merge` first")
        return 1
    d = json.loads(MERGED.read_text())
    meta, cells = d["_meta"], d["cells"]
    bad = 0

    if meta["split"] != "val":
        print(f"  WRONG SPLIT: selected on {meta['split']!r}, must be 'val' (R5)")
        bad += 1
    if meta["n_recordings"] != N_RECORDINGS or meta["n_fit_recordings"] != N_FIT:
        print(f"  WRONG SELECTION SET: {meta['n_recordings']}/{meta['n_fit_recordings']} "
              f"expected {N_RECORDINGS}/{N_FIT}")
        bad += 1

    expected = cells_from_selection()
    missing = [c for c in expected if c not in cells]
    extra = [c for c in cells if c not in expected]
    if missing:
        print(f"  MISSING {len(missing)} cell(s): {missing}")
        bad += len(missing)
    if extra:
        print(f"  UNEXPECTED cell(s) not in e04_selection.json: {extra}")
        bad += len(extra)

    n_ck = {len(v) for v in cells.values()}
    if len(n_ck) > 1:
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
    ap.add_argument("--partition", default="gpuA40x4")
    ap.add_argument("--time-limit", default="04:00:00")
    ap.add_argument("--chunks", type=int, default=4,
                    help="Parallel jobs. Each rebuilds the recording cache, so "
                         "more chunks is not free -- 4 is a reasonable trade "
                         "against a 4 h walltime.")
    args = ap.parse_args()

    if args.action == "merge":
        raise SystemExit(merge())
    if args.action == "verify":
        raise SystemExit(1 if verify() else 0)

    cells = cells_from_selection()
    if args.action == "smoke":
        # One cell, every checkpoint: confirms the per-checkpoint cost and that
        # the curve is smoother than the raw AUC history before 28 cells run.
        groups = [cells[len(cells) // 2:len(cells) // 2 + 1]]
        epochs, action = None, args.sub_action
    else:
        groups = chunk(cells, args.chunks)
        epochs, action = None, args.action

    print(f"cells={len(cells)}  chunks={len(groups)}  "
          f"selection set = {N_RECORDINGS} R5 recordings "
          f"({N_FIT} fit / {N_RECORDINGS - N_FIT} score), alpha={ALPHA}")
    print(f"  checkpoints: every epoch_*.pth.tar per cell\n")

    for i, group in enumerate(groups):
        name = "epochcurve_smoke" if args.action == "smoke" else f"epochcurve_{i}"
        out_name = (f"e04_epoch_curve_smoke.json" if args.action == "smoke"
                    else f"e04_epoch_curve_{i}.json")
        job = Job(
            name=name,
            cluster="delta",
            repo_path=REPO,
            partition=args.partition,
            time_limit=args.time_limit,
            command=build_command(group, out_name, epochs),
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
