"""Early-stopped E0.3: pick each cell's best epoch, probe THAT checkpoint.

Fixed-budget comparison across data scales is invalid, because the optimal
stopping point is itself a function of dataset size (Hestness et al. 2017;
Kaplan et al. 2020 model finite-data overfitting explicitly). The first E0.3 run
probed every cell's final checkpoint and, measured on the in-loop val
diagnostic, every cell was 13-40 % past its peak -- worse the less data it had,
which is exactly the bias that inflates a scaling exponent. See RESULTS.md 2.10.

This script implements the standard fix:

  1. read the per-epoch ``val/clip_scene_auc`` history from the offline wandb
     file (parsed directly -- no wandb server needed),
  2. take each cell's argmax epoch,
  3. snap to the nearest saved ``epoch_N.pth.tar``,
  4. probe that checkpoint on the FULL val set.

**Selection-bias note, stated rather than hidden.** The selection metric is the
in-loop diagnostic, computed on a 10 % subset of val
(``eval.val_recording_fraction``); the reported metric is the probe over all of
val. These are different metrics on overlapping data, so this is early stopping
on validation -- standard, but not free. The clean version selects on val and
reports on test; worth doing before the number goes in a paper.

    uv run --group eeg python experiments/snr_scaling/src/select_and_probe_e03.py \
        --suffix=_es --probe
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling"
OUT_DIR = Path("experiments/snr_scaling/raw_results")   # jobs write measured JSONs straight here   # repo-relative; jobs run from the repo root
SELECT_KEY = "val/clip_scene_auc"
# Rolling window for smoothing the selection metric, ~ the checkpoint spacing.
SMOOTH_WINDOW = 25


def _rolling_mean(v: list[float], w: int) -> list[float]:
    """Centred rolling mean; edges shrink the window rather than pad."""
    import statistics as _st
    return [_st.fmean(v[max(0, i - w // 2):min(len(v), i + w // 2 + 1)])
            for i in range(len(v))]

CELLS = [
    (701, 101), (400, 101), (200, 101), (100, 101), (50, 101),
    (701, 50), (701, 25), (701, 13),
    (400, 50), (200, 25), (100, 13),
]

# Extended-cohort S axis (R7-R10). 400 and 701 overlap the R1-R4 curve on
# purpose -- they are the calibration points that decide whether the two
# segments can be plotted as one curve.
EXTENDED_CELLS = [
    (400, 101), (701, 101), (1000, 101), (1400, 101), (1863, 101),
]


def read_history(wandb_file: str, key: str) -> list[float]:
    """Per-step values of *key* from an offline .wandb datastore.

    Parsed directly because these runs may never have synced, and because a
    server round-trip should not be a dependency of an offline analysis. Note
    this wandb version stores the metric name in ``nested_key``, not ``key`` --
    reading ``item.key`` yields empty strings and silently finds nothing.
    """
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)
    vals: list[float] = []
    while True:
        try:
            raw = ds.scan_data()
        except Exception:
            break
        if raw is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(raw)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        for item in rec.history.item:
            name = item.key or (item.nested_key[0] if item.nested_key else None)
            if name == key:
                try:
                    vals.append(json.loads(item.value_json))
                except Exception:
                    pass
    return vals


def select_epoch(cell_dir: Path, key: str) -> dict:
    """Best epoch for one cell, from the RICHEST wandb run it has.

    A cell can hold several run directories -- a crashed attempt leaves an ~8 KB
    stub beside the real ~9.5 MB run. ``glob`` returns them in arbitrary order,
    so taking ``files[0]`` picks the stub roughly half the time. That is not
    merely a failure mode: a partially-written run would yield a plausible but
    WRONG best epoch, and the wrong checkpoint would then be probed and reported
    as the cell's early-stopped result.

    Choosing by "most values recorded for the selection key" is robust to both
    stubs and glob ordering, and needs no assumption about filenames or mtimes.
    """
    files = glob.glob(str(cell_dir / "wandb" / "run-*" / "run-*.wandb"))
    if not files:
        return {"error": "no wandb file"}
    histories = [(f, read_history(f, key)) for f in files]
    best_file, vals = max(histories, key=lambda t: len(t[1]))
    if not vals:
        return {"error": f"no '{key}' in history (checked {len(files)} run dir(s))"}
    skipped = [Path(f).parent.name for f, v in histories if f != best_file]
    if skipped:
        logger_msg = (f"  [{cell_dir.name}] using {Path(best_file).parent.name} "
                      f"({len(vals)} points); ignored {skipped}")
        print(logger_msg)
    # SMOOTH before argmax. val/clip_scene_auc is an AUC over 29 windows
    # (eval.val_recording_fraction=0.1), with sd ~0.10 and adjacent-epoch swings
    # of ~0.09 -- it is close to pure noise around a slow trend. argmax over 400
    # such samples selects a ~3 sd spike at an essentially arbitrary epoch: it
    # picked epoch 21 for s1000_d22, whose probe then scored 0.0096, near
    # random, because that checkpoint is genuinely undertrained.
    #
    # It also manufactured RESULTS.md 2.10's "overfitting worsens as data
    # shrinks": with equal noise across cells, the same absolute spike is a
    # larger FRACTION of a lower-scoring cell's value, so the apparent drop
    # correlated with score rather than with overfitting. Smoothed, the ordering
    # reverses.
    sm = _rolling_mean(vals, SMOOTH_WINDOW)
    best = max(sm)
    best_ep = sm.index(best)
    raw_best = max(vals)

    saved = sorted(
        (int(p.stem.split("_")[1].split(".")[0]), p)
        for p in cell_dir.glob("epoch_*.pth.tar")
    )
    if not saved:
        # Only latest.pth.tar exists -- the endpoint protocol. Say so loudly
        # rather than silently probing the final checkpoint again.
        return {"error": "no periodic checkpoints; re-run with --save-every",
                "best_epoch": best_ep, "best_value": best}
    nearest_ep, nearest_path = min(saved, key=lambda t: abs(t[0] - best_ep))
    return {
        "best_epoch": best_ep,
        "best_value": best,                 # smoothed
        "raw_best_value": raw_best,         # noisy argmax, for comparison only
        "final_value": sm[-1],              # smoothed, comparable to best
        "raw_final_value": vals[-1],
        "n_epochs_logged": len(vals),
        "smooth_window": SMOOTH_WINDOW,
        "drop_pct": 100.0 * (best - sm[-1]) / best if best else 0.0,
        "selected_epoch": nearest_ep,
        "checkpoint": str(nearest_path),
        "snap_distance": abs(nearest_ep - best_ep),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suffix", default="_es")
    p.add_argument("--extended", action="store_true",
                   help="Use the extended-cohort S-axis cell list.")
    p.add_argument("--cells-glob", default=None,
                   help="Discover cells by directory glob instead of the cell "
                        "lists, e.g. 'e03_s*_a101_nd*'. Needed for replicate "
                        "sweeps where the slug carries a draw seed.")
    p.add_argument("--key", default=SELECT_KEY)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cells whose probe output already exists. Makes a "
                        "broad --cells-glob idempotent: the glob that matches "
                        "new cells also matches already-probed ones, and "
                        "re-probing those is pure waste.")
    p.add_argument("--probe", action="store_true",
                   help="Actually run the probes (otherwise select and report).")
    p.add_argument("--output", default="experiments/snr_scaling/e03_selection.json")
    args = p.parse_args()

    if args.cells_glob:
        # Discover slugs from disk. Enumerating them instead would mean keeping
        # two lists in sync (here and in _submit_e03.py) and would silently skip
        # a cell that failed to produce a directory -- the glob makes a missing
        # cell visible as a missing row.
        slugs = sorted(
            (Path(d).name for d in glob.glob(f"{CKPT_ROOT}/{args.cells_glob}")),
            key=lambda n: (int(re.search(r"_s(\d+)_", n).group(1)), n),
        )
        if not slugs:
            sys.exit(f"No cell directories matched {args.cells_glob!r}")
    else:
        cells = EXTENDED_CELLS if args.extended else CELLS
        slugs = [f"e03_s{s_}_a{a_}{args.suffix}" for s_, a_ in cells]
    sel = {}
    print(f"selection metric: {args.key}\n")
    print(f"  {'cell':<26}{'best_ep':>8}{'best':>8}{'final':>8}{'drop%':>7}"
          f"{'sel_ep':>8}{'snap':>6}")
    for slug in slugs:
        info = select_epoch(Path(CKPT_ROOT) / slug, args.key)
        sel[slug] = info
        if "error" in info and "selected_epoch" not in info:
            print(f"  {slug:<26}  {info['error']}")
            continue
        print(f"  {slug:<26}{info['best_epoch']:>8}{info['best_value']:>8.4f}"
              f"{info['final_value']:>8.4f}{info['drop_pct']:>7.1f}"
              f"{info['selected_epoch']:>8}{info['snap_distance']:>6}")

    Path(args.output).write_text(json.dumps(sel, indent=2))
    print(f"\nWrote {args.output}")

    if not args.probe:
        print("Selection only. Re-run with --probe to probe the chosen checkpoints.")
        return

    for slug in slugs:
        info = sel[slug]
        if "checkpoint" not in info:
            print(f"SKIP {slug}: {info.get('error')}")
            continue
        out = OUT_DIR / f"e03_probe_val_{slug}_best.json"
        if args.skip_existing and out.exists():
            print(f"SKIP {slug}: {out.name} already exists")
            continue
        cmd = [
            "uv", "run", "--group", "eeg", "python",
            "eb_jepa/evaluation/clip_probe/probe.py",
            "--checkpoint", info["checkpoint"],
            "--config", f"{CKPT_ROOT}/{slug}/config_probe.yaml",
            "--split", "val", "--cv-splits", "5",
            "--output", str(out),
        ]
        print(f"\n=== probing {slug} @ epoch {info['selected_epoch']} ===")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"PROBE FAILED for {slug} (rc={r.returncode})", file=sys.stderr)


if __name__ == "__main__":
    main()
