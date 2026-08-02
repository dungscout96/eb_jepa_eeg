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

    uv run --group eeg python experiments/snr_scaling/select_and_probe_e03.py \
        --suffix=_es --probe
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling"
OUT_DIR = Path("experiments/snr_scaling")
SELECT_KEY = "val/clip_scene_auc"

CELLS = [
    (701, 101), (400, 101), (200, 101), (100, 101), (50, 101),
    (701, 50), (701, 25), (701, 13),
    (400, 50), (200, 25), (100, 13),
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
    files = glob.glob(str(cell_dir / "wandb" / "run-*" / "run-*.wandb"))
    if not files:
        return {"error": "no wandb file"}
    vals = read_history(files[0], key)
    if not vals:
        return {"error": f"no '{key}' in history"}
    best = max(vals)
    best_ep = vals.index(best)

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
        "best_value": best,
        "final_value": vals[-1],
        "n_epochs_logged": len(vals),
        "drop_pct": 100.0 * (best - vals[-1]) / best if best else 0.0,
        "selected_epoch": nearest_ep,
        "checkpoint": str(nearest_path),
        "snap_distance": abs(nearest_ep - best_ep),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suffix", default="_es")
    p.add_argument("--key", default=SELECT_KEY)
    p.add_argument("--probe", action="store_true",
                   help="Actually run the probes (otherwise select and report).")
    p.add_argument("--output", default="experiments/snr_scaling/e03_selection.json")
    args = p.parse_args()

    sel = {}
    print(f"selection metric: {args.key}\n")
    print(f"  {'cell':<16}{'best_ep':>8}{'best':>8}{'final':>8}{'drop%':>7}"
          f"{'sel_ep':>8}{'snap':>6}")
    for s, a in CELLS:
        slug = f"e03_s{s}_a{a}{args.suffix}"
        info = select_epoch(Path(CKPT_ROOT) / slug, args.key)
        sel[slug] = info
        if "error" in info and "selected_epoch" not in info:
            print(f"  {slug:<16}  {info['error']}")
            continue
        print(f"  {slug:<16}{info['best_epoch']:>8}{info['best_value']:>8.4f}"
              f"{info['final_value']:>8.4f}{info['drop_pct']:>7.1f}"
              f"{info['selected_epoch']:>8}{info['snap_distance']:>6}")

    Path(args.output).write_text(json.dumps(sel, indent=2))
    print(f"\nWrote {args.output}")

    if not args.probe:
        print("Selection only. Re-run with --probe to probe the chosen checkpoints.")
        return

    for s, a in CELLS:
        slug = f"e03_s{s}_a{a}{args.suffix}"
        info = sel[slug]
        if "checkpoint" not in info:
            print(f"SKIP {slug}: {info.get('error')}")
            continue
        out = OUT_DIR / f"e03_probe_val_{slug}_best.json"
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
