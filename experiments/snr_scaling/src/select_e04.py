"""Pick each e04_reve_scaling (depth-22) cell's best epoch, by the same
smoothed-selection protocol as select_and_probe_e03.py's ``select_epoch``.

Depth-12 (e03_scaling) and depth-22 (e04_reve_scaling, kkokate's checkpoints)
are compared in figures/depth-12-vs-22-subject-scaling.pdf. Reusing the exact
same selection function (read wandb history for ``val/clip_scene_auc``, smooth
with a centred rolling mean, argmax, snap to the nearest saved checkpoint)
means both arms are early-stopped the same way instead of one arm using a
hand-picked epoch.

Unlike select_and_probe_e03.py this script does not also run probes -- it only
selects and writes the JSON. Downstream probing/retrieval is submitted to the
cluster by submit/_submit_traintest_e04.py and submit/_submit_retrieval_e04.py,
which read this script's output to pick each cell's checkpoint.

Must run where the checkpoint directories are mounted (Delta), since
``select_epoch`` parses offline .wandb datastore files directly -- no wandb
server needed, but the files must be on local disk.

Usage (on Delta, from the repo root):
    uv run --group eeg python experiments/snr_scaling/src/select_e04.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_and_probe_e03 import select_epoch  # reuse the exact selection logic

CKPT_ROOT = "/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling"
SELECT_KEY = "val/clip_scene_auc"
# Matches every S x draw cell plus the full-pool e04_s1863_a101_nd cell.
# Deliberately excludes *_FAILED_*, *_smoke*, and s1863_a101_seed7 (a separate
# alternate-seed replicate, not part of the main S sweep).
CELLS_GLOB = "e04_s*_a101_nd*"
OUTPUT = "experiments/snr_scaling/e04_selection.json"


def main() -> None:
    slugs = sorted(
        (Path(d).name for d in glob.glob(f"{CKPT_ROOT}/{CELLS_GLOB}")
         if "FAILED" not in d and "smoke" not in d),
        key=lambda n: (int(re.search(r"_s(\d+)_", n).group(1)), n),
    )
    if not slugs:
        sys.exit(f"No cell directories matched {CELLS_GLOB!r} under {CKPT_ROOT}")

    sel: dict[str, dict] = {}
    print(f"selection metric: {SELECT_KEY}")
    print(f"{len(slugs)} cell(s) matched\n")
    print(f"  {'cell':<28}{'best_ep':>8}{'best':>8}{'final':>8}{'drop%':>7}"
          f"{'sel_ep':>8}{'snap':>6}")
    for slug in slugs:
        info = select_epoch(Path(CKPT_ROOT) / slug, SELECT_KEY)
        sel[slug] = info
        if "error" in info and "selected_epoch" not in info:
            print(f"  {slug:<28}  {info['error']}")
            continue
        print(f"  {slug:<28}{info['best_epoch']:>8}{info['best_value']:>8.4f}"
              f"{info['final_value']:>8.4f}{info['drop_pct']:>7.1f}"
              f"{info['selected_epoch']:>8}{info['snap_distance']:>6}")

    Path(OUTPUT).write_text(json.dumps(sel, indent=2))
    n_ok = sum(1 for v in sel.values() if "selected_epoch" in v)
    print(f"\nWrote {OUTPUT} ({n_ok}/{len(slugs)} cells selected)")


if __name__ == "__main__":
    main()
