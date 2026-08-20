"""Compare three checkpoint selectors on the e04 arm, per cell.

Reads the per-epoch curve written by
``eb_jepa/evaluation/clip_probe/epoch_curve.py`` (merged by
``submit/_submit_epoch_curve.py merge``) and puts three selection rules side by
side:

  auc     smoothed argmax of ``val/clip_scene_auc`` -- the current rule, from
          ``e04_selection.json``. A 29-window AUC, sd ~0.10, adjacent-epoch
          swings ~0.09, so it needs a window-25 smoother to be usable at all.
  probe   argmax of a 12-feature ridge r fit on 60 held-out R5 recordings and
          scored on 20 more. Low variance, and it measures the quantity the
          paper reports.
  ep325   the paper's fixed epoch.

Three questions, in order of how much they could change the paper:

  1. Do the rules disagree, and by how much? Bounded above by the measured
     protocol spread (mean |delta| 0.0019 r, max 0.0064 over 40 probe cells),
     so this is a check, not a re-report.
  2. Do the 8 cells whose AUC argmax pinned at epoch 399 still pin at the last
     saved checkpoint under the probe curve? If yes, the appendix's "the
     4400-step budget is close to binding at high S" gets a second, independent
     confirmation. If no, that paragraph is resting on AUC noise and needs
     rewriting.
  3. Does the epoch chosen by the probe agree with the epoch chosen by
     retrieval? Retrieval fits nothing and is never selected on, so agreement
     is the evidence that probe-based selection is not merely selecting the
     probe.

Also writes ``e04_selection_probe.json`` in the SAME shape as
``e04_selection.json``, so ``_submit_traintest_e04.py`` and
``_submit_retrieval_e04.py`` can consume it by path with no code change.

Usage (locally, once the curve is merged):
    uv run python experiments/snr_scaling/src/analyse_epoch_curve.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # experiments/snr_scaling
CURVE = ROOT / "raw_results" / "e04_epoch_curve.json"
AUC_SELECTION = ROOT / "e04_selection.json"
OUT_SELECTION = ROOT / "e04_selection_probe.json"
FIXED_EPOCH = 325
# The curve has ~15 points, not 400, and a probe r on 20 held-out recordings is
# not the noise process a 29-window AUC is. Smoothing is therefore OFF by
# default: a window-3 mean over 15 points would blur a real early peak (cells
# select as early as epoch 75) to fix a variance problem this metric does not
# have. --smooth exists to test that assumption, not to be used by default.
SMOOTH_DEFAULT = 1


def s_of(cell: str) -> int:
    return int(cell.split("_")[1][1:])


def draw_of(cell: str) -> str:
    return cell.rsplit("_d", 1)[-1] if "_d" in cell else "--"


def smooth(vals: list[float], window: int) -> list[float]:
    """Centred rolling mean, same convention as select_and_probe_e03."""
    if window <= 1:
        return list(vals)
    half = window // 2
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def argmax_epoch(curve: dict, key: str, window: int) -> tuple[int, float]:
    """(epoch, value) maximising `key` over the curve's saved checkpoints."""
    epochs = sorted(int(e) for e in curve)
    vals = [curve[str(e)][key] for e in epochs]
    sm = smooth(vals, window)
    best_i = max(range(len(sm)), key=lambda i: sm[i])
    return epochs[best_i], vals[best_i]


def nearest(curve: dict, epoch: int) -> int:
    return min((int(e) for e in curve), key=lambda e: abs(e - epoch))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curve", default=str(CURVE))
    ap.add_argument("--smooth", type=int, default=SMOOTH_DEFAULT,
                    help="Rolling-mean window over the epoch curve. Default 1 "
                         "(off) -- see the module docstring.")
    ap.add_argument("--write-selection", action="store_true",
                    help=f"Write {OUT_SELECTION.name} for downstream submitters.")
    args = ap.parse_args()

    curve_doc = json.loads(Path(args.curve).read_text())
    meta, cells = curve_doc["_meta"], curve_doc["cells"]
    auc_sel = json.loads(AUC_SELECTION.read_text())

    print(f"selection set: {meta['n_recordings']} {meta['split']}-split recordings "
          f"({meta['n_fit_recordings']} fit / "
          f"{meta['n_recordings'] - meta['n_fit_recordings']} score), "
          f"alpha={meta['alpha']}, smooth={args.smooth}\n")

    print(f"{'cell':<26}{'S':>6}{'auc':>6}{'prb':>6}{'|d|':>5}"
          f"{'r@prb':>9}{'r@auc':>9}{'r@325':>9}{'prb-325':>9}{'retr_ep':>9}")
    print("-" * 94)

    rows, out_sel = [], {}
    for cell in sorted(cells, key=lambda c: (s_of(c), c)):
        cv = cells[cell]
        probe_ep, probe_r = argmax_epoch(cv, "probe_mean_r", args.smooth)
        retr_ep, _ = argmax_epoch(cv, "e2v1_scene", args.smooth)
        auc_ep = auc_sel.get(cell, {}).get("selected_epoch")
        r_at = lambda e: cv[str(nearest(cv, e))]["probe_mean_r"]
        r_auc = r_at(auc_ep) if auc_ep is not None else float("nan")
        r_325 = r_at(FIXED_EPOCH)
        rows.append({
            "cell": cell, "S": s_of(cell), "auc_ep": auc_ep, "probe_ep": probe_ep,
            "retr_ep": retr_ep, "r_probe": probe_r, "r_auc": r_auc, "r_325": r_325,
            "last_epoch": max(int(e) for e in cv),
        })
        d = abs(probe_ep - auc_ep) if auc_ep is not None else float("nan")
        print(f"{cell:<26}{s_of(cell):>6}{auc_ep if auc_ep is not None else -1:>6}"
              f"{probe_ep:>6}{d:>5.0f}{probe_r:>9.4f}{r_auc:>9.4f}{r_325:>9.4f}"
              f"{probe_r - r_325:>+9.4f}{retr_ep:>9}")

        # Same shape as e04_selection.json. No snap_distance entry because there
        # is nothing to snap: the probe curve is only ever evaluated AT saved
        # checkpoints, so the selected epoch IS a checkpoint by construction.
        out_sel[cell] = {
            "best_epoch": probe_ep,
            "best_value": probe_r,
            "selected_epoch": probe_ep,
            "snap_distance": 0,
            "checkpoint": f"{meta['ckpt_root']}/{cell}/epoch_{probe_ep}.pth.tar",
            "selector": "probe_mean_r",
            "smooth_window": args.smooth,
        }

    # --- 1. do the rules disagree -------------------------------------------
    moved = [r for r in rows if r["auc_ep"] is not None and r["probe_ep"] != r["auc_ep"]]
    d_probe_auc = [abs(r["probe_ep"] - r["auc_ep"]) for r in rows if r["auc_ep"] is not None]
    d_r_325 = [r["r_probe"] - r["r_325"] for r in rows]
    print(f"\n1. AGREEMENT. probe vs auc: {len(rows) - len(moved)}/{len(rows)} cells "
          f"pick the same checkpoint; median |delta epoch| {st.median(d_probe_auc):.0f}, "
          f"max {max(d_probe_auc):.0f}.")
    print(f"   Selection-set r at the probe epoch minus at 325: "
          f"mean {st.mean(d_r_325):+.4f}, max {max(d_r_325):+.4f} "
          f"(on {meta['n_recordings'] - meta['n_fit_recordings']} scoring recordings, "
          f"NOT the reported test number).")

    # --- 2. is the budget binding at high S ---------------------------------
    auc_pinned = {c for c, v in auc_sel.items() if v.get("best_epoch") == 399}
    probe_pinned = {r["cell"] for r in rows if r["probe_ep"] == r["last_epoch"]}
    both = auc_pinned & probe_pinned
    print(f"\n2. BUDGET. AUC pinned at the last epoch on {len(auc_pinned)} cells; "
          f"the probe curve pins on {len(probe_pinned)}; {len(both)} agree.")
    hi = [r for r in rows if r["S"] >= 1000]
    hi_pinned = [r for r in hi if r["cell"] in probe_pinned]
    print(f"   Among S>=1000: {len(hi_pinned)}/{len(hi)} pin under the probe curve "
          f"({sorted(r['cell'] for r in hi_pinned)}).")
    if not both and auc_pinned:
        print("   NOTE: no overlap -- the appendix's 'budget close to binding at "
              "high S' claim rests on AUC noise and needs rewriting.")

    # --- 3. is probe selection self-serving ---------------------------------
    d_pr = [abs(r["probe_ep"] - r["retr_ep"]) for r in rows]
    agree = sum(1 for d in d_pr if d == 0)
    print(f"\n3. CROSS-CHECK. probe vs scene-retrieval argmax: {agree}/{len(rows)} "
          f"identical, median |delta epoch| {st.median(d_pr):.0f}, max {max(d_pr):.0f}. "
          "Retrieval fits nothing and is never selected on.")

    by_s: dict[int, list] = {}
    for r in rows:
        by_s.setdefault(r["S"], []).append(r)
    print(f"\n{'S':>6}{'n':>4}{'probe epochs':>28}{'mean r@prb':>12}{'mean r@325':>12}")
    for s in sorted(by_s):
        g = by_s[s]
        print(f"{s:>6}{len(g):>4}{str(sorted(r['probe_ep'] for r in g)):>28}"
              f"{st.mean(r['r_probe'] for r in g):>12.4f}"
              f"{st.mean(r['r_325'] for r in g):>12.4f}")

    if args.write_selection:
        OUT_SELECTION.write_text(json.dumps(out_sel, indent=2))
        print(f"\nWrote {OUT_SELECTION} ({len(out_sel)} cells)")


if __name__ == "__main__":
    main()
