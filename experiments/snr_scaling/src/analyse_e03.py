"""E0.3 analysis: the r(A, S) data-scaling surface.

Reads the 11 cell probes plus the matched random-init baseline and reports
what PLAN.md E0.3 asks for -- fitted exponents on each axis, local exponents at
the full-data corner, and a separability check against the diagonal.

Metric is **delta r-squared above a random encoder of identical shape**, not raw
r-squared. Raw r-squared carries a large constant that any encoder attains, and
a constant offset flattens a log-log slope toward zero: fitting exponents on raw
r-squared would understate BOTH axes and, worse, understate them unequally.

    uv run --group eeg python experiments/snr_scaling/analyse_e03.py
"""

import argparse
import glob
import json
import math
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiments/snr_scaling/src
ROOT = HERE.parent                              # experiments/snr_scaling
RAW = ROOT / "raw_results"                      # measured JSONs
FIGS = ROOT / "figures"                         # rendered figures

FULL_S, FULL_A = 701, 101


def _mean_r2(path: Path) -> tuple[float, dict]:
    d = json.loads(path.read_text())
    feats = d["features"]
    names = sorted(feats)
    per = {n: feats[n]["r2_mean"] for n in names}
    return st.fmean(per.values()), per


def load(suffix: str = "") -> tuple[dict, float, dict]:
    """Load one protocol's cells.

    ``suffix=""`` is the original fixed-budget run (final checkpoint);
    ``suffix="_es_best"`` is the early-stopped run (each cell's selected
    checkpoint). The random baseline is shared -- it is an untrained encoder, so
    it has no stopping point.
    """
    rand_mean, rand_per = _mean_r2(RAW / "e03_probe_val_random.json")
    cells = {}
    pat = f"e03_probe_val_e03_s*_a*{suffix}.json" if suffix else "e03_probe_val_e03_s*_a*.json"
    for f in glob.glob(str(RAW / pat)):
        if not suffix and ("_es" in Path(f).stem):
            continue          # do not mix protocols
        if suffix and not Path(f).stem.endswith(suffix):
            continue
        m = re.search(r"_s(\d+)_a(\d+)", f)
        s, a = int(m.group(1)), int(m.group(2))
        mean, per = _mean_r2(Path(f))
        d = json.loads(Path(f).read_text())
        cells[(s, a)] = {
            "r2": mean,
            "d_r2": mean - rand_mean,
            "per": per,
            "n_rec": d["n_recordings"],
            "n_win": d["n_windows_total"],
        }
    return cells, rand_mean, rand_per


def loglog_slope(xs, ys) -> float:
    """Least-squares exponent b in y ~ x^b. Both axes strictly positive."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = st.fmean(lx), st.fmean(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def local_slope(x0, y0, x1, y1) -> float:
    return math.log(y1 / y0) / math.log(x1 / x0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="",
                    help='"" = fixed-budget final checkpoint; "_es_best" = '
                         "early-stopped selected checkpoint.")
    ap.add_argument("--compare-to", default=None,
                    help="Second suffix to diff against, e.g. \"\" .")
    args = ap.parse_args()
    cells, rand_mean, _ = load(args.suffix)
    print(f"protocol: {args.suffix or 'fixed-budget (final checkpoint)'}\n")

    n_rec = {c["n_rec"] for c in cells.values()}
    n_win = {c["n_win"] for c in cells.values()}
    assert len(n_rec) == 1 and len(n_win) == 1, (
        f"cells were evaluated on different data ({n_rec}, {n_win}) -- the "
        "surface is not comparable")
    print(f"Random-init baseline (identical shape): mean r2 = {rand_mean:.5f}")
    print(f"Every cell evaluated on {n_rec.pop()} val recordings / "
          f"{n_win.pop()} windows -- identical, as E0.3 requires.\n")

    s_axis = [50, 100, 200, 400, FULL_S]
    a_axis = [13, 25, 50, FULL_A]

    print("S axis (anchors fixed at A=101)")
    print(f"  {'S':>5}{'r2':>10}{'d_r2':>10}   local exponent")
    prev = None
    for s in s_axis:
        c = cells[(s, FULL_A)]
        loc = ("" if prev is None else
               f"{local_slope(prev[0], prev[1], s, c['d_r2']):+.3f}")
        print(f"  {s:>5}{c['r2']:>10.5f}{c['d_r2']:>10.5f}   {loc}")
        prev = (s, c["d_r2"])
    b_s = loglog_slope(s_axis, [cells[(s, FULL_A)]["d_r2"] for s in s_axis])
    print(f"  fitted exponent  d log dr2 / d log S = {b_s:+.3f}\n")

    print("A axis (subjects fixed at S=701)")
    print(f"  {'A':>5}{'r2':>10}{'d_r2':>10}   local exponent")
    prev = None
    for a in a_axis:
        c = cells[(FULL_S, a)]
        loc = ("" if prev is None else
               f"{local_slope(prev[0], prev[1], a, c['d_r2']):+.3f}")
        print(f"  {a:>5}{c['r2']:>10.5f}{c['d_r2']:>10.5f}   {loc}")
        prev = (a, c["d_r2"])
    b_a = loglog_slope(a_axis, [cells[(FULL_S, a)]["d_r2"] for a in a_axis])
    print(f"  fitted exponent  d log dr2 / d log A = {b_a:+.3f}\n")

    # The number the paper turns on: the slope AT the operating point, not the
    # slope averaged over a range that includes the cheap early gains.
    corner = cells[(FULL_S, FULL_A)]["d_r2"]
    s_loc = local_slope(400, cells[(400, FULL_A)]["d_r2"], FULL_S, corner)
    a_loc = local_slope(50, cells[(FULL_S, 50)]["d_r2"], FULL_A, corner)
    print("At the full-data corner (the operating point):")
    print(f"  local exponent in S (400 -> 701) = {s_loc:+.3f}")
    print(f"  local exponent in A ( 50 -> 101) = {a_loc:+.3f}")
    print(f"  ratio S:A = {s_loc / a_loc:.2f}x\n")

    print("Separability: does dr2 ~ S^b_s * A^b_a predict the diagonal?")
    print(f"  {'cell':>12}{'measured':>11}{'predicted':>11}{'ratio':>8}")
    k = corner / (FULL_S ** b_s * FULL_A ** b_a)
    for s, a in [(100, 13), (200, 25), (400, 50), (FULL_S, FULL_A)]:
        meas = cells[(s, a)]["d_r2"]
        pred = k * (s ** b_s) * (a ** b_a)
        print(f"  {'S%d A%d' % (s, a):>12}{meas:>11.5f}{pred:>11.5f}"
              f"{meas / pred:>8.2f}")

    # The cleanest statement available: hold the (subjects x anchors) budget
    # roughly fixed and ask which axis to spend it on. This needs no fitted
    # model at all -- it is a direct comparison of measured cells.
    print("\nIso-budget: same number of (subject x anchor) pairs, spent differently")
    print(f"  {'subject-heavy':>18}{'pairs':>8}{'d_r2':>9}   "
          f"{'anchor-heavy':>16}{'pairs':>8}{'d_r2':>9}   gain")
    iso = [((200, 25), (50, FULL_A)),
           ((FULL_S, 13), (100, FULL_A)),
           ((400, 50), (200, FULL_A)),
           ((FULL_S, 50), (400, FULL_A))]
    gains = []
    for (s1, a1), (s2, a2) in iso:
        d1, d2 = cells[(s1, a1)]["d_r2"], cells[(s2, a2)]["d_r2"]
        p1, p2 = s1 * a1, s2 * a2
        gains.append(d1 / d2)
        print(f"  {'S%d A%d' % (s1, a1):>18}{p1:>8}{d1:>9.5f}   "
              f"{'S%d A%d' % (s2, a2):>16}{p2:>8}{d2:>9.5f}   {d1 / d2:>4.2f}x")
    print(f"  subject-heavy wins all {len(iso)}/{len(iso)} comparisons, "
          f"median {st.median(gains):.2f}x")

    out = {
        "random_baseline_mean_r2": rand_mean,
        "iso_budget": [
            {"subject_heavy": f"S{s1}_A{a1}", "anchor_heavy": f"S{s2}_A{a2}",
             "pairs_subject_heavy": s1 * a1, "pairs_anchor_heavy": s2 * a2,
             "d_r2_subject_heavy": cells[(s1, a1)]["d_r2"],
             "d_r2_anchor_heavy": cells[(s2, a2)]["d_r2"],
             "gain": cells[(s1, a1)]["d_r2"] / cells[(s2, a2)]["d_r2"]}
            for (s1, a1), (s2, a2) in iso
        ],
        "exponent_S": b_s,
        "exponent_A": b_a,
        "local_exponent_S_at_corner": s_loc,
        "local_exponent_A_at_corner": a_loc,
        "cells": {f"S{s}_A{a}": {"r2": c["r2"], "d_r2": c["d_r2"]}
                  for (s, a), c in sorted(cells.items())},
    }
    name = f"e03_surface{args.suffix or ''}.json"
    (RAW / name).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RAW / name}")

    if args.compare_to is not None:
        other, _, _ = load(args.compare_to)
        lbl = args.compare_to or "fixed-budget"
        print(f"\nProtocol comparison vs {lbl}:")
        print(f"  {'cell':>12}{'this':>10}{'other':>10}{'ratio':>8}")
        for k in sorted(cells, key=lambda t: (-t[1], -t[0])):
            if k in other:
                a_, b_ = cells[k]["d_r2"], other[k]["d_r2"]
                print(f"  {'S%d A%d' % k:>12}{a_:>10.5f}{b_:>10.5f}"
                      f"{a_ / b_:>8.2f}")


if __name__ == "__main__":
    main()
