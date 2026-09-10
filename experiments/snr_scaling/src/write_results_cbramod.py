"""Generate RESULTS_cbramod.md from the two E1.2 aggregates.

Reads the machine-readable output of ``aggregate_nested.py --output`` for the
warm (``e12cb``) and random-init (``e12cbr``) arms and renders every table from
those numbers -- nothing is hand-transcribed, so a re-run of the sweep only
needs a re-run of this script.

The two questions the document answers are fixed in advance
(PLAN_cbramod.md, "What counts as reproducing the main results"):

  1. LEVEL: warm / random Delta-r^2 at matched S, with the draw spread.
  2. SLOPE: fitted and local subject-scaling exponents per arm, in particular
     the 701 -> 1863 step.

Usage:
    uv run --group eeg python experiments/snr_scaling/src/write_results_cbramod.py \\
        --warm raw_results/e12cb_nested.json --random raw_results/e12cbr_nested.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # experiments/snr_scaling
RAW = ROOT / "raw_results"
OUT = ROOT / "RESULTS_cbramod.md"

TOPKS = ("1", "5", "10")


def _f(x, nd=5) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _pm(g: dict | None, nd=5) -> str:
    """mean ± sd (n) for a by_S group, or a dash when the arm has no cell."""
    if not g or g.get("mean") is None:
        return "—"
    n = g.get("n", 0)
    if n > 1 and g.get("sd") is not None:
        return f"{g['mean']:.{nd}f} ± {g['sd']:.{nd}f} (n={n})"
    return f"{g['mean']:.{nd}f} (n={n})"


def _ratio(a: dict | None, b: dict | None) -> float | None:
    if not a or not b or a.get("mean") is None or not b.get("mean"):
        return None
    return a["mean"] / b["mean"]


def _local_exponent(s0: int, y0: float | None, s1: int, y1: float | None):
    if y0 is None or y1 is None or y0 <= 0 or y1 <= 0:
        return None
    return math.log(y1 / y0) / math.log(s1 / s0)


def _loglog_slope(xs: list[float], ys: list[float]) -> float | None:
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y and y > 0]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return num / den if den else None


def s_axis(*aggs: dict) -> list[int]:
    ss = set()
    for a in aggs:
        ss.update(int(s) for s in a.get("by_S", {}))
    return sorted(ss)


def group(agg: dict, s: int, key: str) -> dict | None:
    return (agg.get("by_S", {}).get(str(s)) or {}).get(key)


def retr_group(agg: dict, s: int, k: str) -> dict | None:
    return ((agg.get("by_S", {}).get(str(s)) or {}).get("retr_test_scene") or {}).get(
        f"top{k}"
    )


def render(warm: dict, rand: dict, today: str | None = None) -> str:
    today = today or date.today().isoformat()
    axis = s_axis(warm, rand)
    lines: list[str] = []
    w = lines.append

    w("# RESULTS_cbramod — E1.2: an HBN-free warm start on the subject-scaling axis")
    w("")
    w(
        f"Generated {today} by `src/write_results_cbramod.py` from "
        f"`{Path(warm.get('_path', 'warm')).name}` and `{Path(rand.get('_path', 'random')).name}`. "
        "Re-run the script after any change to the sweep; do not edit numbers by hand."
    )
    w("")
    w(
        "Two arms of CBraMod (TUEG-pretrained, 4.9M params, 200-d; never saw HBN) on "
        "E0.4's split — pool R1–R4 + R7–R10, val R5, test R6 — at fixed epoch 325. "
        "`e12cb` warm-starts from the released weights; `e12cbr` is the same shape, "
        "recipe, cohorts and seed from random init. Δr² is against an UNTRAINED "
        "encoder of the CBraMod shape; it is not on the same axis as the REVE arms' "
        "Δr² and must not be plotted with them. Design and decision: "
        "[`PLAN_cbramod.md`](PLAN_cbramod.md)."
    )
    w("")

    # ------------------------------------------------------------------ 1
    w("## 1. Level — warm / random Δr² at matched S (12-feature CV probe, val)")
    w("")
    nv_w = (warm.get("null_values") or {}).get("probe_val_mean_r2")
    nv_r = (rand.get("null_values") or {}).get("probe_val_mean_r2")
    w(
        f"Null mean r² (untrained CBraMod shape): warm-arm file {_f(nv_w, 6)}, "
        f"random-arm file {_f(nv_r, 6)} — these must be identical (same measurement, "
        "copied under both prefixes)."
    )
    w("")
    w("| S | warm Δr² | random Δr² | warm / random |")
    w("|---:|---|---|---:|")
    ratios = []
    for s in axis:
        gw, gr = group(warm, s, "d_r2"), group(rand, s, "d_r2")
        r = _ratio(gw, gr)
        if r is not None:
            ratios.append((s, r))
        w(f"| {s} | {_pm(gw)} | {_pm(gr)} | {_f(r, 2)}× |")
    w("")
    if ratios:
        rs = [r for _, r in ratios]
        w(
            f"Ratio across the axis: min {min(rs):.2f}×, max {max(rs):.2f}×, "
            f"median {sorted(rs)[len(rs) // 2]:.2f}× over {len(rs)} S values. "
            "E0.4 vs E0.5 (REVE warm vs random, depth 22) measured ~1.7× at every S."
        )
        w("")

    # ------------------------------------------------------------------ 2
    w("## 2. Slope — subject-scaling exponents per arm (Δr²)")
    w("")
    w("| arm | fitted d log Δr² / d log S | pooled between-draw sd |")
    w("|---|---:|---:|")
    for name, agg in (("warm", warm), ("random", rand)):
        w(
            f"| {name} | {_f(agg.get('fitted_exponent_S_d_r2'), 3)} | "
            f"{_f(agg.get('pooled_between_draw_sd'), 5)} |"
        )
    w("")
    w(
        "Local exponents between adjacent S (and the 701 → 1863 step the claim turns on):"
    )
    w("")
    w("| step | warm exponent | warm Δ in sd | random exponent | random Δ in sd |")
    w("|---|---:|---:|---:|---:|")
    steps_w = {(r["from"], r["to"]): r for r in warm.get("steps_d_r2", [])}
    steps_r = {(r["from"], r["to"]): r for r in rand.get("steps_d_r2", [])}
    for key in sorted(set(steps_w) | set(steps_r)):
        a, b = steps_w.get(key), steps_r.get(key)
        w(
            f"| {key[0]} → {key[1]} | {_f(a and a.get('exponent'), 3)} | "
            f"{_f(a and a.get('in_sd'), 1)} | {_f(b and b.get('exponent'), 3)} | "
            f"{_f(b and b.get('in_sd'), 1)} |"
        )
    w("")
    w(
        "For reference, E0.4 (REVE warm, depth 22) had a 701 → 1863 local exponent of "
        "+0.303, its random-init twin E0.5 +0.265, and depth-12 from scratch +0.049."
    )
    w("")

    # ------------------------------------------------------------------ 3
    w(
        "## 3. Train → test Pearson r (head fit on the 1863-recording pool, evaluated on R6)"
    )
    w("")
    ceil_t = (warm.get("ceilings") or {}).get("test")
    ceil_v = (warm.get("ceilings") or {}).get("val")
    w(
        f"Ceilings on file: val {_f(ceil_v, 3)}, test {_f(ceil_t, 3)}. Per the house "
        "rule, r / ceiling is quoted against the VAL ceiling only; the test ceiling "
        "is a known underestimate (RESULTS.md §2.5)."
    )
    w("")
    w(
        "| S | warm r (test) | warm r / val ceiling | random r (test) | random r / val ceiling |"
    )
    w("|---:|---|---:|---|---:|")
    for s in axis:
        gw, gr = group(warm, s, "tt_test_r"), group(rand, s, "tt_test_r")
        cw = (
            gw["mean"] / ceil_v
            if gw and gw.get("mean") is not None and ceil_v
            else None
        )
        cr = (
            gr["mean"] / ceil_v
            if gr and gr.get("mean") is not None and ceil_v
            else None
        )
        w(f"| {s} | {_pm(gw, 4)} | {_f(cw, 3)} | {_pm(gr, 4)} | {_f(cr, 3)} |")
    w("")
    xs = [float(s) for s in axis]
    for name, agg in (("warm", warm), ("random", rand)):
        ys = [(group(agg, s, "tt_test_r") or {}).get("mean") for s in axis]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if len(pts) >= 2:
            w(
                f"- {name}: fitted d log r / d log S = "
                f"{_f(_loglog_slope([p[0] for p in pts], [p[1] for p in pts]), 3)}"
            )
    w("")

    # ------------------------------------------------------------------ 4
    w("## 4. Scene-level e→v retrieval on TEST (top-1 / 5 / 10)")
    w("")
    nr_w = (warm.get("null_values") or {}).get("retr_test_scene") or {}
    w(
        "Null (untrained CBraMod shape): "
        + ", ".join(
            f"top{k} {_f((nr_w.get(f'top{k}') if isinstance(nr_w.get(f'top{k}'), (int, float)) else (nr_w.get(f'top{k}') or {}).get('mean')), 4)}"
            for k in TOPKS
        )
    )
    w("")
    w(
        "| S | warm top-1 | warm top-5 | warm top-10 | random top-1 | random top-5 | random top-10 |"
    )
    w("|---:|---|---|---|---|---|---|")
    for s in axis:
        cells = []
        for agg in (warm, rand):
            for k in TOPKS:
                cells.append(_pm(retr_group(agg, s, k), 4))
        w(f"| {s} | " + " | ".join(cells) + " |")
    w("")

    # ------------------------------------------------------------------ 5
    missing = [*warm.get("missing", []), *rand.get("missing", [])]
    w("## 5. Coverage")
    w("")
    w(
        f"Cells aggregated: warm {len(warm.get('cells', []))}, random {len(rand.get('cells', []))}. "
        f"Missing readouts: {len(missing)}."
    )
    if missing:
        w("")
        for m in missing:
            w(f"- {m}")
    w("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--warm", default=str(RAW / "e12cb_nested.json"))
    ap.add_argument("--random", default=str(RAW / "e12cbr_nested.json"))
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    warm = json.loads(Path(args.warm).read_text())
    rand = json.loads(Path(args.random).read_text())
    warm["_path"], rand["_path"] = args.warm, args.random
    Path(args.output).write_text(render(warm, rand))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
