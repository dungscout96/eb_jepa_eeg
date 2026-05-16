"""Aggregate per-seed bootstrap JSONs across seeds → L3, t-test vs chance, full table."""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as scipy_t

KEYS = [
    "reg_luminance_mean_corr",
    "reg_contrast_rms_corr",
    "reg_position_in_movie_corr",
    "reg_narrative_event_score_corr",
    "cls_luminance_mean_auc",
    "cls_contrast_rms_auc",
    "cls_position_in_movie_auc",
    "cls_narrative_event_score_auc",
    "age_reg_corr",
    "sex_auc",
    "movie_id_top1",
    "movie_id_top5",
]

CHANCE = {
    "reg_luminance_mean_corr": 0.0,
    "reg_contrast_rms_corr": 0.0,
    "reg_position_in_movie_corr": 0.0,
    "reg_narrative_event_score_corr": 0.0,
    "cls_luminance_mean_auc": 0.5,
    "cls_contrast_rms_auc": 0.5,
    "cls_position_in_movie_auc": 0.5,
    "cls_narrative_event_score_auc": 0.5,
    "age_reg_corr": 0.0,
    "sex_auc": 0.5,
    "movie_id_top1": 0.05,
    "movie_id_top5": 0.25,
}


def _t_test_vs_chance(values, mu0):
    """One-sample two-sided t-test of `values` against `mu0`."""
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if sd == 0:
        return float("inf"), 0.0
    se = sd / math.sqrt(n)
    t = (mean - mu0) / se
    p = 2 * (1 - scipy_t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def aggregate(boot_jsons, out_path=None):
    boots = [json.load(open(p)) for p in boot_jsons]
    metrics = sorted({k for bj in boots for k in bj.get("L2", {})})
    l3 = {}
    l1_agg = {}
    ttest = {}
    for k in metrics:
        l2_means = [
            bj["L2"][k]["mean"]
            for bj in boots
            if k in bj.get("L2", {}) and not np.isnan(bj["L2"][k]["mean"])
        ]
        l2_cis = [
            (bj["L2"][k].get("ci_lo"), bj["L2"][k].get("ci_hi"))
            for bj in boots
            if k in bj.get("L2", {})
        ]
        l1_vals = [
            bj["L1"][k]
            for bj in boots
            if k in bj.get("L1", {}) and not np.isnan(bj["L1"][k])
        ]
        if l2_means:
            l3[k] = {
                "mean": float(np.mean(l2_means)),
                "std": float(np.std(l2_means, ddof=1)) if len(l2_means) > 1 else 0.0,
                "n_seeds": len(l2_means),
                "per_seed_L2": l2_means,
                "per_seed_L2_ci": l2_cis,
            }
        if l1_vals:
            l1_agg[k] = {
                "mean": float(np.mean(l1_vals)),
                "std": float(np.std(l1_vals, ddof=1)) if len(l1_vals) > 1 else 0.0,
                "n_seeds": len(l1_vals),
                "per_seed": l1_vals,
            }
        if l2_means and k in CHANCE:
            t, p = _t_test_vs_chance(l2_means, CHANCE[k])
            ttest[k] = {
                "t": t,
                "p": p,
                "chance": CHANCE[k],
                "sig": bool(p < 0.05) if not math.isnan(p) else False,
            }
    out = {
        "n_seeds": len(boots),
        "boot_jsons": [str(p) for p in boot_jsons],
        "L1_5seed": l1_agg,
        "L3": l3,
        "ttest_vs_chance": ttest,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def fmt(d):
    if d is None:
        return "      -        "
    return f"{d['mean']:+.4f}±{d['std']:.4f}"


def fmt_sig(t):
    if t is None:
        return "  -  "
    return "  ✓  " if t.get("sig") else " ns  "


def print_table(title, agg):
    l1 = agg["L1_5seed"]
    l3 = agg["L3"]
    tt = agg.get("ttest_vs_chance", {})
    n = agg["n_seeds"]
    print(f"=== {title}  (n_seeds={n}) ===")
    for k in KEYS:
        a = l1.get(k)
        b = l3.get(k)
        s = tt.get(k)
        if a or b:
            print(f"  {k:35s} L1={fmt(a)}  L3={fmt(b)} {fmt_sig(s)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"NO FILES match {args.glob}", file=sys.stderr)
        sys.exit(1)
    agg = aggregate(paths, args.out)
    print_table(args.name, agg)


if __name__ == "__main__":
    main()
