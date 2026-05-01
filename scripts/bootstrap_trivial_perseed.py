"""View 2 + View 3 t-test analysis for trivial-stats baselines.

Trivial baselines have no encoder, only probe seed varies. So both views
operate over the 5 probe seeds:

  * View 2 — 1-sample t-test on the 5 raw test corrs (no bootstrap).
  * View 3 — for each probe seed, recording-level bootstrap (B=2000)
             of test predictions → bootstrap-mean per metric per seed;
             then 1-sample t-test on the 5 bootstrap means.

Both tested against chance (corr → 0, AUC → 0.5).

Usage on Delta:
  python scripts/bootstrap_trivial_perseed.py \\
      --pred_root /projects/bbnv/kkokate/eb_jepa_eeg/tier1/predictions \\
      --baseline trivial_corrca_per_chan \\
      --probe_seeds 7,13,42,1234,2025 \\
      --split test \\
      --n_bootstrap 2000 \\
      --out docs/trivial_<baseline>_bootstrap.md
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as scipy_t

sys.path.insert(0, str(Path("/projects/bbnv/kkokate/eb_jepa_eeg/scripts")))
from bootstrap_probe_eval import _bootstrap_movie, _bootstrap_subject  # type: ignore

CHANCE = {
    "reg_position_in_movie_corr": 0.0,
    "reg_luminance_mean_corr": 0.0,
    "reg_contrast_rms_corr": 0.0,
    "reg_narrative_event_score_corr": 0.0,
    "cls_position_in_movie_auc": 0.5,
    "cls_luminance_mean_auc": 0.5,
    "cls_contrast_rms_auc": 0.5,
    "cls_narrative_event_score_auc": 0.5,
    "age_reg_corr": 0.0,
    "sex_auc": 0.5,
}

# Map raw test-metric keys (in the per-clip JSON) to bootstrap output keys.
RAW_TEST_KEYS = {
    "reg_position_in_movie_corr": "test/reg_position_in_movie_corr",
    "reg_luminance_mean_corr": "test/reg_luminance_mean_corr",
    "reg_contrast_rms_corr": "test/reg_contrast_rms_corr",
    "reg_narrative_event_score_corr": "test/reg_narrative_event_score_corr",
    "cls_position_in_movie_auc": "test/cls_position_in_movie_auc",
    "cls_luminance_mean_auc": "test/cls_luminance_mean_auc",
    "cls_contrast_rms_auc": "test/cls_contrast_rms_auc",
    "cls_narrative_event_score_auc": "test/cls_narrative_event_score_auc",
    "age_reg_corr": "test/subject/age_reg/corr",
    "sex_auc": "test/subject/sex/auc",
}


def t_test_against_chance(values, mu0):
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n) if n > 0 else 0.0
    if se == 0:
        return mean, sd, float("nan")
    t = (mean - mu0) / se
    p = 2 * (1 - scipy_t.cdf(abs(t), df=n - 1))
    return mean, sd, p


def bootstrap_one_seed(npz_path, n_bootstrap, seed):
    """Per-seed bootstrap of recordings → {metric_key: bootstrap_mean}."""
    npz = dict(np.load(npz_path, allow_pickle=False))
    rng = np.random.default_rng(seed)
    out = {}
    movie = _bootstrap_movie(npz, n_bootstrap, rng)
    out.update({k: v["mean"] for k, v in movie.items()})
    subj = _bootstrap_subject(npz, n_bootstrap, rng)
    out.update({k: v["mean"] for k, v in subj.items()})
    return out


def load_raw_test(json_path):
    """Load View-2 raw test corrs/AUCs from a tier1 summary JSON."""
    import json
    d = json.load(open(json_path))
    return d["metrics"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", type=Path, required=True)
    ap.add_argument("--summary_root", type=Path, required=True,
                    help="Dir with <baseline>_seed<seed>.json files")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--probe_seeds", required=True,
                    help="comma-separated, e.g. 7,13,42,1234,2025")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.probe_seeds.split(",")]

    # Per-seed bootstrap (View 3 inputs)
    seed_to_boot = {}
    for s in seeds:
        npz = args.pred_root / f"{args.baseline}_seed{s}" / f"{args.split}_seed{s}.npz"
        if not npz.exists():
            print(f"  MISSING {npz}", file=sys.stderr)
            continue
        print(f"seed={s}: bootstrapping {args.n_bootstrap} resamples")
        seed_to_boot[s] = bootstrap_one_seed(npz, args.n_bootstrap, seed=s)

    # Per-seed raw test corrs (View 2 inputs)
    seed_to_raw = {}
    for s in seeds:
        jsonf = args.summary_root / f"{args.baseline}_seed{s}.json"
        if not jsonf.exists():
            print(f"  MISSING {jsonf}", file=sys.stderr)
            continue
        seed_to_raw[s] = load_raw_test(jsonf)

    if not seed_to_boot:
        raise SystemExit("No seeds processed")

    # Assemble per-metric arrays
    metric_keys = sorted({k for m in seed_to_boot.values() for k in m})
    rows = []
    for k in metric_keys:
        v3_vals = [m[k] for m in seed_to_boot.values()
                   if k in m and not math.isnan(m[k])]
        if not v3_vals:
            continue
        mu0 = CHANCE.get(k, 0.5 if k.endswith("auc") else 0.0)
        v3_mean, v3_sd, v3_p = t_test_against_chance(v3_vals, mu0)
        v3_sig = "sig" if (not math.isnan(v3_p) and v3_p < 0.05) else "ns"

        # View 2 — find the matching raw key in summary metrics
        raw_key = RAW_TEST_KEYS.get(k)
        if raw_key is not None:
            v2_vals = [m.get(raw_key, float("nan")) for m in seed_to_raw.values()]
            v2_vals = [v for v in v2_vals if not math.isnan(v)]
        else:
            v2_vals = []
        if v2_vals:
            v2_mean, v2_sd, v2_p = t_test_against_chance(v2_vals, mu0)
            v2_sig = "sig" if (not math.isnan(v2_p) and v2_p < 0.05) else "ns"
        else:
            v2_mean = v2_sd = v2_p = float("nan")
            v2_sig = "?"

        rows.append((k, mu0, v2_mean, v2_sd, v2_p, v2_sig,
                     v3_mean, v3_sd, v3_p, v3_sig, len(v3_vals)))

    # Markdown
    lines = []
    lines.append(f"# Trivial baseline View 2 + View 3 — {args.baseline}")
    lines.append("")
    lines.append(f"Probe seeds: {sorted(seed_to_boot.keys())}")
    lines.append(f"Split: **{args.split}** | B={args.n_bootstrap} | n_seeds={len(seed_to_boot)}")
    lines.append("")
    lines.append("| Metric | Chance | View 2 mean ± σ | V2 p | View 3 mean ± σ | V3 p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for (k, mu0, v2_m, v2_s, v2_p, v2_sig, v3_m, v3_s, v3_p, v3_sig, n) in rows:
        v2 = f"{v2_m:+.4f} ± {v2_s:.4f}" if not math.isnan(v2_m) else "—"
        v2pf = f"{v2_p:.2g} {v2_sig}" if not math.isnan(v2_p) else "—"
        v3 = f"{v3_m:+.4f} ± {v3_s:.4f}"
        v3pf = f"{v3_p:.2g} {v3_sig}"
        lines.append(f"| `{k}` | {mu0} | {v2} | {v2pf} | {v3} | {v3pf} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
