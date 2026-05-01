"""Run paper-style per-encoder bootstrap + t-test on lane-1 perch predictions.

For each of 5 encoders:
  1. Symlink the 5 probe-seed npz files into a tmp dir
  2. Call scripts/bootstrap_probe_eval.run with seeds_glob → averages
     predictions across probe seeds, bootstraps recordings B times,
     returns mean ± CI per metric.
  3. Capture the bootstrap mean per metric.

Then across the 5 encoder-bootstrap-means, run a 1-sample t-test against
chance (corr → 0, AUC → 0.5). Mirrors the View 3 analysis from
docs/significance_analysis_2026-04-29.md.

Usage on Delta:
  python scripts/bootstrap_v10_perch.py \\
      --pred_root /projects/bbnv/kkokate/eb_jepa_eeg/predictions/v10_stage \\
      --tag_prefix nw4ws2_baseline \\
      --tag_suffix _context_enc_perch \\
      --enc_seeds 42,123,456,789,2025 \\
      --probe_seeds 7,13,42,1234,2025 \\
      --split test \\
      --n_bootstrap 1000 \\
      --out docs/v10_lane1_bootstrap.md
"""

import argparse
import math
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import t as scipy_t

# Add scripts/ to path so we can import the existing bootstrap functions
sys.path.insert(0, str(Path("/projects/bbnv/kkokate/eb_jepa_eeg/scripts")))
from bootstrap_probe_eval import (  # type: ignore
    _bootstrap_movie, _bootstrap_movie_id, _bootstrap_subject,
)

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

# Map our metric names → bootstrap_probe_eval's output keys
SUBJECT_KEY_MAP = {
    "age_reg_corr": "age_reg_corr",
    "sex_auc": "sex_auc",
}


def _load_and_average(npz_paths):
    """Load N npz files, return a dict of arrays averaged over the N seeds."""
    npzs = [np.load(p, allow_pickle=False) for p in npz_paths]
    merged = {}
    for k in npzs[0]:
        vals = [d[k] for d in npzs]
        try:
            stacked = np.stack(vals)
            merged[k] = stacked.mean(axis=0) if stacked.dtype.kind == "f" else vals[0]
        except (ValueError, TypeError):
            merged[k] = vals[0]
    return merged


def bootstrap_one_encoder(npz_paths, n_bootstrap, seed):
    """Return {metric_key: bootstrap_mean} for one encoder (avg over probe seeds)."""
    merged = _load_and_average(npz_paths)
    rng = np.random.default_rng(seed)
    out = {}
    movie = _bootstrap_movie(merged, n_bootstrap, rng)
    out.update({k: v["mean"] for k, v in movie.items()})
    subj = _bootstrap_subject(merged, n_bootstrap, rng)
    out.update({k: v["mean"] for k, v in subj.items()})
    return out


def t_test_against_chance(values, mu0):
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    if se == 0:
        return mean, sd, float("nan")
    t = (mean - mu0) / se
    p = 2 * (1 - scipy_t.cdf(abs(t), df=n - 1))
    return mean, sd, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", type=Path, required=True)
    ap.add_argument("--tag_prefix", required=True,
                    help="e.g. nw4ws2_baseline")
    ap.add_argument("--tag_suffix", required=True,
                    help="e.g. _context_enc_perch")
    ap.add_argument("--enc_seeds", required=True,
                    help="comma-separated, e.g. 42,123,456,789,2025")
    ap.add_argument("--probe_seeds", required=True,
                    help="comma-separated, e.g. 7,13,42,1234,2025")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--label", default="lane1_perch")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    enc_seeds = [int(s) for s in args.enc_seeds.split(",")]
    probe_seeds = [int(s) for s in args.probe_seeds.split(",")]

    # Per-encoder bootstrap
    enc_to_metrics = {}
    for enc in enc_seeds:
        npz_paths = []
        for ps in probe_seeds:
            tag = f"{args.tag_prefix}_enc{enc}_p{ps}{args.tag_suffix}"
            d = args.pred_root / tag
            f = d / f"{args.split}_seed{ps}.npz"
            if not f.exists():
                print(f"  MISSING {f}", file=sys.stderr)
                continue
            npz_paths.append(f)
        if len(npz_paths) < 2:
            print(f"  enc={enc}: only {len(npz_paths)} probe-seed files — skip", file=sys.stderr)
            continue
        print(f"enc={enc}: averaging {len(npz_paths)} probe-seed files, bootstrapping {args.n_bootstrap} recs")
        enc_to_metrics[enc] = bootstrap_one_encoder(npz_paths, args.n_bootstrap, seed=enc)

    if not enc_to_metrics:
        raise SystemExit("No encoders processed")

    # T-test across encoder bootstrap means
    metric_keys = sorted({k for m in enc_to_metrics.values() for k in m})
    rows = []
    for key in metric_keys:
        vals = [m[key] for m in enc_to_metrics.values() if key in m and not math.isnan(m[key])]
        if not vals:
            continue
        mu0 = CHANCE.get(key, None)
        if mu0 is None:
            # Try to infer
            mu0 = 0.5 if key.endswith("auc") else 0.0
        mean, sd, p = t_test_against_chance(vals, mu0)
        sig = "✓" if (not math.isnan(p) and p < 0.05) else " "
        rows.append((key, mean, sd, p, len(vals), sig, mu0))

    # Markdown output
    lines = []
    lines.append(f"# v10 lane #1 bootstrap analysis — {args.label}")
    lines.append("")
    lines.append(f"Per-encoder bootstrap (B={args.n_bootstrap} resamples of recordings),")
    lines.append(f"5 probe seeds averaged before bootstrap. Then 1-sample t-test on")
    lines.append(f"the {len(enc_to_metrics)}-encoder bootstrap means against chance.")
    lines.append("")
    lines.append(f"Split: **{args.split}** | tag prefix: `{args.tag_prefix}` | suffix: `{args.tag_suffix}`")
    lines.append("")
    lines.append("| Metric | Chance | Mean ± σ_enc | p (t-test) | n_enc | Sig |")
    lines.append("|---|---:|---:|---:|---:|:-:|")
    for key, mean, sd, p, n, sig, mu0 in rows:
        lines.append(f"| `{key}` | {mu0} | {mean:+.4f} ± {sd:.4f} | {p:.2g} | {n} | {sig} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
