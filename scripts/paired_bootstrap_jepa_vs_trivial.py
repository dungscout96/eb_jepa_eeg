"""Paired recording-level bootstrap of (JEPA + Ridge) vs (Trivial Ridge corrca35).

For each metric, average predictions across all seed-files within each method
to get one ensemble prediction per recording-clip. Then for B bootstrap
resamples of recordings, compute corr(JEPA, target) - corr(Trivial, target)
on the same resampled set. Reports mean Δ ± σ_boot, 95% CI, and one-sided
p-value (P[Δ ≤ 0]) testing 'JEPA > Trivial'.

Both methods must produce per-clip preds in shape (N_rec, T, F) using the
same recording order on the same eval split.

Usage on Delta:
  python scripts/paired_bootstrap_jepa_vs_trivial.py \\
      --jepa_dir_pattern /projects/.../predictions/jepa_ridge_keep_channels_seed*/test_seed*.npz \\
      --trivial_dir_pattern /projects/.../predictions/trivial_ridge_corrca35_seed*/test_seed*.npz \\
      --n_bootstrap 2000 \\
      --out docs/paired_jepa_vs_trivial_corrca35.md
"""

import argparse
import glob
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


def _load_average(pattern: str) -> dict:
    """Load every npz matching pattern, average float arrays across files."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    print(f"  loading {len(files)} files for {pattern}")
    npzs = [dict(np.load(f, allow_pickle=False)) for f in files]
    merged = {}
    for k in npzs[0]:
        vals = [d[k] for d in npzs]
        try:
            stacked = np.stack(vals)
            merged[k] = stacked.mean(axis=0) if stacked.dtype.kind == "f" else vals[0]
        except (ValueError, TypeError):
            merged[k] = vals[0]
    return merged


def _per_metric_corr(preds_NTF: np.ndarray, targets_NTF: np.ndarray,
                     f_mean: np.ndarray, f_std: np.ndarray) -> np.ndarray:
    """Per-feature Pearson r between unnormalized preds and targets."""
    preds_un = preds_NTF * (f_std + 1e-8) + f_mean
    F = preds_un.shape[-1]
    out = np.zeros(F, dtype=np.float64)
    for i in range(F):
        p = preds_un[..., i].ravel()
        t = targets_NTF[..., i].ravel()
        if np.std(p) > 1e-10 and np.std(t) > 1e-10:
            out[i] = float(pearsonr(p, t).statistic)
    return out


def _per_metric_auc(logits_NTF: np.ndarray, targets_NTF: np.ndarray,
                    f_median: np.ndarray) -> np.ndarray:
    """Per-feature ROC-AUC between sigmoid(logits) and (target > median)."""
    probs = 1.0 / (1.0 + np.exp(-logits_NTF))
    F = logits_NTF.shape[-1]
    out = np.full(F, 0.5, dtype=np.float64)
    for i in range(F):
        p = probs[..., i].ravel()
        t_bin = (targets_NTF[..., i].ravel() > f_median[i]).astype(int)
        if len(np.unique(t_bin)) >= 2:
            try:
                out[i] = float(roc_auc_score(t_bin, p))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa_pattern", required=True,
                    help="glob, e.g. '.../jepa_ridge_keep_channels_seed*/test_seed*.npz'")
    ap.add_argument("--trivial_pattern", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print("Loading JEPA predictions...")
    jepa = _load_average(args.jepa_pattern)
    print("Loading Trivial predictions...")
    triv = _load_average(args.trivial_pattern)

    # Validate shapes
    for k in ("movie_reg_preds", "movie_targets", "movie_cls_logits"):
        assert k in jepa, f"JEPA missing {k}"
        assert k in triv, f"Trivial missing {k}"
        assert jepa[k].shape == triv[k].shape, \
            f"shape mismatch {k}: jepa={jepa[k].shape} triv={triv[k].shape}"

    feature_names = list(jepa["feature_names"])
    f_mean = jepa["feature_mean"]
    f_std = jepa["feature_std"]
    f_median = jepa["feature_median"]
    n_rec = jepa["movie_reg_preds"].shape[0]
    F = len(feature_names)

    rng = np.random.default_rng(args.seed)

    # Per-bootstrap: resample recording indices once, compute both methods'
    # corrs/AUCs on the same resample, store the difference.
    deltas_reg = np.zeros((args.n_bootstrap, F), dtype=np.float32)
    deltas_auc = np.zeros((args.n_bootstrap, F), dtype=np.float32)
    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n_rec, n_rec)
        jepa_corr = _per_metric_corr(
            jepa["movie_reg_preds"][idx],
            jepa["movie_targets"][idx],
            f_mean, f_std,
        )
        triv_corr = _per_metric_corr(
            triv["movie_reg_preds"][idx],
            triv["movie_targets"][idx],
            f_mean, f_std,
        )
        deltas_reg[b] = (jepa_corr - triv_corr).astype(np.float32)

        jepa_auc = _per_metric_auc(
            jepa["movie_cls_logits"][idx],
            jepa["movie_targets"][idx],
            f_median,
        )
        triv_auc = _per_metric_auc(
            triv["movie_cls_logits"][idx],
            triv["movie_targets"][idx],
            f_median,
        )
        deltas_auc[b] = (jepa_auc - triv_auc).astype(np.float32)

        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b+1}/{args.n_bootstrap}")

    # Markdown report
    lines = [
        "# Paired bootstrap — JEPA + Ridge `--keep_channels` vs Trivial Ridge corrca35 per_chan",
        "",
        f"**B = {args.n_bootstrap}**, seed = {args.seed}, recording-level resamples on the same set per draw.",
        "Predictions averaged across each method's seed files before bootstrap.",
        "One-sided test: P[Δ ≤ 0] under bootstrap = P-value that JEPA is *not* better than trivial.",
        "",
        "## Regression: Δr = corr(JEPA) − corr(Trivial)",
        "",
        "| Feature | Δr (mean ± σ_boot) | 95% CI | one-sided p (JEPA > triv) |",
        "|---|---:|---:|---:|",
    ]
    for i, fname in enumerate(feature_names):
        d = deltas_reg[:, i]
        m, s = float(d.mean()), float(d.std(ddof=1))
        ci_lo, ci_hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
        p_one = float((d <= 0).mean())
        lines.append(
            f"| `reg_{fname}_corr` | {m:+.4f} ± {s:.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}] | {p_one:.3g} |"
        )
    lines += [
        "",
        "## Classification: Δauc = AUC(JEPA) − AUC(Trivial)",
        "",
        "| Feature | Δauc (mean ± σ_boot) | 95% CI | one-sided p (JEPA > triv) |",
        "|---|---:|---:|---:|",
    ]
    for i, fname in enumerate(feature_names):
        d = deltas_auc[:, i]
        m, s = float(d.mean()), float(d.std(ddof=1))
        ci_lo, ci_hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
        p_one = float((d <= 0).mean())
        lines.append(
            f"| `cls_{fname}_auc` | {m:+.4f} ± {s:.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}] | {p_one:.3g} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
