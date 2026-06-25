"""Hierarchical paired bootstrap of Cine-JEPA vs canonical baselines.

Reads per-seed prediction NPZs (canonical schema with `test_*` keys) and runs
B hierarchical paired bootstrap iterations. Each iteration:
  - samples one seed index s in {0..K-1} with replacement
  - samples 108 recording indices with replacement
  - applies the SAME (s, idx) to every method
  - computes each method's metrics + Δ vs reference

Δ = method − reference (positive Δ means method beats reference).
p_one_sided = P[Δ ≤ 0] (small p ⇒ method significantly better than reference).

Reports per-method point ± hierarchical-bootstrap 95% CI and per-comparison
Δ ± 95% CI + one-sided p in a markdown table. A per-seed stratification
section is emitted as a free supplement.

Usage on Delta:
    python scripts/paired_bootstrap_canonical.py \
        --out=/u/dtyoung/eb_jepa_eeg/paired_bootstrap_headline.md \
        --n_bootstrap=2000

Pass --config_json=/path/to/cfg.json to override the headline preset.
"""

import glob
import json
from pathlib import Path

import fire
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


CANONICAL_ROOT = "/projects/bbnv/kkokate/eb_jepa_eeg/predictions"

HEADLINE_PRESET = {
    "reference": "Cine-JEPA",
    "methods": {
        "Cine-JEPA": f"{CANONICAL_ROOT}/canonical/pB_phaseD_issue10best/seed*/test_seed*.npz",
        "Raw 129-ch stats": f"{CANONICAL_ROOT}/canonical/pB_t1_raw_stats/seed*/test_seed*.npz",
        "CorrCA stats": f"{CANONICAL_ROOT}/canonical/pB_t1_corrca_stats/seed*/test_seed*.npz",
        "TRF (CorrCA-5)": f"{CANONICAL_ROOT}/canonical/pB_t1_trf_corrca5_*/seed*/test_seed*.npz",
        "TRF (129)": f"{CANONICAL_ROOT}/canonical/pB_t1_trf_raw129_*/seed*/test_seed*.npz",
        "Deep4Net (e2e)": f"{CANONICAL_ROOT}/unified/pB_t2_deep4_canonical_*_seed*/test_seed*.npz",
        "BIOT (linear probe)": f"{CANONICAL_ROOT}/unified/pB_t3_biot_canonical_*_seed*/test_seed*.npz",
        "Luna (linear probe)": f"{CANONICAL_ROOT}/unified/pB_t3_luna_canonical_*_seed*/test_seed*.npz",
    },
}

STIM_FEATS = ["luminance_mean", "contrast_rms", "position_in_movie", "narrative_event_score"]
N_PASSES = 20  # clips per recording in flat (2160,) arrays


def _load_method(name, pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{name}: no files matching {pattern}")
    print(f"  {name}: {len(files)} seed files")
    return [dict(np.load(f, allow_pickle=False)) for f in files]


def _check_alignment(method_seeds_dict):
    """Verify all (method, seed) NPZs share the same test_rec_ids order."""
    ref = None
    for name, seeds in method_seeds_dict.items():
        for i, npz in enumerate(seeds):
            rids = npz["test_rec_ids"]
            if ref is None:
                ref = rids
                continue
            if not np.array_equal(rids, ref):
                raise AssertionError(
                    f"{name} seed[{i}] rec_ids do not match reference order"
                )
    return ref


def _safe_corr(p, t):
    if np.std(p) < 1e-10 or np.std(t) < 1e-10:
        return float("nan")
    return float(pearsonr(p, t).statistic)


def _safe_auc(t, p):
    if len(np.unique(t)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(t, p))
    except ValueError:
        return float("nan")


# Metric extractors: each takes (npz, idx) where idx is a (n_rec,) int array of
# recording indices (with replacement) and returns a single float.
def _make_reg_corr(feat):
    pkey = f"test_reg_{feat}_pred"
    tkey = f"test_reg_{feat}_target"

    def fn(npz, idx):
        p = npz[pkey].reshape(-1, N_PASSES)[idx].ravel()
        t = npz[tkey].reshape(-1, N_PASSES)[idx].ravel()
        return _safe_corr(p, t)

    return fn


def _make_cls_auc(feat):
    pkey = f"test_cls_{feat}_proba"
    tkey = f"test_cls_{feat}_target"

    def fn(npz, idx):
        p = npz[pkey].reshape(-1, N_PASSES)[idx].ravel()
        t = npz[tkey].reshape(-1, N_PASSES)[idx].ravel()
        return _safe_auc(t, p)

    return fn


def _age_corr(npz, idx):
    p = npz["test_age_pred"][idx]
    t = npz["test_age_target"][idx]
    mask = ~np.isnan(t) & ~np.isnan(p)
    if mask.sum() < 3:
        return float("nan")
    return _safe_corr(p[mask], t[mask])


def _sex_auc(npz, idx):
    p = npz["test_sex_proba"][idx]
    t = npz["test_sex_target"][idx]
    mask = ~np.isnan(t)
    if mask.sum() < 2:
        return float("nan")
    return _safe_auc(t[mask].astype(int), p[mask])


def _make_movie_id_topk(k):
    def fn(npz, idx):
        proba = npz["test_movie_id_proba"][idx]
        target = npz["test_movie_id_target"][idx]
        kk = min(k, proba.shape[1])
        topk = np.argsort(-proba, axis=1)[:, :kk]
        return float((topk == target[:, None]).any(axis=1).mean())

    return fn


def _build_metrics():
    m = {}
    for f in STIM_FEATS:
        m[f"reg_{f}_corr"] = _make_reg_corr(f)
        m[f"cls_{f}_auc"] = _make_cls_auc(f)
    m["age_corr"] = _age_corr
    m["sex_auc"] = _sex_auc
    m["movie_id_top1"] = _make_movie_id_topk(1)
    m["movie_id_top5"] = _make_movie_id_topk(5)
    return m


def _summarize_iter_array(vals):
    """vals: (B,) array of bootstrap iterates. Returns mean, ci_lo, ci_hi."""
    arr = vals[~np.isnan(vals)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(arr.mean()),
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
    )


def _format_pair_row(name, m_vals, ref_vals):
    mean, lo, hi = _summarize_iter_array(m_vals)
    deltas = m_vals - ref_vals
    valid = ~np.isnan(deltas)
    if valid.sum() == 0:
        return f"| {name} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | nan | nan | nan |"
    d = deltas[valid]
    d_mean = float(d.mean())
    d_lo = float(np.quantile(d, 0.025))
    d_hi = float(np.quantile(d, 0.975))
    p_one = float((d <= 0).mean())  # P[method ≤ ref]; small ⇒ method beats ref
    return (
        f"| {name} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] | "
        f"{d_mean:+.4f} | [{d_lo:+.4f}, {d_hi:+.4f}] | {p_one:.3g} |"
    )


def run(
    out: str = "paired_bootstrap_headline.md",
    config_json: str = "",
    n_bootstrap: int = 2000,
    seed: int = 0,
):
    """Run hierarchical paired bootstrap.

    Args:
        out: Output markdown path.
        config_json: Path to JSON config {reference, methods: {name: glob}}.
                     If empty, uses HEADLINE_PRESET (Cine-JEPA + 7 baselines).
        n_bootstrap: Number of bootstrap iterations.
        seed: RNG seed.
    """
    if config_json:
        cfg = json.loads(Path(config_json).read_text())
    else:
        cfg = HEADLINE_PRESET

    ref_name = cfg["reference"]
    methods_cfg = cfg["methods"]
    assert ref_name in methods_cfg, f"reference '{ref_name}' not in methods"

    print(f"Loading {len(methods_cfg)} methods...")
    method_seeds = {name: _load_method(name, pat) for name, pat in methods_cfg.items()}

    Ks = {name: len(seeds) for name, seeds in method_seeds.items()}
    if len(set(Ks.values())) != 1:
        raise AssertionError(f"seed-count mismatch across methods: {Ks}")
    K = next(iter(Ks.values()))
    print(f"K = {K} seeds per method")

    print("Verifying rec_id alignment across all (method, seed) files...")
    rec_ids = _check_alignment(method_seeds)
    n_rec = len(rec_ids)
    print(f"n_rec = {n_rec}")

    metrics = _build_metrics()
    method_names = list(methods_cfg.keys())
    M = len(method_names)
    ref_idx = method_names.index(ref_name)

    # raw[metric] : (B, M) bootstrap iterate matrix
    raw = {m: np.full((n_bootstrap, M), np.nan, dtype=np.float64) for m in metrics}
    seed_per_iter = np.zeros(n_bootstrap, dtype=np.int32)

    rng = np.random.default_rng(seed)
    print(f"\nRunning B = {n_bootstrap} hierarchical paired bootstrap iterations...")
    for b in range(n_bootstrap):
        s = int(rng.integers(0, K))
        idx = rng.integers(0, n_rec, n_rec)
        seed_per_iter[b] = s
        for mi, name in enumerate(method_names):
            npz = method_seeds[name][s]
            for metric_name, fn in metrics.items():
                raw[metric_name][b, mi] = fn(npz, idx)
        if (b + 1) % 200 == 0:
            print(f"  {b+1}/{n_bootstrap}")

    # ---------- markdown report ----------
    lines = [
        "# Hierarchical paired bootstrap — Cine-JEPA vs Table 1 baselines",
        "",
        f"**B = {n_bootstrap}**, K = {K} seeds per method, n_rec = {n_rec}, RNG seed = {seed}",
        f"Reference: **{ref_name}**",
        "",
        "Each iteration draws (a) one seed index s ∈ {0,..,K-1} with replacement and "
        "(b) 108 recording indices with replacement. The **same** (s, idx) is applied "
        "to every method, so the recording-set noise cancels in the difference.",
        "",
        "Δ = method − reference. p_one = P[Δ ≤ 0] under bootstrap = "
        "**one-sided p-value that the method significantly beats the reference**.",
        "Small p_one (e.g. < 0.025) ⇒ method significantly better than reference. "
        "Large p_one (close to 1) ⇒ reference significantly better than method.",
        "",
    ]

    for metric_name in metrics:
        arr = raw[metric_name]
        ref_vals = arr[:, ref_idx]
        ref_mean, ref_lo, ref_hi = _summarize_iter_array(ref_vals)
        lines += [
            f"## {metric_name}",
            "",
            "| Method | mean | 95% CI | Δ vs ref | Δ 95% CI | p_one (method > ref) |",
            "|---|---:|---:|---:|---:|---:|",
            f"| **{ref_name}** | {ref_mean:.4f} | [{ref_lo:.4f}, {ref_hi:.4f}] | — | — | — |",
        ]
        for mi, name in enumerate(method_names):
            if name == ref_name:
                continue
            lines.append(_format_pair_row(name, arr[:, mi], ref_vals))
        lines.append("")

    # ---------- per-seed stratification supplement ----------
    lines += [
        "## Per-seed stratification (free supplement)",
        "",
        "Stratifies the same iterations by which seed index was drawn. Shows that "
        "the Δ direction is consistent across encoder seeds rather than driven by "
        "a single lucky seed. Table reports seed-conditional mean(Δ) for each pair.",
        "",
    ]
    for metric_name in metrics:
        arr = raw[metric_name]
        ref_vals = arr[:, ref_idx]
        # Header: seed columns
        seed_cols = " | ".join(f"s={s}" for s in range(K))
        lines += [
            f"### {metric_name} — seed-conditional Δ (method − ref)",
            "",
            f"| Method | {seed_cols} |",
            "|---|" + "---:|" * K,
        ]
        for mi, name in enumerate(method_names):
            if name == ref_name:
                continue
            row = [name]
            for s in range(K):
                mask = seed_per_iter == s
                if mask.sum() == 0:
                    row.append("—")
                    continue
                d = arr[mask, mi] - ref_vals[mask]
                d = d[~np.isnan(d)]
                row.append(f"{d.mean():+.4f}" if len(d) else "nan")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    fire.Fire(run)
