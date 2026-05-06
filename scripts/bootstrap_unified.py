"""Recording-level bootstrap (B=2000) on unified_probe_eval.py NPZs.

Computes L1 (raw test r/AUC), L2 (per-seed bootstrap mean + 95% CI),
and aggregates across seeds for L3 (5-seed mean of L2 ± 1σ).

NPZ schema expected (from unified_probe_eval.py):
  - {split}_reg_{feature}_pred / _target   shape (n_rec * n_passes,)
  - {split}_cls_{feature}_proba / _target  shape (n_rec * n_passes,)
  - {split}_age_pred / _target              shape (n_rec,)
  - {split}_sex_proba / _target             shape (n_rec,)
  - {split}_movie_id_proba / _target        shape (n_rec, n_classes) / (n_rec,)
  - {split}_rec_ids                         shape (n_rec,)
  - feature_names                           shape (n_features,)

Layout assumption: per-clip arrays flatten as (rec_0_p_0, rec_0_p_1, ..., rec_0_p_{P-1},
rec_1_p_0, ...) — i.e. recordings contiguous, passes nested. Confirmed against
unified_probe_eval.py:_extract sequential rec×pass val/test path.

Usage:
  python scripts/bootstrap_unified.py \\
      --npz /path/to/test_seed42.npz \\
      --out_json /path/to/bootstrap.json \\
      --b 2000

  # Aggregate across multiple seeds:
  python scripts/bootstrap_unified.py aggregate \\
      --boot_jsons /path/seed42.json /path/seed123.json ... \\
      --out_json /path/aggregate.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

STIM_FEATURES = (
    "luminance_mean",
    "contrast_rms",
    "position_in_movie",
    "narrative_event_score",
)


def _pearson_safe(pred, y):
    if np.std(pred) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(pearsonr(pred, y).statistic)


def _auc_safe(proba, y):
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def _bal_acc(proba, y, threshold=0.5):
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    pred = (proba > threshold).astype(int)
    return float(balanced_accuracy_score(y, pred))


def _topk(proba, y, k):
    """Top-k accuracy for multinomial: proba (N, C), y (N,)."""
    y = y.astype(int)
    if proba.ndim == 1:  # already argmax preds
        return float((proba.astype(int) == y).mean())
    topk_idx = np.argpartition(-proba, kth=min(k - 1, proba.shape[1] - 1), axis=1)[
        :, :k
    ]
    return float(np.mean([y[i] in topk_idx[i] for i in range(len(y))]))


def _compute_metrics(npz, rec_indices=None, n_passes=None):
    """Compute all 18 headline metrics on either full set or a subset of recordings.

    rec_indices: array of recording indices (with replacement OK). When None, use all.
    n_passes: number of clip passes per recording (inferred from clip-array length / n_rec
              when not provided).
    """
    n_rec_full = len(npz["test_rec_ids"])
    if rec_indices is None:
        rec_indices = np.arange(n_rec_full)

    out = {}

    # Per-clip arrays: reshape (n_rec_full * P,) → (n_rec_full, P), index axis 0, flatten back.
    for feat in STIM_FEATURES:
        for kind in ("reg", "cls"):
            pred_key = f"test_{kind}_{feat}_" + ("pred" if kind == "reg" else "proba")
            tgt_key = f"test_{kind}_{feat}_target"
            if pred_key not in npz:
                continue
            pred_flat = npz[pred_key]
            tgt_flat = npz[tgt_key]
            if pred_flat.ndim == 1 and len(pred_flat) % n_rec_full == 0:
                P = len(pred_flat) // n_rec_full
                pred_grp = pred_flat.reshape(n_rec_full, P)
                tgt_grp = tgt_flat.reshape(n_rec_full, P)
                pred_sub = pred_grp[rec_indices].reshape(-1)
                tgt_sub = tgt_grp[rec_indices].reshape(-1)
            else:
                # Per-rec already
                pred_sub = pred_flat[rec_indices]
                tgt_sub = tgt_flat[rec_indices]

            if kind == "reg":
                out[f"reg_{feat}_corr"] = _pearson_safe(pred_sub, tgt_sub)
            else:
                out[f"cls_{feat}_auc"] = _auc_safe(pred_sub, tgt_sub)
                out[f"cls_{feat}_bal_acc"] = _bal_acc(pred_sub, tgt_sub)

    # Subject — already per-recording
    for kind, key_pred, key_tgt, metric in [
        ("age", "test_age_pred", "test_age_target", "age_reg_corr"),
    ]:
        if key_pred in npz:
            pred = npz[key_pred][rec_indices]
            tgt = npz[key_tgt][rec_indices]
            valid = ~np.isnan(pred) & ~np.isnan(tgt)
            out[metric] = (
                _pearson_safe(pred[valid], tgt[valid])
                if valid.sum() >= 2
                else float("nan")
            )

    if "test_sex_proba" in npz:
        pred = npz["test_sex_proba"][rec_indices]
        tgt = npz["test_sex_target"][rec_indices]
        valid = ~np.isnan(pred) & ~np.isnan(tgt)
        out["sex_auc"] = (
            _auc_safe(pred[valid], tgt[valid]) if valid.sum() >= 2 else float("nan")
        )
        out["sex_bal_acc"] = (
            _bal_acc(pred[valid], tgt[valid]) if valid.sum() >= 2 else float("nan")
        )

    if "test_movie_id_proba" in npz:
        proba = npz["test_movie_id_proba"]
        tgt = npz["test_movie_id_target"]
        if proba.ndim == 2:
            proba_sub = proba[rec_indices]
        else:
            proba_sub = proba[rec_indices]
        tgt_sub = tgt[rec_indices]
        out["movie_id_top1"] = _topk(proba_sub, tgt_sub, 1)
        out["movie_id_top5"] = _topk(proba_sub, tgt_sub, 5)

    return out


def bootstrap_one_npz(npz_path, b, seed):
    """Run B=b bootstrap on a single NPZ. Returns L1 (raw) + L2 (boot stats per metric)."""
    npz = dict(np.load(npz_path, allow_pickle=False))
    n_rec = len(npz["test_rec_ids"])
    rng = np.random.default_rng(seed)

    # L1: raw metric on full test set.
    l1 = _compute_metrics(npz)

    # L2: B bootstrap iterations
    samples = {k: [] for k in l1}
    for _ in range(b):
        idx = rng.integers(0, n_rec, size=n_rec)
        m = _compute_metrics(npz, rec_indices=idx)
        for k, v in m.items():
            samples[k].append(v)

    l2 = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            l2[k] = {
                "mean": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "std": float("nan"),
            }
        else:
            l2[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "ci_lo": float(np.percentile(arr, 2.5)),
                "ci_hi": float(np.percentile(arr, 97.5)),
            }
    return l1, l2


def cmd_bootstrap(args):
    l1, l2 = bootstrap_one_npz(args.npz, args.b, args.boot_seed)
    out = {
        "npz": str(args.npz),
        "b": args.b,
        "boot_seed": args.boot_seed,
        "L1": l1,
        "L2": l2,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json}")
    print(f"  L1 narr={l1.get('reg_narrative_event_score_corr', 'n/a'):.4f}")
    if "reg_narrative_event_score_corr" in l2:
        d = l2["reg_narrative_event_score_corr"]
        print(f"  L2 narr={d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")


def cmd_aggregate(args):
    """Aggregate per-seed bootstrap JSONs → L3 (5-seed mean of L2 ± 1σ)."""
    boot_jsons = [json.load(open(p)) for p in args.boot_jsons]
    metrics = list(boot_jsons[0]["L2"].keys())
    l3 = {}
    l1_agg = {}
    for m in metrics:
        l2_means = [
            bj["L2"][m]["mean"]
            for bj in boot_jsons
            if m in bj["L2"] and not np.isnan(bj["L2"][m]["mean"])
        ]
        l1_vals = [
            bj["L1"][m]
            for bj in boot_jsons
            if m in bj.get("L1", {}) and not np.isnan(bj["L1"][m])
        ]
        if l2_means:
            l3[m] = {
                "mean": float(np.mean(l2_means)),
                "std": float(np.std(l2_means, ddof=1)) if len(l2_means) > 1 else 0.0,
                "n_seeds": len(l2_means),
                "per_seed": l2_means,
            }
        if l1_vals:
            l1_agg[m] = {
                "mean": float(np.mean(l1_vals)),
                "std": float(np.std(l1_vals, ddof=1)) if len(l1_vals) > 1 else 0.0,
                "n_seeds": len(l1_vals),
                "per_seed": l1_vals,
            }
    out = {
        "n_seeds": len(boot_jsons),
        "boot_jsons": [str(p) for p in args.boot_jsons],
        "L1_5seed": l1_agg,
        "L3": l3,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json}")
    if "reg_narrative_event_score_corr" in l3:
        d = l3["reg_narrative_event_score_corr"]
        print(f"  L3 narr={d['mean']:.4f} ± {d['std']:.4f}  (n={d['n_seeds']})")
    if "reg_narrative_event_score_corr" in l1_agg:
        d = l1_agg["reg_narrative_event_score_corr"]
        print(f"  L1 narr={d['mean']:.4f} ± {d['std']:.4f}  (n={d['n_seeds']})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("bootstrap")
    p1.add_argument("--npz", type=Path, required=True)
    p1.add_argument("--out_json", type=Path, required=True)
    p1.add_argument("--b", type=int, default=2000)
    p1.add_argument("--boot_seed", type=int, default=42)
    p1.set_defaults(func=cmd_bootstrap)

    p2 = sub.add_parser("aggregate")
    p2.add_argument("--boot_jsons", type=Path, nargs="+", required=True)
    p2.add_argument("--out_json", type=Path, required=True)
    p2.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
