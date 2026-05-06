"""Bootstrap Tier 2/4/6 model-native NPZs.

Tier 2/4/6 NPZ schema (different from unified):
  movie_reg_preds       (n_rec, n_passes, n_features)  — Pearson r predictions per clip
  movie_targets         (n_rec, n_passes, n_features)
  movie_cls_logits      (n_rec, n_passes, n_features)
  feature_names         (n_features,)
  feature_mean / feature_std / feature_median  (n_features,)  — train normalization
  rec_ids               (n_rec,)
  subj_age_reg_pred_norm (n_rec,)  — Tier 4 only
  subj_age_reg_y_mean ()  / subj_age_reg_y_std ()
  subj_age_reg_labels   (n_rec,)
  subj_sex_logits       (n_rec,)
  subj_sex_labels       (n_rec,)

These are model-native predictions (no canonical Ridge re-fit). Bootstrap is
recording-level over the (n_rec, n_passes) grouped arrays.

Outputs the same L1/L2 layout as bootstrap_unified.py for cross-method aggregation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _pearson(p, y):
    if np.std(p) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(pearsonr(p, y).statistic)


def _auc(proba, y):
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def _bal_acc(proba, y, threshold=0.5):
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, (proba > threshold).astype(int)))


def _compute_metrics(npz, rec_indices=None):
    """Bootstrap-compatible metric computation for tier-native NPZ.

    rec_indices: array of recording indices to use (with replacement OK).
    """
    feat_names = [
        s.decode() if isinstance(s, bytes) else str(s) for s in npz["feature_names"]
    ]
    n_rec = len(npz["rec_ids"])
    if rec_indices is None:
        rec_indices = np.arange(n_rec)

    out = {}

    # ---- movie_reg ----
    if "movie_reg_preds" in npz:
        # preds may be normalized or unnormalized; tier4/6 saved the per-clip raw preds.
        preds = npz["movie_reg_preds"][rec_indices]  # (n_sub, n_passes, n_feat)
        targs = npz["movie_targets"][rec_indices]
        # If feature_mean/std are present, we may need to unnormalize. Inspect preds vs targs scale:
        # tier4_full_ft.py saves UNNORMALIZED preds (post-sigmoid not applied) — match by stats.
        # Since both preds and targs are stored together, we just pearson per feature on flattened (n_sub*n_passes,).
        for fi, name in enumerate(feat_names):
            p_flat = preds[:, :, fi].reshape(-1)
            y_flat = targs[:, :, fi].reshape(-1)
            valid = ~np.isnan(p_flat) & ~np.isnan(y_flat)
            if valid.sum() < 2:
                out[f"reg_{name}_corr"] = float("nan")
            else:
                out[f"reg_{name}_corr"] = _pearson(p_flat[valid], y_flat[valid])

    # ---- movie_cls (median-split AUC) ----
    if "movie_cls_logits" in npz and "feature_median" in npz:
        logits = npz["movie_cls_logits"][rec_indices]
        targs = npz["movie_targets"][rec_indices]
        med = npz["feature_median"]
        for fi, name in enumerate(feat_names):
            l_flat = logits[:, :, fi].reshape(-1)
            y_flat = targs[:, :, fi].reshape(-1)
            valid = ~np.isnan(l_flat) & ~np.isnan(y_flat)
            if valid.sum() < 2:
                out[f"cls_{name}_auc"] = float("nan")
                continue
            # Convert logits to proba via sigmoid for AUC.
            proba = 1.0 / (1.0 + np.exp(-l_flat[valid]))
            y_bin = (y_flat[valid] > med[fi]).astype(int)
            out[f"cls_{name}_auc"] = _auc(proba, y_bin)
            out[f"cls_{name}_bal_acc"] = _bal_acc(proba, y_bin)

    # ---- subject age ----
    if "subj_age_reg_pred_norm" in npz:
        pred_n = npz["subj_age_reg_pred_norm"][rec_indices]
        y = npz["subj_age_reg_labels"][rec_indices]
        ym = float(npz["subj_age_reg_y_mean"]) if "subj_age_reg_y_mean" in npz else 0.0
        ys = float(npz["subj_age_reg_y_std"]) if "subj_age_reg_y_std" in npz else 1.0
        pred = pred_n * ys + ym
        valid = ~np.isnan(pred) & ~np.isnan(y)
        if valid.sum() >= 2:
            out["age_reg_corr"] = _pearson(pred[valid], y[valid])

    # ---- subject sex ----
    if "subj_sex_logits" in npz:
        log = npz["subj_sex_logits"][rec_indices]
        y = npz["subj_sex_labels"][rec_indices]
        valid = ~np.isnan(log) & ~np.isnan(y)
        if valid.sum() >= 2:
            proba = 1.0 / (1.0 + np.exp(-log[valid]))
            out["sex_auc"] = _auc(proba, y[valid])
            out["sex_bal_acc"] = _bal_acc(proba, y[valid])

    return out


def bootstrap_one(npz_path, b, seed):
    npz = dict(np.load(npz_path, allow_pickle=False))
    n_rec = len(npz["rec_ids"])
    rng = np.random.default_rng(seed)

    l1 = _compute_metrics(npz)
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
                "std": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
            }
        else:
            l2[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "ci_lo": float(np.percentile(arr, 2.5)),
                "ci_hi": float(np.percentile(arr, 97.5)),
            }
    return l1, l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=42)
    args = ap.parse_args()

    l1, l2 = bootstrap_one(args.npz, args.b, args.boot_seed)
    out = {
        "npz": str(args.npz),
        "b": args.b,
        "boot_seed": args.boot_seed,
        "L1": l1,
        "L2": l2,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json}")
    if "reg_narrative_event_score_corr" in l1:
        print(f"  L1 narr={l1['reg_narrative_event_score_corr']:.4f}")
        d = l2["reg_narrative_event_score_corr"]
        print(f"  L2 narr={d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")


if __name__ == "__main__":
    main()
