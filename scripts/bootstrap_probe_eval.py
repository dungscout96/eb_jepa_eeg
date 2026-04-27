"""Bootstrap probe_eval predictions over recordings to get population CIs.

Reads the per-split .npz files written by probe_eval.py when invoked with
--save_predictions_dir. For each metric (movie corr / cls AUC / movie_id
top-k / subject trait), resamples recording indices with replacement B
times and recomputes the metric → reports mean, std, and 95% CI.

The seed-σ measured by re-running probe_eval with different seeds reflects
*probe-init noise*. The bootstrap σ here reflects *population sampling
noise* — the latter is what you'd want to compare to chance.

Usage
-----
uv run --group eeg python scripts/bootstrap_probe_eval.py \\
    --predictions_dir=/path/to/saved_predictions \\
    --split=test --n_bootstrap=1000
"""

from pathlib import Path

import fire
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _safe_corr(y_true, y_pred):
    if np.std(y_true) < 1e-10 or np.std(y_pred) < 1e-10:
        return float("nan")
    return float(pearsonr(y_pred, y_true).statistic)


def _safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _summarize(samples, alpha=0.05):
    arr = np.array([s for s in samples if not np.isnan(s)])
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci_lo": float(np.quantile(arr, alpha / 2)),
        "ci_hi": float(np.quantile(arr, 1 - alpha / 2)),
        "n": len(arr),
    }


def _bootstrap_movie(npz, n_bootstrap, rng):
    """Bootstrap movie-feature regression corr & cls AUC over recordings.

    movie_reg_preds, movie_cls_logits, movie_targets: [N_rec, T, n_features]
    With one clip per rec and shuffle=False, axis 0 is rec_id.
    """
    if "movie_reg_preds" not in npz:
        return {}
    reg_preds = npz["movie_reg_preds"]    # [N, T, F]
    cls_logits = npz["movie_cls_logits"]
    targets = npz["movie_targets"]
    feature_names = list(npz["feature_names"])
    f_mean = npz["feature_mean"]
    f_std = npz["feature_std"]
    f_median = npz["feature_median"]

    # Unnormalize regression predictions
    reg_preds_un = reg_preds * (f_std + 1e-8) + f_mean

    # Flatten target binary labels
    binary_targets = (targets > f_median).astype(int)
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

    n_rec = reg_preds.shape[0]
    out = {}
    for i, fname in enumerate(feature_names):
        reg_corrs, cls_aucs, cls_baccs = [], [], []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_rec, n_rec)
            yb_pred = reg_preds_un[idx, :, i].ravel()
            yb_targ = targets[idx, :, i].ravel()
            reg_corrs.append(_safe_corr(yb_targ, yb_pred))

            bin_idx = rng.integers(0, n_rec, n_rec)
            yc_true = binary_targets[bin_idx, :, i].ravel()
            yc_prob = cls_probs[bin_idx, :, i].ravel()
            cls_aucs.append(_safe_auc(yc_true, yc_prob))
            yc_pred = (yc_prob > 0.5).astype(int)
            try:
                cls_baccs.append(float(balanced_accuracy_score(yc_true, yc_pred)))
            except ValueError:
                cls_baccs.append(float("nan"))
        out[f"reg_{fname}_corr"] = _summarize(reg_corrs)
        out[f"cls_{fname}_auc"] = _summarize(cls_aucs)
        out[f"cls_{fname}_bal_acc"] = _summarize(cls_baccs)
    return out


def _bootstrap_movie_id(npz, n_bootstrap, rng):
    if "movie_id_logits" not in npz:
        return {}
    logits = npz["movie_id_logits"]            # [N, n_bins]
    positions = npz["movie_id_positions"]      # [N]
    bin_edges = npz["movie_id_bin_edges"]
    n_bins = len(bin_edges) - 1
    bin_labels = np.clip(np.digitize(positions, bin_edges) - 1, 0, n_bins - 1)
    n = logits.shape[0]
    top1, top5 = [], []
    preds = logits.argmax(axis=1)
    topk = np.argsort(-logits, axis=1)[:, :min(5, n_bins)]
    correct1 = (preds == bin_labels).astype(float)
    correct5 = (topk == bin_labels[:, None]).any(axis=1).astype(float)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        top1.append(float(correct1[idx].mean()))
        top5.append(float(correct5[idx].mean()))
    return {"movie_id_top1": _summarize(top1),
            "movie_id_top5": _summarize(top5),
            "movie_id_chance": {"mean": 1.0 / n_bins, "std": 0,
                                "ci_lo": 1.0 / n_bins, "ci_hi": 1.0 / n_bins, "n": 0}}


def _bootstrap_subject(npz, n_bootstrap, rng):
    out = {}
    keys = [k[5:-7] for k in npz.files
            if k.startswith("subj_") and k.endswith("_labels")]
    for label_name in keys:
        labels = npz[f"subj_{label_name}_labels"]
        if f"subj_{label_name}_logits" in npz:
            logits = npz[f"subj_{label_name}_logits"]
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs > 0.5).astype(int)
            valid = ~np.isnan(labels)
            y_true = labels[valid].astype(int)
            y_prob = probs[valid]
            y_pred = preds[valid]
            n = valid.sum()
            aucs, baccs = [], []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, n)
                aucs.append(_safe_auc(y_true[idx], y_prob[idx]))
                try:
                    baccs.append(float(balanced_accuracy_score(y_true[idx], y_pred[idx])))
                except ValueError:
                    baccs.append(float("nan"))
            out[f"{label_name}_auc"] = _summarize(aucs)
            out[f"{label_name}_bal_acc"] = _summarize(baccs)
        elif f"subj_{label_name}_pred_norm" in npz:
            pred_norm = npz[f"subj_{label_name}_pred_norm"]
            y_mean = float(npz[f"subj_{label_name}_y_mean"])
            y_std = float(npz[f"subj_{label_name}_y_std"])
            y_pred = pred_norm * y_std + y_mean
            valid = ~np.isnan(labels)
            y_true = labels[valid]
            y_p = y_pred[valid]
            n = valid.sum()
            corrs, maes = [], []
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, n)
                corrs.append(_safe_corr(y_true[idx], y_p[idx]))
                maes.append(float(np.mean(np.abs(y_p[idx] - y_true[idx]))))
            out[f"{label_name}_corr"] = _summarize(corrs)
            out[f"{label_name}_mae"] = _summarize(maes)
    return out


def run(predictions_dir: str, split: str = "test",
        n_bootstrap: int = 1000, seed: int = 0,
        seeds_glob: str = ""):
    """Bootstrap recording-level CIs from saved predictions.

    Args:
        predictions_dir: Dir containing {split}_seed{S}.npz files.
        split: Split to analyze ("val" or "test").
        n_bootstrap: Number of bootstrap resamples per metric.
        seed: RNG seed for bootstrap.
        seeds_glob: If set, glob pattern matching multiple seed files; their
                    predictions are averaged per recording before bootstrap.
                    e.g. "test_seed*.npz" → avg over all seeds.
    """
    pred_dir = Path(predictions_dir)
    rng = np.random.default_rng(seed)

    if seeds_glob:
        files = sorted(pred_dir.glob(seeds_glob))
        if not files:
            raise FileNotFoundError(f"No files matching {seeds_glob}")
        print(f"Loading {len(files)} seed files: {[f.name for f in files]}")
        npzs = [dict(np.load(f, allow_pickle=False)) for f in files]
        # Average predictions across seeds (subject embeddings/logits identical
        # only if encoder is deterministic; probe-trained heads vary by seed).
        merged = {}
        for k in npzs[0]:
            vals = [d[k] for d in npzs]
            try:
                merged[k] = np.mean(np.stack(vals), axis=0)
            except (ValueError, TypeError):
                merged[k] = npzs[0][k]
        npz = merged
    else:
        path = pred_dir / f"{split}_seed{seed}.npz"
        if not path.exists():
            cands = sorted(pred_dir.glob(f"{split}_seed*.npz"))
            assert cands, f"No predictions found in {pred_dir}"
            path = cands[0]
        print(f"Loading {path}")
        npz = dict(np.load(path, allow_pickle=False))

    print(f"\nBootstrap (B={n_bootstrap}) over recordings on '{split}':\n")
    print(f"{'metric':<40} {'mean':>8} {'std':>8} {'95% CI':>20}")
    print("-" * 78)

    results = {}
    results.update(_bootstrap_movie(npz, n_bootstrap, rng))
    results.update(_bootstrap_movie_id(npz, n_bootstrap, rng))
    results.update(_bootstrap_subject(npz, n_bootstrap, rng))

    for name, s in results.items():
        ci = f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"
        print(f"{name:<40} {s['mean']:>+8.4f} {s['std']:>8.4f} {ci:>20}")

    return results


if __name__ == "__main__":
    fire.Fire(run)
