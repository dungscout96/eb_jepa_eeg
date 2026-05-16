"""Canonical recording-level bootstrap on probe_eval_canonical's saved predictions.

Reads test_*-prefixed keys ONLY from the saved NPZ (val_* are protocol-audit only,
never reported). For each metric the per-clip array of shape (n_rec * n_passes,) is
reshaped to (n_rec, n_passes), then B=2000 iterations resample n_rec indices with
replacement and flatten back to (subset * n_passes,) before recomputing the metric.

Emits a JSON file in the {L1, L2, ...} shape consumed by
`scripts/aggregate_and_print.py`:

    {
      "seed": <encoder seed>,
      "probe_seed": <probe seed>,
      "n_passes": <int>,
      "n_bootstrap": <int>,
      "L1": {metric: float, ...},          # single raw metric on full flat array
      "L2": {metric: {mean, std, ci_lo, ci_hi}, ...}  # bootstrap mean & 95% CI
    }

Usage
-----
uv run --group eeg python -m eb_jepa.evaluation.bootstrap_canonical \\
    --predictions_npz=/abs/path/preds_seed42.npz \\
    --n_bootstrap=2000 \\
    --out_json=/abs/path/L2_seed42.json
"""

import json
from pathlib import Path

import fire
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


# ---------------------------------------------------------------------------
# Metric helpers (safe to NaNs / degenerate inputs)
# ---------------------------------------------------------------------------

def _safe_corr(y, yhat):
    if np.std(y) < 1e-10 or np.std(yhat) < 1e-10 or len(y) < 2:
        return float("nan")
    try:
        return float(pearsonr(yhat, y).statistic)
    except Exception:
        return float("nan")


def _safe_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    valid = ~(np.isnan(y) | np.isnan(p))
    y = y[valid]; p = p[valid]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y.astype(int), p))
    except ValueError:
        return float("nan")


def _safe_bal_acc(y, p, threshold=0.5):
    y = np.asarray(y); p = np.asarray(p)
    valid = ~(np.isnan(y) | np.isnan(p))
    y = y[valid]; p = p[valid]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    yhat = (p > threshold).astype(int)
    try:
        return float(balanced_accuracy_score(y.astype(int), yhat))
    except ValueError:
        return float("nan")


def _summarize(samples, alpha=0.05):
    arr = np.array([s for s in samples if not (s is None or np.isnan(s))])
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci_lo": float(np.quantile(arr, alpha / 2)),
        "ci_hi": float(np.quantile(arr, 1 - alpha / 2)),
        "n": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# Bootstrap drivers
# ---------------------------------------------------------------------------

def _reshape_per_rec(flat, n_rec, n_passes):
    """Reshape (n_rec * n_passes,) → (n_rec, n_passes). The probe_eval_canonical
    extractor uses rec × passes (sequential) ordering for val/test so this is a
    plain reshape, not a permutation."""
    expected = n_rec * n_passes
    assert flat.shape[0] == expected, (
        f"Length mismatch: got {flat.shape[0]}, expected n_rec*n_passes={expected}"
    )
    extra = flat.shape[1:]
    return flat.reshape((n_rec, n_passes) + extra)


def _bootstrap_metric(pred_grp, tgt_grp, n_bootstrap, rng, kind):
    """Resample n_rec recording indices with replacement, recompute metric.

    kind ∈ {"corr", "auc", "bal_acc"}
    """
    n_rec = pred_grp.shape[0]
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_rec, n_rec)
        p = pred_grp[idx].reshape(-1)
        t = tgt_grp[idx].reshape(-1)
        if kind == "corr":
            samples.append(_safe_corr(t, p))
        elif kind == "auc":
            samples.append(_safe_auc(t, p))
        elif kind == "bal_acc":
            samples.append(_safe_bal_acc(t, p))
        else:
            raise ValueError(kind)
    return _summarize(samples)


def _bootstrap_movie_id(probs_grp, target_bin_grp, n_bootstrap, rng):
    """Per-clip probs_grp: (n_rec, n_passes, n_bins). target_bin_grp: (n_rec, n_passes)."""
    n_rec = probs_grp.shape[0]
    n_bins = probs_grp.shape[-1]
    k5 = min(5, n_bins)

    # Precompute per-clip top1/top5 correctness
    flat_probs = probs_grp.reshape(-1, n_bins)
    flat_tgt = target_bin_grp.reshape(-1)
    preds = flat_probs.argmax(axis=1)
    top1 = (preds == flat_tgt).astype(np.float64).reshape(n_rec, -1)
    top5_idx = np.argsort(-flat_probs, axis=1)[:, :k5]
    top5 = (top5_idx == flat_tgt[:, None]).any(axis=1).astype(np.float64).reshape(n_rec, -1)

    s1, s5 = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_rec, n_rec)
        s1.append(float(top1[idx].mean()))
        s5.append(float(top5[idx].mean()))
    return _summarize(s1), _summarize(s5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    predictions_npz: str,
    out_json: str,
    n_bootstrap: int = 2000,
    seed: int = 0,
    wandb_run_id: str = "",
    wandb_project: str = "eb_jepa",
):
    """Bootstrap L2 + emit L1+L2 JSON for downstream aggregate_and_print.py.

    Args
    ----
    predictions_npz : path to NPZ written by probe_eval_canonical
    out_json : where to write the per-seed L1+L2 JSON
    n_bootstrap : bootstrap resamples (default 2000 per spec)
    seed : RNG seed for resampling
    """
    npz_path = Path(predictions_npz)
    assert npz_path.exists(), f"NPZ not found: {npz_path}"
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npz = dict(np.load(npz_path, allow_pickle=False))
    n_passes = int(npz["n_passes"])
    probe_seed = int(npz["probe_seed"])
    enc_seed = int(npz["seed"])
    rec_t = npz["test_rec_ids"]
    n_rec = int(rec_t.max()) + 1
    feature_names = [str(s) for s in npz["feature_names"]]
    rng = np.random.default_rng(seed)

    print(f"Loaded {npz_path.name}: n_rec={n_rec}, n_passes={n_passes}, "
          f"encoder_seed={enc_seed}, probe_seed={probe_seed}")

    L1, L2 = {}, {}

    # --- stim regression + classification --------------------------------
    for fname_feat in feature_names:
        rk = f"reg_{fname_feat}"
        if f"test_{rk}_pred" in npz:
            pred = npz[f"test_{rk}_pred"]
            tgt = npz[f"test_{rk}_target"]
            L1[f"{rk}_corr"] = _safe_corr(tgt, pred)
            pred_g = _reshape_per_rec(pred, n_rec, n_passes)
            tgt_g = _reshape_per_rec(tgt, n_rec, n_passes)
            L2[f"{rk}_corr"] = _bootstrap_metric(pred_g, tgt_g, n_bootstrap, rng, "corr")

        ck = f"cls_{fname_feat}"
        if f"test_{ck}_prob" in npz:
            prob = npz[f"test_{ck}_prob"]
            tgt = npz[f"test_{ck}_target"]
            L1[f"{ck}_auc"] = _safe_auc(tgt, prob)
            L1[f"{ck}_bal_acc"] = _safe_bal_acc(tgt, prob)
            prob_g = _reshape_per_rec(prob, n_rec, n_passes)
            tgt_g = _reshape_per_rec(tgt, n_rec, n_passes)
            L2[f"{ck}_auc"] = _bootstrap_metric(prob_g, tgt_g, n_bootstrap, rng, "auc")
            L2[f"{ck}_bal_acc"] = _bootstrap_metric(prob_g, tgt_g, n_bootstrap, rng, "bal_acc")

    # --- movie_id (top-1 / top-5) ----------------------------------------
    if "test_movie_id_probs" in npz:
        probs = npz["test_movie_id_probs"]            # (N, n_bins)
        ybin = npz["test_movie_id_target_bin"]        # (N,)
        n_bins = probs.shape[1]
        # Flat L1
        preds_flat = probs.argmax(axis=1)
        k5 = min(5, n_bins)
        top5_flat = np.argsort(-probs, axis=1)[:, :k5]
        L1["movie_id_top1"] = float((preds_flat == ybin).mean())
        L1["movie_id_top5"] = float((top5_flat == ybin[:, None]).any(axis=1).mean())

        probs_g = _reshape_per_rec(probs, n_rec, n_passes)         # (n_rec, n_passes, n_bins)
        ybin_g = _reshape_per_rec(ybin.astype(np.int64), n_rec, n_passes)
        s1, s5 = _bootstrap_movie_id(probs_g, ybin_g, n_bootstrap, rng)
        L2["movie_id_top1"] = s1
        L2["movie_id_top5"] = s5

    # --- age regression --------------------------------------------------
    if "test_age_reg_pred" in npz:
        pred = npz["test_age_reg_pred"]
        tgt = npz["test_age_reg_target"]
        valid = ~np.isnan(tgt)
        if valid.any():
            L1["age_reg_corr"] = _safe_corr(tgt[valid], pred[valid])
            # Restrict to valid rows in (n_rec, n_passes) shape. Since age is per-rec
            # (constant within a recording), NaN-validity is also per-rec.
            pred_g = _reshape_per_rec(pred, n_rec, n_passes)
            tgt_g = _reshape_per_rec(tgt, n_rec, n_passes)
            rec_valid = ~np.isnan(tgt_g[:, 0])
            L2["age_reg_corr"] = _bootstrap_metric(
                pred_g[rec_valid], tgt_g[rec_valid], n_bootstrap, rng, "corr",
            )

    # --- sex AUC ---------------------------------------------------------
    if "test_sex_prob" in npz:
        prob = npz["test_sex_prob"]
        tgt = npz["test_sex_target"]
        valid = ~np.isnan(tgt)
        if valid.any():
            L1["sex_auc"] = _safe_auc(tgt[valid], prob[valid])
            prob_g = _reshape_per_rec(prob, n_rec, n_passes)
            tgt_g = _reshape_per_rec(tgt, n_rec, n_passes)
            rec_valid = ~np.isnan(tgt_g[:, 0])
            L2["sex_auc"] = _bootstrap_metric(
                prob_g[rec_valid], tgt_g[rec_valid], n_bootstrap, rng, "auc",
            )

    # --- emit JSON -------------------------------------------------------
    payload = {
        "seed": enc_seed,
        "probe_seed": probe_seed,
        "n_passes": n_passes,
        "n_bootstrap": n_bootstrap,
        "n_rec_test": n_rec,
        "predictions_npz": str(npz_path),
        "L1": L1,
        "L2": L2,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")

    # --- print summary table --------------------------------------------
    print(f"\n{'metric':<40} {'L1':>9} {'L2 mean':>9} {'95% CI':>22}")
    print("-" * 84)
    for k in L1:
        l1v = L1.get(k, float("nan"))
        l2 = L2.get(k, {})
        if l2:
            ci = f"[{l2['ci_lo']:+.3f}, {l2['ci_hi']:+.3f}]"
            print(f"{k:<40} {l1v:>+9.4f} {l2['mean']:>+9.4f} {ci:>22}")
        else:
            print(f"{k:<40} {l1v:>+9.4f}       —")

    # --- W&B append ------------------------------------------------------
    if wandb_run_id:
        try:
            import wandb
            run = wandb.init(project=wandb_project, id=wandb_run_id, resume="must")
            flat = {}
            for k, v in L1.items():
                flat[f"bootstrap_canonical/L1/{k}"] = v
            for k, s in L2.items():
                for stat in ("mean", "std", "ci_lo", "ci_hi"):
                    flat[f"bootstrap_canonical/L2/{k}/{stat}"] = s[stat]
            flat["bootstrap_canonical/n_bootstrap"] = n_bootstrap
            run.log(flat)
            run.finish()
        except Exception as e:
            print(f"[bootstrap_canonical] W&B logging failed: {e}")

    return payload


if __name__ == "__main__":
    fire.Fire(run)
