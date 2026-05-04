"""Unified canonical probe eval — one script, five probe families.

Implements the standard evaluation protocol specified in
``docs/evaluation_guide.md``:

  - Stim regression (4 features × {Pearson r, R²}):     Ridge(α=1.0)
  - Stim classification (4 features × {AUC, bal_acc}):  LogReg(C=1, lbfgs)
  - Subject age (continuous):                           Ridge(α=1.0)
  - Subject sex (binary):                               LogReg(C=1, lbfgs)
  - Movie ID (20-bin top-1, top-5):                     LogReg(C=1, multinomial, lbfgs)

All on kc-pool features (5 channels × 64 dim = 320-d per clip), n_passes
random clip draws averaged per recording. Closed-form / LBFGS solvers
throughout — deterministic given (encoder, probe_seed).

Output:
  - results/<exp_tag>/<seed>/metrics.json — all 18 headline numbers
  - predictions/<exp_tag>/<seed>/test_seed{seed}.npz — per-recording preds
    for bootstrap (B=2000 recording-level)

Usage on Delta:
  PYTHONPATH=. uv run --group eeg python scripts/unified_probe_eval.py \\
      --checkpoint=/path/to/latest.pth.tar \\
      --n_windows=2 --window_size_seconds=4 \\
      --norm_mode=per_recording --corrca_filters=corrca_filters.npz \\
      --n_passes=20 --seed=42 \\
      --output_json=results/unified/<tag>_seed42.json \\
      --save_predictions_dir=predictions/unified/<tag>_seed42

Reference: ``docs/evaluation_guide.md`` (canonical protocol).
"""

import json
import math
import os
from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_seed
from experiments.eeg_jepa.main import resolve_preprocessed_dir

logger = get_logger(__name__)


# ============================================================
# Encoder + feature extraction (kc-pool only — mean-pool dropped per
# evaluation_guide.md anti-patterns).
# ============================================================

def _load_encoder(checkpoint, train_set, cfg, n_windows):
    """Load the context encoder from a JEPA checkpoint."""
    from eb_jepa.architectures import EEGEncoderTokens
    from eb_jepa.training_utils import setup_device

    device = setup_device("auto")
    sd_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = sd_dict.get("model_state_dict", sd_dict)
    encoder = EEGEncoderTokens(
        n_chans=train_set.n_chans,
        n_times=train_set.n_times,
        embed_dim=cfg.model.encoder_embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=n_windows,
        patch_size=cfg.model.get("patch_size", 50),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=train_set.get_chs_info(),
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
    ).to(device)
    ce_sd = {k[len("context_encoder."):]: v
             for k, v in sd.items() if k.startswith("context_encoder.")}
    encoder.load_state_dict(ce_sd, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, device


def _kc_features(eeg, encoder, device):
    """kc-pool: 5 ch × 64 d = 320-d per clip, mean across windows."""
    with torch.no_grad():
        x = eeg.unsqueeze(0).to(device)
        tokens = encoder.encode_tokens(x, mask=None)
        B = tokens.shape[0]
        C = encoder.n_chans
        T = encoder.n_windows
        P = encoder.n_patches_per_window
        D = encoder.embed_dim
        x_tok = tokens.view(B, C, T, P, D)
        pooled = x_tok.mean(dim=3)                       # [B, C, T, D]
        pooled = pooled.permute(0, 2, 1, 3).reshape(B, T, C * D)
        emb = pooled.mean(dim=1).squeeze(0).cpu().numpy()  # [C*D]
    return emb


def _extract(dataset, n_passes, encoder, device, seed):
    """Per-recording: n_passes random clips → kc-pool features + per-clip
    feature labels. Returns features [n_rec, n_passes, 320] and labels
    [n_rec, n_passes, n_features] grouped by recording."""
    rng = torch.Generator().manual_seed(seed)
    n_rec = len(dataset)
    feats_arr = []
    labels_arr = []
    for rec_idx in range(n_rec):
        f_passes = []
        l_passes = []
        for _ in range(n_passes):
            eeg, feats, _ = dataset[rec_idx]
            f_passes.append(_kc_features(eeg, encoder, device))
            l_passes.append(feats.mean(dim=0).numpy())
        feats_arr.append(np.stack(f_passes))
        labels_arr.append(np.stack(l_passes))
    X = np.stack(feats_arr)     # [n_rec, n_passes, D]
    Y = np.stack(labels_arr)    # [n_rec, n_passes, n_features]
    return X, Y


def _subject_labels(dataset):
    """Extract age (float) and sex (0/1, NaN if missing) per recording."""
    n = len(dataset)
    ages = np.full(n, np.nan, dtype=np.float32)
    sexes = np.full(n, np.nan, dtype=np.float32)
    for i, m in enumerate(dataset._recording_metadata):
        a = m.get("age", None)
        if a is not None and not (isinstance(a, float) and math.isnan(a)):
            try:
                ages[i] = float(a)
            except (TypeError, ValueError):
                pass
        s = m.get("sex", m.get("gender", ""))
        if isinstance(s, str):
            s = s.strip().lower()
            if s in ("m", "male", "1", "1.0"):
                sexes[i] = 1.0
            elif s in ("f", "female", "0", "0.0"):
                sexes[i] = 0.0
    return ages, sexes


# ============================================================
# Probe families (canonical heads per evaluation_guide.md).
# ============================================================

def _ridge_reg(Xtr, ytr, Xev, yev):
    """Per-recording Ridge regression. Returns Pearson r + R² + per-rec preds."""
    valid_tr = ~np.isnan(ytr)
    valid_ev = ~np.isnan(yev)
    if valid_tr.sum() < 2 or valid_ev.sum() < 2:
        return float("nan"), float("nan"), np.full(len(yev), np.nan)
    ym, ys = ytr[valid_tr].mean(), ytr[valid_tr].std() + 1e-8
    probe = Ridge(alpha=1.0).fit(Xtr[valid_tr], (ytr[valid_tr] - ym) / ys)
    pred_norm = probe.predict(Xev)
    pred = pred_norm * ys + ym
    r = pearsonr(pred[valid_ev], yev[valid_ev]).statistic
    r2 = r2_score(yev[valid_ev], pred[valid_ev])
    return float(r), float(r2), pred


def _logreg_bin(Xtr, ytr, Xev, yev):
    """Binary LogReg (median-split for continuous targets via caller)."""
    valid_tr = ~np.isnan(ytr)
    valid_ev = ~np.isnan(yev)
    if valid_tr.sum() < 4 or valid_ev.sum() < 4 or len(np.unique(ytr[valid_tr])) < 2:
        return float("nan"), float("nan"), np.full(len(yev), np.nan)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(Xtr[valid_tr], ytr[valid_tr].astype(int))
    proba = clf.predict_proba(Xev)[:, 1]
    pred = (proba > 0.5).astype(int)
    auc = roc_auc_score(yev[valid_ev], proba[valid_ev]) if len(np.unique(yev[valid_ev])) > 1 else float("nan")
    bal = balanced_accuracy_score(yev[valid_ev], pred[valid_ev])
    return float(auc), float(bal), proba


def _logreg_multi(Xtr, ytr, Xev, yev, n_classes):
    """Multinomial LogReg for movie_id top-1 / top-5.

    Note: scikit-learn dropped the `multi_class` kwarg; lbfgs solver
    defaults to multinomial when n_classes > 2."""
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    clf.fit(Xtr, ytr.astype(int))
    proba = clf.predict_proba(Xev)
    # top-1 / top-5
    topk = min(5, n_classes)
    top1 = (proba.argmax(axis=1) == yev.astype(int)).mean()
    top5_idx = np.argpartition(-proba, kth=min(topk - 1, proba.shape[1] - 1), axis=1)[:, :topk]
    top5 = np.array([yev[i].astype(int) in top5_idx[i] for i in range(len(yev))]).mean()
    return float(top1), float(top5), proba


# ============================================================
# Main entry — orchestrates all 5 probe families.
# ============================================================

def run(
    checkpoint: str,
    n_windows: int = 4,
    window_size_seconds: int = 2,
    norm_mode: str = "per_recording",
    corrca_filters: str = "",
    n_passes: int = 20,
    seed: int = 42,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    output_json: str = "",
    save_predictions_dir: str = "",
    movie_id_n_bins: int = 20,
):
    setup_seed(seed)
    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.norm_mode": norm_mode,
    }
    if corrca_filters:
        overrides["data.corrca_filters"] = corrca_filters
    cfg = load_config(fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feature_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))

    logger.info("Loading datasets (n_passes=%d) ...", n_passes)
    train_set = JEPAMovieDataset(
        split="train", n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feature_names, cfg=cfg.data,
        preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )
    eval_sets = {}
    for split in ("val", "test"):
        eval_sets[split] = JEPAMovieDataset(
            split=split, n_windows=n_windows, window_size_seconds=window_size_seconds,
            feature_names=feature_names,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data, preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
        )
    logger.info("n_chans=%d, n_train=%d, n_val=%d, n_test=%d",
                train_set.n_chans, len(train_set), len(eval_sets["val"]), len(eval_sets["test"]))

    encoder, device = _load_encoder(checkpoint, train_set, cfg, n_windows)
    logger.info("Loaded encoder; extracting features ...")

    Xtr_g, Ytr_g = _extract(train_set,         n_passes, encoder, device, seed)
    Xv_g,  Yv_g  = _extract(eval_sets["val"],  n_passes, encoder, device, seed + 1)
    Xt_g,  Yt_g  = _extract(eval_sets["test"], n_passes, encoder, device, seed + 2)
    logger.info("Train: X=%s Y=%s; Val: X=%s; Test: X=%s",
                Xtr_g.shape, Ytr_g.shape, Xv_g.shape, Xt_g.shape)

    # Standardize features (per-feature train stats)
    Xtr_flat = Xtr_g.reshape(-1, Xtr_g.shape[-1])
    mu = Xtr_flat.mean(axis=0, keepdims=True)
    sd = Xtr_flat.std(axis=0, keepdims=True) + 1e-8
    def _stdz(X): return (X - mu) / sd
    Xtr_g = _stdz(Xtr_g); Xv_g = _stdz(Xv_g); Xt_g = _stdz(Xt_g)

    # ===== Stim regression / classification: per-clip semantics (PR #15 protocol)
    # Flatten (n_rec, n_passes) → (n_rec × n_passes,) for Ridge / LogReg fitting
    # so the probe sees ALL clips, not per-recording means. Matches
    # trivial_ridge_baseline.py exactly.
    Xtr_flat_clips = Xtr_g.reshape(-1, Xtr_g.shape[-1])    # [n_train_clips, D]
    Xv_flat_clips  = Xv_g.reshape(-1,  Xv_g.shape[-1])
    Xt_flat_clips  = Xt_g.reshape(-1,  Xt_g.shape[-1])
    Ytr_flat_clips = Ytr_g.reshape(-1, Ytr_g.shape[-1])    # [n_train_clips, n_features]
    Yv_flat_clips  = Yv_g.reshape(-1,  Yv_g.shape[-1])
    Yt_flat_clips  = Yt_g.reshape(-1,  Yt_g.shape[-1])

    # ===== Subject + movie_id: per-recording semantics
    # These are recording-level labels; can't be flattened without changing meaning.
    age_tr, sex_tr = _subject_labels(train_set)
    age_v,  sex_v  = _subject_labels(eval_sets["val"])
    age_t,  sex_t  = _subject_labels(eval_sets["test"])
    Ytr_rec = Ytr_g.mean(axis=1)
    Yv_rec  = Yv_g.mean(axis=1)
    Yt_rec  = Yt_g.mean(axis=1)
    Xtr_rec = Xtr_g.mean(axis=1)  # [n_rec, D]
    Xv_rec  = Xv_g.mean(axis=1)
    Xt_rec  = Xt_g.mean(axis=1)
    pos_idx = feature_names.index("position_in_movie") if "position_in_movie" in feature_names else None
    if pos_idx is not None:
        pos_tr_rec = Ytr_rec[:, pos_idx]
        pos_v_rec  = Yv_rec[:,  pos_idx]
        pos_t_rec  = Yt_rec[:,  pos_idx]
        pos_min = pos_tr_rec.min(); pos_max = pos_tr_rec.max() + 1e-8
        edges = np.linspace(pos_min, pos_max, movie_id_n_bins + 1)
        bin_tr = np.clip(np.digitize(pos_tr_rec, edges) - 1, 0, movie_id_n_bins - 1)
        bin_v  = np.clip(np.digitize(pos_v_rec,  edges) - 1, 0, movie_id_n_bins - 1)
        bin_t  = np.clip(np.digitize(pos_t_rec,  edges) - 1, 0, movie_id_n_bins - 1)

    metrics = {}
    preds_npz = {"feature_names": np.array(feature_names, dtype="<U24")}

    # ---- Stim regression + classification (4 features each) — per-clip Ridge/LogReg
    # Train on FLATTENED clips (n_train × n_passes) → match trivial_ridge_baseline.py
    for fi, fname_feat in enumerate(feature_names):
        ytr = Ytr_flat_clips[:, fi]
        yv  = Yv_flat_clips[:, fi]
        yt  = Yt_flat_clips[:, fi]
        # Regression
        for split, X, y, tag in [("val", Xv_flat_clips, yv, "val"), ("test", Xt_flat_clips, yt, "test")]:
            r, r2, pred = _ridge_reg(Xtr_flat_clips, ytr, X, y)
            metrics[f"{tag}/reg_{fname_feat}_corr"] = r
            metrics[f"{tag}/reg_{fname_feat}_r2"] = r2
            preds_npz[f"{tag}_reg_{fname_feat}_pred"] = pred.astype(np.float32)
            preds_npz[f"{tag}_reg_{fname_feat}_target"] = y.astype(np.float32)
        # Classification (binarize at train median over flattened train labels)
        med = np.nanmedian(ytr)
        ytr_bin = (ytr > med).astype(np.float32)
        for split, X, y, tag in [("val", Xv_flat_clips, yv, "val"), ("test", Xt_flat_clips, yt, "test")]:
            y_bin = (y > med).astype(np.float32)
            auc, bal, proba = _logreg_bin(Xtr_flat_clips, ytr_bin, X, y_bin)
            metrics[f"{tag}/cls_{fname_feat}_auc"] = auc
            metrics[f"{tag}/cls_{fname_feat}_bal_acc"] = bal
            preds_npz[f"{tag}_cls_{fname_feat}_proba"] = proba.astype(np.float32)
            preds_npz[f"{tag}_cls_{fname_feat}_target"] = y_bin.astype(np.float32)

    # ---- Subject — age regression ----
    for split, X, y, tag in [("val", Xv_rec, age_v, "val"), ("test", Xt_rec, age_t, "test")]:
        r, r2, pred = _ridge_reg(Xtr_rec, age_tr, X, y)
        metrics[f"{tag}/subject/age_reg/corr"] = r
        metrics[f"{tag}/subject/age_reg/r2"] = r2
        preds_npz[f"{tag}_age_pred"] = pred.astype(np.float32)
        preds_npz[f"{tag}_age_target"] = y.astype(np.float32)

    # ---- Subject — sex classification ----
    for split, X, y, tag in [("val", Xv_rec, sex_v, "val"), ("test", Xt_rec, sex_t, "test")]:
        auc, bal, proba = _logreg_bin(Xtr_rec, sex_tr, X, y)
        metrics[f"{tag}/subject/sex/auc"] = auc
        metrics[f"{tag}/subject/sex/bal_acc"] = bal
        preds_npz[f"{tag}_sex_proba"] = proba.astype(np.float32)
        preds_npz[f"{tag}_sex_target"] = y.astype(np.float32)

    # ---- Movie-ID 20-class top-1 / top-5 ----
    # Wrapped in try/except so a failure here doesn't drop the other 16 metrics.
    if pos_idx is not None:
        for split, X, y, tag in [("val", Xv_rec, bin_v, "val"), ("test", Xt_rec, bin_t, "test")]:
            try:
                top1, top5, proba = _logreg_multi(Xtr_rec, bin_tr, X, y, movie_id_n_bins)
                metrics[f"{tag}/movie_id/top1"] = top1
                metrics[f"{tag}/movie_id/top5"] = top5
                preds_npz[f"{tag}_movie_id_proba"] = proba.astype(np.float32)
                preds_npz[f"{tag}_movie_id_target"] = y.astype(np.int32)
            except Exception as e:
                logger.warning("movie_id %s probe failed: %s", tag, e)
                metrics[f"{tag}/movie_id/top1"] = float("nan")
                metrics[f"{tag}/movie_id/top5"] = float("nan")

    # rec_ids for bootstrap
    preds_npz["val_rec_ids"]  = np.arange(len(eval_sets["val"]),  dtype=np.int64)
    preds_npz["test_rec_ids"] = np.arange(len(eval_sets["test"]), dtype=np.int64)

    # ---- Persist ----
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump({
                "checkpoint": checkpoint,
                "n_windows": n_windows,
                "window_size_seconds": window_size_seconds,
                "n_passes": n_passes,
                "seed": seed,
                "metrics": metrics,
                "protocol": "unified_probe_eval — kc-pool + Ridge(α=1) + LogReg(C=1, lbfgs); see docs/evaluation_guide.md",
            }, f, indent=2)
        logger.info("Wrote %s", output_json)

    if save_predictions_dir:
        Path(save_predictions_dir).mkdir(parents=True, exist_ok=True)
        npz_path = os.path.join(save_predictions_dir, f"test_seed{seed}.npz")
        np.savez_compressed(npz_path, **preds_npz)
        logger.info("Wrote per-rec predictions: %s", npz_path)

    # ---- Print headline (test split) ----
    logger.info("=== Headline metrics (test split) ===")
    for k in sorted(metrics):
        if k.startswith("test/"):
            logger.info("  %-50s %+.4f", k, metrics[k])


if __name__ == "__main__":
    fire.Fire(run)
