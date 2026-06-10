"""Spec-faithful probe eval: sklearn closed-form heads + n_passes=20.

Design:

  - `sklearn.Ridge(alpha=1)` for stim regression and age regression
  - `sklearn.LogisticRegression(C=1, lbfgs)` for binary cls (sex, stim median-split)
  - `sklearn.LogisticRegression(multinomial, lbfgs, max_iter=2000)` for movie_id (20 bins)
  - Deterministic given (encoder weights, probe_seed, train/val/test data)
  - Per-clip extraction with n_passes=20 augmentation:
      * Train: outer pass × inner shuffled-recording (train_order=True), probe_seed-controlled
      * Val/test: rec × pass (sequential) so saved NPZ reshapes cleanly to (n_rec, n_passes)
  - Per-clip predictions saved as flat (n_rec * n_passes,) arrays under `test_*_pred` /
    `test_*_target` keys for downstream `eb_jepa.evaluation.bootstrap`.
  - L1 = single Pearson r / AUC / top-k on the full flat test array (no resampling)

Usage
-----
uv run --group eeg python -m eb_jepa.evaluation.probe_eval \\
    --checkpoint=/abs/path/latest.pth.tar \\
    --n_windows=2 --window_size_seconds=4 --batch_size=64 \\
    --norm_mode=per_recording --corrca_filters=corrca_filters.npz \\
    --n_passes=20 --probe_seed=42 \\
    --save_predictions_dir=/abs/path/preds --seed=42
"""

import json
import math
from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.stats import pearsonr

from eb_jepa.training.builder import build_jepa, check_old_checkpoint_format
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.logging import get_logger
from eb_jepa.paths import resolve_preprocessed_dir
from eb_jepa.training_utils import load_checkpoint, load_config, setup_device, setup_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Deterministic per-clip extraction with n_passes augmentation
# ---------------------------------------------------------------------------

def _embed_clip(dataset, jepa, rec_idx, device, sample_seed):
    """Sample one clip from `rec_idx` with deterministic random start, encode.

    JEPAMovieDataset.__getitem__ uses torch.randint(0, n-required+1, (1,)) without
    a generator, so it consumes the global torch RNG state. Seeding torch with a
    deterministic per-(rec, pass) value yields reproducible clip starts.

    Returns
    -------
    emb : np.ndarray [D]   clip embedding (mean over n_windows)
    feats_mean : np.ndarray [n_features]   target features (mean over n_windows)
    """
    torch.manual_seed(sample_seed)
    eeg, feats, _, _, _ = dataset[rec_idx]        # eeg: [n_windows, C, T]
    eeg = eeg.unsqueeze(0).to(device)             # [1, n_windows, C, T]
    with torch.no_grad():
        # encode returns [1, D, T, 1, 1]; mean over T windows → [1, D]
        state = jepa.encode(eeg, keep_channels=False)
        emb = state.squeeze(-1).squeeze(-1).mean(dim=2).squeeze(0).cpu().numpy()
    feats_mean = feats.mean(dim=0).numpy()        # [n_features]
    return emb, feats_mean


def _extract(dataset, jepa, device, n_passes, probe_seed, train_order):
    """Encode every recording n_passes times → flat (n_rec*n_passes, ...) arrays.

    Train order: outer pass × inner shuffled-recording (shuffle reseeded each pass
    from the probe_seed RNG). Saved iteration order matches the L1 spec so different
    runs at the same probe_seed produce identical training matrices.

    Val/test order: rec × passes (sequential) so the saved array reshapes cleanly
    to (n_rec, n_passes) for recording-level bootstrap.

    Returns
    -------
    embs : np.ndarray [n_rec * n_passes, D]
    feats : np.ndarray [n_rec * n_passes, n_features]
    rec_ids : np.ndarray [n_rec * n_passes]  (which recording each row came from)
    """
    n_rec = len(dataset)
    embs, feats_out, rec_ids = [], [], []

    if train_order:
        order_gen = torch.Generator().manual_seed(probe_seed)
        for p in range(n_passes):
            order = torch.randperm(n_rec, generator=order_gen).tolist()
            for rec_idx in order:
                sample_seed = probe_seed * 1_000_003 + p * n_rec + rec_idx
                emb, fm = _embed_clip(dataset, jepa, rec_idx, device, sample_seed)
                embs.append(emb)
                feats_out.append(fm)
                rec_ids.append(rec_idx)
    else:
        for rec_idx in range(n_rec):
            for p in range(n_passes):
                sample_seed = probe_seed * 1_000_003 + p * n_rec + rec_idx
                emb, fm = _embed_clip(dataset, jepa, rec_idx, device, sample_seed)
                embs.append(emb)
                feats_out.append(fm)
                rec_ids.append(rec_idx)

    return (
        np.stack(embs).astype(np.float32),
        np.stack(feats_out).astype(np.float32),
        np.array(rec_ids, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Subject-level (age/sex) labels from metadata
# ---------------------------------------------------------------------------

def _subject_labels(dataset):
    """Return arrays of age and sex labels (NaN where unavailable), one per recording."""
    n = len(dataset)
    ages = np.full(n, np.nan, dtype=np.float64)
    sexes = np.full(n, np.nan, dtype=np.float64)
    for i, m in enumerate(dataset._recording_metadata):
        if "age" in m:
            try:
                ages[i] = float(m["age"])
            except (ValueError, TypeError):
                pass
        sex_val = m.get("sex", m.get("gender", ""))
        if isinstance(sex_val, str):
            s = sex_val.strip().lower()
            if s in ("m", "male"):
                sexes[i] = 1.0
            elif s in ("f", "female"):
                sexes[i] = 0.0
    return ages, sexes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    checkpoint: str,
    # Data — must match the checkpoint's training config
    n_windows: int = 2,
    window_size_seconds: int = 4,
    batch_size: int = 64,
    num_workers: int = 4,
    norm_mode: str = "per_recording",
    corrca_filters: str = "corrca_filters.npz",
    add_envelope: bool = False,
    # Spec-faithful eval knobs
    n_passes: int = 20,
    probe_seed: int = 42,
    n_movie_id_bins: int = 20,
    # I/O
    save_predictions_dir: str = "",
    out_json: str = "",
    # W&B
    wandb_run_id: str = "",
    wandb_project: str = "eb_jepa",
    wandb_group: str = "probe_eval",
    # Misc
    fname: str = "config/jepa_pretrain.yaml",
    seed: int = 42,
):
    """Run the spec-faithful probe eval on a frozen MaskedJEPA checkpoint.

    Args
    ----
    checkpoint : abs path to a saved latest.pth.tar
    n_passes : passes per recording; train sees n_rec*n_passes clips total
    probe_seed : RNG seed for sample order + clip starts. Fix to 42 across encoder
        seeds so σ_5seed reflects encoder-training variance only.
    seed : encoder seed tag (used for naming saved NPZ; doesn't affect sklearn fits)
    save_predictions_dir : if set, dumps per-clip flat predictions/targets NPZ per split
    out_json : if set, writes per-seed L1 metrics JSON (informational; bootstrap is the
        sole source of L1+L2 numbers downstream)
    """
    setup_seed(seed)
    device = setup_device("auto")

    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    # Infer encoder_depth and predictor_embed_dim from saved state_dict
    _ckpt_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("model_state_dict", {})
    check_old_checkpoint_format(_ckpt_sd)
    _depth = max(
        int(k.split(".")[3]) + 1
        for k in _ckpt_sd
        if k.startswith("encoder.transformer.layers.")
    )
    if "predictor.input_proj.weight" in _ckpt_sd:
        _pred_dim = int(_ckpt_sd["predictor.input_proj.weight"].shape[0])
    elif "predictor.output_proj.weight" in _ckpt_sd:
        _pred_dim = int(_ckpt_sd["predictor.output_proj.weight"].shape[0])
    else:
        _pred_dim = None

    _overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "data.norm_mode": norm_mode,
        "data.add_envelope": add_envelope,
        "model.encoder_depth": _depth,
        "model.predictor_embed_dim": _pred_dim,
    }
    if corrca_filters:
        _overrides["data.corrca_filters"] = corrca_filters
    cfg = load_config(fname, _overrides)

    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    eval_cfg = cfg.get("eval", {}) or {}
    feature_names = list(eval_cfg.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))
    visual_processing_delay_s = float(eval_cfg.get("visual_processing_delay_s", 0.0))

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading datasets (n_passes=%d, probe_seed=%d)...", n_passes, probe_seed)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        cfg=cfg.data,
        visual_processing_delay_s=visual_processing_delay_s,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    val_set = JEPAMovieDataset(
        split="val",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data,
        visual_processing_delay_s=visual_processing_delay_s,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    test_set = JEPAMovieDataset(
        split="test",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data,
        visual_processing_delay_s=visual_processing_delay_s,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    logger.info(
        "Split sizes — train=%d, val=%d, test=%d",
        len(train_set), len(val_set), len(test_set),
    )

    # ------------------------------------------------------------------
    # Frozen encoder
    # ------------------------------------------------------------------
    n_chans = train_set.n_chans
    n_times = train_set.n_times
    chs_info = train_set.get_chs_info()

    jepa = build_jepa(
        cfg, n_chans=n_chans, n_times=n_times,
        chs_info=chs_info, n_windows=n_windows,
    ).to(device)
    ckpt_info = load_checkpoint(checkpoint_path, jepa, optimizer=None, device=device, strict=False)
    logger.info("Loaded checkpoint at epoch %d", ckpt_info.get("epoch", "?"))
    for p in jepa.parameters():
        p.requires_grad_(False)
    jepa.eval()

    # ------------------------------------------------------------------
    # Extract embeddings + features per split (n_passes augmentation)
    # ------------------------------------------------------------------
    logger.info("Extracting TRAIN clips (train_order=True)...")
    Xtr, Ytr, rec_tr = _extract(train_set, jepa, device, n_passes, probe_seed, train_order=True)
    logger.info("  TRAIN  shape: X=%s  Y=%s", Xtr.shape, Ytr.shape)

    logger.info("Extracting VAL clips (rec × passes)...")
    Xv, Yv, rec_v = _extract(val_set, jepa, device, n_passes, probe_seed, train_order=False)
    logger.info("  VAL    shape: X=%s  Y=%s", Xv.shape, Yv.shape)

    logger.info("Extracting TEST clips (rec × passes)...")
    Xt, Yt, rec_t = _extract(test_set, jepa, device, n_passes, probe_seed, train_order=False)
    logger.info("  TEST   shape: X=%s  Y=%s", Xt.shape, Yt.shape)

    # Per-recording embeddings (mean over the n_passes clip embeddings) for subject probes
    def _per_rec_emb(X, rec_ids, n_rec):
        out = np.zeros((n_rec, X.shape[1]), dtype=np.float32)
        for r in range(n_rec):
            mask = rec_ids == r
            if mask.any():
                out[r] = X[mask].mean(axis=0)
        return out

    Xtr_rec = _per_rec_emb(Xtr, rec_tr, len(train_set))
    Xv_rec = _per_rec_emb(Xv, rec_v, len(val_set))
    Xt_rec = _per_rec_emb(Xt, rec_t, len(test_set))

    # ------------------------------------------------------------------
    # Stim probes (one Ridge + one LogReg per feature)
    # ------------------------------------------------------------------
    save = {}
    L1 = {}
    for fi, fname_feat in enumerate(feature_names):
        ytr = Ytr[:, fi].astype(np.float64)
        yt = Yt[:, fi].astype(np.float64)
        yv = Yv[:, fi].astype(np.float64)

        ym = float(ytr.mean())
        ys = float(ytr.std() + 1e-12)

        # Regression: Ridge(alpha=1) on standardized y
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr, (ytr - ym) / ys)
        pred_t = reg.predict(Xt) * ys + ym
        pred_v = reg.predict(Xv) * ys + ym
        save[f"test_reg_{fname_feat}_pred"] = pred_t.astype(np.float32)
        save[f"test_reg_{fname_feat}_target"] = yt.astype(np.float32)
        save[f"val_reg_{fname_feat}_pred"] = pred_v.astype(np.float32)
        save[f"val_reg_{fname_feat}_target"] = yv.astype(np.float32)
        try:
            r = float(pearsonr(pred_t, yt).statistic)
        except Exception:
            r = float("nan")
        L1[f"reg_{fname_feat}_corr"] = r

        # Classification: LogReg(C=1, lbfgs) on median-split
        med = float(np.median(ytr))
        ytr_bin = (ytr > med).astype(int)
        yt_bin = (yt > med).astype(int)
        yv_bin = (yv > med).astype(int)
        if len(np.unique(ytr_bin)) >= 2:
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            clf.fit(Xtr, ytr_bin)
            prob_t = clf.predict_proba(Xt)[:, 1]
            prob_v = clf.predict_proba(Xv)[:, 1]
            save[f"test_cls_{fname_feat}_prob"] = prob_t.astype(np.float32)
            save[f"test_cls_{fname_feat}_target"] = yt_bin.astype(np.int8)
            save[f"val_cls_{fname_feat}_prob"] = prob_v.astype(np.float32)
            save[f"val_cls_{fname_feat}_target"] = yv_bin.astype(np.int8)
            try:
                auc = float(roc_auc_score(yt_bin, prob_t))
            except ValueError:
                auc = float("nan")
            L1[f"cls_{fname_feat}_auc"] = auc

    # ------------------------------------------------------------------
    # Movie identity (20-bin multinomial LR on position_in_movie)
    # ------------------------------------------------------------------
    if "position_in_movie" in feature_names:
        pos_idx = feature_names.index("position_in_movie")
        pos_tr = Ytr[:, pos_idx].astype(np.float64)
        pos_t = Yt[:, pos_idx].astype(np.float64)
        pos_v = Yv[:, pos_idx].astype(np.float64)
        edges = np.linspace(pos_tr.min(), pos_tr.max() + 1e-8, n_movie_id_bins + 1)
        ybin_tr = np.clip(np.digitize(pos_tr, edges) - 1, 0, n_movie_id_bins - 1)
        ybin_t = np.clip(np.digitize(pos_t, edges) - 1, 0, n_movie_id_bins - 1)
        ybin_v = np.clip(np.digitize(pos_v, edges) - 1, 0, n_movie_id_bins - 1)
        if len(np.unique(ybin_tr)) >= 2:
            # sklearn ≥1.5 dropped the multi_class= kwarg; lbfgs auto-handles
            # multinomial when >2 classes are present in y.
            mid_clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            mid_clf.fit(Xtr, ybin_tr)
            probs_t = mid_clf.predict_proba(Xt)
            probs_v = mid_clf.predict_proba(Xv)
            save["test_movie_id_probs"] = probs_t.astype(np.float32)
            save["test_movie_id_target_bin"] = ybin_t.astype(np.int8)
            save["test_movie_id_bin_edges"] = edges.astype(np.float32)
            save["val_movie_id_probs"] = probs_v.astype(np.float32)
            save["val_movie_id_target_bin"] = ybin_v.astype(np.int8)
            preds = probs_t.argmax(axis=1)
            top5 = np.argsort(-probs_t, axis=1)[:, :min(5, n_movie_id_bins)]
            L1["movie_id_top1"] = float((preds == ybin_t).mean())
            L1["movie_id_top5"] = float((top5 == ybin_t[:, None]).any(axis=1).mean())

    # ------------------------------------------------------------------
    # Subject probes (age regression + sex binary) — per-recording level
    # Predictions are tiled to (n_rec * n_passes,) so bootstrap reshape (n_rec, n_passes)
    # is uniform with the stim/movie_id arrays.
    # ------------------------------------------------------------------
    ages_tr, sex_tr = _subject_labels(train_set)
    ages_t, sex_t = _subject_labels(test_set)
    ages_v, sex_v = _subject_labels(val_set)

    # Age regression: Ridge on per-rec embeddings, fit on train recs with valid age
    valid_tr = ~np.isnan(ages_tr)
    if valid_tr.sum() >= 10:
        atr = ages_tr[valid_tr]
        am, asd = float(atr.mean()), float(atr.std() + 1e-12)
        age_reg = Ridge(alpha=1.0)
        age_reg.fit(Xtr_rec[valid_tr], (atr - am) / asd)
        # Predict on every rec (even NaN labels); we mask out NaN labels per row downstream
        age_pred_t = age_reg.predict(Xt_rec) * asd + am
        age_pred_v = age_reg.predict(Xv_rec) * asd + am
        # Tile to per-clip flat (n_rec * n_passes,) via rec_t indices for per-clip consistency
        save["test_age_reg_pred"] = age_pred_t[rec_t].astype(np.float32)
        save["test_age_reg_target"] = ages_t[rec_t].astype(np.float32)
        save["val_age_reg_pred"] = age_pred_v[rec_v].astype(np.float32)
        save["val_age_reg_target"] = ages_v[rec_v].astype(np.float32)
        valid_t = ~np.isnan(ages_t)
        if valid_t.sum() >= 2:
            try:
                L1["age_reg_corr"] = float(pearsonr(age_pred_t[valid_t], ages_t[valid_t]).statistic)
            except Exception:
                L1["age_reg_corr"] = float("nan")

    # Sex classification: LogReg on per-rec embeddings
    valid_tr = ~np.isnan(sex_tr)
    if valid_tr.sum() >= 10 and len(np.unique(sex_tr[valid_tr])) >= 2:
        sex_clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        sex_clf.fit(Xtr_rec[valid_tr], sex_tr[valid_tr].astype(int))
        sex_prob_t = sex_clf.predict_proba(Xt_rec)[:, 1]
        sex_prob_v = sex_clf.predict_proba(Xv_rec)[:, 1]
        save["test_sex_prob"] = sex_prob_t[rec_t].astype(np.float32)
        save["test_sex_target"] = sex_t[rec_t].astype(np.float32)
        save["val_sex_prob"] = sex_prob_v[rec_v].astype(np.float32)
        save["val_sex_target"] = sex_v[rec_v].astype(np.float32)
        valid_t = ~np.isnan(sex_t)
        if valid_t.sum() >= 2 and len(np.unique(sex_t[valid_t])) >= 2:
            try:
                L1["sex_auc"] = float(roc_auc_score(sex_t[valid_t].astype(int), sex_prob_t[valid_t]))
            except ValueError:
                L1["sex_auc"] = float("nan")

    # ------------------------------------------------------------------
    # Save NPZ + (optional) JSON of L1 metrics
    # ------------------------------------------------------------------
    save["n_passes"] = np.int64(n_passes)
    save["probe_seed"] = np.int64(probe_seed)
    save["seed"] = np.int64(seed)
    save["test_rec_ids"] = rec_t
    save["val_rec_ids"] = rec_v
    save["feature_names"] = np.array(feature_names)

    if save_predictions_dir:
        out_dir = Path(save_predictions_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / f"preds_seed{seed}.npz"
        np.savez_compressed(out_npz, **save)
        logger.info("Saved predictions: %s", out_npz)

    if out_json:
        out_json_path = Path(out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(
                {"L1": L1, "seed": seed, "probe_seed": probe_seed,
                 "n_passes": n_passes, "checkpoint": str(checkpoint_path)},
                f, indent=2,
            )
        logger.info("Wrote per-seed L1 JSON: %s", out_json_path)

    # Print L1 table
    print(f"\n=== Canonical probe eval (L1, seed={seed}) — ckpt={checkpoint_path.parent.name} ===")
    for k, v in L1.items():
        vstr = f"{v:+.4f}" if isinstance(v, float) and not math.isnan(v) else "  nan "
        print(f"  {k:35s}  {vstr}")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    try:
        import wandb
        if wandb_run_id:
            run = wandb.init(project=wandb_project, id=wandb_run_id, resume="must")
        else:
            run = wandb.init(
                project=wandb_project,
                group=wandb_group,
                name=f"probe_eval_s{seed}",
                config={
                    "checkpoint": str(checkpoint_path),
                    "n_windows": n_windows, "window_size_seconds": window_size_seconds,
                    "n_passes": n_passes, "probe_seed": probe_seed,
                    "norm_mode": norm_mode, "corrca_filters": corrca_filters,
                },
            )
        flat = {f"probe_eval/L1/{k}": v for k, v in L1.items()}
        run.log(flat)
        run.finish()
    except Exception as e:
        logger.warning("W&B logging failed: %s", e)

    return L1


if __name__ == "__main__":
    fire.Fire(run)
