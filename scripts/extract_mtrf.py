"""Canonical mTRF backward decoder (Crosse 2016) at lag range 0-500 ms.

For each (input_type, seed):
  1. Build train Toeplitz: at every timepoint t in every train recording,
     stack EEG(t-tau) for tau in {0, 5, 10, ..., 500 ms} = 101 lags.
  2. Fit Ridge α=1 per stim feature (4 features, 4 closed-form fits).
  3. For each test clip: predict stim(t) at every timepoint via the trained
     decoder, average across the clip's timepoints → per-clip prediction.
  4. Save per-recording predictions in NPZ matching unified_probe_eval's
     external_npz consumer schema, plus a per-seed metrics.json.

Differs from feature-level TRF (raw_corrca-style box pooling) by using
sample-level lag stacking with a canonical 500-ms lag range — the
methodology actually published as "mTRF" in Crosse 2016 / mTRF Toolbox.

Usage:
    PYTHONPATH=. python scripts/extract_mtrf.py \\
        --input=corrca5 --seed=42 --out_dir=/abs/path
"""
import json
import sys
from pathlib import Path

import fire
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.training_utils import load_config, setup_seed
from eb_jepa.logging import get_logger
from experiments.eeg_jepa.main import resolve_preprocessed_dir
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

logger = get_logger(__name__)


def _read_full_clip(dataset, rec_idx, start_idx):
    """Read full 8-s clip at original 200 Hz: [n_windows, C, T_samples].

    Applies the dataset's per-recording z-norm AND CorrCA spatial projection
    (when dataset._corrca_W is set), matching JEPAMovieDataset.__getitem__'s
    transformations. Without the CorrCA step, an `input=corrca5` mTRF run
    would silently fall through to raw 129-ch input.
    """
    crop_inds = dataset._crop_inds[rec_idx]
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
    indices = list(range(start_idx, start_idx + required, dataset.temporal_stride))
    eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
    eeg = torch.from_numpy(eeg_np)  # [n_windows, 129, T]
    if dataset._norm_mode == "per_recording":
        rm = eeg.mean(dim=(0, 2), keepdim=True)
        rs = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
        eeg = (eeg - rm) / rs
    elif dataset._norm_mode == "global" and dataset._eeg_mean is not None:
        eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
    # CRITICAL: apply CorrCA spatial projection if configured. Without this
    # step, both input="corrca5" and input="raw129" runs see raw 129-ch data.
    if dataset._corrca_W is not None:
        eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)  # [nw, K, T]
    return eeg.numpy(), indices


def _flatten_clip(eeg_np):
    """[n_windows, C, T_samp] → [C, n_windows*T_samp] concatenated time."""
    nw, C, Ts = eeg_np.shape
    return eeg_np.transpose(1, 0, 2).reshape(C, nw * Ts)  # [C, total_T]


def _toeplitz_lagged(eeg_flat, n_lags):
    """Build sample-level lag matrix: at each valid t, stack [EEG(t), EEG(t-1), ..., EEG(t-(n_lags-1))].

    eeg_flat: [C, T_total]
    Returns features [T_total - n_lags + 1, C * n_lags] and valid sample range.
    """
    C, T = eeg_flat.shape
    valid_T = T - n_lags + 1
    if valid_T <= 0:
        return np.zeros((0, C * n_lags), dtype=np.float32), 0
    out = np.zeros((valid_T, C, n_lags), dtype=np.float32)
    for tau in range(n_lags):
        # at sample t, EEG(t-tau) = eeg_flat[:, t-tau]; valid t in [n_lags-1, T-1]
        out[:, :, tau] = eeg_flat[:, n_lags - 1 - tau : T - tau].T
    return out.reshape(valid_T, C * n_lags), valid_T


def _build_xy(dataset, n_lags, max_clips_per_rec=4):
    """Build (X, y) over all train recordings: X = lagged EEG per timepoint,
    y = stim labels per timepoint. Sub-samples non-overlapping clips per recording
    to keep memory bounded.
    """
    Xs, ys = [], []
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        max_non_overlap = (n_clips - 1) // required + 1
        n_sample = min(max_clips_per_rec, max_non_overlap)
        starts = (np.arange(n_sample) * required).astype(int)
        feats_full = dataset.feature_recordings[rec_idx]  # [n_total, n_features]
        for start in starts:
            eeg, indices = _read_full_clip(dataset, rec_idx, start)
            eeg_flat = _flatten_clip(eeg)
            X_clip, valid_T = _toeplitz_lagged(eeg_flat, n_lags)
            if valid_T <= 0:
                continue
            # Per-sample labels: for each valid t in clip, the stim feature at that
            # timepoint comes from feats_full[indices][window_idx], where window_idx
            # = which window the timepoint t falls into.
            # Approximation: use clip-mean stim (constant across clip timepoints).
            # This collapses temporal labels but matches per-clip eval target.
            label_clip = feats_full[indices].mean(dim=0).numpy().astype(np.float32)
            y_replicated = np.tile(label_clip[None, :], (valid_T, 1))
            Xs.append(X_clip)
            ys.append(y_replicated)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def _predict_per_clip(dataset, ridge_per_feature, n_lags, ym, ys_norm,
                     max_clips_per_rec=4):
    """For each test recording, produce per-recording mean predicted stim.

    Returns preds [n_rec, n_features], targets [n_rec, n_features].
    """
    n_features = ym.shape[0]
    preds = np.full((len(dataset), n_features), np.nan, dtype=np.float32)
    targets = np.full((len(dataset), n_features), np.nan, dtype=np.float32)
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        max_non_overlap = (n_clips - 1) // required + 1
        n_sample = min(max_clips_per_rec, max_non_overlap)
        starts = (np.arange(n_sample) * required).astype(int)
        clip_preds = []
        clip_targets = []
        feats_full = dataset.feature_recordings[rec_idx]
        for start in starts:
            eeg, indices = _read_full_clip(dataset, rec_idx, start)
            eeg_flat = _flatten_clip(eeg)
            X_clip, valid_T = _toeplitz_lagged(eeg_flat, n_lags)
            if valid_T <= 0:
                continue
            # Predict per-sample stim, denormalize, then mean over samples per clip
            sample_preds = np.zeros((valid_T, n_features), dtype=np.float32)
            for fi, ridge in enumerate(ridge_per_feature):
                sample_preds[:, fi] = ridge.predict(X_clip) * ys_norm[fi] + ym[fi]
            clip_pred = sample_preds.mean(axis=0)  # [n_features] per clip
            clip_target = feats_full[indices].mean(dim=0).numpy().astype(np.float32)
            clip_preds.append(clip_pred)
            clip_targets.append(clip_target)
        if clip_preds:
            preds[rec_idx] = np.mean(clip_preds, axis=0)
            targets[rec_idx] = np.mean(clip_targets, axis=0)
    valid = ~np.isnan(preds[:, 0])
    return preds[valid], targets[valid]


def run(input: str, seed: int, out_dir: str,
        n_windows: int = 2, window_size_seconds: int = 4,
        lag_ms: int = 500, fname: str = "experiments/eeg_jepa/cfgs/default.yaml"):
    """Run canonical mTRF backward decoder at the given input type and seed."""
    assert input in ("corrca5", "raw129"), f"input={input!r}"
    setup_seed(seed)

    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.norm_mode": "per_recording",
        "data.corrca_filters": "corrca_filters.npz" if input == "corrca5" else None,
    }
    cfg = load_config(fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feat_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))

    train_set = JEPAMovieDataset(
        split="train", n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feat_names, cfg=cfg.data,
        preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )
    test_set = JEPAMovieDataset(
        split="test", n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feat_names, eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data, preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )

    sfreq = float(train_set.sfreq)
    n_lags = int(round(lag_ms * sfreq / 1000)) + 1  # 0-500 ms inclusive @ 200 Hz = 101 lags
    logger.info("input=%s seed=%d n_lags=%d (%d ms range at %.0f Hz)",
                input, seed, n_lags, lag_ms, sfreq)

    logger.info("Building train (X, y) ...")
    X_train, y_train = _build_xy(train_set, n_lags)
    logger.info("X_train=%s y_train=%s", X_train.shape, y_train.shape)

    # Standardize features and targets (train-only)
    mu_X = X_train.mean(0, keepdims=True); sd_X = X_train.std(0, keepdims=True) + 1e-8
    X_train_n = (X_train - mu_X) / sd_X
    n_features = y_train.shape[1]
    ym = y_train.mean(0); ys_norm = y_train.std(0) + 1e-8
    y_train_n = (y_train - ym[None, :]) / ys_norm[None, :]

    logger.info("Fitting Ridge α=1 per stim feature (n_lags=%d, D=%d, N=%d) ...",
                n_lags, X_train.shape[1], X_train.shape[0])
    ridge_per_feature = []
    for fi in range(n_features):
        r = Ridge(alpha=1.0).fit(X_train_n, y_train_n[:, fi])
        ridge_per_feature.append(r)

    # Wrap predict to handle standardization
    class _Wrapped:
        def __init__(self, ridge): self.ridge = ridge
        def predict(self, X):
            return self.ridge.predict((X - mu_X) / sd_X)
    wrapped = [_Wrapped(r) for r in ridge_per_feature]

    logger.info("Predicting test per-clip ...")
    test_preds, test_targets = _predict_per_clip(test_set, wrapped, n_lags, ym, ys_norm)
    logger.info("test_preds=%s test_targets=%s", test_preds.shape, test_targets.shape)

    # Per-clip Pearson r per feature
    metrics = {}
    for fi, name in enumerate(feat_names):
        r, _ = pearsonr(test_preds[:, fi], test_targets[:, fi])
        metrics[f"test/reg_{name}_corr"] = float(r)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"test_seed{seed}.npz",
        test_preds=test_preds, test_targets=test_targets,
        feat_names=np.array(feat_names),
    )
    (out / "metrics.json").write_text(json.dumps({
        "input": input, "seed": seed, "n_lags": n_lags, "lag_ms": lag_ms,
        "sfreq": sfreq, "n_test_recordings": int(test_preds.shape[0]),
        "metrics": metrics,
    }, indent=2))

    logger.info("=== Per-clip Pearson r (test) ===")
    for k, v in metrics.items():
        logger.info("  %s = %+.4f", k, v)


if __name__ == "__main__":
    fire.Fire(run)
