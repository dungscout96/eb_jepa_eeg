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


def _read_full_clip(dataset, rec_idx, start_idx, downsample_factor=2):
    """Read full 8-s clip and apply dataset transforms + memory-friendly downsample.

    Steps mirror JEPAMovieDataset.__getitem__: per-recording z-norm → CorrCA
    spatial projection (if set). Then box-mean-pool the time axis by
    `downsample_factor` to keep the lag-stacked Toeplitz tractable; default
    factor 2 takes 200 Hz → 100 Hz, halving both sample count and required
    lag count for a fixed lag-time window. Lag spacing of 10 ms at 100 Hz
    is still well within canonical mTRF resolution (Crosse 2016 typically
    uses 5–20 ms spacing).
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
    if dataset._corrca_W is not None:
        eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)  # [nw, K, T]
    if downsample_factor > 1:
        nw, C, T = eeg.shape
        T_trim = (T // downsample_factor) * downsample_factor
        eeg = (
            eeg[..., :T_trim]
            .reshape(nw, C, T_trim // downsample_factor, downsample_factor)
            .mean(-1)
        )
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


def _build_xy(dataset, n_lags, n_passes=2, seed=0):
    """Build (X, y) over all train recordings: X = lagged EEG per timepoint,
    y = stim labels per timepoint. Uses RANDOM clip starts (n_passes per rec)
    so the training set has diverse movie-position content.
    """
    rng = torch.Generator().manual_seed(seed)
    Xs, ys = [], []
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        feats_full = dataset.feature_recordings[rec_idx]  # [n_total, n_features]
        for _ in range(n_passes):
            start = int(torch.randint(0, n_clips, (1,), generator=rng).item())
            eeg, indices = _read_full_clip(dataset, rec_idx, start)
            eeg_flat = _flatten_clip(eeg)
            X_clip, valid_T = _toeplitz_lagged(eeg_flat, n_lags)
            if valid_T <= 0:
                continue
            label_clip = feats_full[indices].mean(dim=0).numpy().astype(np.float32)
            y_replicated = np.tile(label_clip[None, :], (valid_T, 1))
            Xs.append(X_clip)
            ys.append(y_replicated)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def _predict_per_clip(dataset, ridge_per_feature, n_lags, ym, ys_norm,
                     n_passes=20, seed=0):
    """For each (rec, pass) random clip, produce per-clip predicted stim.

    Returns preds [n_rec, n_passes, n_features], targets [n_rec, n_passes, n_features].
    Random clip starts (per rec, per pass) — breaks the deterministic-movie-position
    symmetry that made fixed-start sampling produce identical targets across recs.
    """
    rng = torch.Generator().manual_seed(seed)
    n_features = ym.shape[0]
    n_rec = len(dataset)
    preds = np.full((n_rec, n_passes, n_features), np.nan, dtype=np.float32)
    targets = np.full((n_rec, n_passes, n_features), np.nan, dtype=np.float32)
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
    for rec_idx in range(n_rec):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        feats_full = dataset.feature_recordings[rec_idx]
        for p in range(n_passes):
            start = int(torch.randint(0, n_clips, (1,), generator=rng).item())
            eeg, indices = _read_full_clip(dataset, rec_idx, start)
            eeg_flat = _flatten_clip(eeg)
            X_clip, valid_T = _toeplitz_lagged(eeg_flat, n_lags)
            if valid_T <= 0:
                continue
            sample_preds = np.zeros((valid_T, n_features), dtype=np.float32)
            for fi, ridge in enumerate(ridge_per_feature):
                sample_preds[:, fi] = ridge.predict(X_clip) * ys_norm[fi] + ym[fi]
            preds[rec_idx, p] = sample_preds.mean(axis=0)
            targets[rec_idx, p] = (
                feats_full[indices].mean(dim=0).numpy().astype(np.float32)
            )
    return preds, targets


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

    # Downsample EEG to 100 Hz before lag-stacking for memory tractability.
    sfreq_native = float(train_set.sfreq)
    downsample_factor = 2  # 200 Hz → 100 Hz
    sfreq = sfreq_native / downsample_factor
    n_lags = int(round(lag_ms * sfreq / 1000)) + 1  # 0-500 ms @ 100 Hz = 51 lags
    logger.info("input=%s seed=%d n_lags=%d (%d ms range at %.0f Hz, native %.0f Hz)",
                input, seed, n_lags, lag_ms, sfreq, sfreq_native)

    logger.info("Building train (X, y) with random clip starts (n_passes=2) ...")
    X_train, y_train = _build_xy(train_set, n_lags, n_passes=2, seed=seed)
    logger.info("X_train=%s y_train=%s", X_train.shape, y_train.shape)

    # Standardize features and targets (train-only)
    mu_X = X_train.mean(0, keepdims=True); sd_X = X_train.std(0, keepdims=True) + 1e-8
    X_train_n = (X_train - mu_X) / sd_X
    n_features = y_train.shape[1]
    ym = y_train.mean(0); ys_norm = y_train.std(0) + 1e-8
    y_train_n = (y_train - ym[None, :]) / ys_norm[None, :]

    logger.info("Fitting Ridge α=1 (lsqr) per stim feature (n_lags=%d, D=%d, N=%d) ...",
                n_lags, X_train.shape[1], X_train.shape[0])
    ridge_per_feature = []
    for fi in range(n_features):
        # solver=lsqr uses iterative least-squares — O(N+D) memory per iter,
        # avoids the O(D^2) Cholesky / O(N^2) dual-SVD that crashed at 80 GB.
        r = Ridge(alpha=1.0, solver="lsqr", max_iter=500).fit(X_train_n, y_train_n[:, fi])
        ridge_per_feature.append(r)

    # Wrap predict to handle standardization
    class _Wrapped:
        def __init__(self, ridge): self.ridge = ridge
        def predict(self, X):
            return self.ridge.predict((X - mu_X) / sd_X)
    wrapped = [_Wrapped(r) for r in ridge_per_feature]

    logger.info("Predicting test per-(rec, pass) with random clip starts ...")
    test_preds, test_targets = _predict_per_clip(
        test_set, wrapped, n_lags, ym, ys_norm,
        n_passes=20, seed=seed + 2,
    )
    logger.info(
        "test_preds=%s test_targets=%s", test_preds.shape, test_targets.shape
    )

    # Per-clip flat Pearson r per feature (matches canonical Protocol B)
    pred_flat = test_preds.reshape(-1, len(feat_names))
    targ_flat = test_targets.reshape(-1, len(feat_names))
    valid = ~np.isnan(pred_flat[:, 0])
    pred_flat = pred_flat[valid]
    targ_flat = targ_flat[valid]
    metrics = {}
    for fi, name in enumerate(feat_names):
        r, _ = pearsonr(pred_flat[:, fi], targ_flat[:, fi])
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
