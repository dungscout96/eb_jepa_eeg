"""Linear probe of trivial within-clip EEG features → movie features.

Tests whether position_in_movie (and luminance, contrast, narrative) can
be predicted from simple temporal features that don't require a learned
encoder — channel mean, std, log bandpower in standard bands. If position
is well-decoded by trivial features but luminance/contrast aren't, the
encoder's "position signal" is likely a within-recording temporal-trend
artifact, not stimulus content.

Setup mirrors probe_eval.py exactly (per-rec norm, optional CorrCA,
matching n_windows / window_size_seconds), so the trivial features see the
same EEG the encoder sees.

Usage
-----
PYTHONPATH=. uv run --group eeg python scripts/trivial_position_baseline.py \\
    --norm_mode=per_recording \\
    --corrca_filters=/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz \\
    --n_windows=4 --window_size_seconds=2 \\
    --n_passes=20 --seed=2025
"""

from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.signal import butter, filtfilt, welch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from eb_jepa.datasets.hbn import (
    JEPAMovieDataset, MOVIE_METADATA, _read_raw_windows,
)
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_seed
from experiments.eeg_jepa.main import resolve_preprocessed_dir

logger = get_logger(__name__)

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


def _bandpower(eeg: np.ndarray, sfreq: float) -> np.ndarray:
    """Per-channel log power in 5 bands.

    eeg: [n_windows, C, T] → returns [C * 5] feature vector.
    """
    flat = eeg.reshape(-1, eeg.shape[-1])  # [n_windows*C, T]
    nperseg = min(256, eeg.shape[-1])
    f, psd = welch(flat, fs=sfreq, nperseg=nperseg, axis=-1)
    out = []
    for lo, hi in BANDS.values():
        mask = (f >= lo) & (f < hi)
        bp = np.log(psd[:, mask].mean(axis=-1) + 1e-12)  # [n_windows*C]
        bp = bp.reshape(eeg.shape[0], eeg.shape[1]).mean(axis=0)  # avg across windows → [C]
        out.append(bp)
    return np.concatenate(out)  # [C*5]


def _trivial_features(eeg: torch.Tensor, sfreq: float) -> np.ndarray:
    """Compute trivial within-clip features. eeg: [n_windows, C, T]."""
    eeg_np = eeg.numpy()
    means = eeg_np.mean(axis=(0, 2))           # [C]
    stds = eeg_np.std(axis=(0, 2))             # [C]
    bp = _bandpower(eeg_np, sfreq)             # [C*5]
    return np.concatenate([means, stds, bp])   # [C*7]


def _extract(dataset, n_passes, sfreq, seed):
    """Iterate dataset n_passes times; collect trivial features + features."""
    rng = torch.Generator().manual_seed(seed)
    n_rec = len(dataset)
    feats_list = []
    labels_list = []
    for p in range(n_passes):
        for rec_idx in torch.randperm(n_rec, generator=rng).tolist():
            eeg, feats, _ = dataset[rec_idx]  # eeg [nw,C,T], feats [nw, n_features]
            feats_list.append(_trivial_features(eeg, sfreq))
            labels_list.append(feats.mean(dim=0).numpy())  # [n_features]
        if (p + 1) % 5 == 0 or p == n_passes - 1:
            logger.info("  pass %d/%d", p + 1, n_passes)
    X = np.stack(feats_list)
    Y = np.stack(labels_list)
    return X, Y


def _extract_time_aligned(dataset, K, sfreq):
    """K evenly-spaced (linspace) clips per recording — approximately
    time-aligned across recordings since all recordings cover the same movie.
    Bypasses dataset.__getitem__'s random clip sampling.
    """
    feats_list = []
    labels_list = []
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        starts = np.linspace(0, n_clips - 1, min(K, n_clips), dtype=int)
        for start in starts:
            indices = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
            eeg = torch.from_numpy(eeg_np)
            if dataset._norm_mode == "per_recording":
                rec_mean = eeg.mean(dim=(0, 2), keepdim=True)
                rec_std = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
                eeg = (eeg - rec_mean) / rec_std
            else:
                eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
            if dataset._add_envelope:
                eeg = dataset._append_lowfreq_envelope(eeg)
            if dataset._corrca_W is not None:
                eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)
            feats_list.append(_trivial_features(eeg, sfreq))
            labels_list.append(
                dataset.feature_recordings[rec_idx][indices].mean(dim=0).numpy()
            )
    return np.stack(feats_list), np.stack(labels_list)


def _shuffle_label_globally(datasets, target_label, feature_names, seed,
                            movie="ThePresent"):
    """Permute one label column across movie frames, consistently across
    all recordings. At each movie frame f, assign the original label value
    from a random other frame: new_label[f] = original_label[perm[f]].

    Recover the original frame for each window from position_in_movie
    (= frame_idx / (n_frames-1)).
    """
    import pandas as pd
    target_idx = feature_names.index(target_label)
    pos_idx = feature_names.index("position_in_movie")
    parquet_path = MOVIE_METADATA[movie]["feature_parquet"]
    df = pd.read_parquet(parquet_path)
    n_frames = len(df)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_frames)
    permuted_label_at_frame = df[target_label].values[perm]

    for ds in datasets:
        for rec_idx in range(len(ds._fif_paths)):
            feats = ds.feature_recordings[rec_idx]
            positions = feats[:, pos_idx].numpy()
            frame_indices = np.clip(
                np.round(positions * (n_frames - 1)).astype(int), 0, n_frames - 1
            )
            new_vals = permuted_label_at_frame[frame_indices].astype(np.float32)
            feats[:, target_idx] = torch.from_numpy(new_vals)


def run(
    n_windows: int = 4,
    window_size_seconds: int = 2,
    norm_mode: str = "per_recording",
    corrca_filters: str = "",
    n_passes: int = 20,
    seed: int = 2025,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    # Test 1: globally permute a label column across movie frames
    shuffle_label: str = "",
    # Test 2: K evenly-spaced clips per recording (deterministic, not random)
    time_aligned_K: int = 0,
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

    logger.info("Loading datasets (n_passes=%d)...", n_passes)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    eval_sets = {}
    for split in ("val", "test"):
        eval_sets[split] = JEPAMovieDataset(
            split=split,
            n_windows=n_windows,
            window_size_seconds=window_size_seconds,
            feature_names=feature_names,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
        )

    sfreq = train_set.n_times / window_size_seconds
    logger.info("sfreq=%.1f Hz, n_chans=%d", sfreq, train_set.n_chans)

    if shuffle_label:
        logger.info("Shuffling label '%s' globally across movie frames (seed=%d)",
                    shuffle_label, seed)
        _shuffle_label_globally(
            [train_set, eval_sets["val"], eval_sets["test"]],
            shuffle_label, feature_names, seed,
        )

    if time_aligned_K > 0:
        logger.info("Extracting time-aligned clips (K=%d per recording)...", time_aligned_K)
        X_train, Y_train = _extract_time_aligned(train_set, time_aligned_K, sfreq)
        X_val, Y_val = _extract_time_aligned(eval_sets["val"], time_aligned_K, sfreq)
        X_test, Y_test = _extract_time_aligned(eval_sets["test"], time_aligned_K, sfreq)
    else:
        logger.info("Extracting train features (%d recordings × %d passes)...",
                    len(train_set), n_passes)
        X_train, Y_train = _extract(train_set, n_passes, sfreq, seed)
        logger.info("  X_train: %s, Y_train: %s", X_train.shape, Y_train.shape)

        logger.info("Extracting val features...")
        X_val, Y_val = _extract(eval_sets["val"], n_passes, sfreq, seed + 1)
        logger.info("Extracting test features...")
        X_test, Y_test = _extract(eval_sets["test"], n_passes, sfreq, seed + 2)

    # Standardize features (helps ridge)
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True).clip(min=1e-8)
    X_train_n = (X_train - mu) / sd
    X_val_n = (X_val - mu) / sd
    X_test_n = (X_test - mu) / sd

    mode = []
    if shuffle_label:
        mode.append(f"shuffle={shuffle_label}")
    if time_aligned_K > 0:
        mode.append(f"time_aligned_K={time_aligned_K}")
    if not mode:
        mode.append(f"random_n_passes={n_passes}")
    print(f"\n=== Trivial-feature linear probe → movie features [{' + '.join(mode)}] ===")
    print(f"Setup: n_windows={n_windows}, ws={window_size_seconds}s, "
          f"norm_mode={norm_mode}, corrca={'yes' if corrca_filters else 'no'}")
    print(f"Features: {X_train.shape[1]} dims (per-channel mean+std + 5 bandpowers)")
    print(f"Train clips: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    print()
    print(f"{'feature':<25} {'val corr':>10} {'test corr':>10}")
    print("-" * 50)
    for i, fname_ in enumerate(feature_names):
        y_tr = Y_train[:, i]
        y_va = Y_val[:, i]
        y_te = Y_test[:, i]
        if np.std(y_tr) < 1e-10:
            print(f"{fname_:<25} {'(const)':>10} {'(const)':>10}")
            continue
        # Standardize target for stable training
        ym, ys = y_tr.mean(), y_tr.std()
        probe = Ridge(alpha=1.0).fit(X_train_n, (y_tr - ym) / ys)
        pred_va = probe.predict(X_val_n) * ys + ym
        pred_te = probe.predict(X_test_n) * ys + ym
        c_va = pearsonr(pred_va, y_va).statistic if np.std(pred_va) > 0 else 0.0
        c_te = pearsonr(pred_te, y_te).statistic if np.std(pred_te) > 0 else 0.0
        print(f"{fname_:<25} {c_va:>+10.4f} {c_te:>+10.4f}")


if __name__ == "__main__":
    fire.Fire(run)
