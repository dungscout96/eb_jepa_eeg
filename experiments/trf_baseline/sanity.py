"""Single-subject TRF sanity check.

Three diagnostics, fast to run on a few recordings:
    A. In-sample fit on one rec → expect high train r. Validates math/alignment.
    B. Time-split (first 80% train, last 20% eval) on one rec → expect modest
       positive eval r if any decodable signal exists at the single-rec scale.
    C. Phase-randomized feature control (shuffle y in time before fit) → expect
       eval r ≈ 0. Validates no leakage.

If A is high and B is low, the TRF is overfitting within a recording — pooled
fitting needs more recordings.
If A is low, there is a bug in alignment or the design matrix.

Usage
-----
PYTHONPATH=. uv run --group eeg python experiments/trf_baseline/sanity.py \
    --n_recs=3 --n_lags_ms=1000 --fs_target=50
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from eb_jepa.datasets.hbn import (
    JEPAMovieDataset, MOVIE_METADATA, _preload_movie_features,
)
from eb_jepa.training_utils import load_config

from experiments.trf_baseline.run_trf import (
    DEFAULT_FEATURES, RidgeSolver, feature_timeseries, load_continuous,
    make_lagged, pearson_r,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fit_one_rec(eeg, y, n_lags, alpha):
    X = make_lagged(eeg, n_lags)
    if X is None:
        return None
    y_trim = y[: X.shape[0]]
    solver = RidgeSolver(X.shape[1], y_trim.shape[1])
    solver.add(X, y_trim)
    W = solver.solve(alpha)
    pred = RidgeSolver.predict(X, W)
    return X, y_trim, W, pred


def time_split_fit(eeg, y, n_lags, alpha, train_frac=0.8):
    X = make_lagged(eeg, n_lags)
    if X is None:
        return None
    y_trim = y[: X.shape[0]]
    n_split = int(train_frac * X.shape[0])
    Xtr, ytr = X[:n_split], y_trim[:n_split]
    Xte, yte = X[n_split:], y_trim[n_split:]
    solver = RidgeSolver(X.shape[1], y_trim.shape[1])
    solver.add(Xtr, ytr)
    W = solver.solve(alpha)
    pred_tr = RidgeSolver.predict(Xtr, W)
    pred_te = RidgeSolver.predict(Xte, W)
    return ytr, pred_tr, yte, pred_te


def run(
    n_recs: int = 3,
    n_lags_ms: int = 1000,
    fs_target: int = 50,
    alpha: float = 1e3,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    movie: str = "ThePresent",
):
    cfg = load_config(fname, {})
    n_lags = int(round(n_lags_ms / 1000.0 * fs_target))

    dummy_mean = torch.zeros((1, 129, 1), dtype=torch.float32)
    dummy_std = torch.ones((1, 129, 1), dtype=torch.float32)
    ds = JEPAMovieDataset(
        split="train", n_windows=1, window_size_seconds=2,
        feature_names=list(DEFAULT_FEATURES),
        eeg_norm_stats={"mean": dummy_mean, "std": dummy_std},
        cfg=cfg.data,
        preprocessed=cfg.data.get("preprocessed", True),
        preprocessed_dir=cfg.data.get("preprocessed_dir", None),
    )
    paths = ds._fif_paths[:n_recs]
    crops = ds._crop_inds[:n_recs]
    logger.info("Sanity check on %d recs, n_lags=%d (%dms @ %dHz), alpha=%g",
                n_recs, n_lags, n_lags_ms, fs_target, alpha)

    mv_feats = _preload_movie_features(movie)[movie]

    print("\n" + "=" * 70)
    print(f"Diagnostic A: in-sample fit (train r should be high)")
    print(f"Diagnostic B: 80/20 time split (eval r reveals true single-rec signal)")
    print(f"Diagnostic C: shuffled-y control (eval r should be ~0)")
    print("=" * 70)

    for i, (fp, ci) in enumerate(zip(paths, crops)):
        print(f"\n--- rec {i}: {Path(fp).name} ---")
        t0 = time.time()
        eeg, mt = load_continuous(fp, ci, movie, fs_target, corrca_W=None)
        y = feature_timeseries(mt, mv_feats, list(DEFAULT_FEATURES), movie)
        print(f"  eeg shape: {eeg.shape}, y shape: {y.shape}, load: {time.time()-t0:.1f}s")

        # A — in-sample
        out = fit_one_rec(eeg, y, n_lags, alpha)
        if out is None:
            print("  [A] not enough samples")
            continue
        _, y_trim, _, pred = out
        print("  [A] in-sample r per feature:")
        for fi, name in enumerate(DEFAULT_FEATURES):
            r = pearson_r(pred[:, fi], y_trim[:, fi])
            print(f"        {name:25s}  train r = {r:+.3f}")

        # B — 80/20 time split
        ytr, ptr, yte, pte = time_split_fit(eeg, y, n_lags, alpha)
        print("  [B] time-split (train_r → eval_r):")
        for fi, name in enumerate(DEFAULT_FEATURES):
            rtr = pearson_r(ptr[:, fi], ytr[:, fi])
            rte = pearson_r(pte[:, fi], yte[:, fi])
            print(f"        {name:25s}  train = {rtr:+.3f}   eval = {rte:+.3f}")

        # C — circular shift y by half its length, then time-split fit
        shift = y.shape[0] // 2
        y_shuf = np.roll(y, shift=shift, axis=0)
        ytr_s, ptr_s, yte_s, pte_s = time_split_fit(eeg, y_shuf, n_lags, alpha)
        print("  [C] shuffled-y control (train_r → eval_r):")
        for fi, name in enumerate(DEFAULT_FEATURES):
            rtr = pearson_r(ptr_s[:, fi], ytr_s[:, fi])
            rte = pearson_r(pte_s[:, fi], yte_s[:, fi])
            print(f"        {name:25s}  train = {rtr:+.3f}   eval = {rte:+.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_recs", type=int, default=3)
    p.add_argument("--n_lags_ms", type=int, default=1000)
    p.add_argument("--fs_target", type=int, default=50)
    p.add_argument("--alpha", type=float, default=1e3)
    args = p.parse_args()
    run(n_recs=args.n_recs, n_lags_ms=args.n_lags_ms, fs_target=args.fs_target,
        alpha=args.alpha)
