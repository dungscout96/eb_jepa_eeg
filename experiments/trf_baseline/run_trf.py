"""TRF baseline: predict stimulus features from raw EEG with Ridge.

Backward (decoding) TRF: feature_hat[t] = sum_{tau} W[tau] @ EEG[t+tau].
One pooled cross-subject Ridge fit (covariance form, streamed over recordings)
matches the way the frozen-encoder probe is trained on R1-R4 and evaluated
on R5/R6.

Usage
-----
PYTHONPATH=. uv run --group eeg python experiments/trf_baseline/run_trf.py \
    --input=raw \
    --max_train_recs=100 \
    --output_dir=outputs/trf_prototype

Output: a JSON metrics file and a markdown report row, ready to be merged
into the SIGReg-vs-VICReg probe table for apples-to-apples comparison.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate
from sklearn.metrics import balanced_accuracy_score

from eb_jepa.datasets.hbn import (
    JEPAMovieDataset, MOVIE_METADATA, _preload_movie_features,
)
from eb_jepa.training_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Probe-matched constants (see experiments/eeg_jepa/cfgs/default.yaml).
SFREQ_RAW = 200
VISUAL_DELAY_S = 0.3                # delay applied at dataset construction time
DEFAULT_FEATURES = ("contrast_rms", "luminance_mean", "position_in_movie")
DEFAULT_TASK = "ThePresent"


# --------------------------------------------------------------------------
# Continuous data extraction
# --------------------------------------------------------------------------

def _video_start_sample(crop_inds: np.ndarray, sfreq: int) -> int:
    """Recover the video_start absolute sample index from crop_inds.

    JEPAMovieDataset's first window starts ``visual_processing_delay_s * sfreq``
    samples after video_start (trial_start_offset_samples), so:
        video_start = crop_inds[0, 1] - delay_samples
    """
    delay_samples = int(round(VISUAL_DELAY_S * sfreq))
    return int(crop_inds[0, 1]) - delay_samples


def load_continuous(
    fif_path: str,
    crop_inds: np.ndarray,
    movie: str,
    fs_target: int,
    corrca_W: np.ndarray | None = None,
):
    """Read continuous EEG aligned to movie time, per-rec z-score, decimate.

    Returns
    -------
    eeg : np.ndarray [T_target, C]   (already per-rec z-scored, optionally CorrCA)
    movie_time : np.ndarray [T_target] in seconds (0..duration)
    """
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
    sfreq = int(round(raw.info["sfreq"]))
    assert sfreq == SFREQ_RAW, f"unexpected sfreq {sfreq}"

    vs = _video_start_sample(crop_inds, sfreq)
    duration_s = MOVIE_METADATA[movie]["duration"]
    n_samples = int(duration_s * sfreq)
    start = max(0, vs)
    stop = min(raw.n_times, vs + n_samples)
    if start >= stop:
        del raw
        return None, None

    data = raw._getitem((slice(None), slice(start, stop)), return_times=False)
    del raw
    data = data.astype(np.float32)  # [C, T]

    # Per-recording z-score (across time, per channel) — the user's request.
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-8)
    data = (data - mean) / std

    if corrca_W is not None:
        # data: [C, T], W: [C, k]  →  [k, T]
        data = corrca_W.T @ data

    # Decimate to fs_target (factor must be int).
    factor = sfreq // fs_target
    assert sfreq == factor * fs_target, f"sfreq {sfreq} not divisible by {fs_target}"
    if factor > 1:
        data = decimate(data, factor, axis=1, ftype="fir", zero_phase=True).astype(np.float32)

    eeg = data.T  # [T_target, C]
    T = eeg.shape[0]
    movie_time = np.arange(T) / fs_target
    return eeg, movie_time


def feature_timeseries(
    movie_time: np.ndarray,
    movie_features: pd.DataFrame,
    feature_names: tuple[str, ...],
    movie: str,
) -> np.ndarray:
    """Per-frame movie features sampled at ``movie_time``. Returns [T, n_feat]."""
    fps = MOVIE_METADATA[movie]["fps"]
    n_frames = len(movie_features)
    frame_idx = np.clip((movie_time * fps).astype(np.int64), 0, n_frames - 1)
    cols = []
    for name in feature_names:
        cols.append(movie_features[name].to_numpy()[frame_idx].astype(np.float32))
    return np.stack(cols, axis=1)


# --------------------------------------------------------------------------
# Time-lagged design matrix and covariance-form Ridge
# --------------------------------------------------------------------------

def make_lagged(eeg: np.ndarray, n_lags: int) -> np.ndarray:
    """Build [T-n_lags+1, n_lags*C] backward-decoder design matrix.

    Row t holds [EEG[t], EEG[t+1], ..., EEG[t+n_lags-1]] flattened.
    Backward decoder predicts feature[t] from future EEG samples (the
    post-stimulus neural response).
    """
    T, C = eeg.shape
    if T < n_lags:
        return None
    rows = T - n_lags + 1
    out = np.empty((rows, n_lags * C), dtype=np.float32)
    for L in range(n_lags):
        out[:, L * C : (L + 1) * C] = eeg[L : L + rows]
    return out


class RidgeSolver:
    """Streaming covariance-form Ridge with unregularized intercept."""

    def __init__(self, n_features_in: int, n_outputs: int):
        d = n_features_in + 1
        self.d_features = n_features_in
        self.XtX = np.zeros((d, d), dtype=np.float64)
        self.Xty = np.zeros((d, n_outputs), dtype=np.float64)
        self.n_samples = 0

    def add(self, X: np.ndarray, y: np.ndarray):
        # Augment X with a constant column so the bias is solved jointly.
        T = X.shape[0]
        Xd = np.empty((T, self.d_features + 1), dtype=np.float64)
        Xd[:, :-1] = X
        Xd[:, -1] = 1.0
        self.XtX += Xd.T @ Xd
        self.Xty += Xd.T @ y.astype(np.float64)
        self.n_samples += T

    def solve(self, alpha: float) -> np.ndarray:
        d = self.d_features + 1
        reg = alpha * np.eye(d, dtype=np.float64)
        reg[-1, -1] = 0.0  # do not regularize the intercept
        A = self.XtX + reg
        W = np.linalg.solve(A, self.Xty)
        return W.astype(np.float32)  # [d+1, n_out]

    @staticmethod
    def predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        # X: [T, d_features], W: [d_features+1, n_out]
        return X @ W[:-1] + W[-1]


# --------------------------------------------------------------------------
# Evaluation: window-matched + continuous
# --------------------------------------------------------------------------

def predict_recording(eeg: np.ndarray, W: np.ndarray, n_lags: int) -> np.ndarray:
    """Apply Ridge weights to a recording. Returns [T_pred, n_out]."""
    X = make_lagged(eeg, n_lags)
    if X is None:
        return None
    return RidgeSolver.predict(X, W)


def window_avg(pred: np.ndarray, fs_target: int, window_s: float, n_windows: int):
    """Aggregate predictions into [n_clips, n_features] matching probe windows.

    A "clip" of n_windows windows of window_s seconds is averaged into one
    prediction per clip — matching the frozen-encoder probe protocol where
    the encoder pools across the n_windows windows in a clip.
    """
    samples_per_window = int(round(window_s * fs_target))
    samples_per_clip = samples_per_window * n_windows
    T = pred.shape[0]
    n_clips = T // samples_per_clip
    if n_clips == 0:
        return None
    pred = pred[: n_clips * samples_per_clip]
    pred = pred.reshape(n_clips, samples_per_clip, -1).mean(axis=1)
    return pred  # [n_clips, n_out]


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def collect_recordings(split: str, cfg, max_recs: int | None):
    """Build a JEPAMovieDataset just to harvest per-recording metadata.

    We pass dummy norm stats to skip the expensive per-channel mean/std pass
    (per-rec z-score is computed from the continuous EEG in load_continuous).
    """
    dummy_mean = torch.zeros((1, 129, 1), dtype=torch.float32)
    dummy_std = torch.ones((1, 129, 1), dtype=torch.float32)
    ds = JEPAMovieDataset(
        split=split,
        n_windows=1,
        window_size_seconds=2,
        feature_names=list(DEFAULT_FEATURES),
        eeg_norm_stats={"mean": dummy_mean, "std": dummy_std},
        cfg=cfg.data,
        preprocessed=cfg.data.get("preprocessed", True),
        preprocessed_dir=cfg.data.get("preprocessed_dir", None),
    )
    paths = list(ds._fif_paths)
    crops = list(ds._crop_inds)
    if max_recs is not None and len(paths) > max_recs:
        rng = np.random.default_rng(2025)
        idx = rng.choice(len(paths), size=max_recs, replace=False)
        idx = sorted(idx.tolist())
        paths = [paths[i] for i in idx]
        crops = [crops[i] for i in idx]
    return paths, crops


def run(
    input: str = "raw",                         # "raw" or "corrca"
    corrca_path: str = "",
    max_train_recs: int | None = 100,
    n_lags_ms: int = 1000,
    fs_target: int = 50,
    feature_names: tuple[str, ...] = DEFAULT_FEATURES,
    alphas: tuple[float, ...] = (1e-1, 1e1, 1e3, 1e5),
    eval_window_configs: tuple[tuple[int, float], ...] = ((1, 1.0), (2, 1.0), (4, 4.0)),
    output_dir: str = "outputs/trf_prototype",
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    movie: str = DEFAULT_TASK,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(fname, {})
    n_lags = int(round((n_lags_ms / 1000.0) * fs_target))
    logger.info("TRF: input=%s, n_lags=%d (%.0f ms @ %d Hz), features=%s",
                input, n_lags, n_lags_ms, fs_target, feature_names)

    corrca_W = None
    if input == "corrca":
        assert corrca_path, "corrca_path required for input=corrca"
        z = np.load(corrca_path)
        corrca_W = z["W"].astype(np.float32)
        logger.info("CorrCA: %d → %d components, ISC=%s",
                    corrca_W.shape[0], corrca_W.shape[1],
                    z["isc_values"].tolist() if "isc_values" in z.files else "n/a")

    # Per-frame movie features (loaded once, indexed per-recording).
    mv_feats = _preload_movie_features(movie)[movie]
    logger.info("Loaded %d movie frames from features parquet", len(mv_feats))

    # ----------------------------------------------------------------
    # 1. Train: streaming covariance accumulation
    # ----------------------------------------------------------------
    train_paths, train_crops = collect_recordings("train", cfg, max_train_recs)
    logger.info("Training on %d recordings", len(train_paths))

    n_chans_eff = corrca_W.shape[1] if corrca_W is not None else 129
    d_in = n_lags * n_chans_eff
    n_out = len(feature_names)
    solver = RidgeSolver(d_in, n_out)

    t0 = time.time()
    for i, (fp, ci) in enumerate(zip(train_paths, train_crops)):
        eeg, mt = load_continuous(fp, ci, movie, fs_target, corrca_W)
        if eeg is None:
            continue
        y = feature_timeseries(mt, mv_feats, feature_names, movie)
        X = make_lagged(eeg, n_lags)
        if X is None:
            continue
        # Trim y to match X rows (decoder uses EEG[t..t+L-1] for feature[t]).
        y_trim = y[: X.shape[0]]
        # Remove per-recording mean from y (same as standardising via demean).
        # Don't z-score y — the Pearson r is scale-invariant; bal_acc uses train median.
        solver.add(X, y_trim)
        if (i + 1) % 25 == 0:
            logger.info("  trained on %d/%d recs in %.1fs (cum samples=%d)",
                        i + 1, len(train_paths), time.time() - t0, solver.n_samples)
    logger.info("Train accumulation done in %.1fs (samples=%d, d_in=%d)",
                time.time() - t0, solver.n_samples, d_in)

    # Solve Ridge for each alpha.
    Ws = {alpha: solver.solve(alpha) for alpha in alphas}
    logger.info("Ridge solved for %d alphas", len(Ws))

    # Movie features are a deterministic function of movie time, so the
    # train-set distribution is identical across recordings — compute a
    # per-config clip-level median once on the canonical feature time series.
    # (The probe uses train-clip median; matching that exactly requires
    # a per-window-config median.)
    T_full = int(MOVIE_METADATA[movie]["duration"] * fs_target)
    mt_full = np.arange(T_full) / fs_target
    y_full = feature_timeseries(mt_full, mv_feats, feature_names, movie)
    medians_per_config: dict[str, np.ndarray] = {}
    for nw, ws in eval_window_configs:
        clip_feats = window_avg(y_full, fs_target, ws, nw)
        if clip_feats is None:
            continue
        cfg_key = f"nw{nw}_ws{int(ws)}"
        medians_per_config[cfg_key] = np.median(clip_feats, axis=0)
        logger.info("Median (%s): %s", cfg_key,
                    {n: float(m) for n, m in zip(feature_names, medians_per_config[cfg_key])})

    # ----------------------------------------------------------------
    # 2. Eval per split, per alpha, per window config
    # ----------------------------------------------------------------
    results = {}
    for split in ("val", "test"):
        paths, crops = collect_recordings(split, cfg, None)
        logger.info("Eval split=%s: %d recordings", split, len(paths))

        # Cache per-recording continuous predictions per alpha, plus targets.
        per_rec_preds = {alpha: [] for alpha in alphas}   # list of [T, n_out]
        per_rec_targets = []                              # list of [T, n_out]
        for fp, ci in zip(paths, crops):
            eeg, mt = load_continuous(fp, ci, movie, fs_target, corrca_W)
            if eeg is None:
                continue
            y = feature_timeseries(mt, mv_feats, feature_names, movie)
            X = make_lagged(eeg, n_lags)
            if X is None:
                continue
            y_trim = y[: X.shape[0]]
            per_rec_targets.append(y_trim)
            for alpha in alphas:
                per_rec_preds[alpha].append(RidgeSolver.predict(X, Ws[alpha]))

        # Continuous metric: per-rec Pearson r per feature, mean across recs.
        cont_r = {alpha: np.zeros(n_out) for alpha in alphas}
        for alpha in alphas:
            rs_per_feat = [[] for _ in range(n_out)]
            for pr, tg in zip(per_rec_preds[alpha], per_rec_targets):
                for fi in range(n_out):
                    rs_per_feat[fi].append(pearson_r(pr[:, fi], tg[:, fi]))
            cont_r[alpha] = np.array([np.mean(rs) for rs in rs_per_feat])

        # Window-matched metric: aggregate to clip-level, pool across recs.
        win_metrics = {alpha: {} for alpha in alphas}
        for alpha in alphas:
            for n_windows, window_s in eval_window_configs:
                preds_clip, tgts_clip = [], []
                for pr, tg in zip(per_rec_preds[alpha], per_rec_targets):
                    pc = window_avg(pr, fs_target, window_s, n_windows)
                    tc = window_avg(tg, fs_target, window_s, n_windows)
                    if pc is None:
                        continue
                    preds_clip.append(pc)
                    tgts_clip.append(tc)
                if not preds_clip:
                    continue
                preds_clip = np.concatenate(preds_clip, axis=0)
                tgts_clip = np.concatenate(tgts_clip, axis=0)
                cfg_key = f"nw{n_windows}_ws{int(window_s)}"
                med = medians_per_config.get(cfg_key)
                m = {}
                for fi, name in enumerate(feature_names):
                    r = pearson_r(preds_clip[:, fi], tgts_clip[:, fi])
                    threshold = float(med[fi]) if med is not None else float(np.median(tgts_clip[:, fi]))
                    p_bin = (preds_clip[:, fi] > threshold).astype(int)
                    t_bin = (tgts_clip[:, fi] > threshold).astype(int)
                    if t_bin.sum() == 0 or t_bin.sum() == len(t_bin):
                        bal_acc = float("nan")
                    else:
                        bal_acc = balanced_accuracy_score(t_bin, p_bin)
                    m[name] = {"corr": r, "bal_acc": bal_acc}
                win_metrics[alpha][cfg_key] = m

        results[split] = {
            "continuous_r_per_alpha": {
                str(a): {name: float(cont_r[a][fi]) for fi, name in enumerate(feature_names)}
                for a in alphas
            },
            "window_metrics_per_alpha": {
                str(a): win_metrics[a] for a in alphas
            },
            "n_recs": len(per_rec_targets),
        }

    # ----------------------------------------------------------------
    # 3. Pick best alpha by val mean continuous r (averaged over features),
    #    report val + test under that alpha.
    # ----------------------------------------------------------------
    best_alpha = max(
        alphas,
        key=lambda a: np.mean([results["val"]["continuous_r_per_alpha"][str(a)][n]
                               for n in feature_names]),
    )
    logger.info("Best alpha (val mean r): %g", best_alpha)

    out = {
        "config": {
            "input": input,
            "corrca_path": corrca_path,
            "n_lags": n_lags,
            "n_lags_ms": n_lags_ms,
            "fs_target": fs_target,
            "feature_names": list(feature_names),
            "alphas": list(alphas),
            "eval_window_configs": [list(c) for c in eval_window_configs],
            "max_train_recs": max_train_recs,
            "movie": movie,
            "n_chans_eff": n_chans_eff,
        },
        "feature_medians": {
            cfg_key: {n: float(med[fi]) for fi, n in enumerate(feature_names)}
            for cfg_key, med in medians_per_config.items()
        },
        "best_alpha": float(best_alpha),
        "results": results,
    }
    out_path = out_dir / f"trf_{input}_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", out_path)

    # Print a quick summary table to stdout for the prototype run.
    print("\n=== TRF prototype summary ===")
    print(f"input={input}, best_alpha={best_alpha}, n_train_recs={len(train_paths)}")
    for split in ("val", "test"):
        print(f"\n[{split}] continuous Pearson r per feature (best alpha):")
        for n in feature_names:
            r = results[split]["continuous_r_per_alpha"][str(best_alpha)][n]
            print(f"  {n:25s}  r = {r:+.3f}")
        print(f"\n[{split}] window-matched (nw1_ws1) corr / bal_acc:")
        wm = results[split]["window_metrics_per_alpha"][str(best_alpha)].get("nw1_ws1", {})
        for n in feature_names:
            mm = wm.get(n, {})
            print(f"  {n:25s}  r = {mm.get('corr', float('nan')):+.3f}   "
                  f"bal_acc = {mm.get('bal_acc', float('nan')):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="raw", choices=["raw", "corrca"])
    p.add_argument("--corrca_path", default="")
    p.add_argument("--max_train_recs", type=int, default=100)
    p.add_argument("--n_lags_ms", type=int, default=1000)
    p.add_argument("--fs_target", type=int, default=50)
    p.add_argument("--output_dir", default="outputs/trf_prototype")
    p.add_argument("--fname", default="experiments/eeg_jepa/cfgs/default.yaml")
    args = p.parse_args()
    run(
        input=args.input,
        corrca_path=args.corrca_path,
        max_train_recs=args.max_train_recs,
        n_lags_ms=args.n_lags_ms,
        fs_target=args.fs_target,
        output_dir=args.output_dir,
        fname=args.fname,
    )
