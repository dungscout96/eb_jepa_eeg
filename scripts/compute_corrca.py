"""Compute CorrCA spatial filters for stimulus-driven component extraction.

CorrCA (Correlated Component Analysis) finds spatial filters that maximize
inter-subject correlation — the extracted components are, by definition,
stimulus-driven (subject-specific patterns cancel out across subjects).

Usage (on Delta):
    export HBN_PREPROCESS_DIR=/projects/bbnv/kkokate/hbn_preprocessed

    # Standard broadband CorrCA (5 components):
    PYTHONPATH=. python scripts/compute_corrca.py --output_path corrca_5.npz --n_components 5

    # 10 components:
    PYTHONPATH=. python scripts/compute_corrca.py --output_path corrca_10.npz --n_components 10

    # Bandpass 1-8Hz + OAS shrinkage:
    PYTHONPATH=. python scripts/compute_corrca.py --output_path corrca_bp.npz \
        --bandpass_low 1 --bandpass_high 8 --shrinkage oas

    # Band-specific CorrCA (4 bands x 3 components = 12 channels):
    PYTHONPATH=. python scripts/compute_corrca.py --output_path corrca_bands.npz \
        --band_specific --n_components 3

References:
    Parra et al. "Correlated Components Analysis" (NBDT, 2019)
    Dmochowski et al. (2012, Frontiers in Human Neuroscience)
"""

import argparse
import logging
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from eb_jepa.datasets.hbn import (
    SPLIT_RELEASES,
    _load_dataset,
    _read_raw_windows,
    PREPROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 20),
}


def _bandpass(data, low, high, sfreq, order=4):
    """Apply zero-phase Butterworth bandpass filter. data: [C, T]."""
    sos = butter(order, [low, high], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def _apply_shrinkage(R, method="oas"):
    """Apply shrinkage to a covariance matrix."""
    if method == "oas":
        from sklearn.covariance import OAS
        p = R.shape[0]
        oas = OAS()
        oas.fit(np.eye(p))  # dummy fit to initialize
        # Manual OAS: shrink R toward scaled identity
        n = max(p, 100)  # effective sample size
        trace_R = np.trace(R)
        trace_R2 = np.trace(R @ R)
        rho_num = (1 - 2.0 / p) * trace_R2 + trace_R ** 2
        rho_den = (n + 1 - 2.0 / p) * (trace_R2 - trace_R ** 2 / p)
        rho = min(1.0, max(0.0, rho_num / max(rho_den, 1e-10)))
        target = trace_R / p * np.eye(p)
        R_shrunk = (1 - rho) * R + rho * target
        logger.info("OAS shrinkage: rho=%.4f", rho)
        return R_shrunk
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        p = R.shape[0]
        trace_R = np.trace(R)
        trace_R2 = np.trace(R @ R)
        n = max(p, 100)
        rho = min(1.0, max(0.0, ((1 - 2.0 / p) * trace_R2 + trace_R ** 2)
                  / ((n + 1 - 2.0 / p) * (trace_R2 - trace_R ** 2 / p) + 1e-10)))
        target = trace_R / p * np.eye(p)
        R_shrunk = (1 - rho) * R + rho * target
        logger.info("Ledoit-Wolf shrinkage: rho=%.4f", rho)
        return R_shrunk
    else:
        # Ridge fallback
        R += np.eye(R.shape[0]) * 1e-6
        return R


def _load_movie_data(task, preprocessed_dir, n_time_bins, bandpass_low=None, bandpass_high=None):
    """Load movie-aligned EEG from all training subjects, return binned data."""
    train_releases = SPLIT_RELEASES.get("train", {})
    logger.info("Loading training data for task '%s' from %d releases", task, len(train_releases))

    bin_data = {b: [] for b in range(n_time_bins)}
    n_subjects_loaded = 0
    sfreq_detected = None

    for release in train_releases:
        logger.info("Loading release %s...", release)
        try:
            ds = _load_dataset(release, task, preprocessed=True,
                               preprocessed_dir=Path(preprocessed_dir), preload=False)
        except FileNotFoundError:
            logger.warning("Preprocessed data not found for %s/%s, skipping", release, task)
            continue

        for rec in tqdm(ds.datasets, desc=f"{release}"):
            try:
                raw = rec.raw
            except (ValueError, OSError):
                continue

            sfreq = raw.info["sfreq"]
            if sfreq_detected is None:
                sfreq_detected = sfreq

            try:
                events, event_id = mne.events_from_annotations(raw, verbose=False)
            except Exception:
                continue

            if "video_start" not in event_id or "video_stop" not in event_id:
                continue

            try:
                events_filtered = mne.pick_events(
                    events, include=[event_id["video_start"], event_id["video_stop"]]
                )
            except Exception:
                continue

            if len(events_filtered) < 2:
                continue

            start_samp = int(events_filtered[0, 0])
            end_samp = int(events_filtered[1, 0])

            try:
                data = raw.get_data(start=start_samp, stop=min(end_samp, raw.n_times))
            except (ValueError, IndexError):
                continue

            if data.shape[1] < 100:
                continue

            # Optional bandpass filter
            if bandpass_low is not None and bandpass_high is not None:
                data = _bandpass(data, bandpass_low, bandpass_high, sfreq)

            # Bin into temporal bins
            samples_per_bin = data.shape[1] // n_time_bins
            for b in range(n_time_bins):
                s = b * samples_per_bin
                e = s + samples_per_bin
                if e <= data.shape[1]:
                    bin_data[b].append(data[:, s:e].astype(np.float32))

            n_subjects_loaded += 1

    logger.info("Loaded %d subjects across %d releases", n_subjects_loaded, len(train_releases))
    return bin_data, n_subjects_loaded, sfreq_detected


def _compute_covariance(bin_data, n_time_bins):
    """Compute between-subject (R_b) and within-subject (R_w) covariance matrices."""
    logger.info("Computing covariance matrices...")

    n_chans_detected = None
    R_b = None
    R_w = None
    n_pairs = 0
    n_samples_total = 0

    for b in tqdm(range(n_time_bins), desc="Time bins"):
        subjects_at_bin = bin_data[b]
        K = len(subjects_at_bin)
        if K < 2:
            continue

        if n_chans_detected is None:
            n_chans_detected = subjects_at_bin[0].shape[0]
            R_b = np.zeros((n_chans_detected, n_chans_detected), dtype=np.float64)
            R_w = np.zeros((n_chans_detected, n_chans_detected), dtype=np.float64)

        min_t = min(s.shape[1] for s in subjects_at_bin if s.shape[0] == n_chans_detected)
        valid = [s[:, :min_t] for s in subjects_at_bin if s.shape[0] == n_chans_detected]
        if len(valid) < 2:
            continue

        X = np.stack(valid)  # [K, C, T]
        K = X.shape[0]
        T = X.shape[2]

        for i in range(K):
            R_w += X[i] @ X[i].T / T

        X_sum = X.sum(axis=0)
        R_b += (X_sum @ X_sum.T / T)
        for i in range(K):
            R_b -= X[i] @ X[i].T / T

        n_pairs += K * (K - 1)
        n_samples_total += K

    R_b /= max(n_pairs, 1)
    R_w /= max(n_samples_total, 1)

    return R_b, R_w, n_chans_detected


def _solve_corrca(R_b, R_w, n_components, shrinkage="ridge"):
    """Solve generalized eigenvalue problem with optional shrinkage."""
    R_w = _apply_shrinkage(R_w, shrinkage)

    logger.info("Solving generalized eigenvalue problem (R_b w = λ R_w w)...")
    eigenvalues, eigenvectors = eigh(R_b, R_w)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    W = eigenvectors[:, :n_components]
    isc_values = eigenvalues[:n_components]

    logger.info("CorrCA ISC values (top %d): %s", n_components,
                ", ".join(f"{v:.4f}" for v in isc_values))

    return W, isc_values


def compute_corrca(
    task: str = "ThePresent",
    n_components: int = 5,
    n_time_bins: int = 100,
    output_path: str = "corrca_filters.npz",
    preprocessed_dir: str | None = None,
    bandpass_low: float | None = None,
    bandpass_high: float | None = None,
    shrinkage: str = "ridge",
    band_specific: bool = False,
):
    """Compute CorrCA spatial filters from training data.

    Args:
        bandpass_low/high: If set, bandpass filter before CorrCA (e.g., 1-8 Hz).
        shrinkage: Covariance regularization: "ridge" (1e-6), "oas", or "ledoit_wolf".
        band_specific: If True, compute separate CorrCA per frequency band
            (delta/theta/alpha/beta), concatenate filters. Output has 4*n_components channels.
    """
    if preprocessed_dir is None:
        preprocessed_dir = PREPROCESSED_DIR

    if band_specific:
        # Compute CorrCA separately for each frequency band, concatenate
        all_W = []
        all_isc = []
        band_labels = []

        for band_name, (low, high) in BANDS.items():
            logger.info("=== Band: %s (%d-%d Hz) ===", band_name, low, high)
            bin_data, n_subj, sfreq = _load_movie_data(
                task, preprocessed_dir, n_time_bins, bandpass_low=low, bandpass_high=high)
            R_b, R_w, n_chans = _compute_covariance(bin_data, n_time_bins)
            W, isc = _solve_corrca(R_b, R_w, n_components, shrinkage)
            all_W.append(W)
            all_isc.extend(isc)
            band_labels.extend([band_name] * n_components)
            del bin_data  # free memory

        W_concat = np.concatenate(all_W, axis=1)  # [129, 4*n_components]
        isc_concat = np.array(all_isc, dtype=np.float32)

        logger.info("Band-specific CorrCA: %d total components across %d bands",
                     W_concat.shape[1], len(BANDS))
        logger.info("All ISC values: %s", ", ".join(f"{v:.4f}" for v in isc_concat))

        np.savez(
            output_path,
            W=W_concat.astype(np.float32),
            isc_values=isc_concat,
            n_subjects=n_subj,
            n_chans=n_chans,
            task=task,
            band_labels=np.array(band_labels),
        )
        logger.info("Saved band-specific CorrCA filters to %s (shape: %s)",
                     output_path, W_concat.shape)
        return

    # Standard single-band CorrCA
    bin_data, n_subjects_loaded, sfreq = _load_movie_data(
        task, preprocessed_dir, n_time_bins, bandpass_low, bandpass_high)
    R_b, R_w, n_chans_detected = _compute_covariance(bin_data, n_time_bins)
    W, isc_values = _solve_corrca(R_b, R_w, n_components, shrinkage)

    np.savez(
        output_path,
        W=W.astype(np.float32),
        isc_values=isc_values.astype(np.float32),
        n_subjects=n_subjects_loaded,
        n_chans=n_chans_detected,
        task=task,
    )
    logger.info("Saved CorrCA filters to %s (shape: %s)", output_path, W.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="corrca_filters.npz")
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--task", default="ThePresent")
    parser.add_argument("--n_time_bins", type=int, default=100)
    parser.add_argument("--preprocessed_dir", default=None)
    parser.add_argument("--bandpass_low", type=float, default=None)
    parser.add_argument("--bandpass_high", type=float, default=None)
    parser.add_argument("--shrinkage", default="ridge", choices=["ridge", "oas", "ledoit_wolf"])
    parser.add_argument("--band_specific", action="store_true")
    args = parser.parse_args()
    compute_corrca(**vars(args))
