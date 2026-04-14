"""Compute CorrCA spatial filters for stimulus-driven component extraction.

CorrCA (Correlated Component Analysis) finds spatial filters that maximize
inter-subject correlation — the extracted components are, by definition,
stimulus-driven (subject-specific patterns cancel out across subjects).

Usage (on Delta):
    export HBN_PREPROCESS_DIR=/projects/bbnv/kkokate/hbn_preprocessed
    PYTHONPATH=. .venv/bin/python scripts/compute_corrca.py \
        --output corrca_filters.npz \
        --n_components 5 \
        --task ThePresent

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
from tqdm import tqdm

from eb_jepa.datasets.hbn import (
    SPLIT_RELEASES,
    _load_dataset,
    _read_raw_windows,
    PREPROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_corrca(
    task: str = "ThePresent",
    n_components: int = 5,
    window_size_seconds: float = 2.0,
    n_time_bins: int = 100,
    output_path: str = "corrca_filters.npz",
    preprocessed_dir: str | None = None,
):
    """Compute CorrCA spatial filters from training data.

    Algorithm:
        1. Load all training subjects' time-aligned EEG for the movie
        2. Bin into n_time_bins temporal bins
        3. Compute between-subject covariance R_b (shared stimulus response)
        4. Compute within-subject covariance R_w (total variance)
        5. Solve generalized eigenvalue problem: R_b w = λ R_w w
        6. Save top-k spatial filter matrix W [n_chans, n_components]
    """
    if preprocessed_dir is None:
        preprocessed_dir = PREPROCESSED_DIR

    # Load training split releases
    train_releases = SPLIT_RELEASES.get("train", {})
    logger.info("Loading training data for task '%s' from %d releases", task, len(train_releases))

    # Collect per-subject time-binned EEG
    # subject_data[bin_idx] = list of [C, T_samples] arrays from different subjects
    bin_data = {b: [] for b in range(n_time_bins)}
    n_subjects_loaded = 0

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
            n_chans = len(raw.ch_names)

            # Find video_start annotation
            video_start = None
            for ann in raw.annotations:
                if ann["description"] == "video_start":
                    video_start = ann["onset"]
                    video_duration = ann["duration"]
                    break
            if video_start is None:
                continue

            # Extract movie-aligned data
            start_samp = int(video_start * sfreq)
            end_samp = int((video_start + video_duration) * sfreq)
            data = raw.get_data(start=start_samp, stop=end_samp)  # [C, T]

            if data.shape[1] < 100:
                continue

            # Bin into temporal bins
            samples_per_bin = data.shape[1] // n_time_bins
            for b in range(n_time_bins):
                s = b * samples_per_bin
                e = s + samples_per_bin
                if e <= data.shape[1]:
                    bin_data[b].append(data[:, s:e].astype(np.float32))

            n_subjects_loaded += 1

    logger.info("Loaded %d subjects across %d releases", n_subjects_loaded, len(train_releases))

    # Compute R_b (between-subject) and R_w (within-subject) covariance
    # R_b = (1/K(K-1)) sum_{i!=j} cov(X_i, X_j)
    # R_w = (1/K) sum_i cov(X_i, X_i)
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

        # Ensure all subjects have same n_chans
        if n_chans_detected is None:
            n_chans_detected = subjects_at_bin[0].shape[0]
            R_b = np.zeros((n_chans_detected, n_chans_detected), dtype=np.float64)
            R_w = np.zeros((n_chans_detected, n_chans_detected), dtype=np.float64)

        # Stack subjects: [K, C, T]
        min_t = min(s.shape[1] for s in subjects_at_bin if s.shape[0] == n_chans_detected)
        valid = [s[:, :min_t] for s in subjects_at_bin if s.shape[0] == n_chans_detected]
        if len(valid) < 2:
            continue

        X = np.stack(valid)  # [K, C, T]
        K = X.shape[0]
        T = X.shape[2]

        # Within-subject: average of X_i @ X_i^T
        for i in range(K):
            R_w += X[i] @ X[i].T / T

        # Between-subject: average of X_i @ X_j^T
        X_sum = X.sum(axis=0)  # [C, T]
        # sum_{i!=j} X_i @ X_j^T = (sum_i X_i) @ (sum_j X_j)^T - sum_i X_i @ X_i^T
        R_b += (X_sum @ X_sum.T / T)
        for i in range(K):
            R_b -= X[i] @ X[i].T / T

        n_pairs += K * (K - 1)
        n_samples_total += K

    # Normalize
    R_b /= max(n_pairs, 1)
    R_w /= max(n_samples_total, 1)

    # Regularize R_w for numerical stability
    R_w += np.eye(n_chans_detected) * 1e-6

    logger.info("Solving generalized eigenvalue problem (R_b w = λ R_w w)...")
    eigenvalues, eigenvectors = eigh(R_b, R_w)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top-k components
    W = eigenvectors[:, :n_components]  # [n_chans, n_components]
    isc_values = eigenvalues[:n_components]

    logger.info("CorrCA ISC values (top %d): %s", n_components,
                ", ".join(f"{v:.4f}" for v in isc_values))

    # Save
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
    args = parser.parse_args()
    compute_corrca(**vars(args))
