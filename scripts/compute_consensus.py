"""Precompute the ISC-consensus signal (across-subject mean of CorrCA-5 at each
movie timestep) for use as the JEPA teacher-branch input in Exp 10.

The consensus acts as a denoised, stimulus-locked reference: each training
recording's CorrCA output is ~93% within-subject noise (see raw_data_analysis),
so averaging across ~700 subjects at matched movie time gives a much cleaner
approximation of the stimulus-driven component of EEG.

Algorithm:
    for each training recording:
        read raw EEG between video_start and video_start + movie_samples
        per-recording z-normalize over the full movie portion (mean & std over [C, T])
        project with CorrCA W: [129, T_rec] -> [5, T_rec]
        clip/pad to exactly movie_samples
    consensus = mean across recordings over axis 0 -> [5, T_movie]

Output: consensus.npz with
    data:       [5, T_movie] float32
    sfreq:      int (200)
    task:       str ("ThePresent")
    n_subjects: int (count of recordings averaged)

Usage on Delta:
    export HBN_PREPROCESS_DIR=/projects/bbnv/kkokate/hbn_preprocessed
    PYTHONPATH=. uv run --group eeg python scripts/compute_consensus.py \
        --corrca_path corrca_filters.npz \
        --output_path consensus.npz \
        --task ThePresent
"""

import argparse
import logging
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm

from eb_jepa.datasets.hbn import (
    MOVIE_METADATA,
    PREPROCESSED_DIR,
    SPLIT_RELEASES,
    _load_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_consensus(
    corrca_path: str,
    output_path: str = "consensus.npz",
    task: str = "ThePresent",
    preprocessed_dir: str | None = None,
):
    if preprocessed_dir is None:
        preprocessed_dir = PREPROCESSED_DIR

    # Load CorrCA filters [C_in, k]
    corrca = np.load(corrca_path)
    W = corrca["W"].astype(np.float32)
    n_chans_W, k = W.shape
    logger.info("Loaded CorrCA %s: %d -> %d components (ISC: %s)",
                corrca_path, n_chans_W, k,
                ", ".join(f"{v:.3f}" for v in corrca["isc_values"]))

    train_releases = SPLIT_RELEASES.get("train", {})
    logger.info("Building consensus over training releases: %s",
                list(train_releases.keys()))

    # Target length: movie_duration * sfreq (sfreq detected from first raw)
    movie_duration_s = MOVIE_METADATA[task]["duration"]
    sfreq = None
    movie_samples = None

    accum = None        # [k, T] sum of per-subject CorrCA projections
    count = None        # [T] per-timestep count of contributing recordings
    n_subjects = 0
    n_rejected = 0

    for release in train_releases:
        logger.info("Loading release %s...", release)
        try:
            ds = _load_dataset(release, task, preprocessed=True,
                               preprocessed_dir=Path(preprocessed_dir), preload=False)
        except FileNotFoundError:
            logger.warning("Preprocessed data missing for %s/%s, skipping",
                           release, task)
            continue

        for rec in tqdm(ds.datasets, desc=f"{release}"):
            try:
                raw = rec.raw
            except (ValueError, OSError):
                n_rejected += 1
                continue

            try:
                events, event_id = mne.events_from_annotations(raw, verbose=False)
            except Exception:
                n_rejected += 1
                continue
            if "video_start" not in event_id or "video_stop" not in event_id:
                n_rejected += 1
                continue

            events_filtered = mne.pick_events(
                events, include=[event_id["video_start"], event_id["video_stop"]]
            )
            if len(events_filtered) < 2:
                n_rejected += 1
                continue

            if sfreq is None:
                sfreq = int(round(raw.info["sfreq"]))
                movie_samples = int(round(movie_duration_s * sfreq))
                accum = np.zeros((k, movie_samples), dtype=np.float64)
                count = np.zeros(movie_samples, dtype=np.int64)
                logger.info("Target consensus shape: [%d, %d] @ %d Hz",
                            k, movie_samples, sfreq)

            start_samp = int(events_filtered[0, 0])
            end_samp = int(events_filtered[1, 0])
            # Clip end to video_start + movie_samples to enforce shared grid
            end_capped = min(end_samp, start_samp + movie_samples,
                             raw.n_times)

            try:
                data = raw.get_data(start=start_samp, stop=end_capped)
            except (ValueError, IndexError):
                n_rejected += 1
                continue

            if data.shape[0] != n_chans_W:
                n_rejected += 1
                continue
            T_rec = data.shape[1]
            if T_rec < sfreq * 30:  # require at least 30s of movie
                n_rejected += 1
                continue

            data = data.astype(np.float32)

            # Per-recording z-normalize over the full movie portion
            # (matches training behaviour modulo window-batch effects; see
            #  JEPAMovieDataset.__getitem__)
            rec_mean = data.mean()
            rec_std = data.std()
            if rec_std < 1e-8:
                n_rejected += 1
                continue
            data = (data - rec_mean) / rec_std

            # CorrCA project: [C, T] -> [k, T]
            proj = W.T @ data  # [k, T_rec]

            accum[:, :T_rec] += proj
            count[:T_rec] += 1
            n_subjects += 1

    if n_subjects == 0:
        raise RuntimeError("No recordings contributed to consensus (all rejected).")

    # Divide sums by per-timestep counts (handle the case where some tail
    # samples had fewer contributors if recordings are slightly short).
    count_safe = np.maximum(count, 1)
    consensus = (accum / count_safe).astype(np.float32)

    logger.info("Consensus built from %d recordings (%d rejected)",
                n_subjects, n_rejected)
    logger.info("Per-channel variance of consensus: %s",
                np.var(consensus, axis=1))
    logger.info("Per-timestep count range: [%d, %d], median %d",
                int(count.min()), int(count.max()), int(np.median(count)))

    np.savez(
        output_path,
        data=consensus,
        sfreq=np.int64(sfreq),
        task=task,
        n_subjects=np.int64(n_subjects),
        count_per_sample=count.astype(np.int64),
    )
    logger.info("Saved %s (shape: %s)", output_path, consensus.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrca_path", required=True,
                        help="Path to corrca_filters.npz")
    parser.add_argument("--output_path", default="consensus.npz")
    parser.add_argument("--task", default="ThePresent")
    parser.add_argument("--preprocessed_dir", default=None)
    args = parser.parse_args()
    compute_consensus(**vars(args))
