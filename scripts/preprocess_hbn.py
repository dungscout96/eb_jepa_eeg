"""Preprocess HBN EEG data following REVE's practice.

Preprocessing pipeline (Défossez et al. 2023; REVE, arxiv:2510.21585):
  1. Remove recordings shorter than 10 seconds
  2. Resample to 200 Hz
  3. Apply 0.5–99.5 Hz band-pass filter
  4. Convert to float32
  5. Z-score normalization (per-channel, stats computed across all recordings)
  6. Clip values exceeding ±15 standard deviations

Usage:
    # Activate conda env first, then run with uv:
    conda activate eb_jepa
    uv run scripts/preprocess_hbn.py
    uv run scripts/preprocess_hbn.py release=R2 task=DespicableMe
    uv run scripts/preprocess_hbn.py --multirun \
        task=ThePresent,DespicableMe,RestingState,ContrastChangeDetection
"""

import json
import logging
import sys
import time
from pathlib import Path

import hydra
import mne
import numpy as np
from omegaconf import DictConfig, OmegaConf

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, preprocess

# Ensure project root is on sys.path so eb_jepa imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eb_jepa.datasets.hbn import load_or_download  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELEASES = {
    "R1": "ds005505",
    "R2": "ds005506",
    "R6": "ds005510",
}

TASKS = ["ThePresent", "DespicableMe", "RestingState", "ContrastChangeDetection"]


# ---------------------------------------------------------------------------
# Pass 1: reject, resample, filter, float32
# ---------------------------------------------------------------------------


def reject_short_recordings(
    dataset: BaseConcatDataset,
    min_duration_s: float,
) -> tuple[BaseConcatDataset, int]:
    """Remove recordings shorter than *min_duration_s*.

    Returns the filtered dataset and the number of rejected recordings.
    """
    kept = []
    rejected = 0
    for ds in dataset.datasets:
        try:
            duration = ds.raw.times[-1]
        except (ValueError, OSError) as exc:
            logger.warning("Skipping unloadable recording: %s", exc)
            rejected += 1
            continue
        if duration < min_duration_s:
            logger.info(
                "Rejected recording (%.1fs < %.1fs)", duration, min_duration_s
            )
            rejected += 1
        else:
            kept.append(ds)

    logger.info(
        "Kept %d / %d recordings (rejected %d short)",
        len(kept),
        len(kept) + rejected,
        rejected,
    )
    return BaseConcatDataset(kept), rejected


def run_pass1(
    dataset: BaseConcatDataset,
    save_dir: Path,
    *,
    target_sfreq: float,
    l_freq: float,
    h_freq: float,
) -> BaseConcatDataset:
    """Resample, bandpass filter, and convert to float32.

    Preprocessed data is saved to *save_dir* via braindecode serialization.
    """
    preprocessors = [
        Preprocessor("resample", sfreq=target_sfreq),
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
        Preprocessor(
            lambda data, factor: data * factor,
            factor=1.0,  # identity — forces float64→float64 but we cast below
            apply_on_array=True,
        ),
    ]

    save_dir.mkdir(parents=True, exist_ok=True)
    dataset = preprocess(
        dataset,
        preprocessors,
        save_dir=str(save_dir),
        overwrite=True,
    )
    # Ensure float32 on disk — braindecode may keep float64 after MNE ops.
    _cast_to_float32(dataset, save_dir)
    return dataset


def _cast_to_float32(dataset: BaseConcatDataset, save_dir: Path) -> None:
    """Ensure all raw data in the dataset is stored as float32."""
    for ds in dataset.datasets:
        raw = ds.raw
        if raw.get_data().dtype != np.float32:
            raw.apply_function(lambda x: x.astype(np.float32), dtype=np.float32)


# ---------------------------------------------------------------------------
# Statistics computation (online / two-pass)
# ---------------------------------------------------------------------------


def compute_channel_stats(
    dataset: BaseConcatDataset,
) -> dict:
    """Compute per-channel mean and std across all recordings.

    Uses Chan-Golub-LeVeque parallel combination of batch statistics for
    numerical stability without loading all data into memory at once.

    Returns dict with keys: ``mean``, ``std`` (arrays of shape ``(n_channels,)``),
    ``n_samples``, ``n_recordings``.
    """
    n_channels = None
    count = 0  # total time-samples seen
    mean = None
    M2 = None  # running sum of squared deviations

    for ds in dataset.datasets:
        data = ds.raw.get_data().astype(np.float64)  # (C, T)
        C, T = data.shape

        if n_channels is None:
            n_channels = C
            mean = np.zeros(C, dtype=np.float64)
            M2 = np.zeros(C, dtype=np.float64)
        elif C != n_channels:
            logger.warning(
                "Channel count mismatch: expected %d, got %d — skipping recording",
                n_channels,
                C,
            )
            continue

        # Batch statistics for this recording
        batch_mean = data.mean(axis=1)  # (C,)
        batch_var = data.var(axis=1)  # (C,)
        n = T

        # Parallel combination (Chan-Golub-LeVeque)
        new_count = count + n
        delta = batch_mean - mean
        mean = (count * mean + n * batch_mean) / new_count
        M2 = M2 + n * batch_var + delta**2 * count * n / max(new_count, 1)
        count = new_count

    std = np.sqrt(M2 / max(count, 1))

    logger.info(
        "Stats over %d recordings, %d total samples: "
        "mean range [%.4f, %.4f], std range [%.4f, %.4f]",
        len(dataset.datasets),
        count,
        mean.min(),
        mean.max(),
        std.min(),
        std.max(),
    )

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "n_samples": int(count),
        "n_recordings": len(dataset.datasets),
    }


# ---------------------------------------------------------------------------
# Pass 2: z-score normalize + clip
# ---------------------------------------------------------------------------


def run_pass2(
    dataset: BaseConcatDataset,
    stats: dict,
    clip_std: float,
    save_dir: Path,
) -> BaseConcatDataset:
    """Apply z-score normalization and clipping, then save.

    Normalization: ``(x - mean) / std`` per channel.
    Clipping: values beyond ``±clip_std`` are clamped.
    """
    ch_mean = stats["mean"][:, np.newaxis].astype(np.float64)  # (C, 1)
    ch_std = stats["std"][:, np.newaxis].astype(np.float64)  # (C, 1)
    # Guard against zero-std channels
    ch_std = np.where(ch_std < 1e-8, 1.0, ch_std)

    def _zscore_clip(data: np.ndarray) -> np.ndarray:
        normed = (data.astype(np.float64) - ch_mean) / ch_std
        normed = np.clip(normed, -clip_std, clip_std)
        return normed.astype(np.float32)

    preprocessors = [
        Preprocessor(_zscore_clip, apply_on_array=True),
    ]

    save_dir.mkdir(parents=True, exist_ok=True)
    dataset = preprocess(
        dataset,
        preprocessors,
        save_dir=str(save_dir),
        overwrite=True,
    )
    return dataset


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def process_release_task(release: str, task: str, cfg: DictConfig) -> dict:
    """Run the full two-pass preprocessing pipeline for one (release, task).

    Returns a summary dict with recording counts and timing.
    """
    output_base = Path(cfg.output_dir)
    intermediate_dir = output_base / release / task / "intermediate"
    final_dir = output_base / release / task / "preprocessed"
    meta_dir = output_base / release / task

    # --- Download / load raw data ---
    logger.info("Loading release=%s, task=%s ...", release, task)
    dataset = load_or_download(release, task=task)
    total_recordings = len(dataset.datasets)
    logger.info("Loaded %d recordings", total_recordings)

    # --- Step 1: Reject short recordings ---
    dataset, n_rejected = reject_short_recordings(dataset, cfg.min_duration_s)

    summary = {
        "release": release,
        "task": task,
        "total_recordings": total_recordings,
        "rejected_short": n_rejected,
        "kept": len(dataset.datasets),
    }

    if len(dataset.datasets) == 0:
        logger.warning("No recordings left after rejection — nothing to process.")
        return summary

    # --- Pass 1: Resample, filter, float32 ---
    logger.info("Pass 1: resample (%.0f Hz), filter (%.1f–%.1f Hz), float32 ...",
                cfg.target_sfreq, cfg.l_freq, cfg.h_freq)
    t0 = time.time()
    dataset = run_pass1(
        dataset,
        intermediate_dir,
        target_sfreq=cfg.target_sfreq,
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
    )
    summary["pass1_elapsed_s"] = round(time.time() - t0, 1)
    logger.info("Pass 1 complete in %.1fs", summary["pass1_elapsed_s"])

    # --- Compute normalization stats ---
    logger.info("Computing per-channel normalization statistics ...")
    stats = compute_channel_stats(dataset)

    stats_path = meta_dir / "normalization_stats.npz"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        stats_path,
        mean=stats["mean"],
        std=stats["std"],
        n_samples=stats["n_samples"],
        n_recordings=stats["n_recordings"],
    )
    logger.info("Saved normalization stats to %s", stats_path)

    # --- Pass 2: Z-score + clip ---
    logger.info("Pass 2: z-score normalize + clip at ±%.0f std ...", cfg.clip_std)
    t0 = time.time()
    dataset = run_pass2(dataset, stats, cfg.clip_std, final_dir)
    summary["pass2_elapsed_s"] = round(time.time() - t0, 1)
    logger.info("Pass 2 complete in %.1fs", summary["pass2_elapsed_s"])

    summary["normalization"] = {
        "n_samples": stats["n_samples"],
        "n_recordings": stats["n_recordings"],
        "mean_range": [float(stats["mean"].min()), float(stats["mean"].max())],
        "std_range": [float(stats["std"].min()), float(stats["std"].max())],
    }

    return summary


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../config", config_name="preprocess_hbn")
def main(cfg: DictConfig) -> None:
    """Preprocess HBN EEG data for one (release, task) combination."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    logger.info("HBN EEG Preprocessing")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    release = cfg.release
    task = cfg.task

    if release not in RELEASES:
        raise ValueError(
            f"Unknown release '{release}'. Must be one of {list(RELEASES.keys())}."
        )
    if task not in TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Must be one of {TASKS}."
        )

    t0 = time.time()
    summary = process_release_task(release, task, cfg)
    summary["total_elapsed_s"] = round(time.time() - t0, 1)

    # Persist summary
    summary_path = Path(cfg.output_dir) / release / task / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done in %.1fs", summary["total_elapsed_s"])
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
