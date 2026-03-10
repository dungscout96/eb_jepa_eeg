"""HBN (Healthy Brain Network) EEG dataset for self-supervised learning and movie probe tasks."""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import mne
import pandas as pd
import torch
from torch.utils.data import Dataset

from braindecode.datasets import BaseConcatDataset  # noqa: F401
from braindecode.datautil.serialization import load_concat_dataset
from braindecode.preprocessing import (
    create_fixed_length_windows,
    create_windows_from_events,
)
from eegdash import EEGDashDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = Path(
    os.environ.get("HBN_CACHE_DIR",
                   str(Path.home() / ".cache" / "eb_jepa" / "datasets" / "eegdash_cache"))
)


SPLIT_RELEASES = {
    "train": {
        "R2": "ds005506",  # 152 subjects
        "R3": "ds005507",  # 184 subjects
        "R4": "ds005508",  # 324 subjects
    },
    "val": {"R1": "ds005505"},  # 136 subjects
    "test": {"R6": "ds005510"},  # 134 subjects
}

DEFAULT_TASK = "ThePresent"

# Default tolerances & thresholds (overridable via dataset constructor)
# TODO: Verify
VISUAL_PROCESSING_DELAY_S = 0.1
DEFAULT_ANNOTATION_DURATION_TOLERANCE_S = 0.5
DEFAULT_POST_MOVIE_VISUAL_PROCESSING_S = 2.0
DEFAULT_MAX_RECORDING_OVERSHOOT_S = 60.0
DEFAULT_TRIAL_STOP_OFFSET_S = 0.1

MOVIE_METADATA = {
    "ThePresent": {
        "duration": 203.3,  # 3 minutes 23 seconds
        "fps": 24,
        "frame_count": 4878,
        "feature_csv": str(
            PROJECT_ROOT / "data" / "output" / "The_Present" / "features.csv"
        ),
    }
}

_MOVIE_PATHS = {
    "ThePresent": PROJECT_ROOT / "data" / "movies" / "The_Present.mp4",
    "RestingState": PROJECT_ROOT / "data" / "movies" / "Resting_State.mp4",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_releases(split: str) -> dict:
    """Return the release mapping for the given split."""
    if split not in SPLIT_RELEASES:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of {list(SPLIT_RELEASES.keys())}."
        )
    return SPLIT_RELEASES[split]


def get_movie_recording_duration(
    raw: mne.io.BaseRaw,
    movie: str,
    *,
    max_recording_overshoot_s: float = DEFAULT_MAX_RECORDING_OVERSHOOT_S,
) -> float:
    """Compute the duration of the movie portion of a recording (in seconds)."""
    events, event_id = mne.events_from_annotations(raw)
    events_filtered = mne.pick_events(
        events,
        include=[event_id["video_start"], event_id["video_stop"]],
    )
    duration_samples = events_filtered[1, 0] - events_filtered[0, 0]
    duration_seconds = duration_samples / raw.info["sfreq"]

    max_expected = MOVIE_METADATA[movie]["duration"] + max_recording_overshoot_s
    if duration_seconds > max_expected:
        raise ValueError(
            f"Recording duration {duration_seconds:.1f}s exceeds "
            f"expected maximum {max_expected:.1f}s for movie '{movie}'."
        )
    return duration_seconds


def reject_recording(
    raw: mne.io.BaseRaw,
    movie: str,
    *,
    annotation_duration_tolerance_s: float = DEFAULT_ANNOTATION_DURATION_TOLERANCE_S,
    max_recording_overshoot_s: float = DEFAULT_MAX_RECORDING_OVERSHOOT_S,
) -> bool:
    """Return True if the recording should be excluded from the dataset.

    Rejection criteria:
    - Missing ``video_start`` or ``video_stop`` annotations
    - Recording duration shorter than movie duration (minus *annotation_duration_tolerance_s*)
    """
    events, event_id = mne.events_from_annotations(raw)
    if "video_start" not in event_id or "video_stop" not in event_id:
        logger.warning("Recording missing 'video_start' or 'video_stop' annotations: %s", raw)
        return True

    duration_seconds = get_movie_recording_duration(
        raw, movie, max_recording_overshoot_s=max_recording_overshoot_s
    )
    min_duration = MOVIE_METADATA[movie]["duration"] - annotation_duration_tolerance_s
    if duration_seconds < min_duration:
        logger.warning(
            "Recording duration %.1fs is shorter than expected minimum %.1fs for movie '%s'",
            duration_seconds, min_duration, movie
        )
        return True

    return False


def _release_to_dataset_id(release: str) -> str:
    """Look up the OpenNeuro dataset ID for a given release key."""
    for split_releases in SPLIT_RELEASES.values():
        if release in split_releases:
            return split_releases[release]
    raise ValueError(
        f"Unknown release '{release}'. Known releases: "
        f"{[r for splits in SPLIT_RELEASES.values() for r in splits]}"
    )


def load_or_download(release, task=DEFAULT_TASK):
    """Load an EEGDashDataset from cache, downloading if necessary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_id = _release_to_dataset_id(release)
    dataset = EEGDashDataset(
        cache_dir=DATA_DIR,
        dataset=dataset_id,
    )

    return dataset


# Default directory for preprocessed data (override via HBN_PREPROCESS_DIR env var).
PREPROCESSED_DIR = Path(
    os.environ.get("HBN_PREPROCESS_DIR", str(PROJECT_ROOT / "data" / "hbn_preprocessed"))
)


def load_preprocessed(
    release: str,
    task: str,
    preprocessed_dir: Path | None = None,
) -> BaseConcatDataset:
    """Load preprocessed EEG data saved by ``scripts/preprocess_hbn.py``.

    Parameters
    ----------
    release : str
        HBN release identifier (e.g. ``"R1"``).
    task : str
        EEG task name (e.g. ``"ThePresent"``).
    preprocessed_dir : Path or None
        Root directory of preprocessed data.  Defaults to :data:`PREPROCESSED_DIR`.

    Returns
    -------
    BaseConcatDataset
        Dataset loaded via braindecode serialization (preload=True).
    """
    if preprocessed_dir is None:
        preprocessed_dir = PREPROCESSED_DIR
    data_path = preprocessed_dir / release / task / "preprocessed"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}. "
            f"Run `scripts/preprocess_hbn.py release={release} task={task}` first."
        )
    return load_concat_dataset(str(data_path), preload=True)


def _has_preprocessed(release: str, task: str, preprocessed_dir: Path | None = None) -> bool:
    """Check if preprocessed data exists for a (release, task) combination."""
    if preprocessed_dir is None:
        preprocessed_dir = PREPROCESSED_DIR
    return (preprocessed_dir / release / task / "preprocessed").exists()


def _load_dataset(
    release: str,
    task: str,
    preprocessed: bool = False,
    preprocessed_dir: Path | None = None,
) -> BaseConcatDataset:
    """Load dataset for a (release, task), auto-detecting preprocessed data.

    If *preprocessed* is True, forces loading from preprocessed directory
    (raises if missing).  Otherwise, automatically uses preprocessed data
    when available and falls back to downloading raw data.
    """
    if preprocessed:
        return load_preprocessed(release, task, preprocessed_dir)
    if _has_preprocessed(release, task, preprocessed_dir):
        logger.info("Found preprocessed data for %s/%s, loading...", release, task)
        return load_preprocessed(release, task, preprocessed_dir)
    return load_or_download(release, task=task)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class HBNDataset(Dataset):
    """Self-supervised EEG dataset: each item is a random crop of contiguous windows (WxCxT).

    n_windows are contiguously selected from a random start point in the recording.
    This resembles the memory constraint of video clip in JEPA.
    More number of windows increases the memory requirement but also increases the
    temporal context available to the model.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    n_windows : int
        Number of contiguous windows per item.
    window_size_seconds : float
        Duration of each window in seconds.
    task : str or list[str]
        EEG task name(s).  When a list is given, recordings from all tasks
        are combined into a single dataset.
    preprocessed : bool
        If True, force loading from preprocessed data (error if missing).
        If False (default), preprocessed data is used automatically when
        available and raw data is downloaded otherwise.
    preprocessed_dir : Path or None
        Override the default preprocessed data directory.
    """

    def __init__(
        self,
        split="train",
        n_windows=16,
        window_size_seconds=2,
        task=DEFAULT_TASK,
        preprocessed=False,
        preprocessed_dir=None,
    ):
        releases = _resolve_releases(split)
        self.n_windows = n_windows
        self.window_size_seconds = window_size_seconds
        tasks = [task] if isinstance(task, str) else list(task)

        self.recordings = []
        for release in releases:
            for t in tasks:
                dataset = _load_dataset(release, t, preprocessed, preprocessed_dir)
                sfreq = dataset.datasets[0].raw.info["sfreq"]
                window_samples = int(window_size_seconds * sfreq)
                windowed_dataset = create_fixed_length_windows(
                    dataset,
                    window_size_samples=window_samples,
                    window_stride_samples=window_samples,
                    drop_last_window=True,
                )
                for recording_ds in windowed_dataset.datasets:
                    if len(recording_ds) >= n_windows:
                        self.recordings.append(recording_ds)

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        recording = self.recordings[idx]
        start = torch.randint(0, len(recording) - self.n_windows + 1, (1,)).item()
        windows = torch.stack([
            torch.from_numpy(recording[start + i][0])
            for i in range(self.n_windows)
        ])
        return windows


# ---------------------------------------------------------------------------
# Movie-probe dataset
# ---------------------------------------------------------------------------

def _preload_movie_features(task: str) -> dict:
    """Load movie feature CSVs for the given task."""
    csv_path = MOVIE_METADATA[task]["feature_csv"]
    return {task: pd.read_csv(csv_path)}


def get_movie_metadata(task):
    """Compute movie duration, fps, and frame count from the video file.

    This is a utility for recomputing MOVIE_METADATA values; results are
    typically hardcoded above after initial computation.
    """
    import cv2  # lazy: heavy dependency, only needed for metadata recomputation

    movie_path = str(_MOVIE_PATHS[task])
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    movie_duration_seconds = frame_count / fps
    cap.release()
    return movie_duration_seconds, fps, frame_count


def get_window_movie_metadata(
    window_onset: int,
    sfreq: int,
    movie: str,
    movie_features: pd.DataFrame,
    *,
    post_movie_visual_processing_s: float = DEFAULT_POST_MOVIE_VISUAL_PROCESSING_S,
    visual_processing_delay_s: float = VISUAL_PROCESSING_DELAY_S,
) -> dict:
    """Map an EEG window onset (in samples) to the corresponding movie frame features.

    The EEG at *window_onset* reflects visual input from
    ``visual_processing_delay_s`` seconds earlier, so the frame index is
    computed from ``(window_onset - delay) / sfreq * fps``.

    Assumes the movie starts at the same time as the recording with no dropped
    frames.  Windows up to *post_movie_visual_processing_s* past the movie end
    are clamped to the last available frame.
    """
    delay_samples = int(visual_processing_delay_s * sfreq)
    movie_timestamp = (window_onset - delay_samples) / sfreq
    frame_index = int(movie_timestamp * MOVIE_METADATA[movie]["fps"])

    # Clamp to valid range (negative indices can occur when onset < delay).
    frame_index = max(0, frame_index)

    # Clamp frames that fall past the movie end but within the visual-processing
    # tolerance window (viewers still process the last frame briefly after it ends).
    if frame_index >= len(movie_features):
        max_overshoot_frames = int(
            MOVIE_METADATA[movie]["fps"] * post_movie_visual_processing_s
        )
        if (frame_index - MOVIE_METADATA[movie]["frame_count"]) < max_overshoot_frames:
            frame_index = len(movie_features) - 1

    return movie_features.iloc[frame_index].to_dict()


class HBNMovieDataset(Dataset):
    """Supervised EEG dataset: each window is paired with movie features at its timestamp."""

    def __init__(
        self,
        split="train",
        window_size_seconds=2,
        task=DEFAULT_TASK,
        *,
        cfg: DictConfig | dict,
        preprocessed: bool = False,
        preprocessed_dir: Path | None = None,
    ):
        self.window_size_seconds = window_size_seconds
        tasks = [task] if isinstance(task, str) else list(task)
        self.tasks = tasks
        self.task = tasks[0]  # primary task (backwards compat)
        self.movie_features = {}
        for t in tasks:
            self.movie_features.update(_preload_movie_features(t))
        self.post_movie_visual_processing_s = cfg.get("post_movie_visual_processing_s") if isinstance(cfg, dict) else cfg.post_movie_visual_processing_s
        self.visual_processing_delay_s = cfg.get("visual_processing_delay_s") if isinstance(cfg, dict) else cfg.visual_processing_delay_s
        releases = _resolve_releases(split)

        self.sfreq = None
        self.data = []
        self.labels = []
        total_recordings = 0
        total_rejected = 0

        for t in tasks:
            # Load data for this task across all releases
            datasets = []
            for release in releases:
                ds = _load_dataset(release, t, preprocessed, preprocessed_dir)
                datasets.append(ds)
            data = BaseConcatDataset(datasets)

            if self.sfreq is None:
                self.sfreq = data.datasets[0].raw.info["sfreq"]
            sfreq = self.sfreq

            selected_recordings = []
            rejected = 0
            for recording_ds in data.datasets:
                try:
                    raw = recording_ds.raw
                except (ValueError, OSError) as exc:
                    logger.warning("Skipping unloadable recording %s: %s", recording_ds, exc)
                    rejected += 1
                    continue

                for idx, ann in enumerate(raw.annotations):
                    if ann["description"] == "video_start":
                        movie_recording_duration = get_movie_recording_duration(
                            raw, movie=t,
                            max_recording_overshoot_s=cfg.get("max_recording_overshoot_s") if isinstance(cfg, dict) else cfg.max_recording_overshoot_s,
                        )
                        movie_recording_duration = min(
                            movie_recording_duration, MOVIE_METADATA[t]["duration"]
                        )
                        logger.info(
                            "Setting 'video_start' duration to %.2fs for %s",
                            movie_recording_duration, recording_ds,
                        )
                        raw.annotations.duration[idx] = movie_recording_duration

                if reject_recording(
                    raw, movie=t,
                    annotation_duration_tolerance_s=cfg.get("annotation_duration_tolerance_s") if isinstance(cfg, dict) else cfg.annotation_duration_tolerance_s,
                    max_recording_overshoot_s=cfg.get("max_recording_overshoot_s") if isinstance(cfg, dict) else cfg.max_recording_overshoot_s,
                ):
                    rejected += 1
                    continue
                selected_recordings.append(recording_ds)

            total_recordings += len(data.datasets)
            total_rejected += rejected
            logger.info("Task %s: rejected %d/%d recordings", t, rejected, len(data.datasets))

            window_size_samples = int(window_size_seconds * sfreq)
            visual_processing_delay = cfg.get("visual_processing_delay_s") if isinstance(cfg, dict) else cfg.visual_processing_delay_s
            trial_stop_offset = cfg.get("trial_stop_offset_s") if isinstance(cfg, dict) else cfg.trial_stop_offset_s
            trial_start_offset_samples = int(visual_processing_delay * sfreq)

            # Temporarily set self.task for _get_movie_features_for_window
            self.task = t
            for rec in selected_recordings:
                # Offset the trial start by the visual processing delay so the first
                # EEG window corresponds to the neural response to the first movie frame.
                window_ds = create_windows_from_events(
                    BaseConcatDataset([rec]),
                    mapping={"video_start": 0},
                    trial_start_offset_samples=trial_start_offset_samples,
                    trial_stop_offset_samples=-int(trial_stop_offset * sfreq),
                    window_size_samples=window_size_samples,
                    window_stride_samples=window_size_samples,
                    drop_last_window=True,
                )
                self.data.append(window_ds)

                window_onsets = window_ds.get_metadata().apply(lambda row: row["i_start_in_trial"], axis=1)
                movie_features_for_windows = window_onsets.apply(self._get_movie_features_for_window)
                self.labels.append(movie_features_for_windows)

        self.task = tasks[0]  # reset to primary task
        logger.info("Total: rejected %d/%d recordings across %d task(s)",
                     total_rejected, total_recordings, len(tasks))

    def _get_movie_features_for_window(self, window_onset) -> dict:
        return get_window_movie_metadata(
            window_onset=window_onset,
            sfreq=self.sfreq,
            movie=self.task,
            movie_features=self.movie_features[self.task],
            post_movie_visual_processing_s=self.post_movie_visual_processing_s,
            visual_processing_delay_s=self.visual_processing_delay_s,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window_ds = self.data[idx]
        X = torch.stack([torch.from_numpy(window_ds[i][0]) for i in range(len(window_ds))])
        features = self.labels[idx]

        return X.float(), features


class JEPAMovieDataset(HBNMovieDataset):
    """JEPA-ready EEG dataset extending HBNMovieDataset.

    Pre-extracts EEG windows and movie features into tensors, then returns
    fixed-length contiguous chunks.  Each ``__getitem__`` randomly crops a
    chunk of ``n_windows`` consecutive windows from one recording, so every
    epoch sees different temporal slices.
    """

    DEFAULT_FEATURES = [
        "contrast_rms",
        "luminance_mean",
        "entropy",
        "scene_natural_score",
    ]

    def __init__(
        self,
        split="train",
        n_windows=16,
        window_size_seconds=2,
        task=DEFAULT_TASK,
        feature_names=None,
        eeg_norm_stats=None,
        temporal_stride=1,
        *,
        cfg: DictConfig | dict,
    ):
        super().__init__(split, window_size_seconds, task, cfg=cfg)
        self.n_windows = n_windows
        self.temporal_stride = temporal_stride
        self.feature_names = feature_names or self.DEFAULT_FEATURES
        self._precompute_tensors(eeg_norm_stats=eeg_norm_stats)

    def _precompute_tensors(self, eeg_norm_stats=None):
        """Convert braindecode windows and feature dicts into tensors.

        Args:
            eeg_norm_stats: Optional dict with 'mean' and 'std' tensors for
                EEG normalization. If None, stats are computed from this dataset.
                Pass train set stats when building val/test sets.
        """
        eeg_recordings = []
        feature_recordings = []

        for rec_idx in range(len(self.data)):
            window_ds = self.data[rec_idx]
            labels = self.labels[rec_idx]
            n_win = len(window_ds)

            required_windows = (self.n_windows - 1) * self.temporal_stride + 1
            if n_win < required_windows:
                continue

            eeg = torch.stack(
                [torch.from_numpy(window_ds[i][0]) for i in range(n_win)]
            ).float()  # [n_win, C, W]

            feats = []
            for i in range(n_win):
                d = labels.iloc[i]
                feats.append([float(d.get(f, 0.0)) for f in self.feature_names])
            feats = torch.tensor(feats, dtype=torch.float32)
            feats = torch.nan_to_num(feats, nan=0.0)

            eeg_recordings.append(eeg)
            feature_recordings.append(feats)

        # Per-channel z-normalization of EEG data
        if eeg_norm_stats is not None:
            self._eeg_mean = eeg_norm_stats["mean"]
            self._eeg_std = eeg_norm_stats["std"]
        else:
            all_eeg = torch.cat(eeg_recordings, dim=0)  # [total_windows, C, W]
            self._eeg_mean = all_eeg.mean(dim=(0, 2), keepdim=True)  # [1, C, 1]
            self._eeg_std = all_eeg.std(dim=(0, 2), keepdim=True)    # [1, C, 1]
            self._eeg_std = torch.clamp(self._eeg_std, min=1e-8)
        for i in range(len(eeg_recordings)):
            eeg_recordings[i] = (eeg_recordings[i] - self._eeg_mean) / self._eeg_std

        self.eeg_recordings = eeg_recordings
        self.feature_recordings = feature_recordings
        logger.info(
            "JEPAMovieDataset: %d recordings with >= %d windows (stride=%d, effective span=%.1fs, EEG z-normalized per channel)",
            len(self.eeg_recordings),
            required_windows,
            self.temporal_stride,
            self.n_windows * self.temporal_stride * self.window_size_seconds,
        )

    @property
    def n_chans(self):
        return self.eeg_recordings[0].shape[1]

    @property
    def n_times(self):
        return self.eeg_recordings[0].shape[2]

    def get_chs_info(self):
        """Return MNE-compatible chs_info with channel positions from GSN-HydroCel-129 montage."""
        import numpy as np

        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        info = mne.create_info(montage.ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((len(montage.ch_names), 1)), info, verbose=False)
        raw.set_montage(montage, verbose=False)
        return raw.info["chs"]

    def get_eeg_norm_stats(self):
        """Return EEG normalization stats dict with 'mean' and 'std' tensors."""
        return {"mean": self._eeg_mean, "std": self._eeg_std}

    def compute_feature_stats(self):
        all_feats = torch.cat(self.feature_recordings, dim=0)
        return {"mean": all_feats.mean(0), "std": all_feats.std(0)}

    def compute_feature_median(self):
        all_feats = torch.cat(self.feature_recordings, dim=0)
        return all_feats.median(0).values

    def __len__(self):
        return len(self.eeg_recordings)

    def __getitem__(self, idx):
        eeg = self.eeg_recordings[idx]
        feats = self.feature_recordings[idx]
        n = len(eeg)
        required = (self.n_windows - 1) * self.temporal_stride + 1
        start = torch.randint(0, n - required + 1, (1,)).item()
        indices = list(range(start, start + required, self.temporal_stride))
        return (
            eeg[indices],  # [n_windows, C, W]
            feats[indices],  # [n_windows, n_features]
        )


class HBNMovieProbeDataset(HBNMovieDataset):
    def __init__(
        self,
        split="train",
        window_size_seconds=2,
        task=DEFAULT_TASK,
        *,
        cfg: DictConfig | dict,
        preprocessed: bool = False,
        preprocessed_dir: Path | None = None,
    ):
        super().__init__(
            split, window_size_seconds, task,
            cfg=cfg,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
        )
        self.data = [window[0] for window_ds in self.data for window in window_ds]
        self.labels = [label for labels_series in self.labels for label in labels_series]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        features = self.labels[idx]
        return torch.from_numpy(X).float(), features
