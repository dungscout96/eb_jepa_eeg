"""HBN (Healthy Brain Network) EEG dataset for self-supervised learning and movie probe tasks."""

import logging
from pathlib import Path

import mne
import pandas as pd
import torch
from torch.utils.data import Dataset

from braindecode.datasets import BaseConcatDataset  # noqa: F401
from braindecode.preprocessing import (
    create_fixed_length_windows,
    create_windows_from_events,
)
from eegdash.dataset import EEGChallengeDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = Path.home() / ".cache" / "eb_jepa" / "datasets" / "eegdash_cache"


SPLIT_RELEASES = {
    "train": {"R1": "ds005505"},
    "val": {"R1": "ds005505"}, # TODO
    "test": {"R1": "ds005505"}, # TODO
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
        return True

    duration_seconds = get_movie_recording_duration(
        raw, movie, max_recording_overshoot_s=max_recording_overshoot_s
    )
    min_duration = MOVIE_METADATA[movie]["duration"] - annotation_duration_tolerance_s
    if duration_seconds < min_duration:
        return True

    return False


def load_or_download(release, task=DEFAULT_TASK):
    """Load an EEGChallengeDataset from cache, downloading if necessary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_id = f"EEG2025r{release[1:]}mini"
    data_dir = DATA_DIR / dataset_id
    needs_download = not data_dir.exists() or not list(data_dir.glob("**/*.bdf"))

    dataset = EEGChallengeDataset(
        cache_dir=DATA_DIR,
        release=release,
        download=needs_download,
        task=task,
        mini=True,
    )
    if needs_download:
        dataset.download_all(n_jobs=-1)
    return dataset


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class HBNDataset(Dataset):
    """Self-supervised EEG dataset: each item is a random crop of contiguous windows (WxCxT)."""

    def __init__(self, split="train", n_windows=16, window_size_seconds=2, task=DEFAULT_TASK):
        releases = _resolve_releases(split)
        self.n_windows = n_windows
        self.window_size_seconds = window_size_seconds

        self.recordings = []
        for release, dataset_name in releases.items():
            dataset = load_or_download(release, dataset_name, task=task)
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

        self.movie_features = _preload_movie_features(task)

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


class HBNMovieProbeDataset(Dataset):
    """Supervised EEG dataset: each window is paired with movie features at its timestamp."""

    def __init__(
        self,
        split="train",
        window_size_seconds=2,
        task=DEFAULT_TASK,
        *,
        annotation_duration_tolerance_s=DEFAULT_ANNOTATION_DURATION_TOLERANCE_S,
        post_movie_visual_processing_s=DEFAULT_POST_MOVIE_VISUAL_PROCESSING_S,
        max_recording_overshoot_s=DEFAULT_MAX_RECORDING_OVERSHOOT_S,
        trial_stop_offset_s=DEFAULT_TRIAL_STOP_OFFSET_S,
        visual_processing_delay_s=VISUAL_PROCESSING_DELAY_S,
    ):
        self.window_size_seconds = window_size_seconds
        self.task = task
        self.post_movie_visual_processing_s = post_movie_visual_processing_s
        self.visual_processing_delay_s = visual_processing_delay_s
        releases = _resolve_releases(split)

        data = BaseConcatDataset([
            load_or_download(release, task=task)
            for release, dataset_name in releases.items()
        ])

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
                        raw, movie=task,
                        max_recording_overshoot_s=max_recording_overshoot_s,
                    )
                    movie_recording_duration = min(
                        movie_recording_duration, MOVIE_METADATA[task]["duration"]
                    )
                    logger.info(
                        "Setting 'video_start' duration to %.2fs for %s",
                        movie_recording_duration, recording_ds,
                    )
                    raw.annotations.duration[idx] = movie_recording_duration

            if reject_recording(
                raw, movie=task,
                annotation_duration_tolerance_s=annotation_duration_tolerance_s,
                max_recording_overshoot_s=max_recording_overshoot_s,
            ):
                rejected += 1
                continue
            selected_recordings.append(recording_ds)

        logger.info("Rejected %d/%d recordings", rejected, len(data.datasets))

        data = BaseConcatDataset(selected_recordings)
        sfreq = data.datasets[0].raw.info["sfreq"]
        window_samples = int(window_size_seconds * sfreq)
        # Offset the trial start by the visual processing delay so the first
        # EEG window corresponds to the neural response to the first movie frame.
        self.trial_start_offset_samples = int(visual_processing_delay_s * sfreq)
        self.data = create_windows_from_events(
            data,
            mapping={"video_start": 0},
            trial_start_offset_samples=self.trial_start_offset_samples,
            trial_stop_offset_samples=-int(trial_stop_offset_s * sfreq),
            window_size_samples=window_samples,
            window_stride_samples=window_samples,
            drop_last_window=True,
        )

        self.sfreq = sfreq
        self.movie_features = _preload_movie_features(task)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y, crop_inds = self.data[idx]
        window_onset = crop_inds[1]  # i_start_in_trial (samples)
        features = get_window_movie_metadata(
            window_onset=window_onset,
            sfreq=self.sfreq,
            movie=self.task,
            movie_features=self.movie_features[self.task],
            post_movie_visual_processing_s=self.post_movie_visual_processing_s,
            visual_processing_delay_s=self.visual_processing_delay_s,
        )
        return torch.from_numpy(X).float(), features
