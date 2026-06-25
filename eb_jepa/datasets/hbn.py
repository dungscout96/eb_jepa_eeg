"""HBN (Healthy Brain Network) EEG dataset for self-supervised learning and movie probe tasks."""

import gc
import logging
import math
import os
from pathlib import Path

import numpy as np
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

def _resolve_hbn_cache_dir() -> Path:
    """Return HBN cache dir: env var > auto-detect known paths > default."""
    env = os.environ.get("HBN_CACHE_DIR")
    if env:
        return Path(env)
    known_paths = [
        Path("/mnt/v1/dtyoung/data/eb_jepa_eeg/hbn_cache"),
        Path("/expanse/projects/nemar/openneuro"),
    ]
    for p in known_paths:
        if p.exists():
            return p
    return Path.home() / ".cache" / "eb_jepa" / "datasets" / "eegdash_cache"


DATA_DIR = _resolve_hbn_cache_dir()

if os.environ.get("environment") == "development":
    SPLIT_RELEASES = {
        "train": {
            "R1": "ds005505",  # 136 subjects
        },
        "val": {"R1": "ds005505"},  # 136 subjects
        "test": {"R1": "ds005505"},  # 134 subjects
    }
else:
    SPLIT_RELEASES = {
        "train": {
            "R1": "ds005505",  # 136 subjects
            "R2": "ds005506",  # 152 subjects
            "R3": "ds005507",  # 184 subjects
            "R4": "ds005508",  # 324 subjects
        },
        "val": {"R5": "ds005509"},  # 136 subjects
        "test": {"R6": "ds005510"},  # 134 subjects
    }

DEFAULT_TASK = "ThePresent"

# Default tolerances & thresholds (overridable via dataset constructor)
# TODO: Verify
VISUAL_PROCESSING_DELAY_S = 0.0
DEFAULT_ANNOTATION_DURATION_TOLERANCE_S = 0.5
DEFAULT_MAX_RECORDING_OVERSHOOT_S = 60.0

MOVIE_METADATA = {
    "ThePresent": {
        "duration": 203.3,  # 3 minutes 23 seconds
        "fps": 24,
        "frame_count": 4878,
        "feature_parquet": str(
            PROJECT_ROOT / "movie_annotation" / "output" / "The_Present" / "features_enriched.parquet"
        ),
        "vjepa2_embeddings": str(
            PROJECT_ROOT / "movie_annotation" / "output" / "The_Present" / "vjepa2_embeddings.npz"
        ),
        "shot_boundaries": str(
            PROJECT_ROOT / "movie_annotation" / "output" / "The_Present"
            / "shot_detection" / "transnetv2_boundaries.csv"
        ),
        "scene_map": str(
            PROJECT_ROOT / "experiments" / "clip_pretraining" / "embedding_feature_correlation" / "vjepa2_scenes_map.csv"
        ),
        "vjepa2_recipe": str(
            PROJECT_ROOT / "movie_annotation" / "output" / "The_Present" / "vjepa2_recipe.npz"
        ),
    },
    "DespicableMe": {
        "duration": 170.6,
        "fps": 25,
        "frame_count": 4266,
        "feature_parquet": str(
            PROJECT_ROOT / "movie_annotation" / "output" / "despicable_me" / "features_enriched.parquet"
        ),
        "vjepa2_embeddings": str(
            PROJECT_ROOT / "movie_annotation" / "output" / "despicable_me" / "vjepa2_embeddings.npz"
        ),
        # Shot detection not yet run for DespicableMe; loader returns None
        # (shot_id targets will be filled with -1 for these recordings).
        "shot_boundaries": None,
        "scene_map": None,
        "vjepa2_recipe": None,
    },
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
        logger.debug("Recording missing 'video_start' or 'video_stop' annotations: %s", raw)
        return True

    duration_seconds = get_movie_recording_duration(
        raw, movie, max_recording_overshoot_s=max_recording_overshoot_s
    )
    min_duration = MOVIE_METADATA[movie]["duration"] - annotation_duration_tolerance_s
    if duration_seconds < min_duration:
        logger.debug(
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


def _load_participants_metadata(
    releases: dict[str, str],
    cache_dir: Path | None = None,
) -> dict[str, dict]:
    """Fetch and cache participants.tsv from OpenNeuro, return subject → metadata dict.

    Parameters
    ----------
    releases : dict
        Mapping of release key → OpenNeuro dataset ID (e.g. {"R1": "ds005505"}).
    cache_dir : Path or None
        Directory to cache downloaded TSV files. Falls back to
        ``~/.cache/eb_jepa_eeg/hbn_participants/``.

    Returns
    -------
    dict mapping subject ID (without ``sub-`` prefix) → {"age": float, "sex": str, ...}
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "eb_jepa_eeg" / "hbn_participants"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_dataset_ids = set(releases.values())
    subject_meta = {}

    for dataset_id in sorted(all_dataset_ids):
        tsv_path = cache_dir / f"{dataset_id}_participants.tsv"

        if not tsv_path.exists():
            url = f"https://openneuro.org/crn/datasets/{dataset_id}/files/participants.tsv"
            logger.info("Downloading participants.tsv for %s ...", dataset_id)
            try:
                import urllib.request
                urllib.request.urlretrieve(url, tsv_path)
            except Exception as exc:
                logger.warning("Failed to download %s: %s", url, exc)
                continue

        try:
            df = pd.read_csv(tsv_path, sep="\t")
        except Exception as exc:
            logger.warning("Failed to read %s: %s", tsv_path, exc)
            continue

        for _, row in df.iterrows():
            pid = str(row.get("participant_id", ""))
            subj_id = pid.replace("sub-", "")
            if not subj_id:
                continue
            meta = {}
            if pd.notna(row.get("age")):
                try:
                    meta["age"] = float(row["age"])
                except (ValueError, TypeError):
                    pass
            if pd.notna(row.get("sex")):
                meta["sex"] = str(row["sex"])
            if meta:
                subject_meta[subj_id] = meta

    logger.info(
        "Loaded participant metadata for %d subjects from %d dataset(s)",
        len(subject_meta), len(all_dataset_ids),
    )
    return subject_meta


def load_or_download(release, task=DEFAULT_TASK):
    """Load an EEGDashDataset from cache, downloading if necessary.

    Filters to the specified *task* so only matching recordings are returned.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_id = _release_to_dataset_id(release)
    dataset = EEGDashDataset(
        cache_dir=DATA_DIR,
        dataset=dataset_id,
        task=task,
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
    preload: bool = False,
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
    preload : bool
        If True, load all data into memory.  If False (default), use
        memory-mapped access for lower memory usage.

    Returns
    -------
    BaseConcatDataset
        Dataset loaded via braindecode serialization.
    """
    if preprocessed_dir is None:
        preprocessed_dir = PREPROCESSED_DIR
    data_path = preprocessed_dir / release / task / "preprocessed"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}. "
            f"Run `scripts/preprocess_hbn.py release={release} task={task}` first."
        )
    return load_concat_dataset(str(data_path), preload=preload)


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
    preload: bool = False,
) -> BaseConcatDataset:
    """Load dataset for a (release, task), auto-detecting preprocessed data.

    If *preprocessed* is True, forces loading from preprocessed directory
    (raises if missing).  Otherwise, automatically uses preprocessed data
    when available and falls back to downloading raw data.
    """
    if preprocessed:
        return load_preprocessed(release, task, preprocessed_dir, preload=preload)
    if _has_preprocessed(release, task, preprocessed_dir):
        logger.info("Found preprocessed data for %s/%s, loading...", release, task)
        return load_preprocessed(release, task, preprocessed_dir, preload=preload)
    return load_or_download(release, task=task)


# ---------------------------------------------------------------------------
# Subject metadata helpers
# ---------------------------------------------------------------------------


def _extract_recording_metadata(window_ds_metadata: "pd.DataFrame") -> dict:
    """Pull subject-level attributes from a braindecode window metadata DataFrame.

    ``window_ds.get_metadata()`` returns one row per window, but subject
    attributes (age, sex, participant_id, …) are constant across all windows
    in the same recording, so we only inspect the first row.

    Returns a dict of whatever subject-level columns are present; missing or
    null values are omitted.
    """
    subject_cols = ("subject", "age", "sex", "gender", "participant_id", "site", "diagnosis")
    row = window_ds_metadata.iloc[0]
    meta = {}
    for col in subject_cols:
        if col in window_ds_metadata.columns:
            val = row[col]
            if pd.notna(val):
                meta[col] = val
    return meta


def _compute_probe_labels(
    metadata_list: list[dict],
) -> tuple[list[float], str]:
    """Derive a binary probe label from per-recording subject metadata.

    Priority order:
      1. **age** — binarised as ``age > median(age)`` across the split.
      2. **sex / gender** — Male/M → 1.0, Female/F → 0.0.
      3. Fallback → all ``float('nan')`` (hook will use luminance instead).

    A label column is accepted if at least 50 % of recordings have a valid value.

    Returns
    -------
    labels : list[float]
        Per-recording binary label (0.0/1.0) or ``float('nan')`` when unknown.
    label_name : str
        Human-readable description of the label used.
    """
    n = len(metadata_list)
    if n == 0:
        return [], "none"

    # ---- Try age ----
    ages: list[float] = []
    for m in metadata_list:
        try:
            ages.append(float(m["age"]))
        except (KeyError, TypeError, ValueError):
            ages.append(float("nan"))
    valid_ages = [a for a in ages if not math.isnan(a)]
    if len(valid_ages) >= n * 0.5:
        median_age = float(np.median(valid_ages))
        labels = [
            float(a > median_age) if not math.isnan(a) else float("nan")
            for a in ages
        ]
        return labels, f"age_gt_{median_age:.1f}"

    # ---- Try sex / gender ----
    male_terms = {"m", "male"}
    female_terms = {"f", "female"}
    sex_labels: list[float] = []
    for m in metadata_list:
        raw = str(m.get("sex", m.get("gender", ""))).strip().lower()
        if raw in male_terms:
            sex_labels.append(1.0)
        elif raw in female_terms:
            sex_labels.append(0.0)
        else:
            sex_labels.append(float("nan"))
    valid_sex = [s for s in sex_labels if not math.isnan(s)]
    if len(valid_sex) >= n * 0.5:
        return sex_labels, "sex_male_vs_female"

    # ---- No usable metadata ----
    return [float("nan")] * n, "none"


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
        preprocessed_dir: Path | str | None = None,
    ):
        releases = _resolve_releases(split)
        self.n_windows = n_windows
        self.window_size_seconds = window_size_seconds
        tasks = [task] if isinstance(task, str) else list(task)

        preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir is not None else None

        self._fif_paths = []
        self._crop_inds = []
        for release in releases:
            for t in tasks:
                dataset = _load_dataset(release, t, preprocessed, preprocessed_dir, preload=False)
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
                        fif_path = str(recording_ds.raw.filenames[0])
                        crop_inds = recording_ds.crop_inds.copy()
                        self._fif_paths.append(fif_path)
                        self._crop_inds.append(crop_inds)
                    recording_ds.raw.close()
                # Close Raw handles from original dataset too
                for ds in dataset.datasets:
                    try:
                        ds.raw.close()
                    except Exception:
                        pass
                del windowed_dataset, dataset
        gc.collect()

    def __len__(self):
        return len(self._fif_paths)

    def __getitem__(self, idx):
        crop_inds = self._crop_inds[idx]
        start = torch.randint(0, len(crop_inds) - self.n_windows + 1, (1,)).item()
        X = _read_raw_windows(
            self._fif_paths[idx],
            crop_inds[start:start + self.n_windows],
        )
        return torch.from_numpy(X)


# ---------------------------------------------------------------------------
# Movie-probe dataset
# ---------------------------------------------------------------------------

def _preload_movie_features(task: str) -> dict:
    """Load movie feature parquets for the given task."""
    parquet_path = MOVIE_METADATA[task]["feature_parquet"]
    return {task: pd.read_parquet(parquet_path)}


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
    visual_processing_delay_s: float = VISUAL_PROCESSING_DELAY_S,
) -> dict | None:
    """Map an EEG window onset (in samples) to the corresponding movie frame features.

    The EEG at *window_onset* reflects visual input from
    ``visual_processing_delay_s`` seconds earlier, so the frame index is
    computed from ``(window_onset - delay) / sfreq * fps``.

    Returns ``None`` when the computed frame index falls outside the valid
    range; callers are expected to discard such windows.
    """
    delay_samples = int(visual_processing_delay_s * sfreq)
    movie_timestamp = (window_onset - delay_samples) / sfreq
    frame_index = int(movie_timestamp * MOVIE_METADATA[movie]["fps"])

    if frame_index < 0 or frame_index >= len(movie_features):
        return None

    return movie_features.iloc[frame_index].to_dict()


# ---------------------------------------------------------------------------
# Window-anchored target extraction (V-JEPA 2 embeddings + shot IDs)
# ---------------------------------------------------------------------------

_MOVIE_EMBEDDINGS_CACHE: dict[str, dict] = {}
_SHOT_BOUNDARIES_CACHE: dict[str, np.ndarray] = {}
_VJEPA2_RECIPE_CACHE: dict[str, dict] = {}


def _load_movie_embeddings(movie: str) -> dict | None:
    """Load and cache pretrained frame embeddings for *movie* (e.g. V-JEPA 2).

    Returns ``{"embeddings": [N, D], "timestamps": [N], "dim": D, "freq": float}``
    or ``None`` if the movie has no embeddings configured.
    """
    if movie in _MOVIE_EMBEDDINGS_CACHE:
        return _MOVIE_EMBEDDINGS_CACHE[movie]
    path = MOVIE_METADATA.get(movie, {}).get("vjepa2_embeddings")
    if path is None or not Path(path).exists():
        _MOVIE_EMBEDDINGS_CACHE[movie] = None
        return None
    data = np.load(path)
    bundle = {
        "embeddings": data["embeddings"].astype(np.float32),
        "timestamps": data["timestamps"].astype(np.float64),
        "freq": float(data["frequency"]) if "frequency" in data.files else None,
        "dim": int(data["embeddings"].shape[1]),
    }
    _MOVIE_EMBEDDINGS_CACHE[movie] = bundle
    logger.info("Loaded %s embeddings from %s: %s", movie, path, bundle["embeddings"].shape)
    return bundle


def _load_shot_boundaries(movie: str) -> np.ndarray | None:
    """Load and cache shot-boundary frames (first frame of each new shot) for *movie*."""
    if movie in _SHOT_BOUNDARIES_CACHE:
        return _SHOT_BOUNDARIES_CACHE[movie]
    path = MOVIE_METADATA.get(movie, {}).get("shot_boundaries")
    if path is None or not Path(path).exists():
        _SHOT_BOUNDARIES_CACHE[movie] = None
        return None
    df = pd.read_csv(path)
    boundaries = np.sort(df["boundary_frame"].to_numpy().astype(np.int64))
    _SHOT_BOUNDARIES_CACHE[movie] = boundaries
    logger.info("Loaded %s shot boundaries from %s: %d shots", movie, path, len(boundaries) + 1)
    return boundaries


def _load_vjepa2_recipe(movie: str) -> dict | None:
    """Load and cache V-JEPA-2 recipe artifacts (global mean, shot means, scene IDs).

    Built by ``experiments/clip_pretraining/embedding_feature_correlation/precompute_vjepa2_recipe.py``.
    Returns ``{"global_mean": [D], "shot_means": [n_shots, D], "scene_id_per_shot": [n_shots]}``
    or ``None`` if not configured for *movie*.
    """
    if movie in _VJEPA2_RECIPE_CACHE:
        return _VJEPA2_RECIPE_CACHE[movie]
    path = MOVIE_METADATA.get(movie, {}).get("vjepa2_recipe")
    if path is None or not Path(path).exists():
        _VJEPA2_RECIPE_CACHE[movie] = None
        return None
    data = np.load(path)
    bundle = {
        "global_mean": data["global_mean"].astype(np.float32),
        "shot_means": data["shot_means"].astype(np.float32),
        "scene_id_per_shot": data["scene_id_per_shot"].astype(np.int64),
    }
    _VJEPA2_RECIPE_CACHE[movie] = bundle
    logger.info(
        "Loaded %s V-JEPA-2 recipe from %s: %d shots, %d scenes, dim=%d",
        movie, path,
        bundle["shot_means"].shape[0],
        len(set(bundle["scene_id_per_shot"].tolist())),
        bundle["global_mean"].shape[0],
    )
    return bundle


def get_window_frame_embedding(
    frame_idx: int,
    fps: float,
    window_size_seconds: float,
    embeddings: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray | None:
    """Mean-pool pretrained frame embeddings over the EEG window's movie span.

    The window covers movie time ``[frame_idx / fps, frame_idx / fps + window_size_seconds]``.
    Returns ``None`` if no embedding timestamps fall in that range.
    """
    t_start = frame_idx / fps
    t_end = t_start + window_size_seconds
    lo = int(np.searchsorted(timestamps, t_start, side="left"))
    hi = int(np.searchsorted(timestamps, t_end, side="left"))
    if hi <= lo:
        return None
    return embeddings[lo:hi].mean(axis=0).astype(np.float32)


def get_window_shot_id(
    frame_idx: int,
    n_window_frames: int,
    boundary_frames: np.ndarray,
) -> tuple[int, bool]:
    """Dominant-shot rule: return the shot occupying the longest fraction of the window.

    Shots are 0-indexed. Shot ``k`` covers frames ``[boundary_frames[k-1], boundary_frames[k])``,
    with shot 0 covering ``[0, boundary_frames[0])`` and the last shot extending to +inf.

    Returns ``(dominant_shot_id, crosses_boundary)``.
    """
    start = int(frame_idx)
    end = start + int(n_window_frames)
    first_shot = int(np.searchsorted(boundary_frames, start, side="right"))
    last_shot = int(np.searchsorted(boundary_frames, end - 1, side="right"))
    if last_shot == first_shot:
        return first_shot, False
    best_shot = first_shot
    best_overlap = -1
    for shot in range(first_shot, last_shot + 1):
        shot_start = int(boundary_frames[shot - 1]) if shot > 0 else 0
        shot_end = int(boundary_frames[shot]) if shot < len(boundary_frames) else end
        overlap = min(shot_end, end) - max(shot_start, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_shot = shot
    return best_shot, True


def _read_raw_windows(fif_path, crop_inds):
    """Read multiple windows from a FIF file by absolute sample indices.

    Opens the file with memory-mapping, reads the requested slices, and
    releases the Raw object so no file handles or mmap state persist.

    Parameters
    ----------
    fif_path : str
        Path to the ``-raw.fif`` file.
    crop_inds : array-like of shape (n_windows, 3)
        Each row is ``(i_window_in_trial, i_start, i_stop)`` with absolute
        sample indices (as stored by braindecode ``EEGWindowsDataset``).

    Returns
    -------
    np.ndarray of shape (n_windows, n_channels, n_times)
    """
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
    windows = []
    for _, i_start, i_stop in crop_inds:
        data = raw._getitem(
            (slice(None), slice(int(i_start), int(i_stop))),
            return_times=False,
        )
        windows.append(data)
    del raw
    return np.stack(windows).astype("float32")


class HBNMovieDataset(Dataset):
    """Supervised EEG dataset: each window is paired with movie features at its timestamp.

    Only lightweight metadata (FIF file paths, sample indices, labels) is kept
    in memory.  EEG data is read on-demand from memory-mapped FIF files in
    ``__getitem__``.
    """

    def __init__(
        self,
        split="train",
        window_size_seconds=2,
        task=DEFAULT_TASK,
        *,
        cfg: DictConfig | dict,
        visual_processing_delay_s: float = 0.0,
        preprocessed: bool = False,
        preprocessed_dir: Path | str | None = None,
    ):
        self.window_size_seconds = window_size_seconds
        tasks = [task] if isinstance(task, str) else list(task)
        self.tasks = tasks
        self.task = tasks[0]  # primary task (backwards compat)
        self.movie_features = {}
        for t in tasks:
            self.movie_features.update(_preload_movie_features(t))
        self.visual_processing_delay_s = visual_processing_delay_s
        releases = _resolve_releases(split)

        self.sfreq = None
        # Lightweight storage: file paths + absolute sample indices per recording
        self._fif_paths = []             # str per recording
        self._crop_inds = []             # np.ndarray (n_windows, 3) per recording
        self.labels = []                 # pd.Series per recording
        self._recording_metadata = []    # dict per recording (subject-level attributes)
        self._recording_tasks = []       # task name per recording — required by
                                         # JEPAMovieDataset to look up the right
                                         # V-JEPA-2 vectors / fps / shot bounds.
        total_recordings = 0
        total_rejected = 0

        preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir is not None else None
        for t in tasks:
            # Load data for this task across all releases.
            # preload=False so Raw objects use memory-mapping.
            datasets = []
            for release in releases:
                ds = _load_dataset(release, t, preprocessed, preprocessed_dir, preload=False)
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
                    logger.debug("Skipping unloadable recording %s: %s", recording_ds, exc)
                    rejected += 1
                    continue

                if reject_recording(
                    raw, movie=t,
                    annotation_duration_tolerance_s=cfg.get("annotation_duration_tolerance_s") if isinstance(cfg, dict) else cfg.annotation_duration_tolerance_s,
                    max_recording_overshoot_s=cfg.get("max_recording_overshoot_s") if isinstance(cfg, dict) else cfg.max_recording_overshoot_s,
                ):
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
                        logger.debug(
                            "Setting 'video_start' duration to %.2fs for %s",
                            movie_recording_duration, recording_ds,
                        )
                        raw.annotations.duration[idx] = movie_recording_duration

                selected_recordings.append(recording_ds)

            total_recordings += len(data.datasets)
            total_rejected += rejected
            logger.info("Task %s: rejected %d/%d recordings", t, rejected, len(data.datasets))

            window_size_samples = int(window_size_seconds * sfreq)

            # Temporarily set self.task for _get_movie_features_for_window
            self.task = t
            for rec in selected_recordings:
                fif_path = str(rec.raw.filenames[0])

                # Get the video_start event sample so we can convert absolute
                # braindecode indices to onsets relative to the movie start.
                events, event_id = mne.events_from_annotations(rec.raw, verbose=False)
                video_start_sample = events[events[:, 2] == event_id["video_start"]][0, 0]

                # Window grid starts exactly at video_start. Out-of-bound
                # windows (label is None) are filtered below, so the EEG
                # window at time t aligns with the movie frame at time
                # t - visual_processing_delay_s.
                window_ds = create_windows_from_events(
                    BaseConcatDataset([rec]),
                    mapping={"video_start": 0},
                    trial_start_offset_samples=0,
                    trial_stop_offset_samples=0,
                    window_size_samples=window_size_samples,
                    window_stride_samples=window_size_samples,
                    drop_last_window=True,
                    preload=False,
                )

                # Extract lightweight metadata and discard braindecode objects
                wds = window_ds.datasets[0]
                crop_inds = wds.crop_inds.copy()  # np.ndarray (n_windows, 3)

                # braindecode's i_start_in_trial uses absolute sample indices;
                # get_window_movie_metadata expects onsets relative to video_start.
                window_meta_df = window_ds.get_metadata()
                abs_onsets = window_meta_df["i_start_in_trial"]
                window_onsets = abs_onsets - video_start_sample
                movie_features_for_windows = window_onsets.apply(self._get_movie_features_for_window)

                # Drop windows whose computed frame index fell out of bounds
                # (returned None from get_window_movie_metadata).
                keep_mask = movie_features_for_windows.notna().to_numpy()
                if not keep_mask.any():
                    logger.debug("All windows out-of-bound for %s; skipping", fif_path)
                    continue
                crop_inds = crop_inds[keep_mask]
                movie_features_for_windows = movie_features_for_windows[keep_mask].reset_index(drop=True)

                # Extract subject-level attributes (age, sex, …) from braindecode metadata.
                # These are the same for every window in a recording, so we only look
                # at the first row.  Missing / null columns are silently skipped.
                try:
                    subj_meta = _extract_recording_metadata(window_meta_df)
                except Exception:
                    subj_meta = {}

                self._fif_paths.append(fif_path)
                self._crop_inds.append(crop_inds)
                self.labels.append(movie_features_for_windows)
                self._recording_metadata.append(subj_meta)
                self._recording_tasks.append(t)

                # Explicitly close Raw file handles before moving on
                rec.raw.close()
                del window_ds, wds

            # Close any remaining Raw handles from rejected recordings
            for ds in data.datasets:
                try:
                    ds.raw.close()
                except Exception:
                    pass
            del data, datasets, selected_recordings

        self.task = tasks[0]  # reset to primary task

        # Enrich recording metadata with age/sex from OpenNeuro participants.tsv.
        # The preprocessed description.json files only carry subject ID + task;
        # age and sex must be fetched from the source BIDS dataset.
        n_missing = sum(1 for m in self._recording_metadata if "age" not in m)
        if n_missing > 0:
            participants_cache = preprocessed_dir if preprocessed_dir else None
            all_releases = {}
            for r in releases:
                all_releases[r] = _release_to_dataset_id(r)
            participants = _load_participants_metadata(all_releases, cache_dir=participants_cache)
            enriched = 0
            for meta in self._recording_metadata:
                subj_id = meta.get("subject", meta.get("participant_id", ""))
                if subj_id and subj_id in participants:
                    for k, v in participants[subj_id].items():
                        if k not in meta:
                            meta[k] = v
                            enriched += 1
            logger.info(
                "Enriched %d metadata fields from participants.tsv (%d/%d had missing age)",
                enriched, n_missing, len(self._recording_metadata),
            )

        # Force cleanup of any remaining MNE/mmap objects before returning
        gc.collect()
        logger.info("Total: rejected %d/%d recordings across %d task(s)",
                     total_rejected, total_recordings, len(tasks))

    def _get_movie_features_for_window(self, window_onset) -> dict | None:
        return get_window_movie_metadata(
            window_onset=window_onset,
            sfreq=self.sfreq,
            movie=self.task,
            movie_features=self.movie_features[self.task],
            visual_processing_delay_s=self.visual_processing_delay_s,
        )

    def __len__(self):
        return len(self._fif_paths)

    def __getitem__(self, idx):
        X = _read_raw_windows(self._fif_paths[idx], self._crop_inds[idx])
        features = self.labels[idx]
        return torch.from_numpy(X), features


class JEPAMovieDataset(HBNMovieDataset):
    """JEPA-ready EEG dataset extending HBNMovieDataset.

    Lazily loads EEG windows from memory-mapped FIF files on demand, keeping
    only file paths, sample indices, and movie feature tensors in memory.
    Each ``__getitem__`` randomly crops ``n_windows`` consecutive windows from
    one recording, so every epoch sees different temporal slices.
    """

    DEFAULT_FEATURES = [
        "contrast_rms",
        "luminance_mean",
        "position_in_movie",
        "narrative_event_score",
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
        visual_processing_delay_s: float = 0.0,
        preprocessed: bool = False,
        preprocessed_dir: Path | str | None = None,
        recipe_mode: bool = False,
        recipe_target_kind: str = "shot_mean",
        recipe_mean_center: bool = True,
    ):
        super().__init__(
            split, window_size_seconds, task, cfg=cfg,
            visual_processing_delay_s=visual_processing_delay_s,
            preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
        )
        self.n_windows = n_windows
        self.temporal_stride = temporal_stride
        # None → use class defaults; [] → skip feature extraction (e.g. JEPA
        # pretraining doesn't need labels).
        self.feature_names = (
            list(self.DEFAULT_FEATURES) if feature_names is None else list(feature_names)
        )
        self._norm_mode = cfg.get("norm_mode", "global") if not isinstance(cfg, dict) else cfg.get("norm_mode", "global")
        self._add_envelope = cfg.get("add_envelope", False) if not isinstance(cfg, dict) else cfg.get("add_envelope", False)
        # CorrCA spatial filters: project 129 channels → k stimulus-driven components
        corrca_path = cfg.get("corrca_filters", None) if not isinstance(cfg, dict) else cfg.get("corrca_filters", None)
        self._corrca_W = None
        if corrca_path is not None:
            corrca = np.load(corrca_path)
            self._corrca_W = torch.from_numpy(corrca["W"]).float()  # [n_chans, k]
            logger.info("Loaded CorrCA filters from %s: %d → %d components (ISC: %s)",
                        corrca_path, corrca["W"].shape[0], corrca["W"].shape[1],
                        ", ".join(f"{v:.3f}" for v in corrca["isc_values"]))

        required_windows = (n_windows - 1) * temporal_stride + 1

        # Window-anchored prediction targets: pretrained frame embedding (V-JEPA 2)
        # and shot ID. Loaded once per task — multi-movie runs need the right
        # vision vectors / fps / shot bounds per recording.
        self._frame_embeddings = {
            t: _load_movie_embeddings(t) for t in self.tasks
        }
        self._shot_boundaries = {
            t: _load_shot_boundaries(t) for t in self.tasks
        }
        missing_emb = [t for t, b in self._frame_embeddings.items() if b is None]
        if missing_emb:
            raise FileNotFoundError(
                f"Frame embeddings missing for task(s) {missing_emb}. "
                "All tasks must have pretrained vision embeddings configured."
            )
        empty_emb = [
            t for t, b in self._frame_embeddings.items()
            if b["embeddings"].size == 0 or b["dim"] == 0
        ]
        if empty_emb:
            raise ValueError(
                f"Frame embeddings are empty for task(s) {empty_emb}."
            )
        dims = {t: b["dim"] for t, b in self._frame_embeddings.items()}
        unique_dims = set(dims.values())
        if len(unique_dims) > 1:
            raise ValueError(
                f"Frame embedding dims differ across tasks: {dims}. "
                "All tasks must share the same embedding dimensionality."
            )
        self._frame_embedding_dim = unique_dims.pop()
        # n_shots: max across tasks (per-task shot_id labels still collide at 0;
        # ShotIDHead is not used by the CLIP path so deferring a fix).
        non_null_shots = [b for b in self._shot_boundaries.values() if b is not None]
        self._n_shots = (
            max(len(b) for b in non_null_shots) + 1 if non_null_shots else 0
        )

        # Recipe artifacts (global mean, shot means, scene IDs). Loaded per task.
        # Required when recipe_mode=True; missing → fail loud (the trainer can
        # only verify the recipe's sanity numbers if these stats exist).
        self.recipe_mode = recipe_mode
        self.recipe_target_kind = recipe_target_kind
        self.recipe_mean_center = recipe_mean_center
        self._vjepa2_recipes = {t: _load_vjepa2_recipe(t) for t in self.tasks}
        if recipe_mode:
            missing = [t for t, b in self._vjepa2_recipes.items() if b is None]
            if missing:
                raise FileNotFoundError(
                    "recipe_mode=True but vjepa2_recipe missing for task(s) "
                    f"{missing}. Run: "
                    f"python -m experiments.clip_pretraining.embedding_feature_correlation.precompute_vjepa2_recipe "
                    f"--task {missing[0]}"
                )
        # Scene-id namespacing across tasks: scene_id_global = task_idx * SCENE_NAMESPACE + scene_id
        self._SCENE_NAMESPACE = 100_000
        self._task_to_idx = {t: i for i, t in enumerate(self.tasks)}

        # Filter recordings with enough windows and pre-extract feature tensors.
        # Parent already stored _fif_paths, _crop_inds, labels, _recording_tasks.
        filtered_paths = []
        filtered_crops = []
        filtered_metadata = []
        filtered_tasks = []
        self.feature_recordings = []
        self.embedding_recordings = []   # [n_win, D] per recording (float32)
        self.shot_id_recordings = []     # [n_win] per recording (int64)
        # Recipe-mode-only side tables (empty lists when recipe_mode=False).
        self.scene_id_recordings = []    # [n_win] per recording (int64, global namespace)
        self.t_start_recordings = []     # [n_win] per recording (float32, seconds in movie)
        self._n_chans = None
        self._n_times = None

        for rec_idx in range(len(self._fif_paths)):
            crop_inds = self._crop_inds[rec_idx]
            labels = self.labels[rec_idx]
            rec_task = self._recording_tasks[rec_idx]
            n_win = len(crop_inds)

            if n_win < required_windows:
                continue

            # Per-task lookups — DespicableMe is 25 fps and uses its own V-JEPA-2.
            movie_fps = MOVIE_METADATA[rec_task]["fps"]
            window_n_frames = int(window_size_seconds * movie_fps)
            frame_emb_bundle = self._frame_embeddings[rec_task]
            shot_bnds = self._shot_boundaries[rec_task]
            recipe = self._vjepa2_recipes[rec_task]

            # Extract feature tensor (small: n_win x n_features floats).
            # Windows whose movie span falls outside the frame-embedding
            # timestamps (e.g. EEG runs past the end of the movie embeddings)
            # are dropped — never zero-filled, since they would otherwise become
            # silent zero CLIP/recipe targets.
            feats = []
            embeds = []
            shot_ids = []
            t_starts = []
            keep_idx = []
            for i in range(n_win):
                d = labels.iloc[i]
                frame_idx = int(d["frame_idx"])
                e = get_window_frame_embedding(
                    frame_idx, movie_fps, window_size_seconds,
                    frame_emb_bundle["embeddings"],
                    frame_emb_bundle["timestamps"],
                )
                if e is None:
                    continue
                feats.append([float(d.get(f, 0.0)) for f in self.feature_names])
                t_starts.append(frame_idx / movie_fps)
                embeds.append(e)
                if shot_bnds is not None:
                    sid, _ = get_window_shot_id(frame_idx, window_n_frames, shot_bnds)
                    shot_ids.append(sid)
                keep_idx.append(i)

            n_dropped = n_win - len(keep_idx)
            if n_dropped:
                logger.warning(
                    "Recording %s (task=%s): dropped %d/%d windows with no frame embedding coverage",
                    self._fif_paths[rec_idx], rec_task, n_dropped, n_win,
                )
                crop_inds = crop_inds[np.asarray(keep_idx, dtype=np.int64)]
                n_win = len(crop_inds)
                if n_win < required_windows:
                    continue

            feats = torch.tensor(feats, dtype=torch.float32)
            feats = torch.nan_to_num(feats, nan=0.0)
            embeds_t = torch.from_numpy(np.stack(embeds))
            if shot_ids:
                shot_ids_t = torch.tensor(shot_ids, dtype=torch.long)
            else:
                shot_ids_t = torch.full((n_win,), -1, dtype=torch.long)
            t_starts_t = torch.tensor(t_starts, dtype=torch.float32)

            # Recipe target replacement and centering (init-time, per-clip cost is one-off).
            scene_ids_t = torch.full((n_win,), -1, dtype=torch.long)
            if recipe is not None:
                task_idx = self._task_to_idx[rec_task]
                shot_means_t = torch.from_numpy(recipe["shot_means"])
                global_mean_t = torch.from_numpy(recipe["global_mean"])
                scene_per_shot_t = torch.from_numpy(recipe["scene_id_per_shot"])
                valid_shot = shot_ids_t >= 0
                if self.recipe_mode and not valid_shot.all():
                    raise ValueError(
                        f"recipe_mode=True but {(~valid_shot).sum().item()}/{n_win} "
                        f"windows in {self._fif_paths[rec_idx]} (task={rec_task}) "
                        "have no shot id. Shot boundaries must cover every window — "
                        "fix the boundaries or disable recipe_mode."
                    )
                if self.recipe_mode and self.recipe_target_kind == "shot_mean":
                    # Replace per-window mean V-JEPA-2 with the shot-mean target.
                    embeds_t = shot_means_t[shot_ids_t].clone()
                if self.recipe_mode and self.recipe_mean_center:
                    embeds_t = embeds_t - global_mean_t
                # Always populate scene IDs when the recipe is available so
                # downstream code can opt in without recomputing.
                scene_ids_t = torch.full((n_win,), -1, dtype=torch.long)
                if valid_shot.any():
                    local_scene = scene_per_shot_t[shot_ids_t.clamp(min=0)]
                    scene_ids_t = (
                        local_scene + task_idx * self._SCENE_NAMESPACE
                    )
                    scene_ids_t[~valid_shot] = -1

            filtered_paths.append(self._fif_paths[rec_idx])
            filtered_crops.append(crop_inds)
            filtered_metadata.append(self._recording_metadata[rec_idx])
            filtered_tasks.append(rec_task)
            self.feature_recordings.append(feats)
            self.embedding_recordings.append(embeds_t)
            self.shot_id_recordings.append(shot_ids_t)
            self.scene_id_recordings.append(scene_ids_t)
            self.t_start_recordings.append(t_starts_t)

            # Capture shape from first valid recording (one cheap read)
            if self._n_chans is None:
                sample = _read_raw_windows(
                    self._fif_paths[rec_idx], crop_inds[:1]
                )
                self._n_chans = sample.shape[1]
                self._n_times = sample.shape[2]

        # Replace parent's storage with filtered set
        self._fif_paths = filtered_paths
        self._crop_inds = filtered_crops
        self._recording_metadata = filtered_metadata
        self._recording_tasks = filtered_tasks
        del self.labels  # labels are now in feature_recordings

        # Derive binary probe labels from subject metadata (age / sex).
        # Falls back to all-NaN if metadata is not available; the sanity
        # check hook will then use luminance as a fallback label instead.
        self._probe_labels, self.probe_label_name = _compute_probe_labels(
            self._recording_metadata
        )
        n_valid = sum(1 for v in self._probe_labels if not math.isnan(v))
        logger.info(
            "Probe label: '%s', valid for %d/%d recordings",
            self.probe_label_name,
            n_valid,
            len(self._probe_labels),
        )

        # Compute or set per-channel normalization stats. Skip entirely under
        # per_recording mode -- __getitem__ computes mean/std per clip.
        if self._norm_mode == "per_recording":
            self._eeg_mean = None
            self._eeg_std = None
        elif eeg_norm_stats is not None:
            self._eeg_mean = eeg_norm_stats["mean"]
            self._eeg_std = eeg_norm_stats["std"]
        else:
            self._compute_norm_stats(cache_dir=preprocessed_dir)

        logger.info(
            "JEPAMovieDataset: %d recordings with >= %d windows "
            "(stride=%d, effective span=%.1fs, lazy loading, EEG z-normalized per channel)",
            len(self._fif_paths),
            required_windows,
            self.temporal_stride,
            self.n_windows * self.temporal_stride * self.window_size_seconds,
        )

    def _compute_norm_stats(self, cache_dir=None):
        """Compute per-channel mean/std by streaming through all preprocessed recordings.

        NOTE: The preprocessing pipeline saves a ``normalization_stats.npz`` next to
        each release directory, but those stats are computed on the *intermediate*
        (pre-normalization) FIF files in raw EEG units (~1e-5 V).  The FIF files
        actually read at training time are the *already z-scored* outputs of pass 2,
        which have unit scale.  Loading the cached stats and dividing again by ~1e-5
        would multiply embeddings by ~100,000.

        This method computes from the actual z-scored data and caches the result in
        ``{cache_dir}/jepa_norm_stats_train.npz`` to avoid re-reading 700+ FIF files
        on every run (~20 min on NFS without cache).
        """
        # Try disk cache first (keyed to training split data, not pre-norm FIFs)
        cache_file = None
        if cache_dir is not None:
            cache_file = Path(cache_dir) / "jepa_norm_stats_train.npz"
            if cache_file.exists():
                cached = np.load(cache_file)
                self._eeg_mean = torch.from_numpy(cached["mean"])
                self._eeg_std = torch.from_numpy(cached["std"])
                logger.info("Loaded norm stats from cache: %s", cache_file)
                return

        logger.info("Computing norm stats from %d recordings...", len(self._fif_paths))
        channel_sum = None
        channel_sum_sq = None
        total_timepoints = 0

        for fif_path, crop_inds in zip(self._fif_paths, self._crop_inds):
            windows = _read_raw_windows(fif_path, crop_inds)  # (n_win, C, T)
            x = torch.from_numpy(windows).double()
            if channel_sum is None:
                n_ch = x.shape[1]
                channel_sum = torch.zeros(n_ch, dtype=torch.float64)
                channel_sum_sq = torch.zeros(n_ch, dtype=torch.float64)
            channel_sum += x.sum(dim=(0, 2))
            channel_sum_sq += (x ** 2).sum(dim=(0, 2))
            total_timepoints += x.shape[0] * x.shape[2]
            del windows, x  # free before next recording

        mean = (channel_sum / total_timepoints).float()
        var = (channel_sum_sq / total_timepoints).float() - mean ** 2
        std = torch.sqrt(var.clamp(min=0)).clamp(min=1e-8)
        self._eeg_mean = mean[None, :, None]  # [1, C, 1]
        self._eeg_std = std[None, :, None]    # [1, C, 1]

        if cache_file is not None:
            np.savez(cache_file, mean=self._eeg_mean.numpy(), std=self._eeg_std.numpy())
            logger.info("Saved norm stats cache: %s", cache_file)

    @property
    def n_chans(self):
        if self._corrca_W is not None:
            return self._corrca_W.shape[1]
        if self._add_envelope:
            return self._n_chans * 2
        return self._n_chans

    @property
    def n_times(self):
        return self._n_times

    def get_chs_info(self):
        """Return MNE-compatible chs_info with channel positions from GSN-HydroCel-129 montage."""
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        info = mne.create_info(montage.ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((len(montage.ch_names), 1)), info, verbose=False)
        raw.set_montage(montage, verbose=False)
        chs = raw.info["chs"]
        if self._corrca_W is not None:
            # For each CorrCA component, use the position of its peak-weight channel
            W = self._corrca_W.numpy()
            k = W.shape[1]
            corrca_chs = []
            for i in range(k):
                peak_idx = int(np.argmax(np.abs(W[:, i])))
                corrca_chs.append(chs[peak_idx])
            return corrca_chs
        if self._add_envelope:
            # Duplicate channel positions for envelope channels
            chs = chs + chs
        return chs

    def get_eeg_norm_stats(self):
        """Return EEG normalization stats dict with 'mean' and 'std' tensors."""
        return {"mean": self._eeg_mean, "std": self._eeg_std}

    @property
    def frame_embedding_dim(self) -> int:
        """Dim of the per-window pretrained frame embedding target (0 if disabled)."""
        return self._frame_embedding_dim

    @property
    def n_shots(self) -> int:
        """Number of distinct shot IDs in the movie (0 if shot boundaries missing)."""
        return self._n_shots

    def compute_feature_stats(self):
        all_feats = torch.cat(self.feature_recordings, dim=0)
        return {"mean": all_feats.mean(0), "std": all_feats.std(0)}

    def compute_feature_median(self):
        all_feats = torch.cat(self.feature_recordings, dim=0)
        return all_feats.median(0).values

    def __len__(self):
        return len(self._fif_paths)

    def __getitem__(self, idx):
        crop_inds = self._crop_inds[idx]
        feats = self.feature_recordings[idx]
        embeds = self.embedding_recordings[idx]
        shot_ids = self.shot_id_recordings[idx]
        scene_ids = self.scene_id_recordings[idx] if self.scene_id_recordings else None
        t_starts = self.t_start_recordings[idx] if self.t_start_recordings else None
        n = len(crop_inds)
        required = (self.n_windows - 1) * self.temporal_stride + 1
        start = torch.randint(0, n - required + 1, (1,)).item()
        indices = list(range(start, start + required, self.temporal_stride))

        # Load only the needed windows from disk
        eeg = torch.from_numpy(
            _read_raw_windows(self._fif_paths[idx], crop_inds[indices])
        )

        # Normalization: per-recording removes subject fingerprint, global preserves it
        if self._norm_mode == "per_recording":
            rec_mean = eeg.mean(dim=(0, 2), keepdim=True)  # [1, C, 1]
            rec_std = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            eeg = (eeg - rec_mean) / rec_std
        else:
            eeg = (eeg - self._eeg_mean) / self._eeg_std

        # Optional: append 1-8 Hz envelope channels (where ISC is highest)
        if self._add_envelope:
            eeg = self._append_lowfreq_envelope(eeg)

        # CorrCA spatial projection: 129 channels → k stimulus-driven components
        if self._corrca_W is not None:
            # eeg: [n_windows, C, T] → [n_windows, k, T]
            eeg = torch.einsum("wct,ck->wkt", eeg, self._corrca_W)

        # Binary subject label (age > median, sex, …) — scalar float tensor.
        # NaN means metadata was unavailable for this recording.
        probe_label = torch.tensor(self._probe_labels[idx], dtype=torch.float32)

        if self.recipe_mode:
            return (
                eeg,
                feats[indices],
                embeds[indices],
                shot_ids[indices],
                scene_ids[indices],
                t_starts[indices],
                probe_label,
            )
        return eeg, feats[indices], embeds[indices], shot_ids[indices], probe_label

    @staticmethod
    def _append_lowfreq_envelope(eeg: torch.Tensor) -> torch.Tensor:
        """Append 1-8 Hz analytic amplitude envelope as extra channels.

        The delta/theta band carries the highest inter-subject correlation
        (ISC 0.10-0.28) for stimulus-driven responses during movie watching.

        Args:
            eeg: [n_windows, C, T] z-normalized EEG

        Returns:
            [n_windows, 2C, T] with original + envelope channels
        """
        from scipy.signal import butter, filtfilt, hilbert

        x = eeg.numpy()
        sfreq = x.shape[-1] / 2.0  # T samples / window_size_seconds(2s) = sfreq
        # Bandpass 1-8 Hz (4th order Butterworth)
        b, a = butter(4, [1, 8], btype="band", fs=sfreq)
        # filtfilt needs sufficient length; skip if windows too short
        if x.shape[-1] < 27:  # min padlen for order-4 butter
            return eeg
        filtered = filtfilt(b, a, x, axis=-1).astype("float32")
        envelope = np.abs(hilbert(filtered, axis=-1)).astype("float32")
        # Z-normalize envelope per-window to remove amplitude differences
        env_mean = envelope.mean(axis=(0, 2), keepdims=True)
        env_std = envelope.std(axis=(0, 2), keepdims=True)
        env_std = np.clip(env_std, 1e-8, None)
        envelope = (envelope - env_mean) / env_std
        return torch.from_numpy(np.concatenate([x, envelope], axis=1))


class HBNMovieProbeDataset(HBNMovieDataset):
    def __init__(
        self,
        split="train",
        window_size_seconds=2,
        task=DEFAULT_TASK,
        *,
        cfg: DictConfig | dict,
        visual_processing_delay_s: float = 0.0,
        preprocessed: bool = False,
        preprocessed_dir: Path | None = None,
    ):
        super().__init__(
            split, window_size_seconds, task,
            cfg=cfg,
            visual_processing_delay_s=visual_processing_delay_s,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
        )
        # Build a flat index of (recording_idx, window_idx) for random access
        # without materializing all EEG data in memory.
        self._flat_index = []
        self._flat_labels = []
        for rec_idx in range(len(self._fif_paths)):
            labels = self.labels[rec_idx]
            for win_idx in range(len(self._crop_inds[rec_idx])):
                self._flat_index.append((rec_idx, win_idx))
                self._flat_labels.append(labels.iloc[win_idx])

    def __len__(self):
        return len(self._flat_index)

    def __getitem__(self, idx):
        rec_idx, win_idx = self._flat_index[idx]
        X = _read_raw_windows(
            self._fif_paths[rec_idx],
            self._crop_inds[rec_idx][win_idx:win_idx + 1],
        )
        features = self._flat_labels[idx]
        return torch.from_numpy(X[0]), features
