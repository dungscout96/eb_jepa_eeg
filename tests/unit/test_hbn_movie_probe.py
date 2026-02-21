"""Tests for HBNMovieProbeDataset data input/output contract.

Documents and verifies the assumptions about:
- Movie feature CSV structure (columns, types, alignment with frames)
- Window-to-frame mapping (EEG window onset -> movie frame index)
- End-of-movie handling (windows near/beyond movie end)
- Recording rejection criteria (missing annotations, too-short recordings)
- __getitem__ output format (tensor shape, dtype, feature dict keys)
- DataLoader collation behavior
"""

import sys
from unittest.mock import patch, MagicMock
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Stub heavy transitive dependencies so the module can be imported without
# eegdash / braindecode (mirrors tests/unit/test_hbn_utils.py)
# ---------------------------------------------------------------------------
_STUB_MODULES = [
    "eegdash", "eegdash.dataset",
    "braindecode", "braindecode.datasets",
    "braindecode.preprocessing",
]
for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        stub = ModuleType(_mod_name)
        if _mod_name == "eegdash.dataset":
            stub.EEGChallengeDataset = MagicMock()
        if _mod_name == "braindecode.datasets":
            stub.BaseConcatDataset = MagicMock()
        if _mod_name == "braindecode.preprocessing":
            stub.create_fixed_length_windows = MagicMock()
            stub.create_windows_from_events = MagicMock()
        sys.modules[_mod_name] = stub

from eb_jepa.datasets import hbn  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures & constants
# ---------------------------------------------------------------------------
# Columns we expect in the features CSV, grouped by semantic type.
EXPECTED_NUMERIC_COLUMNS = [
    "luminance_mean", "contrast_rms",
    "color_r_mean", "color_g_mean", "color_b_mean",
    "saturation_mean", "edge_density", "spatial_freq_energy",
    "entropy", "motion_energy",
    "frame_idx", "timestamp_s",
    "n_faces", "face_area_frac",
    "depth_mean", "depth_std", "depth_range",
    "n_objects",
    "scene_category_score", "scene_natural_score", "scene_open_score",
]
EXPECTED_CATEGORICAL_COLUMNS = ["object_categories", "scene_category"]
EXPECTED_BOOLEAN_COLUMNS = ["scene_cut"]
ALL_EXPECTED_COLUMNS = (
    EXPECTED_NUMERIC_COLUMNS + EXPECTED_CATEGORICAL_COLUMNS + EXPECTED_BOOLEAN_COLUMNS
)

SFREQ = 256  # typical EEG sampling frequency (Hz)
N_CHANNELS = 129  # EGI HydroCel Geodesic Sensor Net
FPS = hbn.MOVIE_METADATA["ThePresent"]["fps"]  # 24
FRAME_COUNT = hbn.MOVIE_METADATA["ThePresent"]["frame_count"]  # 4878
MOVIE_DURATION = hbn.MOVIE_METADATA["ThePresent"]["duration"]


def _make_movie_features(n_frames: int = 100) -> pd.DataFrame:
    """Create a minimal mock movie-features DataFrame with the correct schema."""
    rng = np.random.default_rng(42)
    data = {col: rng.random(n_frames) for col in EXPECTED_NUMERIC_COLUMNS}
    data["frame_idx"] = np.arange(n_frames)
    data["timestamp_s"] = data["frame_idx"] / FPS
    data["n_faces"] = rng.integers(0, 3, size=n_frames)
    data["n_objects"] = rng.integers(0, 5, size=n_frames)
    data["scene_cut"] = rng.choice([True, False], size=n_frames)
    data["object_categories"] = ["{}" for _ in range(n_frames)]
    data["scene_category"] = rng.choice(["bathroom", "bedroom", "office"], size=n_frames)
    return pd.DataFrame(data)


# ===================================================================
# 1. Features CSV structure assumptions
# ===================================================================
class TestFeatureCSVAssumptions:
    """Verify that the real features CSV matches the assumptions baked
    into the dataset and downstream code."""

    @pytest.fixture()
    def features_df(self):
        return pd.read_csv(hbn.MOVIE_METADATA["ThePresent"]["feature_csv"])

    def test_all_expected_columns_present(self, features_df):
        """The CSV must contain every column the dataset code and probes expect."""
        for col in ALL_EXPECTED_COLUMNS:
            assert col in features_df.columns, f"Missing expected column: {col}"

    def test_no_unexpected_columns(self, features_df):
        """Guard against silent schema drift — flag any new columns."""
        unexpected = set(features_df.columns) - set(ALL_EXPECTED_COLUMNS)
        assert unexpected == set(), f"Unexpected columns in features CSV: {unexpected}"

    def test_numeric_columns_are_numeric(self, features_df):
        """Columns used in regression losses must be numeric (float or int)."""
        for col in EXPECTED_NUMERIC_COLUMNS:
            assert pd.api.types.is_numeric_dtype(features_df[col]), (
                f"Column '{col}' should be numeric, got {features_df[col].dtype}"
            )

    def test_frame_idx_sequential_from_zero(self, features_df):
        """frame_idx must be a contiguous 0-based sequence — we use iloc indexing."""
        expected = np.arange(len(features_df))
        np.testing.assert_array_equal(features_df["frame_idx"].values, expected)

    def test_timestamps_monotonically_increasing(self, features_df):
        """Timestamps should strictly increase (no duplicates or decreases)."""
        diffs = features_df["timestamp_s"].diff().iloc[1:]
        assert (diffs > 0).all(), "Timestamps are not strictly monotonically increasing"

    def test_row_count_matches_frame_count_minus_one(self, features_df):
        """The CSV has one row per frame. frame_count from cv2 is 4878 but the
        CSV covers frames 0..4876, yielding 4877 rows (the last frame at index
        4877 is never captured because the movie ends mid-frame).
        We tolerate a difference of at most 1 between frame_count and CSV rows."""
        n_rows = len(features_df)
        assert abs(n_rows - FRAME_COUNT) <= 1, (
            f"CSV has {n_rows} rows but MOVIE_METADATA frame_count is {FRAME_COUNT}. "
            f"Difference of {abs(n_rows - FRAME_COUNT)} exceeds tolerance of 1."
        )

    def test_no_nan_in_key_numeric_columns(self, features_df):
        """Core visual features should not have NaN values."""
        key_cols = [
            "luminance_mean", "contrast_rms", "entropy",
            "motion_energy", "edge_density",
        ]
        for col in key_cols:
            assert not features_df[col].isna().any(), (
                f"Column '{col}' has NaN values"
            )


# ===================================================================
# 2. Window-to-frame mapping (get_window_movie_metadata)
# ===================================================================
class TestWindowToFrameMapping:
    """Verify the core assumption: EEG window onset (samples) maps to the
    correct movie frame via  frame_index = int(onset / sfreq * fps)."""

    def test_first_window_maps_to_frame_zero(self):
        """Window at onset=0 should map to frame 0."""
        features = _make_movie_features(100)
        result = hbn.get_window_movie_metadata(
            window_onset=0, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0,
        )
        assert result["frame_idx"] == 0

    @pytest.mark.parametrize("onset_samples,expected_frame", [
        # onset/sfreq * fps = frame (truncated to int)
        (0, 0),
        (256, 24),       # 1.0s * 24fps = frame 24
        (512, 48),       # 2.0s * 24fps = frame 48
        (128, 12),       # 0.5s * 24fps = frame 12
        (10, 0),         # 10/256 * 24 = 0.9375 -> frame 0
        (11, 1),         # 11/256 * 24 = 1.03125 -> frame 1
    ])
    def test_onset_to_frame_parametric(self, onset_samples, expected_frame):
        """Assumption: frame_index = int(window_onset / sfreq * fps)."""
        n_frames = max(expected_frame + 10, 100)
        features = _make_movie_features(n_frames)
        result = hbn.get_window_movie_metadata(
            window_onset=onset_samples, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0,
        )
        assert result["frame_idx"] == expected_frame

    def test_features_dict_has_all_columns(self):
        """Returned dict should contain every column from the DataFrame."""
        features = _make_movie_features(10)
        result = hbn.get_window_movie_metadata(
            window_onset=0, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0,
        )
        assert set(result.keys()) == set(features.columns)

    def test_features_values_match_dataframe_row(self):
        """The returned dict values should exactly match the corresponding
        DataFrame row (no transformations or rounding)."""
        features = _make_movie_features(50)
        # onset=256 at sfreq=256 -> 1.0s -> frame 24
        result = hbn.get_window_movie_metadata(
            window_onset=256, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0,
        )
        expected_row = features.iloc[24].to_dict()
        for key in expected_row:
            assert result[key] == expected_row[key], (
                f"Mismatch on '{key}': {result[key]} != {expected_row[key]}"
            )


# ===================================================================
# 3. End-of-movie edge case handling
# ===================================================================
class TestEndOfMovieHandling:
    """Windows that fall near or slightly past the movie end should be
    clamped to the last frame's features (within 2-second tolerance)."""

    def test_window_slightly_past_movie_end_uses_last_frame(self):
        """A window onset that maps to a frame just past frame_count should
        still return valid features (the last row), as long as it's within
        2 seconds (= 2*fps = 48 frames) past the movie end."""
        n_frames = 100
        features = _make_movie_features(n_frames)

        # Onset that maps to frame_count + 10 (within 48-frame tolerance)
        # frame_index = int(onset / sfreq * fps) should exceed FRAME_COUNT
        # but be within FRAME_COUNT + fps*2
        target_frame = FRAME_COUNT + 10  # 4888
        onset = int(target_frame / FPS * SFREQ)  # back-compute onset

        result = hbn.get_window_movie_metadata(
            window_onset=onset, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0,
        )
        # Should clamp to last row
        expected = features.iloc[-1].to_dict()
        assert result["frame_idx"] == expected["frame_idx"]

    def test_window_at_exact_last_frame(self):
        """Window at the exact last valid frame should work normally."""
        n_frames = 50
        features = _make_movie_features(n_frames)
        last_frame = n_frames - 1
        onset = int(last_frame / FPS * SFREQ)
        result = hbn.get_window_movie_metadata(
            window_onset=onset, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0,
        )
        computed_frame = int(onset / SFREQ * FPS)
        assert result["frame_idx"] == features.iloc[computed_frame]["frame_idx"]


# ===================================================================
# 3b. Visual processing delay
# ===================================================================
class TestVisualProcessingDelay:
    """Verify that visual_processing_delay_s shifts the frame lookup
    backwards in time: EEG at onset T reflects the frame at T - delay."""

    def test_delay_shifts_frame_backward(self):
        """With a 0.1s delay at 256 Hz, onset=256 (1.0s) should map to
        frame at 0.9s = int(0.9 * 24) = 21, not frame 24."""
        features = _make_movie_features(100)
        result = hbn.get_window_movie_metadata(
            window_onset=256, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0.1,
        )
        # (256 - 25) / 256 * 24 = 21.65 -> frame 21
        assert result["frame_idx"] == 21

    def test_zero_delay_matches_raw_mapping(self):
        """With zero delay, the mapping is simply int(onset / sfreq * fps)."""
        features = _make_movie_features(100)
        result = hbn.get_window_movie_metadata(
            window_onset=256, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0,
        )
        assert result["frame_idx"] == 24  # int(256/256 * 24)

    def test_delay_clamps_negative_frame_to_zero(self):
        """When onset < delay_samples, frame index would be negative;
        it should be clamped to 0."""
        features = _make_movie_features(100)
        # delay_samples = int(0.1 * 256) = 25, onset=10 < 25
        result = hbn.get_window_movie_metadata(
            window_onset=10, sfreq=SFREQ, movie="ThePresent",
            movie_features=features, visual_processing_delay_s=0.1,
        )
        assert result["frame_idx"] == 0


# ===================================================================
# 4. Recording rejection criteria
# ===================================================================
class TestRejectRecording:
    """Recordings are rejected if they lack video_start/video_stop annotations
    or if the movie portion is too short."""

    def _make_mock_raw(self, event_ids, duration_samples, sfreq=500.0):
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": sfreq}
        events = np.array([
            [1000, 0, 1],
            [1000 + duration_samples, 0, 2],
        ])
        return mock_raw, events, event_ids

    def test_reject_missing_video_start(self):
        """Recordings without 'video_start' annotation should be rejected."""
        mock_raw = MagicMock()
        events = np.array([[0, 0, 1]])
        event_id = {"some_other_event": 1}  # no video_start
        with patch("mne.events_from_annotations", return_value=(events, event_id)):
            assert hbn.reject_recording(mock_raw, "ThePresent") is True

    def test_reject_missing_video_stop(self):
        """Recordings without 'video_stop' annotation should be rejected."""
        mock_raw = MagicMock()
        events = np.array([[0, 0, 1]])
        event_id = {"video_start": 1}  # no video_stop
        with patch("mne.events_from_annotations", return_value=(events, event_id)):
            assert hbn.reject_recording(mock_raw, "ThePresent") is True

    def test_reject_too_short_recording(self):
        """Recordings shorter than movie_duration - 0.5s should be rejected."""
        sfreq = 500.0
        # Make recording 10 seconds shorter than movie
        short_duration = MOVIE_DURATION - 10.0
        duration_samples = int(short_duration * sfreq)

        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": sfreq}
        events = np.array([
            [1000, 0, 1],
            [1000 + duration_samples, 0, 2],
        ])
        event_id = {"video_start": 1, "video_stop": 2}

        with patch("mne.events_from_annotations", return_value=(events, event_id)), \
             patch("mne.pick_events", return_value=events):
            assert hbn.reject_recording(mock_raw, "ThePresent") is True

    def test_accept_recording_within_tolerance(self):
        """Recordings within 0.5s of movie duration should be accepted."""
        sfreq = 500.0
        # Make recording just 0.3s shorter than movie (within 0.5s tolerance)
        ok_duration = MOVIE_DURATION - 0.3
        duration_samples = int(ok_duration * sfreq)

        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": sfreq}
        events = np.array([
            [1000, 0, 1],
            [1000 + duration_samples, 0, 2],
        ])
        event_id = {"video_start": 1, "video_stop": 2}

        with patch("mne.events_from_annotations", return_value=(events, event_id)), \
             patch("mne.pick_events", return_value=events):
            result = hbn.reject_recording(mock_raw, "ThePresent")
            assert result is False

    def test_accept_recording_longer_than_movie(self):
        """Recordings longer than the movie are fine (common due to annotation
        timing)."""
        sfreq = 500.0
        longer_duration = MOVIE_DURATION + 5.0
        duration_samples = int(longer_duration * sfreq)

        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": sfreq}
        events = np.array([
            [1000, 0, 1],
            [1000 + duration_samples, 0, 2],
        ])
        event_id = {"video_start": 1, "video_stop": 2}

        with patch("mne.events_from_annotations", return_value=(events, event_id)), \
             patch("mne.pick_events", return_value=events):
            result = hbn.reject_recording(mock_raw, "ThePresent")
            assert result is False


# ===================================================================
# 5. __getitem__ output contract
# ===================================================================
class TestGetItemOutputContract:
    """Verify the shape, dtype, and structure of what __getitem__ returns,
    using a fully mocked HBNMovieProbeDataset."""

    @pytest.fixture()
    def mock_dataset(self):
        """Build an HBNMovieProbeDataset with mocked internals."""
        ds = object.__new__(hbn.HBNMovieProbeDataset)
        ds.window_size_seconds = 2
        ds.task = "ThePresent"
        ds.post_movie_visual_processing_s = hbn.DEFAULT_POST_MOVIE_VISUAL_PROCESSING_S
        ds.visual_processing_delay_s = hbn.VISUAL_PROCESSING_DELAY_S
        ds.sfreq = SFREQ
        n_samples = ds.window_size_seconds * SFREQ  # 512

        # Mock self.data (the windowed braindecode dataset)
        # Each item returns (X_array, y, crop_inds)
        # crop_inds = (i_window_in_trial, i_start_in_trial, i_stop_in_trial)
        mock_windows = []
        for i in range(50):
            X = np.random.randn(N_CHANNELS, n_samples).astype(np.float64)
            y = 0
            onset_samples = i * n_samples  # non-overlapping windows
            crop_inds = (i, onset_samples, onset_samples + n_samples)
            mock_windows.append((X, y, crop_inds))

        mock_data = MagicMock()
        mock_data.__len__ = lambda self: len(mock_windows)
        mock_data.__getitem__ = lambda self, idx: mock_windows[idx]
        ds.data = mock_data

        ds.movie_features = {
            "ThePresent": _make_movie_features(5000),
        }
        return ds

    def test_returns_tuple_of_two(self, mock_dataset):
        """__getitem__ returns exactly (tensor, dict)."""
        result = mock_dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_X_is_float_tensor(self, mock_dataset):
        X, _ = mock_dataset[0]
        assert isinstance(X, torch.Tensor)
        assert X.dtype == torch.float32

    def test_X_shape_is_channels_by_time(self, mock_dataset):
        """X shape: (C, T) = (129, window_size_seconds * sfreq)."""
        X, _ = mock_dataset[0]
        expected_T = mock_dataset.window_size_seconds * SFREQ
        assert X.shape == (N_CHANNELS, expected_T)

    def test_features_is_dict(self, mock_dataset):
        _, features = mock_dataset[0]
        assert isinstance(features, dict)

    def test_features_has_all_expected_keys(self, mock_dataset):
        _, features = mock_dataset[0]
        for col in ALL_EXPECTED_COLUMNS:
            assert col in features, f"Missing key '{col}' in features dict"

    def test_features_numeric_values_are_scalar(self, mock_dataset):
        """Numeric feature values should be Python scalars (not arrays),
        so that DataLoader can collate them into tensors."""
        _, features = mock_dataset[0]
        for col in EXPECTED_NUMERIC_COLUMNS:
            val = features[col]
            assert np.isscalar(val) or isinstance(val, (int, float, np.integer, np.floating)), (
                f"Feature '{col}' is not a scalar: {type(val)}"
            )

    def test_different_indices_give_different_onsets(self, mock_dataset):
        """Consecutive windows should correspond to different movie timestamps."""
        _, feat0 = mock_dataset[0]
        _, feat1 = mock_dataset[1]
        # Because windows are non-overlapping, they should map to different frames
        # (unless they happen to land on the same frame, which is unlikely for 2s windows)
        assert feat0["frame_idx"] != feat1["frame_idx"]

    def test_len_matches_underlying_data(self, mock_dataset):
        assert len(mock_dataset) == 50


# ===================================================================
# 6. DataLoader collation behavior
# ===================================================================
class TestDataLoaderCollation:
    """Verify that the output of __getitem__ can be collated by a standard
    DataLoader into the expected batch format."""

    @pytest.fixture()
    def mock_dataset(self):
        ds = object.__new__(hbn.HBNMovieProbeDataset)
        ds.window_size_seconds = 2
        ds.task = "ThePresent"
        ds.post_movie_visual_processing_s = hbn.DEFAULT_POST_MOVIE_VISUAL_PROCESSING_S
        ds.visual_processing_delay_s = hbn.VISUAL_PROCESSING_DELAY_S
        ds.sfreq = SFREQ
        n_samples = ds.window_size_seconds * SFREQ

        mock_windows = []
        for i in range(10):
            X = np.random.randn(N_CHANNELS, n_samples).astype(np.float64)
            y = 0
            onset_samples = i * n_samples
            crop_inds = (i, onset_samples, onset_samples + n_samples)
            mock_windows.append((X, y, crop_inds))

        mock_data = MagicMock()
        mock_data.__len__ = lambda self: len(mock_windows)
        mock_data.__getitem__ = lambda self, idx: mock_windows[idx]
        ds.data = mock_data

        ds.movie_features = {"ThePresent": _make_movie_features(5000)}
        return ds

    def test_batch_X_shape(self, mock_dataset):
        """Batched X should be (B, C, T)."""
        loader = DataLoader(mock_dataset, batch_size=4, shuffle=False)
        X, _ = next(iter(loader))
        expected_T = mock_dataset.window_size_seconds * SFREQ
        assert X.shape == (4, N_CHANNELS, expected_T)

    def test_batch_X_dtype(self, mock_dataset):
        loader = DataLoader(mock_dataset, batch_size=4, shuffle=False)
        X, _ = next(iter(loader))
        assert X.dtype == torch.float32

    def test_batch_features_is_dict_of_tensors_or_lists(self, mock_dataset):
        """After collation, numeric features become tensors; string features
        become lists of strings."""
        loader = DataLoader(mock_dataset, batch_size=4, shuffle=False)
        _, features = next(iter(loader))
        assert isinstance(features, dict)

        # Numeric columns should be tensors of length B
        for col in ["luminance_mean", "contrast_rms", "entropy"]:
            assert isinstance(features[col], torch.Tensor), (
                f"Expected tensor for '{col}', got {type(features[col])}"
            )
            assert features[col].shape == (4,)

        # String columns should be lists (or tuples) of strings
        for col in ["scene_category", "object_categories"]:
            assert isinstance(features[col], (list, tuple)), (
                f"Expected list/tuple for '{col}', got {type(features[col])}"
            )
            assert len(features[col]) == 4

    def test_numeric_features_usable_in_loss(self, mock_dataset):
        """Regression workflow: a numeric feature batch should be directly
        usable in a loss function without extra conversion."""
        loader = DataLoader(mock_dataset, batch_size=4, shuffle=False)
        _, features = next(iter(loader))
        target = features["luminance_mean"].float()
        dummy_pred = torch.randn(4)
        loss = torch.nn.functional.mse_loss(dummy_pred, target)
        assert loss.item() >= 0  # just check it's computable


# ===================================================================
# 7. MOVIE_METADATA consistency
# ===================================================================
class TestMovieMetadataConsistency:
    """Verify internal consistency of the hardcoded MOVIE_METADATA."""

    def test_fps_is_positive_integer(self):
        assert hbn.MOVIE_METADATA["ThePresent"]["fps"] > 0
        assert isinstance(hbn.MOVIE_METADATA["ThePresent"]["fps"], int)

    def test_frame_count_consistent_with_duration(self):
        """frame_count / fps should approximately equal duration."""
        meta = hbn.MOVIE_METADATA["ThePresent"]
        computed_duration = meta["frame_count"] / meta["fps"]
        assert computed_duration == pytest.approx(meta["duration"], abs=0.1)

    def test_feature_csv_path_exists(self):
        from pathlib import Path
        csv_path = Path(hbn.MOVIE_METADATA["ThePresent"]["feature_csv"])
        assert csv_path.exists(), f"Feature CSV not found at {csv_path}"
