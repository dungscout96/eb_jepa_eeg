import sys
from unittest.mock import patch, MagicMock
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

# Stub out heavy transitive dependencies so hbn can be imported even when
# torchaudio / braindecode have ABI issues or are not installed.
_STUB_MODULES = [
    "eegdash", "eegdash.dataset",
    "braindecode", "braindecode.datasets",
    "braindecode.preprocessing",
]
_saved = {}
for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        stub = ModuleType(_mod_name)
        # Provide the names that hbn.py imports at module level
        if _mod_name == "eegdash.dataset":
            stub.EEGChallengeDataset = MagicMock()
        if _mod_name == "braindecode.datasets":
            stub.BaseConcatDataset = MagicMock()
        if _mod_name == "braindecode.preprocessing":
            stub.create_fixed_length_windows = MagicMock()
            stub.create_windows_from_events = MagicMock()
        sys.modules[_mod_name] = stub
        _saved[_mod_name] = stub

from eb_jepa.datasets import hbn  # noqa: E402


class TestGetMovieMetadata:
    @patch("cv2.VideoCapture")
    def test_returns_duration_fps_frame_count(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            # cv2.CAP_PROP_FPS == 5, CAP_PROP_FRAME_COUNT == 7
            5: 24.0,
            7: 4800.0,
        }[prop]
        mock_cap_cls.return_value = mock_cap

        duration, fps, frame_count = hbn.get_movie_metadata("ThePresent")

        assert fps == 24.0
        assert frame_count == 4800.0
        assert duration == pytest.approx(200.0)  # 4800 / 24

    @patch("cv2.VideoCapture")
    def test_invalid_task_raises(self, mock_cap_cls):
        with pytest.raises(KeyError):
            hbn.get_movie_metadata("NonexistentMovie")


class TestGetWindowMovieMetadata:
    def test_correct_frame_lookup(self):
        movie_features = pd.DataFrame({
            "brightness": [0.1, 0.2, 0.3, 0.4, 0.5],
            "contrast": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        # window_onset=10, sfreq=250 -> timestamp=0.04s -> frame_index = int(0.04 * 24) = 0
        features = hbn.get_window_movie_metadata(
            window_onset=10, sfreq=250, movie="ThePresent",
            movie_features=movie_features, visual_processing_delay_s=0,
        )
        assert features["brightness"] == pytest.approx(0.1)
        assert features["contrast"] == pytest.approx(1.0)

    def test_later_frame(self):
        movie_features = pd.DataFrame({
            "val": [10, 20, 30, 40, 50],
        })
        # int(31/250 * 24) = int(2.976) = 2
        features = hbn.get_window_movie_metadata(
            window_onset=31, sfreq=250, movie="ThePresent",
            movie_features=movie_features, visual_processing_delay_s=0,
        )
        assert features["val"] == 30  # iloc[2]


class TestLoadOrDownload:
    @patch.object(hbn, "EEGChallengeDataset")
    def test_downloads_when_dir_missing(self, mock_dataset_cls, tmp_path):
        mock_dataset = MagicMock()
        mock_dataset_cls.return_value = mock_dataset

        with patch.object(hbn, "DATA_DIR", tmp_path):
            hbn.load_or_download("R1", "ds005505")

        mock_dataset_cls.assert_called_once()
        mock_dataset.download_all.assert_called_once_with(n_jobs=-1)

    @patch.object(hbn, "EEGChallengeDataset")
    def test_skips_download_when_data_exists(self, mock_dataset_cls, tmp_path):
        mock_dataset = MagicMock()
        mock_dataset_cls.return_value = mock_dataset

        data_dir = tmp_path / "EEG2025r1mini"
        data_dir.mkdir()
        (data_dir / "recording.bdf").touch()

        with patch.object(hbn, "DATA_DIR", tmp_path):
            hbn.load_or_download("R1", "ds005505")

        mock_dataset_cls.assert_called_once()
        _, kwargs = mock_dataset_cls.call_args
        assert kwargs["download"] is False
        mock_dataset.download_all.assert_not_called()


class TestGetMovieRecordingDuration:
    def test_computes_duration_from_events(self):
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 500.0}

        events = np.array([
            [1000, 0, 1],  # video_start at sample 1000
            [51000, 0, 2],  # video_stop at sample 51000
        ])
        event_id = {"video_start": 1, "video_stop": 2}

        with patch("mne.events_from_annotations", return_value=(events, event_id)), \
             patch("mne.pick_events", return_value=events):
            duration = hbn.get_movie_recording_duration(mock_raw, movie="ThePresent")

        assert duration == pytest.approx(100.0)  # (51000-1000) / 500
