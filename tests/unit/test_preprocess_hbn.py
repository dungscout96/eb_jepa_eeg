"""Unit tests for scripts/preprocess_hbn.py preprocessing functions."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Stub heavy transitive dependencies so the script can be imported even when
# braindecode / eegdash are not installed or were partially stubbed by another
# test module that ran earlier in the same process.


class _FakeBaseConcatDataset:
    """Lightweight stand-in for braindecode.datasets.BaseConcatDataset."""

    def __init__(self, datasets):
        self.datasets = list(datasets)


_STUB_ATTRS = {
    "eegdash": {},
    "eegdash.dataset": {"EEGChallengeDataset": MagicMock()},
    "braindecode": {},
    "braindecode.datasets": {"BaseConcatDataset": _FakeBaseConcatDataset},
    "braindecode.preprocessing": {
        "Preprocessor": MagicMock(),
        "preprocess": MagicMock(),
        "create_fixed_length_windows": MagicMock(),
        "create_windows_from_events": MagicMock(),
    },
    "braindecode.datautil": {},
    "braindecode.datautil.serialization": {
        "load_concat_dataset": MagicMock(),
        "save_concat_dataset": MagicMock(),
    },
}

for _mod_name, _attrs in _STUB_ATTRS.items():
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = ModuleType(_mod_name)
    _mod = sys.modules[_mod_name]
    for _attr_name, _attr_val in _attrs.items():
        setattr(_mod, _attr_name, _attr_val)

# Now we can import the preprocessing script functions.
# We need to add the scripts dir to path and import.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the individual functions from preprocess_hbn
from preprocess_hbn import (  # noqa: E402
    compute_channel_stats,
    reject_short_recordings,
)


# ---------------------------------------------------------------------------
# Helpers to build mock MNE-like objects
# ---------------------------------------------------------------------------


def _make_mock_raw(duration_s: float, sfreq: float = 250.0, n_channels: int = 4):
    """Create a mock Raw-like object with .times and .get_data()."""
    n_samples = int(duration_s * sfreq)
    raw = MagicMock()
    raw.times = np.linspace(0, duration_s, n_samples, endpoint=False)
    raw.info = {"sfreq": sfreq}
    # Random data (C, T) in microvolts-like range
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_samples)).astype(np.float64)
    raw.get_data.return_value = data
    return raw, data


def _make_mock_dataset(raws):
    """Create a mock BaseConcatDataset-like object wrapping a list of raws."""
    ds = MagicMock()
    recordings = []
    for raw in raws:
        rec = MagicMock()
        rec.raw = raw
        recordings.append(rec)
    ds.datasets = recordings
    return ds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRejectShortRecordings:
    def test_rejects_short(self):
        raw_short, _ = _make_mock_raw(5.0)   # 5 seconds — below threshold
        raw_long, _ = _make_mock_raw(30.0)    # 30 seconds — above threshold
        dataset = _make_mock_dataset([raw_short, raw_long])

        filtered, n_rejected = reject_short_recordings(dataset, min_duration_s=10.0)

        assert n_rejected == 1
        assert len(filtered.datasets) == 1

    def test_keeps_all_when_long_enough(self):
        raw1, _ = _make_mock_raw(15.0)
        raw2, _ = _make_mock_raw(20.0)
        dataset = _make_mock_dataset([raw1, raw2])

        filtered, n_rejected = reject_short_recordings(dataset, min_duration_s=10.0)

        assert n_rejected == 0
        assert len(filtered.datasets) == 2

    def test_rejects_all_when_too_short(self):
        raw1, _ = _make_mock_raw(3.0)
        raw2, _ = _make_mock_raw(5.0)
        dataset = _make_mock_dataset([raw1, raw2])

        filtered, n_rejected = reject_short_recordings(dataset, min_duration_s=10.0)

        assert n_rejected == 2
        assert len(filtered.datasets) == 0

    def test_handles_unloadable_recording(self):
        raw_good, _ = _make_mock_raw(20.0)
        dataset = _make_mock_dataset([raw_good])

        # Make second recording raise on .raw access
        bad_rec = MagicMock()
        type(bad_rec).raw = property(lambda self: (_ for _ in ()).throw(OSError("corrupt")))
        dataset.datasets.append(bad_rec)

        filtered, n_rejected = reject_short_recordings(dataset, min_duration_s=10.0)

        assert n_rejected == 1
        assert len(filtered.datasets) == 1


class TestComputeChannelStats:
    def test_single_recording_stats(self):
        """Stats from a single recording should match numpy directly."""
        rng = np.random.default_rng(123)
        n_channels, n_samples = 4, 1000
        data = rng.standard_normal((n_channels, n_samples)).astype(np.float64)

        raw = MagicMock()
        raw.get_data.return_value = data
        dataset = _make_mock_dataset([raw])

        stats = compute_channel_stats(dataset)

        np.testing.assert_allclose(stats["mean"], data.mean(axis=1), atol=1e-6)
        np.testing.assert_allclose(stats["std"], data.std(axis=1), atol=1e-6)
        assert stats["n_samples"] == n_samples
        assert stats["n_recordings"] == 1

    def test_multiple_recordings_stats(self):
        """Stats across two recordings should equal stats over concatenated data."""
        rng = np.random.default_rng(456)
        n_channels = 3
        data1 = rng.standard_normal((n_channels, 500)).astype(np.float64)
        data2 = rng.standard_normal((n_channels, 800)).astype(np.float64)

        raw1 = MagicMock()
        raw1.get_data.return_value = data1
        raw2 = MagicMock()
        raw2.get_data.return_value = data2
        dataset = _make_mock_dataset([raw1, raw2])

        stats = compute_channel_stats(dataset)

        # Ground truth: concatenated
        combined = np.concatenate([data1, data2], axis=1)
        expected_mean = combined.mean(axis=1)
        expected_std = combined.std(axis=1)

        np.testing.assert_allclose(stats["mean"], expected_mean, atol=1e-6)
        np.testing.assert_allclose(stats["std"], expected_std, atol=1e-6)
        assert stats["n_samples"] == 1300
        assert stats["n_recordings"] == 2

    def test_channel_mismatch_skips(self):
        """Recording with different channel count is skipped."""
        rng = np.random.default_rng(789)
        data1 = rng.standard_normal((4, 500)).astype(np.float64)
        data2 = rng.standard_normal((3, 500)).astype(np.float64)  # different n_channels

        raw1 = MagicMock()
        raw1.get_data.return_value = data1
        raw2 = MagicMock()
        raw2.get_data.return_value = data2
        dataset = _make_mock_dataset([raw1, raw2])

        stats = compute_channel_stats(dataset)

        # Only data1 should be included
        np.testing.assert_allclose(stats["mean"], data1.mean(axis=1), atol=1e-6)
        assert stats["n_recordings"] == 2  # counts all, but only 1 contributed


class TestNormalizeAndClip:
    def test_zscore_and_clip(self):
        """Verify that z-score normalization followed by clipping works correctly."""
        # Simulate what run_pass2 does internally
        rng = np.random.default_rng(101)
        n_channels, n_samples = 4, 1000
        data = rng.standard_normal((n_channels, n_samples)).astype(np.float32) * 10 + 5

        # Compute stats
        mean = data.mean(axis=1)
        std = data.std(axis=1)

        # Apply z-score
        ch_mean = mean[:, np.newaxis]
        ch_std = std[:, np.newaxis]
        normed = (data - ch_mean) / ch_std

        # Clip
        clip_std = 15.0
        clipped = np.clip(normed, -clip_std, clip_std)

        # After z-score, mean ≈ 0, std ≈ 1
        for c in range(n_channels):
            assert abs(normed[c].mean()) < 0.05
            assert abs(normed[c].std() - 1.0) < 0.1

        # After clipping, all values within bounds
        assert clipped.min() >= -clip_std
        assert clipped.max() <= clip_std

    def test_extreme_values_clipped(self):
        """Values beyond clip_std are clamped."""
        data = np.array([[0.0, 0.0, 0.0, 100.0, -100.0]], dtype=np.float32)
        mean = np.array([0.0], dtype=np.float32)
        std = np.array([1.0], dtype=np.float32)

        normed = (data - mean[:, None]) / std[:, None]
        clipped = np.clip(normed, -15.0, 15.0)

        assert clipped[0, 3] == 15.0
        assert clipped[0, 4] == -15.0
