"""Unit tests for window-anchored frame embedding + shot ID helpers."""
import numpy as np

from eb_jepa.datasets.hbn import (
    get_window_frame_embedding,
    get_window_shot_id,
)


def test_frame_embedding_mean_pool_full_window():
    # 2 Hz clips, 5 clips covering 0..2s (timestamps 0.0, 0.5, 1.0, 1.5, 2.0)
    embeddings = np.arange(20, dtype=np.float32).reshape(5, 4)
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    # Window [0, 2): half-open → includes clips at 0.0, 0.5, 1.0, 1.5 (NOT 2.0)
    out = get_window_frame_embedding(0, 24.0, 2.0, embeddings, timestamps)
    np.testing.assert_allclose(out, embeddings[:4].mean(axis=0))


def test_frame_embedding_partial_window():
    embeddings = np.arange(20, dtype=np.float32).reshape(5, 4)
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    # Window starts at frame 12 (0.5s), 1s long → covers [0.5, 1.5)
    out = get_window_frame_embedding(12, 24.0, 1.0, embeddings, timestamps)
    np.testing.assert_allclose(out, embeddings[1:3].mean(axis=0))


def test_frame_embedding_out_of_bounds():
    embeddings = np.zeros((5, 4), dtype=np.float32)
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    # Window starts at 100s, far past the last timestamp
    out = get_window_frame_embedding(2400, 24.0, 2.0, embeddings, timestamps)
    assert out is None


def test_shot_id_inside_single_shot():
    # Shots: [0, 100) [100, 250) [250, +inf)
    boundaries = np.array([100, 250], dtype=np.int64)

    # Window fully inside shot 0
    sid, crossed = get_window_shot_id(10, 48, boundaries)
    assert sid == 0 and crossed is False


def test_shot_id_dominant_on_boundary_window():
    # Boundary at 100; window [90, 138) → shot 0 overlap=10, shot 1 overlap=38
    boundaries = np.array([100, 250], dtype=np.int64)
    sid, crossed = get_window_shot_id(90, 48, boundaries)
    assert sid == 1 and crossed is True


def test_shot_id_dominant_when_window_mostly_in_first_shot():
    # Boundary at 100; window [60, 108) → shot 0 overlap=40, shot 1 overlap=8
    boundaries = np.array([100, 250], dtype=np.int64)
    sid, crossed = get_window_shot_id(60, 48, boundaries)
    assert sid == 0 and crossed is True


def test_shot_id_final_shot():
    boundaries = np.array([100, 250], dtype=np.int64)
    # Window deep in shot 2 (last shot, unbounded)
    sid, crossed = get_window_shot_id(500, 48, boundaries)
    assert sid == 2 and crossed is False
