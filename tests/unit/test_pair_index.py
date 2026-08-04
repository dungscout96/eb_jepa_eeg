"""Unit tests for PairedSubjectJEPADataset's (movie, time) -> [recording] index.

Runs with no HBN data: the object is created via ``object.__new__`` and the
per-recording side tables that ``JEPAMovieDataset.__init__`` would have built
are populated by hand, then ``_build_pair_index()`` is called directly.

NOTE ON THE INTERIOR-HOLE TESTS — DO NOT DELETE THEM AS UNREACHABLE.
With the currently shipped data, window drops are a pure prefix/suffix trim
(both drop paths are monotone in movie time, and both V-JEPA-2 timestamp grids
are uniform 0.5 s with no gaps), so a recording cannot actually have an interior
hole today. These tests are deliberately *defensive*: they pin the tripwire
behaviour that keeps pairing correct — rather than silently wrong — if the
embedding grid is ever re-extracted at a coarser stride, or a movie whose grid is
not uniform is wired into MOVIE_METADATA.
"""

import pytest
import torch

from eb_jepa.datasets.paired import PairedSubjectJEPADataset


def _make(t_starts_per_rec, subjects, tasks=None, *, n_windows=1,
          temporal_stride=1, window_size_seconds=2.0,
          pairs_per_recording=1, pair_min_partners=1):
    """Build a PairedSubjectJEPADataset with hand-populated side tables."""
    ds = object.__new__(PairedSubjectJEPADataset)
    n = len(t_starts_per_rec)
    ds._fif_paths = [f"/fake/rec{i}-raw.fif" for i in range(n)]
    ds._crop_inds = [
        torch.zeros(len(t), 3, dtype=torch.long).numpy() for t in t_starts_per_rec
    ]
    ds.t_start_recordings = [
        torch.tensor(t, dtype=torch.float32) for t in t_starts_per_rec
    ]
    ds._recording_tasks = list(tasks) if tasks else ["ThePresent"] * n
    ds._recording_metadata = [{"subject": s} for s in subjects]
    ds.n_windows = n_windows
    ds.temporal_stride = temporal_stride
    ds.window_size_seconds = window_size_seconds
    ds.pairs_per_recording = pairs_per_recording
    ds.pair_min_partners = pair_min_partners
    ds._build_pair_index()
    return ds


# -- happy path -------------------------------------------------------------


def test_three_recordings_fully_overlapping():
    ds = _make(
        [[0.0, 2.0, 4.0, 6.0]] * 3,
        subjects=["s1", "s2", "s3"],
    )
    assert len(ds._anchor_to_recs) == 4
    assert sorted(ds._anchor_to_recs) == [("ThePresent", k) for k in (0, 1, 2, 3)]
    for recs in ds._anchor_to_recs.values():
        assert recs == [0, 1, 2]
    assert ds._a_pool == [0, 1, 2]
    assert len(ds) == 3


def test_len_scales_with_pairs_per_recording():
    ds = _make(
        [[0.0, 2.0]] * 3, subjects=["s1", "s2", "s3"], pairs_per_recording=4
    )
    assert len(ds) == 12
    # Index wraps around the A-pool.
    assert ds._a_pool[11 % len(ds._a_pool)] == ds._a_pool[2]


def test_window_ordinal_keys_are_movie_time_not_window_index():
    """Two recordings trimmed differently must still align on movie time."""
    ds = _make(
        [
            [0.0, 2.0, 4.0, 6.0],   # full
            [4.0, 6.0],             # leading trim: its window 0 is t=4
        ],
        subjects=["s1", "s2"],
    )
    # Recording 1's window 0 maps to key 2 (t=4), not key 0.
    assert ds._tkey_to_win[1] == {2: 0, 3: 1}
    # Only t=4 and t=6 are shared, so only those two anchors survive pruning.
    assert sorted(k for _, k in ds._anchor_to_recs) == [2, 3]
    assert ds._anchor_to_recs[("ThePresent", 2)] == [0, 1]


# -- subject constraints ----------------------------------------------------


def test_same_subject_only_anchors_are_pruned():
    """Two recordings from ONE subject cannot form a cross-subject pair."""
    ds = _make(
        [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
        subjects=["s1", "s1", "s2"],
    )
    # Anchors survive (3 recordings, 2 subjects), but rec 0/1 can only pair
    # with rec 2.
    for _ in range(20):
        _, _, rec_ids = _getitem_meta(ds, 0)
        assert rec_ids[1] == 2


def test_no_pairable_anchor_raises():
    with pytest.raises(ValueError, match="No pairable"):
        _make([[0.0, 2.0], [0.0, 2.0]], subjects=["s1", "s1"])


def test_missing_subject_metadata_does_not_merge_recordings():
    """Recordings with no subject id must not be treated as the same subject."""
    ds = object.__new__(PairedSubjectJEPADataset)
    ds._recording_metadata = [{}, {}, {"participant_id": "sub-3"}]
    assert ds._subject_of(0) != ds._subject_of(1)
    assert ds._subject_of(2) == "sub-3"


def test_pair_min_partners_prunes_thin_anchors():
    ds = _make(
        [
            [0.0, 2.0, 4.0],
            [0.0, 2.0],        # no t=4
            [0.0, 2.0],        # no t=4
        ],
        subjects=["s1", "s2", "s3"],
        pair_min_partners=2,
    )
    # t=4 has only 1 recording -> pruned. t=0 and t=2 have 3 -> kept.
    assert sorted(k for _, k in ds._anchor_to_recs) == [0, 1]


# -- multi-window clips -----------------------------------------------------


def test_multi_window_clip_requires_every_window_present():
    """DEFENSIVE (see module docstring): synthetic interior hole at t=4."""
    ds = _make(
        [
            [0.0, 2.0, 4.0, 6.0],
            [0.0, 2.0, 4.0, 6.0],
            [0.0, 2.0, 6.0],          # interior hole at t=4
        ],
        subjects=["s1", "s2", "s3"],
        n_windows=2,
    )
    # Recording 2 can only anchor a 2-window clip at k0=0 (needs {0,1}).
    # k0=1 needs {1,2} and k0=2 needs {2,3} — both miss key 2 (t=4).
    assert ds._rec_anchors[2] == [("ThePresent", 0)]
    assert ds._anchor_to_recs[("ThePresent", 0)] == [0, 1, 2]
    assert ds._anchor_to_recs[("ThePresent", 1)] == [0, 1]
    assert ds._anchor_to_recs[("ThePresent", 2)] == [0, 1]


def test_temporal_stride_skips_keys():
    ds = _make(
        [[0.0, 2.0, 4.0, 6.0, 8.0]] * 2,
        subjects=["s1", "s2"],
        n_windows=2,
        temporal_stride=2,
    )
    # A 2-window clip at stride 2 needs {k0, k0+2}, so k0 in {0, 1, 2}.
    assert sorted(k for _, k in ds._anchor_to_recs) == [0, 1, 2]


def test_clip_longer_than_any_recording_leaves_no_anchors():
    with pytest.raises(ValueError, match="No pairable"):
        _make([[0.0, 2.0]] * 2, subjects=["s1", "s2"], n_windows=5)


# -- movies -----------------------------------------------------------------


def test_anchors_never_cross_movies():
    ds = _make(
        [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
        subjects=["s1", "s2", "s3", "s4"],
        tasks=["ThePresent", "ThePresent", "DespicableMe", "DespicableMe"],
    )
    assert ds._anchor_to_recs[("ThePresent", 0)] == [0, 1]
    assert ds._anchor_to_recs[("DespicableMe", 0)] == [2, 3]
    for _ in range(20):
        _, _, rec_ids = _getitem_meta(ds, ds._a_pool.index(0))
        assert set(rec_ids.tolist()) <= {0, 1}


# -- quantizer soundness ----------------------------------------------------


def test_duplicate_t_start_raises_collision():
    with pytest.raises(ValueError, match="collision"):
        _make(
            [[0.0, 2.0, 2.0], [0.0, 2.0, 4.0]],
            subjects=["s1", "s2"],
        )


def test_sub_window_drift_is_absorbed_by_window_ordinal_keying():
    """t_start values drifting by < ws/2 must still map to one anchor.

    This is why the key is a window ordinal rather than a fixed epsilon bucket:
    at any window size where ``ws * fps`` is not an integer, ``int()``
    truncation in the frame-index computation makes t_start wobble.
    """
    ds = _make(
        [[0.0, 1.96, 3.92], [0.0, 2.04, 4.08]],
        subjects=["s1", "s2"],
    )
    assert sorted(k for _, k in ds._anchor_to_recs) == [0, 1, 2]
    assert ds._anchor_to_recs[("ThePresent", 1)] == [0, 1]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pairs_per_recording": 0}, "pairs_per_recording"),
        ({"pair_min_partners": 0}, "pair_min_partners"),
    ],
)
def test_invalid_constructor_args_rejected_before_data_load(kwargs, match):
    """These guards run before super().__init__, so no HBN data is needed.

    That ordering is deliberate: the parent __init__ spends minutes loading and
    windowing releases, and there is no reason to do that only to reject an
    out-of-range argument afterwards.
    """
    with pytest.raises(ValueError, match=match):
        PairedSubjectJEPADataset(split="train", cfg={}, **kwargs)


# -- helpers ----------------------------------------------------------------


def _getitem_meta(ds, idx):
    """Run __getitem__'s selection logic without touching disk.

    Patches ``_load_clip`` to return a shaped zero tensor so the pairing logic
    (anchor choice, distinct-subject rejection sampling, window lookup) can be
    exercised with no FIF files present.
    """
    ds._load_clip = lambda rec_idx, indices: torch.zeros(
        len(indices), 4, 8, dtype=torch.float32
    )
    return ds[idx]


def test_getitem_shapes_and_distinct_subjects():
    ds = _make(
        [[0.0, 2.0, 4.0]] * 3,
        subjects=["s1", "s2", "s3"],
        n_windows=2,
    )
    eeg, t_starts, rec_ids = _getitem_meta(ds, 0)

    assert eeg.shape == (2, 2, 4, 8)          # [2 subjects, n_windows, C, T]
    assert t_starts.shape == (2,)
    assert rec_ids.shape == (2,)
    assert rec_ids.dtype == torch.long
    assert ds._subject_of(rec_ids[0].item()) != ds._subject_of(rec_ids[1].item())
    # Both subjects' clips are drawn at the same movie time.
    assert t_starts[1] - t_starts[0] == pytest.approx(2.0)


def test_getitem_partner_is_always_a_different_subject():
    ds = _make(
        [[0.0, 2.0]] * 5,
        subjects=["s1", "s1", "s1", "s1", "s2"],
    )
    for idx in range(len(ds)):
        for _ in range(10):
            _, _, rec_ids = _getitem_meta(ds, idx)
            a, b = rec_ids.tolist()
            assert ds._subject_of(a) != ds._subject_of(b)
