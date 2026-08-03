"""Tests for the E0.3 (anchors x subjects) scaling knobs in JEPAMovieDataset.

These guard properties the experiment's interpretability depends on, not just
crashes. If ``max_anchors`` leaks extra movie moments, or ``max_subjects``
misaligns the per-recording lists, the resulting r(A, S) surface looks fine and
means nothing -- which is the failure mode worth paying for tests to avoid.

The dataset's real __init__ reads FIFs, so these drive ``_apply_scaling_subsample``
and ``__getitem__`` against a hand-built object instead.
"""

import numpy as np
import pytest
import torch

from eb_jepa.datasets.hbn import JEPAMovieDataset


def _fake(n_rec=8, n_win=20, n_windows=4, subjects=None, stride=1):
    """A JEPAMovieDataset with only the per-recording state the knobs touch."""
    ds = object.__new__(JEPAMovieDataset)
    ds.n_windows = n_windows
    ds.temporal_stride = stride
    ds.recipe_mode = True
    ds._norm_mode = "per_recording"
    ds._add_envelope = False
    ds._corrca_W = None
    subjects = subjects or [f"sub-{i}" for i in range(n_rec)]
    ds._fif_paths = [f"/fake/{i}.fif" for i in range(n_rec)]
    ds._crop_inds = [np.zeros((n_win, 3), dtype=np.int64) for _ in range(n_rec)]
    ds._recording_metadata = [{"subject": s} for s in subjects]
    ds._recording_tasks = ["ThePresent"] * n_rec
    ds.feature_recordings = [torch.zeros(n_win, 2) for _ in range(n_rec)]
    ds.embedding_recordings = [torch.zeros(n_win, 8) for _ in range(n_rec)]
    ds.shot_id_recordings = [torch.zeros(n_win, dtype=torch.int64)] * n_rec
    ds.scene_id_recordings = [torch.zeros(n_win, dtype=torch.int64)] * n_rec
    # Movie time: 2 s windows starting at 0. Same grid in every recording.
    ds.t_start_recordings = [torch.arange(n_win, dtype=torch.float32) * 2.0
                             for _ in range(n_rec)]
    ds._probe_labels = [0.0] * n_rec
    return ds


# ---------------------------------------------------------------- no-ops


def test_all_none_is_a_no_op():
    """An unmodified config must behave exactly as before -- contiguous crops,
    length == recording count."""
    ds = _fake()
    ds._apply_scaling_subsample(None, None, None, seed=0)
    assert ds._epoch_size is None
    assert ds._allowed_anchor_idx is None
    assert len(ds) == 8


# ---------------------------------------------------------------- subjects (S)


def test_max_subjects_keeps_whole_subjects_not_recordings():
    """A subject with two recordings must contribute both or neither; otherwise
    the S axis is confounded by recordings-per-subject."""
    subs = ["a", "a", "b", "b", "c", "c", "d", "d"]
    ds = _fake(n_rec=8, subjects=subs)
    ds._apply_scaling_subsample(2, None, None, seed=0)
    kept = [m["subject"] for m in ds._recording_metadata]
    assert len(set(kept)) == 2
    for s in set(kept):
        assert kept.count(s) == 2          # both recordings, never one


def test_max_subjects_filters_every_per_recording_list_in_step():
    ds = _fake(n_rec=10)
    ds.feature_recordings = [torch.full((20, 2), float(i)) for i in range(10)]
    ds.embedding_recordings = [torch.full((20, 8), float(i)) for i in range(10)]
    ds._apply_scaling_subsample(4, None, None, seed=1)
    n = len(ds._fif_paths)
    assert n == 4
    for attr in JEPAMovieDataset._PER_RECORDING_ATTRS:
        assert len(getattr(ds, attr)) == n, attr
    # The join must survive: recording i's path, features and embeddings all
    # still describe the same original recording.
    for i in range(n):
        tag = float(ds._fif_paths[i].split("/")[-1].split(".")[0])
        assert ds.feature_recordings[i][0, 0].item() == tag
        assert ds.embedding_recordings[i][0, 0].item() == tag


def test_max_subjects_is_deterministic_given_seed():
    a, b, c = _fake(n_rec=12), _fake(n_rec=12), _fake(n_rec=12)
    a._apply_scaling_subsample(5, None, None, seed=7)
    b._apply_scaling_subsample(5, None, None, seed=7)
    c._apply_scaling_subsample(5, None, None, seed=8)
    assert a._fif_paths == b._fif_paths
    assert a._fif_paths != c._fif_paths


def test_max_subjects_above_cohort_size_is_a_no_op():
    ds = _fake(n_rec=6)
    ds._apply_scaling_subsample(99, None, None, seed=0)
    assert len(ds._fif_paths) == 6


def test_misaligned_list_is_refused_rather_than_silently_subsampled():
    """The whole point of _PER_RECORDING_ATTRS: a list that has drifted out of
    step must raise, not produce a dataset pairing one recording's EEG with
    another recording's targets."""
    ds = _fake(n_rec=8)
    ds.embedding_recordings = ds.embedding_recordings[:5]   # drift
    with pytest.raises(RuntimeError, match="expected 8"):
        ds._apply_scaling_subsample(4, None, None, seed=0)


# ---------------------------------------------------------------- anchors (A)


def test_max_anchors_restricts_the_moments_ever_seen():
    """The load-bearing property. Over many draws the model must never see a
    movie moment outside the selected anchor set -- if a contiguous crop were
    reused, a block starting at an allowed anchor would run straight through
    disallowed ones."""
    ds = _fake(n_rec=4, n_win=40, n_windows=4)
    ds._apply_scaling_subsample(None, 8, None, seed=0)
    torch.manual_seed(0)
    seen = set()
    for _ in range(300):
        for i in range(len(ds._fif_paths)):
            allowed = ds._allowed_anchor_idx[i]
            sel = torch.randperm(len(allowed))[:ds.n_windows].numpy()
            seen.update(allowed[np.sort(sel)].tolist())
    assert len(seen) <= 8
    assert seen <= set(ds._allowed_anchor_idx[0].tolist())


def test_max_anchors_selects_evenly_spaced_moments():
    """Evenly spaced, not random: a random subset clumps and would confound the
    A axis with coverage of the movie."""
    ds = _fake(n_rec=2, n_win=100, n_windows=4)
    ds._apply_scaling_subsample(None, 10, None, seed=0)
    idx = np.sort(ds._allowed_anchor_idx[0])
    assert len(idx) == 10
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1        # uniform to within rounding
    assert idx[0] == 0 and idx[-1] == 99       # spans the whole movie


def test_max_anchors_joins_on_movie_time_not_window_index():
    """Recordings with different dropped prefixes must select the SAME movie
    moments, at different window indices."""
    ds = _fake(n_rec=2, n_win=20, n_windows=2)
    # Recording 1 lost its first two windows: same moments, shifted indices.
    ds.t_start_recordings[1] = torch.arange(20, dtype=torch.float32) * 2.0 + 4.0
    ds._apply_scaling_subsample(None, 5, None, seed=0)
    t0 = ds.t_start_recordings[0][ds._allowed_anchor_idx[0]].tolist()
    t1 = ds.t_start_recordings[1][ds._allowed_anchor_idx[1]].tolist()
    shared = sorted(set(t0) & set(t1))
    assert shared, "no shared moments -- the join fell back to window index"
    for t in shared:
        i0 = ds._allowed_anchor_idx[0][t0.index(t)]
        i1 = ds._allowed_anchor_idx[1][t1.index(t)]
        assert i0 != i1 or t == 0.0            # same moment, different index


def test_max_anchors_below_n_windows_raises():
    """Silently sampling with replacement would duplicate anchors inside one
    item and quietly weaken the InfoNCE batch."""
    ds = _fake(n_rec=2, n_win=40, n_windows=8)
    with pytest.raises(RuntimeError, match="fewer than n_windows"):
        ds._apply_scaling_subsample(None, 4, None, seed=0)


def test_max_anchors_above_movie_length_is_a_no_op():
    ds = _fake(n_rec=2, n_win=30, n_windows=4)
    ds._apply_scaling_subsample(None, 999, None, seed=0)
    assert len(ds._allowed_anchor_idx[0]) == 30


# ---------------------------------------------------------------- epoch_size


def test_epoch_size_holds_steps_per_epoch_constant_across_S():
    """The protocol requirement from PLAN.md: data scale must not change the
    optimisation budget."""
    big = _fake(n_rec=64)
    big._apply_scaling_subsample(None, None, 640, seed=0)
    small = _fake(n_rec=64)
    small._apply_scaling_subsample(8, None, 640, seed=0)
    assert len(big) == len(small) == 640
    assert len(small._fif_paths) == 8      # far fewer recordings...
    assert len(small) == len(big)          # ...but the same steps per epoch


def test_epoch_size_wraps_indices_over_the_recordings_present():
    ds = _fake(n_rec=5, n_win=20, n_windows=2)
    ds._apply_scaling_subsample(None, None, 50, seed=0)
    assert len(ds) == 50
    # Every index in [0, 50) must map into [0, 5) -- no IndexError at the tail.
    for i in range(len(ds)):
        assert 0 <= i % len(ds._fif_paths) < 5


def test_epoch_size_none_falls_back_to_recording_count():
    ds = _fake(n_rec=7)
    ds._apply_scaling_subsample(3, None, None, seed=0)
    assert len(ds) == len(ds._fif_paths) == 3


# ---------------------------------------------------------------- combined


def test_axes_compose_independently():
    ds = _fake(n_rec=20, n_win=60, n_windows=4)
    ds._apply_scaling_subsample(6, 12, 200, seed=3)
    assert len(ds._fif_paths) == 6
    assert len(ds._allowed_anchor_idx) == 6
    assert all(len(a) == 12 for a in ds._allowed_anchor_idx)
    assert len(ds) == 200


# ------------------------------------- epoch_size smaller than the cohort


def _index_sampler(ds, n_draws, seed=0):
    """Indices __getitem__ actually resolves to, mimicking the DataLoader:
    it only ever emits indices in [0, len(ds))."""
    torch.manual_seed(seed)
    n_rec = len(ds._fif_paths)
    seen = []
    for i in range(n_draws):
        idx = i % len(ds)
        if ds._epoch_size is not None and ds._epoch_size < n_rec:
            idx = int(torch.randint(0, n_rec, (1,)).item())
        else:
            idx = idx % n_rec
        seen.append(idx)
    return seen


def test_epoch_size_smaller_than_cohort_still_reaches_every_recording():
    """The E0.3 extension needs epoch_size=703 with 1863 recordings, to hold
    steps matched at 4400. Plain `idx % n` is the IDENTITY there -- __len__
    reports 703 so the DataLoader never emits an index above 702, and
    recordings 703..1862 would never be sampled. That silently turns an S=1863
    cell into an S=703 one that still looks like a valid data point."""
    ds = _fake(n_rec=50, n_win=20, n_windows=2)
    ds._apply_scaling_subsample(None, None, 10, seed=0)
    assert len(ds) == 10 and len(ds._fif_paths) == 50
    seen = set(_index_sampler(ds, 4000))
    assert len(seen) == 50, f"only {len(seen)}/50 recordings ever sampled"


def test_epoch_size_at_least_cohort_keeps_exact_wrapping():
    """The case E0.3 actually ran must not change: with epoch_size == n_rec,
    an epoch covers each recording exactly once."""
    ds = _fake(n_rec=20, n_win=20, n_windows=2)
    ds._apply_scaling_subsample(None, None, 20, seed=0)
    seen = _index_sampler(ds, 20)
    assert sorted(seen) == list(range(20))


def test_epoch_size_multiple_of_cohort_repeats_uniformly():
    ds = _fake(n_rec=10, n_win=20, n_windows=2)
    ds._apply_scaling_subsample(None, None, 30, seed=0)
    from collections import Counter
    counts = Counter(_index_sampler(ds, 30))
    assert set(counts) == set(range(10))
    assert set(counts.values()) == {3}


def test_getitem_itself_reaches_every_recording_when_epoch_size_is_smaller():
    """Drives the REAL __getitem__ rather than a re-implementation of its index
    logic -- a mimicking test would keep passing if the production code drifted.
    _load_clip is stubbed because the fake paths have no FIFs behind them."""
    ds = _fake(n_rec=40, n_win=12, n_windows=2)
    ds._apply_scaling_subsample(None, None, 8, seed=0)
    assert len(ds) == 8 and len(ds._fif_paths) == 40

    used = []

    def _spy(rec_idx, indices):
        used.append(rec_idx)
        return torch.zeros(len(indices), 4, 10)

    ds._load_clip = _spy
    torch.manual_seed(0)
    for i in range(3000):
        ds[i % len(ds)]            # exactly what a DataLoader can emit

    assert max(used) >= 8, (
        "no recording at or beyond epoch_size was ever loaded -- the "
        "S=1863-with-epoch_size=703 cell would silently be an S=703 cell")
    assert len(set(used)) == 40, f"only {len(set(used))}/40 recordings loaded"


def test_getitem_covers_each_recording_once_when_epoch_size_equals_cohort():
    """The configuration E0.3 actually ran must be unchanged: exact coverage."""
    ds = _fake(n_rec=12, n_win=12, n_windows=2)
    ds._apply_scaling_subsample(None, None, 12, seed=0)
    used = []
    ds._load_clip = lambda r, i: (used.append(r), torch.zeros(len(i), 4, 10))[1]
    for i in range(12):
        ds[i]
    assert sorted(used) == list(range(12))
