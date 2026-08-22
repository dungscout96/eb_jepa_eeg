"""Tests for the cached per-epoch selector.

Two failures are guarded here, both silent.

The first is the refactor that makes the selector affordable. Reading and
normalising a recording's windows was inlined in
``embed_recording_all_windows``; an epoch sweep needs it separately so the
result can be cached across checkpoints. If the split path normalises even
slightly differently from the original, EVERY number the selector produces
shifts, and nothing about the output would look wrong.

The second is the multi-target ridge in ``probe_score``. Features are grouped by
identical NaN mask and fitted as one solve for speed. That is only legitimate if
it is arithmetically identical to fitting each feature on its own valid rows,
which is what ``probe_traintest.py`` does.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from eb_jepa.evaluation.clip_probe.probe import (
    embed_recording_all_windows,
    encode_windows,
    load_recording_windows,
)

_SPEC = importlib.util.spec_from_file_location(
    "epoch_curve",
    Path(__file__).resolve().parents[2] / "eb_jepa" / "evaluation" / "clip_probe"
    / "epoch_curve.py",
)
ec = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ec)

_ANA = importlib.util.spec_from_file_location(
    "analyse_epoch_curve",
    Path(__file__).resolve().parents[2] / "experiments" / "snr_scaling" / "src"
    / "analyse_epoch_curve.py",
)
ana = importlib.util.module_from_spec(_ANA)
_ANA.loader.exec_module(ana)


class _FakeDataset:
    """Minimum surface embed_recording_all_windows touches."""

    def __init__(self, raw, norm_mode="per_recording"):
        self._raw = raw
        self._crop_inds = [None]
        self._fif_paths = ["unused.fif"]
        self._norm_mode = norm_mode
        self._eeg_mean = torch.tensor(0.5)
        self._eeg_std = torch.tensor(2.0)
        self.feature_recordings = [torch.zeros(len(raw), 12)]


class _FakeEncoder:
    """Deterministic, order-preserving stand-in for the real encoder."""

    def encode_tokens(self, batch, mask=None):
        return batch

    def pool_to_windows(self, tokens):
        # [B, 1, C, T] -> [B, D, 1, 1, 1] with D = C, mean over time.
        return tokens.mean(dim=-1).transpose(1, 2)[..., None, None]


@pytest.fixture
def fake(monkeypatch):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(7, 5, 16)).astype(np.float32)
    monkeypatch.setattr(
        "eb_jepa.evaluation.clip_probe.probe._read_raw_windows",
        lambda path, crop_inds: raw,
    )
    return raw


@pytest.mark.parametrize("norm_mode", ["per_recording", "global"])
def test_split_load_encode_matches_the_original_path(fake, norm_mode):
    """The refactor must be bit-for-bit, not merely close.

    This is the whole basis for caching: the sweep encodes a tensor produced by
    load_recording_windows, while every previously published number came from
    embed_recording_all_windows. A drift here silently rewrites the comparison.
    """
    ds = _FakeDataset(fake, norm_mode)
    enc = _FakeEncoder()

    X_ref, Y_ref = embed_recording_all_windows(enc, ds, 0, "cpu", batch_size=3)
    X_split = encode_windows(enc, load_recording_windows(ds, 0), "cpu", batch_size=3)

    assert X_split.shape == X_ref.shape
    np.testing.assert_array_equal(X_split, X_ref)
    assert Y_ref.shape == (len(fake), 12)


def test_cached_tensor_is_reusable_across_encoders(fake):
    """Cache once, encode many -- the property the cost argument rests on."""
    ds = _FakeDataset(fake)
    cached = load_recording_windows(ds, 0)
    before = cached.clone()
    for _ in range(3):
        encode_windows(_FakeEncoder(), cached, "cpu", batch_size=4)
    torch.testing.assert_close(cached, before)


def test_batch_size_does_not_change_the_embedding(fake):
    ds = _FakeDataset(fake)
    cached = load_recording_windows(ds, 0)
    a = encode_windows(_FakeEncoder(), cached, "cpu", batch_size=1)
    b = encode_windows(_FakeEncoder(), cached, "cpu", batch_size=64)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_mask_groups_splits_by_nan_pattern():
    Y = np.zeros((6, 4))
    Y[0, 1] = np.nan          # feature 1 has its own mask
    Y[2, 3] = np.nan          # feature 3 has another
    groups = ec._mask_groups(Y)
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 1, 2]  # {0, 2} share the all-valid mask
    assert sorted(sum(groups.values(), [])) == [0, 1, 2, 3]


def test_grouped_ridge_equals_per_feature_ridge():
    """The speed trick must not change the number.

    probe_score fits every feature sharing a NaN mask in one multi-target solve.
    Ridge is separable across targets, so that is exactly per-feature fitting --
    this pins it, because a future switch to a solver that couples targets (or
    an accidental mask intersection) would silently bias every selection.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(7)
    # Both halves must clear probe_score's min-rows guard (100 fit, 10 score),
    # which is the same threshold probe_traintest.py applies.
    n, d, k = 400, 12, 4
    X = rng.normal(size=(n, d))
    W = rng.normal(size=(d, k))
    Y = X @ W + 0.1 * rng.normal(size=(n, k))
    Y[3, 2] = np.nan                       # one feature gets a different mask
    is_fit = np.zeros(n, dtype=bool)
    is_fit[: n // 2] = True

    got = ec.probe_score(X, {"Y": Y, "is_fit": is_fit}, alpha=1.0)
    assert len(got["probe_per_feature"]) == k

    # Reference: one fit per feature, on that feature's own valid rows.
    scaler = StandardScaler().fit(X[is_fit])
    Xn = scaler.transform(X)
    for i in range(k):
        valid = ~np.isnan(Y[:, i])
        tr, te = valid & is_fit, valid & ~is_fit
        pred = Ridge(alpha=1.0).fit(Xn[tr], Y[tr, i]).predict(Xn[te])
        want = float(np.corrcoef(pred, Y[te, i])[0, 1])
        assert got["probe_per_feature"][ec.SCALAR_FEATURES_DEFAULT[i]] == pytest.approx(
            want, abs=1e-10)


def test_probe_score_scores_on_held_out_recordings_only():
    """A head fitted and scored on the same rows would rank checkpoints by how
    well they memorise, which is the failure the fit/score split exists to
    prevent."""
    rng = np.random.default_rng(1)
    n, d = 300, 8
    X = rng.normal(size=(n, d))
    Y = np.repeat(rng.normal(size=(n, 1)), 12, axis=1)
    is_fit = np.zeros(n, dtype=bool)
    is_fit[:200] = True
    # Make the scored half's target pure noise: a genuine held-out score must
    # collapse toward 0 no matter how well the fit half is memorised.
    Y[~is_fit] = rng.normal(size=(n - 200, 12))
    out = ec.probe_score(X, {"Y": Y, "is_fit": is_fit}, alpha=1.0)
    assert abs(out["probe_mean_r"]) < 0.5


def test_smooth_window_one_is_identity():
    vals = [0.1, 0.9, 0.2, 0.8]
    assert ana.smooth(vals, 1) == vals


def test_argmax_epoch_reads_saved_checkpoints_only():
    """The probe selector never snaps: it only ever evaluates at checkpoints,
    so the epoch it returns is a checkpoint by construction."""
    curve = {"25": {"probe_mean_r": 0.10}, "50": {"probe_mean_r": 0.30},
             "75": {"probe_mean_r": 0.20}}
    ep, val = ana.argmax_epoch(curve, "probe_mean_r", window=1)
    assert (ep, val) == (50, 0.30)
    assert str(ep) in curve


def test_argmax_epoch_smoothing_can_move_the_pick():
    curve = {"25": {"probe_mean_r": 0.10}, "50": {"probe_mean_r": 0.31},
             "75": {"probe_mean_r": 0.30}, "100": {"probe_mean_r": 0.29}}
    assert ana.argmax_epoch(curve, "probe_mean_r", window=1)[0] == 50
    assert ana.argmax_epoch(curve, "probe_mean_r", window=3)[0] == 75
