"""Tests for the E0.2 K-averaging measurement.

The load-bearing claim is that ``empirical_reliability`` measures R(K) without
assuming Spearman-Brown, so that comparing the two is a real validation rather
than a tautology. These tests check it recovers R(K) on data with a planted
rho1, and that it degrades safely.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "snr_scaling"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ka = _load("k_averaging")


def _synth_embeddings(rho1, n_rec=64, n_anchors=101, n_dim=16, seed=0):
    """X[r,a,d] = g[a,d] + noise, with per-dimension signal fraction rho1."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_anchors, n_dim)) * math.sqrt(rho1)
    n = rng.standard_normal((n_rec, n_anchors, n_dim)) * math.sqrt(1.0 - rho1)
    return (g[None] + n).astype(np.float32)


def _sb(rho1, k):
    return k * rho1 / (1.0 + (k - 1) * rho1)


@pytest.mark.parametrize("rho1", [0.05, 0.10, 0.30])
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_empirical_reliability_recovers_spearman_brown(rho1, k):
    """The measurement must land on the prediction when the model holds.

    This is the check that makes the E0.2 comparison meaningful: if these two
    could not agree on synthetic data, agreement on real data would prove
    nothing.
    """
    X = _synth_embeddings(rho1, n_rec=64, n_anchors=400, n_dim=32, seed=k)
    rng = np.random.default_rng(1)
    out = ka.empirical_reliability(X, [k], n_draws=40, rng=rng)
    assert out[str(k)]["mean"] == pytest.approx(_sb(rho1, k), abs=0.03)


def test_empirical_reliability_skips_k_that_would_overlap():
    """2K > n_recordings cannot give disjoint halves; those K must be skipped,
    not silently computed on overlapping subject sets."""
    X = _synth_embeddings(0.1, n_rec=10, n_anchors=200, n_dim=8, seed=3)
    out = ka.empirical_reliability(X, [1, 2, 5, 6, 8], n_draws=5,
                                   rng=np.random.default_rng(0))
    assert set(out) == {"1", "2", "5"}


def test_empirical_reliability_is_zero_without_shared_signal():
    X = _synth_embeddings(0.0, n_rec=40, n_anchors=300, n_dim=16, seed=4)
    out = ka.empirical_reliability(X, [4], n_draws=30, rng=np.random.default_rng(2))
    assert abs(out["4"]["mean"]) < 0.05


def test_empirical_reliability_increases_with_k():
    X = _synth_embeddings(0.08, n_rec=64, n_anchors=400, n_dim=32, seed=5)
    out = ka.empirical_reliability(X, [1, 2, 4, 8, 16], n_draws=30,
                                   rng=np.random.default_rng(3))
    means = [out[str(k)]["mean"] for k in (1, 2, 4, 8, 16)]
    assert all(b > a for a, b in zip(means, means[1:]))


def test_predicted_ceiling_matches_sqrt_spearman_brown():
    ks = [1, 2, 10, 50]
    pred = ka.predicted_ceiling(0.05, ks)
    for k in ks:
        assert pred[str(k)] == pytest.approx(math.sqrt(_sb(0.05, k)))


def test_predicted_ceiling_is_nan_for_degenerate_rho1():
    for bad in (0.0, -0.1, float("nan")):
        assert math.isnan(ka.predicted_ceiling(bad, [1, 4])["4"])


def test_probe_curve_uses_disjoint_draws_and_single_draw_at_full_n():
    """At K == n_recordings only one draw exists; the code must not pretend to
    have n_draws independent samples (which would fake a tiny error bar)."""
    n_rec, n_anchors, n_dim = 8, 50, 4
    rng = np.random.default_rng(0)
    X = _synth_embeddings(0.2, n_rec=n_rec, n_anchors=n_anchors, n_dim=n_dim, seed=6)
    Y = X.mean(axis=0)[:, :1]      # a feature the embedding can predict

    class _Id:
        def transform(self, a):
            return a

    class _Reg:
        def predict(self, a):
            return a[:, 0]

    out = ka.probe_curve(lambda sel: X[sel].mean(axis=0), n_rec,
                         [2, n_rec], n_draws=7, rng=rng,
                         models={"luminance_mean": _Reg()}, scaler=_Id(),
                         Y=Y, feature_names=["luminance_mean"])
    assert out["2"]["n_draws"] == 7
    assert out[str(n_rec)]["n_draws"] == 1
    assert out[str(n_rec)]["std_r"] == 0.0


def test_probe_curve_skips_k_above_n_recordings():
    n_rec = 6
    X = _synth_embeddings(0.2, n_rec=n_rec, n_anchors=40, n_dim=4, seed=7)
    Y = X.mean(axis=0)[:, :1]

    class _Id:
        def transform(self, a):
            return a

    class _Reg:
        def predict(self, a):
            return a[:, 0]

    out = ka.probe_curve(lambda sel: X[sel].mean(axis=0), n_rec, [2, 6, 16],
                         n_draws=3, rng=np.random.default_rng(0),
                         models={"luminance_mean": _Reg()}, scaler=_Id(),
                         Y=Y, feature_names=["luminance_mean"])
    assert set(out) == {"2", "6"}


def test_align_anchors_joins_on_movie_time_not_window_index():
    """Recordings with different dropped prefixes must still align by movie
    time -- the failure this join exists to prevent."""
    import torch

    class _DS:
        # rec 0 starts at t=0.0, rec 1 has its first window dropped.
        t_start_recordings = [
            torch.tensor([0.0, 2.0, 4.0, 6.0]),
            torch.tensor([2.0, 4.0, 6.0]),
        ]
        _recording_metadata = [{"subject": "A"}, {"subject": "B"}]

    anchors, idx, subjects = ka.align_anchors(_DS())
    assert anchors == [2000, 4000, 6000]
    # Same movie time maps to DIFFERENT window indices in the two recordings.
    assert idx[0].tolist() == [1, 2, 3]
    assert idx[1].tolist() == [0, 1, 2]
    assert subjects == ["A", "B"]
