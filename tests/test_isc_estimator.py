"""Recovery tests for the rho1 estimator in experiments/snr_scaling/measure_isc.py.

The whole ceiling argument rests on "mean pairwise ISC == rho1", so that
identity is tested against synthetic data with a known ground-truth rho1
rather than assumed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "measure_isc",
    Path(__file__).resolve().parents[1] / "experiments" / "snr_scaling" / "measure_isc.py",
)
measure_isc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(measure_isc)


def _synth(rho1, n_subjects=60, n_obs=4000, seed=0):
    """x_s = g + n_s with Var(g)/(Var(g)+Var(n)) == rho1 exactly in expectation."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n_obs) * np.sqrt(rho1)
    n = rng.standard_normal((n_subjects, n_obs)) * np.sqrt(1.0 - rho1)
    return g[None, :] + n


@pytest.mark.parametrize("rho1", [0.02, 0.05, 0.10, 0.28, 0.5])
def test_pairwise_isc_recovers_rho1(rho1):
    est, _, n_pairs = measure_isc.mean_pairwise_isc(_synth(rho1))
    assert n_pairs == 60 * 59 // 2
    assert est == pytest.approx(rho1, abs=0.01)


def test_isc_is_zero_without_shared_signal():
    rng = np.random.default_rng(1)
    est, _, _ = measure_isc.mean_pairwise_isc(rng.standard_normal((40, 3000)))
    assert abs(est) < 0.02


def test_isc_is_scale_invariant_per_subject():
    """Pearson must not care about per-recording gain -- impedance differences
    between subjects must not masquerade as reliability."""
    x = _synth(0.1, seed=3)
    gains = np.linspace(0.1, 10.0, x.shape[0])[:, None]
    plain, _, _ = measure_isc.mean_pairwise_isc(x)
    scaled, _, _ = measure_isc.mean_pairwise_isc(x * gains)
    assert scaled == pytest.approx(plain, abs=1e-9)


def test_zero_variance_subject_is_dropped_not_nan():
    x = _synth(0.1, n_subjects=20, seed=4)
    x[3] = 0.0
    est, _, n_pairs = measure_isc.mean_pairwise_isc(x)
    assert np.isfinite(est)
    assert n_pairs == 19 * 18 // 2


def test_spearman_brown_matches_direct_averaging():
    """R(K) must predict the reliability of an actual K-subject average."""
    rho1, k = 0.05, 8
    rng = np.random.default_rng(5)
    n_obs, n_groups = 20000, 40
    g = rng.standard_normal(n_obs) * np.sqrt(rho1)
    # Each "group" is an independent average of k subjects.
    groups = np.stack([
        g + rng.standard_normal((k, n_obs)).mean(axis=0) * np.sqrt(1.0 - rho1)
        for _ in range(n_groups)
    ])
    measured, _, _ = measure_isc.mean_pairwise_isc(groups)
    assert measured == pytest.approx(measure_isc.reliability(rho1, k), abs=0.02)


def test_ceiling_is_sqrt_of_reliability_and_monotone():
    assert measure_isc.ceiling(0.05, 1) == pytest.approx(np.sqrt(0.05))
    ks = [1, 2, 5, 10, 50, 100]
    vals = [measure_isc.ceiling(0.05, k) for k in ks]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 1.0


def test_band_log_power_isolates_the_right_band():
    """A pure 10 Hz sinusoid must land in alpha, not delta/theta or beta."""
    sfreq, n_times = 200.0, 400
    t = np.arange(n_times) / sfreq
    x = np.sin(2 * np.pi * 10.0 * t)[None, None, None, :]
    p = measure_isc.band_log_power(x, sfreq)[0, 0, 0]
    names = list(measure_isc.BANDS)
    assert names[int(np.argmax(p[:3]))] == "alpha"


def test_flat_reference_channel_would_fake_a_huge_isc():
    """Regression: the EGI 'Cz' reference is stored as exact zeros, and the
    same denormal residue appears in EVERY recording. Z-scoring it against an
    epsilon floor turns that shared bit pattern into a signal, yielding a
    spurious ISC near 1. This is why load_aligned drops flat channels BEFORE
    normalising -- CorrCA maximises across-subject correlation and would
    otherwise select this channel as its top component.
    """
    n_rec, n_obs = 30, 2000
    rng = np.random.default_rng(11)
    # One deterministic pattern, identical across recordings, at denormal scale.
    residue = rng.standard_normal(n_obs) * 3.45e-29
    flat = np.tile(residue, (n_rec, 1))

    z = (flat - flat.mean(axis=1, keepdims=True)) / np.maximum(
        flat.std(axis=1, keepdims=True), 1e-8)
    spurious, _, _ = measure_isc.mean_pairwise_isc(z)
    assert spurious > 0.9, "the failure mode this guard exists for is gone"

    # The guard: median std is ~1e-29 vs ~1.0 for real channels, so the
    # ratio test excludes it by many orders of magnitude.
    stds = np.stack([flat.std(axis=1), rng.standard_normal((n_rec, n_obs)).std(axis=1)], axis=1)
    med = np.median(stds, axis=0)
    good = med > 1e-6 * float(np.median(med))
    assert not good[0] and good[1]


def test_corrca_covariances_recover_a_planted_spatial_filter():
    """CorrCA must find the direction carrying the shared response."""
    rng = np.random.default_rng(7)
    n_rec, n_anchors, n_chans, n_times = 24, 30, 8, 100
    w_true = np.zeros(n_chans)
    w_true[2] = 1.0
    shared = rng.standard_normal((n_anchors, n_times))
    x = rng.standard_normal((n_rec, n_anchors, n_chans, n_times)).astype(np.float32)
    x += (w_true[None, None, :, None] * shared[None, :, None, :] * 3.0).astype(np.float32)

    r_b, r_w = measure_isc.corrca_covariances(x)
    w, eigs = measure_isc.solve_corrca_eigenproblem(r_b, r_w, n_components=2)
    top = np.abs(w[:, 0])
    assert int(np.argmax(top)) == 2
    assert eigs[0] > eigs[1]
