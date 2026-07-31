"""Recovery tests for the literature noise-ceiling estimators.

Each estimator is checked against synthetic data with a known ground-truth
signal fraction, and the three are checked against each other. The point of
implementing all three is that they should agree; these tests establish that
they do on data where the answer is known.
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


nc = _load("noise_ceiling")
measure_isc = _load("measure_isc")


def _synth(rho1, n_trials=80, n_obs=3000, seed=0):
    """x_n = s + e_n with Var(s)/(Var(s)+Var(e)) == rho1 in expectation."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(n_obs) * math.sqrt(rho1)
    e = rng.standard_normal((n_trials, n_obs)) * math.sqrt(1.0 - rho1)
    return s[None, :] + e


# --- Sahani & Linden (2003) ------------------------------------------------

@pytest.mark.parametrize("rho1", [0.02, 0.05, 0.10, 0.30, 0.60])
def test_sahani_linden_recovers_explainable_fraction(rho1):
    out = nc.sahani_linden_power(_synth(rho1, seed=int(rho1 * 1000)))
    assert out["explainable_fraction"] == pytest.approx(rho1, abs=0.015)
    assert out["SP"] + out["NP"] == pytest.approx(out["TP"], rel=1e-9)


def test_sahani_linden_is_unbiased_at_zero_signal():
    """With no shared signal SP must average ~0, and may go slightly negative.
    An estimator that clipped at 0 would bias every ceiling upward."""
    ests = [
        nc.sahani_linden_power(_synth(0.0, n_trials=20, n_obs=500, seed=s))["SP"]
        for s in range(40)
    ]
    assert abs(float(np.mean(ests))) < 0.02
    assert min(ests) < 0, "estimator appears clipped; it must be unbiased"


def test_sahani_linden_rejects_single_trial():
    with pytest.raises(ValueError):
        nc.sahani_linden_power(np.zeros((1, 100)))


# --- Schoppe et al. (2016) -------------------------------------------------

@pytest.mark.parametrize("rho1", [0.02, 0.05, 0.10, 0.28])
@pytest.mark.parametrize("k", [1, 2, 10, 50])
def test_cc_max_equals_spearman_brown(rho1, k):
    """Schoppe's CC_max and the Spearman-Brown ceiling are the same quantity.

    This is why measure_isc.py's ceiling and this module's are interchangeable
    -- and why agreement between them checks the ESTIMATORS, not the algebra.
    """
    sp, np_ = rho1, 1.0 - rho1
    assert nc.cc_max(sp, np_, k) == pytest.approx(measure_isc.ceiling(rho1, k), rel=1e-12)


def test_cc_norm_is_one_for_a_perfect_model():
    rho1, n = 0.05, 1
    sp, np_ = rho1, 1.0 - rho1
    perfect = nc.cc_max(sp, np_, n)
    assert nc.cc_norm(perfect, sp, np_, n) == pytest.approx(1.0)


def test_cc_max_increases_with_repeats_and_stays_below_one():
    sp, np_ = 0.05, 0.95
    vals = [nc.cc_max(sp, np_, k) for k in (1, 2, 5, 20, 100, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 1.0


def test_cc_max_is_nan_without_signal():
    assert math.isnan(nc.cc_max(0.0, 1.0, 10))
    assert math.isnan(nc.cc_max(-0.01, 1.0, 10))


# --- Split-half / Hsu et al. (2004) ---------------------------------------

@pytest.mark.parametrize("rho1", [0.05, 0.10, 0.30])
def test_split_half_recovers_single_trial_rho1(rho1):
    out = nc.split_half_reliability(_synth(rho1, seed=7), n_repeats=40, seed=1)
    assert out["rho1_corrected"] == pytest.approx(rho1, abs=0.02)


def test_split_half_rejects_too_few_trials():
    with pytest.raises(ValueError):
        nc.split_half_reliability(np.zeros((3, 100)))


# --- Cross-estimator agreement (the actual point) --------------------------

@pytest.mark.parametrize("rho1", [0.02, 0.05, 0.10, 0.28])
def test_all_three_estimators_agree(rho1):
    """Sahani-Linden power ratio, split-half extrapolation, and mean pairwise
    ISC must land on the same rho1. They fail differently, so agreement on real
    data is meaningful evidence."""
    r = _synth(rho1, n_trials=80, n_obs=4000, seed=13)
    sl = nc.sahani_linden_power(r)["explainable_fraction"]
    sh = nc.split_half_reliability(r, n_repeats=40, seed=2)["rho1_corrected"]
    isc, _, _ = measure_isc.mean_pairwise_isc(r)
    assert sl == pytest.approx(rho1, abs=0.02)
    assert sh == pytest.approx(rho1, abs=0.02)
    assert isc == pytest.approx(rho1, abs=0.02)
    assert max(sl, sh, isc) - min(sl, sh, isc) < 0.02


@pytest.mark.parametrize("rho1", [0.02, 0.05, 0.10, 0.30])
def test_standardised_sahani_linden_equals_mean_pairwise_isc(rho1):
    """After per-repeat z-scoring the two estimators are the SAME quantity.

    This is what explains the raw-vs-ISC gap on real data: Sahani-Linden is
    scale-sensitive across repeats, Pearson is not.
    """
    r = _synth(rho1, n_trials=60, n_obs=3000, seed=31)
    sl = nc.sahani_linden_power(nc.standardize_rows(r))["explainable_fraction"]
    isc, _, _ = measure_isc.mean_pairwise_isc(r)
    assert sl == pytest.approx(isc, abs=1e-6)


def test_heterogeneous_subject_gain_splits_the_two_estimators():
    """Per-subject amplitude differences deflate raw SL but leave ISC alone --
    the mechanism behind the discrepancy observed on HBN."""
    r = _synth(0.10, n_trials=60, n_obs=3000, seed=33)
    gains = np.exp(np.linspace(-1.5, 1.5, r.shape[0]))[:, None]
    scaled = r * gains

    isc_plain, _, _ = measure_isc.mean_pairwise_isc(r)
    isc_scaled, _, _ = measure_isc.mean_pairwise_isc(scaled)
    raw = nc.sahani_linden_power(scaled)["explainable_fraction"]
    fixed = nc.sahani_linden_power(nc.standardize_rows(scaled))["explainable_fraction"]

    assert isc_scaled == pytest.approx(isc_plain, abs=1e-9)   # ISC unaffected
    assert raw < 0.6 * isc_scaled                              # raw SL deflated
    assert fixed == pytest.approx(isc_scaled, abs=1e-6)        # standardising restores it


def test_summarise_returns_all_three_and_normalises_observed():
    r = _synth(0.05, seed=21)
    out = nc.summarise(r, observed_cc={"probe": 0.1517}, n_repeats=20)
    assert set(out) >= {"sahani_linden", "split_half_hsu", "schoppe_cc_max_by_k",
                        "schoppe_cc_norm"}
    assert out["schoppe_cc_max_by_k"]["1"] == pytest.approx(math.sqrt(0.05), abs=0.02)
    assert out["schoppe_cc_norm"]["probe"] > 0
