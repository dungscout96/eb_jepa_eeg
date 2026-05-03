"""Tests for eb_jepa.evaluation.bootstrap helpers.

Exercises the three numpy/scipy-only helpers (_safe_corr, _safe_auc,
_summarize) directly. These are the primitives every higher-level
bootstrap function (movie / movie_id / subject) builds on.
"""

import numpy as np
import pytest

from eb_jepa.evaluation.bootstrap import _safe_auc, _safe_corr, _summarize


class TestSafeCorr:
    def test_identity_correlation_is_one(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        assert _safe_corr(x, x) == pytest.approx(1.0, abs=1e-9)

    def test_negation_correlation_is_minus_one(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(200)
        assert _safe_corr(x, -x) == pytest.approx(-1.0, abs=1e-9)

    def test_returns_nan_when_truth_is_constant(self):
        x = np.zeros(50)
        y = np.arange(50, dtype=float)
        assert np.isnan(_safe_corr(x, y))

    def test_returns_nan_when_pred_is_constant(self):
        x = np.arange(50, dtype=float)
        y = np.full(50, 3.14)
        assert np.isnan(_safe_corr(x, y))


class TestSafeAuc:
    def test_perfect_separation_is_one(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert _safe_auc(y, s) == pytest.approx(1.0)

    def test_random_score_is_near_half(self):
        rng = np.random.default_rng(2)
        n = 2000
        y = rng.integers(0, 2, n)
        s = rng.random(n)
        assert abs(_safe_auc(y, s) - 0.5) < 0.05

    def test_returns_nan_when_only_one_class(self):
        y = np.zeros(20, dtype=int)
        s = np.linspace(0, 1, 20)
        assert np.isnan(_safe_auc(y, s))


class TestSummarize:
    def test_empty_input_returns_nan_summary(self):
        out = _summarize([])
        assert out["n"] == 0
        assert np.isnan(out["mean"])
        assert np.isnan(out["std"])
        assert np.isnan(out["ci_lo"])
        assert np.isnan(out["ci_hi"])

    def test_filters_nans_from_summary(self):
        out = _summarize([1.0, 2.0, 3.0, float("nan")])
        assert out["n"] == 3
        assert out["mean"] == pytest.approx(2.0)

    def test_ci_brackets_mean_and_widens_with_alpha(self):
        rng = np.random.default_rng(3)
        samples = list(rng.standard_normal(2000))
        narrow = _summarize(samples, alpha=0.5)  # 25th-75th percentile
        wide = _summarize(samples, alpha=0.05)   # 2.5th-97.5th percentile

        assert narrow["ci_lo"] < narrow["mean"] < narrow["ci_hi"]
        assert wide["ci_lo"] < narrow["ci_lo"]
        assert wide["ci_hi"] > narrow["ci_hi"]

    def test_ci_width_shrinks_with_more_samples(self):
        """Mean CI width across replicates: more samples -> tighter quantiles
        on average. A single 50-sample CI can happen to be narrower than a
        5000-sample CI; averaging removes the seed sensitivity."""
        rng = np.random.default_rng(4)

        def avg_width(n_samples, replicates=30):
            widths = []
            for _ in range(replicates):
                s = _summarize(list(rng.standard_normal(n_samples)))
                widths.append(s["ci_hi"] - s["ci_lo"])
            return float(np.mean(widths))

        # Theoretical 95% CI width for N(0,1) is ~3.92; with N=20 the quantile
        # estimator is much noisier and averages substantially narrower than
        # the asymptote (typical mean ~3.0). With N=5000 it sits near 3.92.
        assert avg_width(20) < avg_width(5000)
