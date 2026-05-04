"""Tests for eb_jepa.preprocessing.corrca.solve_corrca_eigenproblem.

Exercises the linear-algebra core that compute_corrca relies on -- the
data-loading half (mne / preprocessed FIFs) needs cluster fixtures and
is covered by integration runs, not unit tests.
"""

import numpy as np
import pytest

from eb_jepa.preprocessing import solve_corrca_eigenproblem


def test_recovers_known_stimulus_direction():
    """Synthetic data: K subjects share a 1-D stimulus signal along the
    first channel; rest is independent per-subject noise. The top CorrCA
    filter should put almost all weight on channel 0."""
    rng = np.random.default_rng(0)
    K, C, T = 12, 8, 500

    stim = rng.standard_normal(T)            # shared stimulus signal
    X = []
    for _ in range(K):
        noise = rng.standard_normal((C, T)) * 0.1
        sig = np.zeros((C, T))
        sig[0] = stim                        # channel 0 carries the stimulus
        X.append(sig + noise)
    X = np.stack(X)                           # [K, C, T]

    # Standard CorrCA matrices (matches compute_corrca's loop math).
    R_w = np.zeros((C, C))
    for i in range(K):
        R_w += X[i] @ X[i].T / T
    R_w /= K

    X_sum = X.sum(axis=0)
    R_b = X_sum @ X_sum.T / T
    for i in range(K):
        R_b -= X[i] @ X[i].T / T
    R_b /= K * (K - 1)

    W, isc = solve_corrca_eigenproblem(R_b, R_w, n_components=3)

    assert W.shape == (C, 3)
    assert isc.shape == (3,)
    # Eigenvalues are sorted descending.
    assert isc[0] >= isc[1] >= isc[2]
    # Top filter is concentrated on channel 0.
    top = W[:, 0]
    top = top / np.linalg.norm(top)
    assert abs(top[0]) > 0.95


def test_eigenvalues_sorted_descending():
    """Output ordering contract."""
    rng = np.random.default_rng(1)
    C = 6
    A = rng.standard_normal((C, C))
    R_b = A @ A.T + np.eye(C)
    B = rng.standard_normal((C, C))
    R_w = B @ B.T + np.eye(C)

    _, isc = solve_corrca_eigenproblem(R_b, R_w, n_components=C)
    assert all(isc[i] >= isc[i + 1] for i in range(C - 1))


def test_n_components_clipping():
    """Asking for fewer components returns a narrower W."""
    rng = np.random.default_rng(2)
    C = 5
    A = rng.standard_normal((C, C))
    R_b = A @ A.T + np.eye(C)
    R_w = np.eye(C) * 2.0

    W2, isc2 = solve_corrca_eigenproblem(R_b, R_w, n_components=2)
    W4, isc4 = solve_corrca_eigenproblem(R_b, R_w, n_components=4)

    assert W2.shape == (C, 2)
    assert W4.shape == (C, 4)
    # The first 2 components match across calls (ordering is deterministic).
    np.testing.assert_allclose(isc2, isc4[:2])


def test_ridge_handles_singular_R_w():
    """Singular R_w would blow up an unregularized solve; ridge keeps it
    finite."""
    rng = np.random.default_rng(3)
    C = 4
    R_w = np.zeros((C, C))                   # rank-zero -- must rely on ridge
    R_b = rng.standard_normal((C, C))
    R_b = R_b @ R_b.T

    W, isc = solve_corrca_eigenproblem(R_b, R_w, n_components=2, ridge=1e-3)

    assert np.all(np.isfinite(W))
    assert np.all(np.isfinite(isc))
