"""Literature-standard noise-ceiling estimators, for cross-checking rho1.

Three independent routes to the same underlying quantity -- the fraction of
response variance that is stimulus-locked rather than trial-specific. Agreement
between them is evidence the ceiling in ``measure_isc.py`` is real; disagreement
is diagnostic.

    1. Sahani & Linden (2003), "How linear are auditory cortical responses?"
       NeurIPS 15. Decomposes total power into signal power (SP) and noise
       power (NP) from repeated presentations. The "explainable variance
       fraction" is SP/TP.

    2. Split-half reliability with Spearman-Brown correction, the normalisation
       shared by Hsu, Borst & Theunissen (2004), "Quantifying variability and
       prediction accuracy in electrophysiological data", Network 15(2).

    3. Schoppe, Harper, Willmore, King & Schnupp (2016), "Measuring the
       Performance of Neural Models", Front. Comput. Neurosci. 10:10. Defines
       CC_max, the highest correlation any model can achieve against a
       finite-trial mean, and CC_norm = CC_abs / CC_max.

IMPORTANT -- what "trial" means here
------------------------------------
All three papers assume *repeated presentations of the same stimulus to the
same subject*, so that between-presentation variability is measurement noise.
HBN movie-watching gives exactly ONE presentation per subject, so we use
**subjects as the repeat axis**: the "signal" is the response component shared
across subjects, and the subject fingerprint is absorbed into the noise term.

That is the correct choice for a stimulus-decoding ceiling -- we want to bound
how well the *stimulus* can be read out, and subject-specific structure is
precisely what must not count. It is also what makes these estimators
commensurable with the inter-subject correlation in ``measure_isc.py``. But it
IS a departure from the papers' setting and must be stated when citing them:
these are cross-subject, not cross-trial, noise ceilings.

Consequence worth knowing: because the fingerprint is ~96% of single-trial
variance in this data, the cross-subject ceiling is far lower than a
within-subject repeated-trial ceiling would be. Papers reporting the latter
(e.g. trial-averaged image-decoding work) are not directly comparable.

Relationship to Spearman-Brown
------------------------------
Schoppe's CC_max and the Spearman-Brown ceiling used in ``measure_isc.py`` are
algebraically the same quantity -- see ``test_noise_ceiling.py``, which asserts
it. Implementing both is not redundant: it checks the *power-based* estimate of
rho1 against the *correlation-based* one, which fail in different ways.
"""

import math

import numpy as np


def standardize_rows(r: np.ndarray) -> np.ndarray:
    """Z-score each repeat independently, so per-subject scale cannot count
    as noise.

    Why this matters here. Sahani-Linden is scale-*sensitive* across repeats:
    a subject with unusually large response variance inflates the average
    per-trial variance (TP) and therefore deflates SP/TP. Pearson correlation
    -- and hence mean pairwise ISC -- is scale-*invariant*. With heterogeneous
    subject amplitudes (different impedance, different alpha power) the two
    estimators therefore disagree, and the gap between them measures the
    amplitude heterogeneity rather than any disagreement about signal.

    After row standardisation the two coincide exactly. With Var(x_n) = 1 for
    every n we get TP = 1 and Var(mean) = (1 + (N-1)*rho)/N for mean pairwise
    correlation rho, so

        SP = (N*Var(mean) - 1)/(N - 1) = rho

    i.e. ``sahani_linden_power(standardize_rows(r))["explainable_fraction"]``
    IS the mean pairwise ISC. ``test_noise_ceiling.py`` asserts this.

    Which one to quote depends on the downstream model. If the pipeline
    normalises per recording (as this repo's ``norm_mode=per_recording``
    does), the encoder never sees per-subject scale, so the scale-invariant
    estimate is the correct ceiling. Quote the raw one only for a model fed
    un-normalised amplitudes.
    """
    r = np.asarray(r, dtype=np.float64)
    mu = r.mean(axis=1, keepdims=True)
    sd = r.std(axis=1, ddof=1, keepdims=True)
    return (r - mu) / np.where(sd > 0, sd, 1.0)


def sahani_linden_power(r: np.ndarray) -> dict:
    """Signal / noise power decomposition, Sahani & Linden (2003).

    Parameters
    ----------
    r : ndarray [n_trials, n_observations]
        One row per repeat (here: per subject), one column per observation
        (here: per movie anchor, or per time sample).

    Returns
    -------
    dict with TP (total power), SP (signal power), NP (noise power) and
    ``explainable_fraction`` = SP/TP -- the fraction of the response variance
    that is reproducible across repeats.

    The estimator is unbiased: writing x_n = s + e_n with Var(s) = SP and
    Var(e) = NP, the average per-trial variance estimates TP = SP + NP, while
    the variance of the mean estimates SP + NP/N. Solving the pair gives

        SP_hat = (N * Var(mean) - <Var(trial)>) / (N - 1)

    ``explainable_fraction`` can come out slightly negative when the true SP is
    near zero and N is small; that is expected of an unbiased estimator and is
    reported rather than clipped, because silently clipping to 0 would bias a
    ceiling upward.
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"expected [n_trials, n_obs], got shape {r.shape}")
    n_trials = r.shape[0]
    if n_trials < 2:
        raise ValueError(f"need >= 2 trials, got {n_trials}")

    total_power = float(r.var(axis=1, ddof=1).mean())
    var_of_mean = float(r.mean(axis=0).var(ddof=1))
    signal_power = (n_trials * var_of_mean - total_power) / (n_trials - 1)
    noise_power = total_power - signal_power
    return {
        "n_trials": int(n_trials),
        "TP": total_power,
        "SP": float(signal_power),
        "NP": float(noise_power),
        "explainable_fraction": float(signal_power / total_power)
        if total_power > 0 else float("nan"),
    }


def cc_max(signal_power: float, noise_power: float, n_trials: int) -> float:
    """Schoppe et al. (2016) CC_max: the ceiling on correlation with an
    ``n_trials``-repeat mean.

    A perfect model predicts the noise-free signal s, but is scored against the
    measured mean, which still carries NP/N of noise. Hence

        CC_max = sqrt( N*SP / (N*SP + NP) )

    which is exactly sqrt of the Spearman-Brown reliability at rho1 = SP/(SP+NP).
    """
    denom = n_trials * signal_power + noise_power
    if signal_power <= 0 or denom <= 0:
        return float("nan")
    return math.sqrt(n_trials * signal_power / denom)


def cc_norm(cc_abs: float, signal_power: float, noise_power: float,
            n_trials: int) -> float:
    """Schoppe et al. (2016) normalised correlation, CC_abs / CC_max.

    CC_norm = 1 means the model is as good as the data allow. Values above 1
    indicate the ceiling was underestimated (too few repeats, or SP estimated
    on data the model also saw).
    """
    ceiling = cc_max(signal_power, noise_power, n_trials)
    if not math.isfinite(ceiling) or ceiling <= 0:
        return float("nan")
    return cc_abs / ceiling


def split_half_reliability(r: np.ndarray, n_repeats: int = 50,
                           seed: int = 0) -> dict:
    """Split-half reliability, Spearman-Brown corrected to a single trial.

    The normalisation shared by Hsu, Borst & Theunissen (2004). Randomly halves
    the repeats, correlates the two half-means, and averages over
    ``n_repeats`` random splits.

    A split-half correlation r_hh is the reliability of a (N/2)-trial average,
    so inverting Spearman-Brown recovers the single-trial value

        rho1 = r_hh / (k - (k-1) * r_hh),      k = N/2

    Returns the raw split-half correlation and the corrected single-trial
    reliability. This is an estimator of the same rho1 that mean pairwise ISC
    estimates, by a different route -- they should agree.
    """
    r = np.asarray(r, dtype=np.float64)
    n_trials = r.shape[0]
    if n_trials < 4:
        raise ValueError(f"need >= 4 trials to split, got {n_trials}")
    rng = np.random.default_rng(seed)
    half = n_trials // 2

    corrs = []
    for _ in range(n_repeats):
        perm = rng.permutation(n_trials)
        a = r[perm[:half]].mean(axis=0)
        b = r[perm[half:2 * half]].mean(axis=0)
        a = a - a.mean()
        b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            corrs.append(float(a @ b / (na * nb)))
    if not corrs:
        return {"split_half_r": float("nan"), "rho1_corrected": float("nan"),
                "n_splits": 0, "half_size": half}

    r_hh = float(np.mean(corrs))
    denom = half - (half - 1) * r_hh
    rho1 = r_hh / denom if denom != 0 else float("nan")
    return {
        "split_half_r": r_hh,
        "split_half_std": float(np.std(corrs)),
        "rho1_corrected": float(rho1),
        "n_splits": len(corrs),
        "half_size": half,
    }


def summarise(r: np.ndarray, observed_cc: dict[str, float] | None = None,
              n_repeats: int = 50, seed: int = 0,
              ks=(1, 2, 5, 10, 20, 50, 100)) -> dict:
    """Run all three estimators on one [n_trials, n_obs] response matrix."""
    sl = sahani_linden_power(r)
    sh = split_half_reliability(r, n_repeats=n_repeats, seed=seed)
    out = {
        "sahani_linden": sl,
        "split_half_hsu": sh,
        "schoppe_cc_max_by_k": {
            str(k): cc_max(sl["SP"], sl["NP"], k) for k in ks
        },
    }
    if observed_cc:
        out["schoppe_cc_norm"] = {
            name: cc_norm(v, sl["SP"], sl["NP"], 1)
            for name, v in observed_cc.items()
        }
    return out
