"""E0.1 -- measure rho1, the single-trial inter-subject reliability of EEG.

Why this number decides the paper
---------------------------------
Model the EEG of subject *s* at movie moment *t* as

    x_s(t) = g(t) + n_s(t)

with ``g`` the stimulus response (identical across subjects, since they watch
the same movie) and ``n`` the subject fingerprint plus noise (independent
across subjects). Write

    rho1 = Var(g) / (Var(g) + Var(n))

Two facts make ``rho1`` the quantity that bounds every probe result in this
repo.

**1. The mean pairwise inter-subject correlation IS rho1.** For two subjects
i != j, ``Cov(x_i, x_j) = Var(g)`` because the noise terms are independent,
and each has variance ``Var(g) + Var(n)``, so ``corr(x_i, x_j) = rho1``
exactly. No modelling assumption beyond the decomposition -- we can just
measure it.

**2. Spearman-Brown gives the ceiling.** Averaging K subjects at the same
movie moment leaves ``g`` untouched and shrinks the noise variance by K, so
the averaged signal has reliability

    R(K) = K*rho1 / (1 + (K-1)*rho1)

A frozen V-JEPA-2 movie feature is a deterministic function of the stimulus
and therefore carries no measurement noise. The attenuation formula then
bounds the correlation any probe can achieve:

    r(K) <= sqrt(R(K)) * corr(g, y)  <=  sqrt(R(K))

That bound holds for *any* encoder and *any* training objective, so
``sqrt(R(K))`` is the ceiling against which every number in
RESULTS_jul7 / RESULTS_jul9 should be read. Our best from-scratch result
(r = 0.1517) implies rho1 >= 0.023; whether the true value is 0.02 (we are at
the ceiling) or 0.10 (we are at half of it) is exactly what this script
settles. See ``experiments/snr_scaling/PLAN.md`` section E0.1.

Three readout spaces
--------------------
``rho1`` is not one number -- it depends on what the downstream model is
allowed to read. We report all three, because they bound different things.

``waveform``
    Per-channel Pearson correlation between subject pairs over the
    concatenated window time-course. The classical Hasson / Parra ISC, and
    the number comparable to the literature.

``band``
    Per-channel, per-band log power computed *per 2 s window*, correlated
    across windows. This matches the probe's unit of analysis (one embedding
    per window) and separates the delta/theta vs alpha split that
    ``experiments.md`` lines 47-70 asserts as the mechanism.

``corrca``
    CorrCA finds the spatial filters that maximise across-subject
    correlation, so the top component's ISC is the *multivariate* reliability
    -- the one a ridge probe on a 512-d embedding can actually exploit. A
    per-channel number understates it, which would make the ceiling look
    falsely low. Filters are fit on one disjoint half of the subjects and
    evaluated on the other, because the training eigenvalue is upward-biased
    and an optimistic ceiling is worse than no ceiling.

Usage
-----
Data lives on the cluster; see ``_submit_isc.py`` for the Delta job.

    uv run --group eeg python experiments/snr_scaling/measure_isc.py \
        --split=test --task=ThePresent \
        --output=experiments/snr_scaling/isc_test_TP.json

Memory: holds a dense ``[n_recordings, n_anchors, n_chans, n_times]`` float32
array -- ~2.3 GB for R6 (108 recordings), ~6 GB for R5 (293). Use
``--max-recordings`` to cap it.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.logging import get_logger
from eb_jepa.preprocessing.corrca import solve_corrca_eigenproblem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_ceiling  # noqa: E402  (sibling module, not an installed package)

logger = get_logger(__name__)

# Measured probe results this ceiling is meant to bound, from
# RESULTS_jul7.md 3.4 / 4.3 (R6 test, mean Pearson r over 12 movie features).
OBSERVED_PROBE_R = {
    "random_init": 0.0527,
    "best_from_scratch_soft_tau0.05": 0.1517,
    "reve_warmstart": 0.1715,
}

# Bands. delta/theta is where experiments.md reports ISC 0.10-0.28; alpha is
# where the variance is but ISC < 0.05. Broadband is the 1-40 Hz reference.
BANDS = {
    "delta_theta": (1.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "broadband": (1.0, 40.0),
}

# Dataset construction knobs that must match the training configs, so the
# ceiling is comparable to the probe numbers it is meant to bound.
DATASET_CFG = {
    "annotation_duration_tolerance_s": 1.0,
    "max_recording_overshoot_s": 60.0,
    "norm_mode": "per_recording",
    "add_envelope": False,
    "corrca_filters": None,
}


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def mean_pairwise_isc(m: np.ndarray) -> tuple[float, float, int]:
    """Mean off-diagonal correlation of rows of ``m`` [n_subjects, n_obs].

    This is the rho1 estimator: each entry of the correlation matrix between
    two distinct subjects estimates Var(g)/(Var(g)+Var(n)).

    Returns (mean, std, n_pairs) over the upper triangle.
    """
    m = np.asarray(m, dtype=np.float64)
    m = m - m.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    # A subject with zero variance on this observable carries no information;
    # drop rather than divide by zero.
    keep = norm[:, 0] > 0
    m, norm = m[keep], norm[keep]
    if len(m) < 2:
        return float("nan"), float("nan"), 0
    c = (m / norm) @ (m / norm).T
    iu = np.triu_indices(len(c), k=1)
    vals = c[iu]
    return float(vals.mean()), float(vals.std()), int(vals.size)


def reliability(rho1: float, k: int) -> float:
    """Spearman-Brown reliability of a K-subject average."""
    if not math.isfinite(rho1) or rho1 <= 0:
        return float("nan")
    return k * rho1 / (1.0 + (k - 1) * rho1)


def ceiling(rho1: float, k: int) -> float:
    """Upper bound on |corr(K-subject average, a noiseless stimulus feature)|."""
    r = reliability(rho1, k)
    return math.sqrt(r) if math.isfinite(r) else float("nan")


def band_log_power(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Per-window log band power.

    ``x`` is [n_rec, n_anchors, n_chans, n_times]; returns
    [n_rec, n_anchors, n_chans, n_bands] in the key order of ``BANDS``.
    """
    n_times = x.shape[-1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    out = np.empty(x.shape[:3] + (len(BANDS),), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS.values()):
        sel = (freqs >= lo) & (freqs < hi)
        # +eps guards against an all-zero band on a flat/rejected channel.
        out[..., i] = np.log(spec[..., sel].sum(axis=-1) + 1e-20)
    return out


def corrca_covariances(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Between- and within-subject covariance, pooled over anchors.

    ``x`` is [n_rec, n_anchors, n_chans, n_times].

    Uses the identity ``sum_{i!=j} X_i X_j^T = S S^T - Q`` with
    ``S = sum_i X_i`` and ``Q = sum_i X_i X_i^T``, which avoids materialising
    the O(K^2) pairwise products.
    """
    n_rec, n_anchors, n_chans, n_times = x.shape
    r_b = np.zeros((n_chans, n_chans), dtype=np.float64)
    r_w = np.zeros((n_chans, n_chans), dtype=np.float64)
    for a in range(n_anchors):
        xa = x[:, a].astype(np.float64)            # [K, C, T]
        s = xa.sum(axis=0)                          # [C, T]
        q = np.einsum("kct,kdt->cd", xa, xa)        # sum_i X_i X_i^T
        r_b += (s @ s.T - q) / (n_rec * (n_rec - 1) * n_times)
        r_w += q / (n_rec * n_times)
    return r_b / n_anchors, r_w / n_anchors


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_aligned(split: str, task: str, window_size_seconds: float,
                 preprocessed_dir: str | None, max_recordings: int | None,
                 min_channel_std_ratio: float = 1e-6):
    """Return (x, subjects, anchors, sfreq, dropped_channels), x dense [R, A, C, T].

    Recordings are joined on *movie time* (``t_start_recordings``), not on
    window index -- the same join ``paired.py`` and ``retrieval.py`` use. We
    keep the intersection of anchors so the array is dense and every pairwise
    correlation is computed on identical movie moments.
    """
    ds = JEPAMovieDataset(
        split=split,
        n_windows=1,
        window_size_seconds=window_size_seconds,
        task=task,
        feature_names=[],          # no probe targets needed
        cfg=DATASET_CFG,
        preprocessed=True,
        preprocessed_dir=preprocessed_dir,
        recipe_mode=False,
    )
    n_rec = len(ds._fif_paths)
    if n_rec < 2:
        raise RuntimeError(f"Need >= 2 recordings to compute ISC, got {n_rec}.")

    # Round to ms to make float movie times hashable without drift.
    keyed = [
        {int(round(float(t) * 1000)): i for i, t in enumerate(ts)}
        for ts in ds.t_start_recordings
    ]
    subjects = [
        str((m or {}).get("subject", (m or {}).get("participant_id", f"__rec{i}")))
        for i, m in enumerate(ds._recording_metadata)
    ]

    if max_recordings is not None and n_rec > max_recordings:
        # Keep one recording per subject first, so a cap never silently turns
        # an S-subject estimate into a repeated-measures one.
        order, seen = [], set()
        for i, s in enumerate(subjects):
            if s not in seen:
                seen.add(s)
                order.append(i)
        order += [i for i in range(n_rec) if i not in set(order)]
        keep = sorted(order[:max_recordings])
        keyed = [keyed[i] for i in keep]
        subjects = [subjects[i] for i in keep]
        fif_paths = [ds._fif_paths[i] for i in keep]
        crop_inds = [ds._crop_inds[i] for i in keep]
        n_rec = len(keep)
        logger.info("Capped to %d recordings (%d unique subjects)",
                    n_rec, len(set(subjects)))
    else:
        fif_paths, crop_inds = ds._fif_paths, ds._crop_inds

    anchors = sorted(set.intersection(*(set(k) for k in keyed)))
    if not anchors:
        raise RuntimeError("No movie time is shared by all recordings.")
    logger.info("%d recordings, %d unique subjects, %d shared anchors "
                "(%.1f s of movie)", n_rec, len(set(subjects)), len(anchors),
                len(anchors) * window_size_seconds)

    sfreq = float(ds.sfreq)
    x, stds = None, None
    for r in range(n_rec):
        idx = np.array([keyed[r][a] for a in anchors], dtype=np.int64)
        w = _read_raw_windows(fif_paths[r], crop_inds[r][idx])  # [A, C, T]
        w = w.astype(np.float32)
        if x is None:
            x = np.empty((n_rec,) + w.shape, dtype=np.float32)
            stds = np.empty((n_rec, w.shape[1]), dtype=np.float64)
        x[r] = w
        stds[r] = w.std(axis=(0, 2))
        if (r + 1) % 25 == 0:
            logger.info("  loaded %d/%d recordings", r + 1, n_rec)

    # Drop numerically-flat channels BEFORE normalising. The EGI HydroCel
    # montage stores its reference ('Cz', index 128) as exact zeros -- std
    # ~3e-29, and bit-identical across recordings because it is the same
    # denormal residue of the same referencing operation. Z-scoring such a
    # channel divides that deterministic pattern by the epsilon floor and
    # amplifies it into a signal every subject shares, producing a spurious
    # ISC near 0.4 and, worse, giving CorrCA a component to latch onto.
    med = np.median(stds, axis=0)
    scale = float(np.median(med))
    good = med > min_channel_std_ratio * scale
    dropped = np.flatnonzero(~good).tolist()
    if dropped:
        logger.warning(
            "Dropping %d numerically-flat channel(s) %s "
            "(median std <= %.1e x the across-channel median)",
            len(dropped), dropped, min_channel_std_ratio,
        )
        x = x[:, :, good, :]

    # Per-recording per-channel z-score, matching norm_mode=per_recording in
    # the training configs. Pearson is scale-invariant so this does not affect
    # the waveform/band ISC, but it does set the CorrCA geometry.
    mu = x.mean(axis=(1, 3), keepdims=True)
    sd = x.std(axis=(1, 3), keepdims=True)
    np.subtract(x, mu, out=x)
    np.divide(x, np.maximum(sd, 1e-12), out=x)
    return x, subjects, anchors, sfreq, dropped


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def analyse_waveform(x: np.ndarray) -> dict:
    """Per-channel ISC over the concatenated window time-course."""
    n_rec, n_anchors, n_chans, n_times = x.shape
    flat = x.transpose(2, 0, 1, 3).reshape(n_chans, n_rec, n_anchors * n_times)
    per_chan = [mean_pairwise_isc(flat[c])[0] for c in range(n_chans)]
    per_chan = np.asarray(per_chan)
    best = int(np.nanargmax(per_chan))
    return {
        "per_channel_mean": per_chan.tolist(),
        "mean_over_channels": float(np.nanmean(per_chan)),
        "max_channel_isc": float(per_chan[best]),
        "max_channel_index": best,
    }


def analyse_bands(x: np.ndarray, sfreq: float) -> dict:
    """Per-channel, per-band ISC of window-level log power."""
    power = band_log_power(x, sfreq)          # [R, A, C, B]
    out = {}
    for bi, band in enumerate(BANDS):
        per_chan = np.asarray([
            mean_pairwise_isc(power[:, :, c, bi])[0]
            for c in range(x.shape[2])
        ])
        best = int(np.nanargmax(per_chan))
        out[band] = {
            "mean_over_channels": float(np.nanmean(per_chan)),
            "max_channel_isc": float(per_chan[best]),
            "max_channel_index": best,
            "per_channel_mean": per_chan.tolist(),
        }
    return out


def analyse_corrca(x: np.ndarray, subjects: list[str], sfreq: float,
                   n_components: int, seed: int) -> dict:
    """Cross-validated multivariate ISC via CorrCA spatial filters.

    Filters are fit on one half of the *subjects* and scored on the disjoint
    other half. The in-sample eigenvalue is reported too, purely to show the
    size of the optimism gap -- it must not be used as a ceiling.
    """
    uniq = sorted(set(subjects))
    if len(uniq) < 4:
        return {"error": f"need >= 4 subjects for a split, got {len(uniq)}"}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fit_subj = {uniq[i] for i in perm[: len(uniq) // 2]}
    fit_idx = [i for i, s in enumerate(subjects) if s in fit_subj]
    val_idx = [i for i, s in enumerate(subjects) if s not in fit_subj]
    if len(fit_idx) < 2 or len(val_idx) < 2:
        return {"error": "subject split left a half with < 2 recordings"}

    r_b, r_w = corrca_covariances(x[fit_idx])
    w, eigenvalues = solve_corrca_eigenproblem(r_b, r_w, n_components)

    # Project the held-out half and score each component.
    y = np.einsum("ract,cn->rant", x[val_idx], w)
    n_rec, n_anchors, n_comp, n_times = y.shape

    held_wave, held_band = [], []
    for n in range(n_comp):
        flat = y[:, :, n, :].reshape(n_rec, n_anchors * n_times)
        held_wave.append(mean_pairwise_isc(flat)[0])
        p = band_log_power(y[:, :, n:n + 1, :], sfreq)[:, :, 0, :]
        held_band.append({
            b: mean_pairwise_isc(p[:, :, bi])[0]
            for bi, b in enumerate(BANDS)
        })

    # Independent noisy measurements add in SNR, so a probe reading several
    # components at once gets the combined reliability, not the best single
    # one. With SNR_i = rho_i/(1-rho_i), R_combined = sum(SNR)/(1+sum(SNR)).
    # This is a LOWER bound on the multivariate ceiling: only n_components
    # are computed, and more components would raise it further.
    snr = sum(r / (1.0 - r) for r in held_wave if 0.0 < r < 1.0)
    combined = snr / (1.0 + snr) if snr > 0 else float("nan")

    return {
        "n_components": n_components,
        "n_fit_recordings": len(fit_idx),
        "n_val_recordings": len(val_idx),
        "in_sample_eigenvalues": eigenvalues.tolist(),
        "held_out_waveform_isc": held_wave,
        "held_out_band_isc": held_band,
        # Component 1 empirically fails to generalise (its in-sample
        # eigenvalue is much the largest but its held-out ISC is near zero),
        # so index 0 is NOT a usable summary -- take the max instead.
        "held_out_top_component_isc": float(held_wave[0]) if held_wave else float("nan"),
        "held_out_max_component_isc": float(max(held_wave)) if held_wave else float("nan"),
        "held_out_combined_reliability": float(combined),
    }


# ---------------------------------------------------------------------------

def analyse_noise_ceiling(x: np.ndarray, sfreq: float, corrca: dict,
                          seed: int, n_splits: int = 50) -> dict:
    """Sahani-Linden / split-half / Schoppe ceilings on the same data.

    Independent of the pairwise-ISC route in ``analyse_waveform`` /
    ``analyse_bands``: those estimate rho1 from correlations, these from a
    variance decomposition. Agreement is the point -- see
    ``noise_ceiling.py`` for why the subject axis stands in for repeated
    trials, and what that means when citing the papers.
    """
    n_rec, n_anchors, n_chans, n_times = x.shape
    out: dict = {"observed_probe_r": dict(OBSERVED_PROBE_R)}

    # (1) Window-level band power -- the probe's unit of analysis.
    power = band_log_power(x, sfreq)                       # [R, A, C, B]
    per_band = {}
    for bi, band in enumerate(BANDS):
        frac, frac_std, rho_sh = [], [], []
        for c in range(n_chans):
            r = power[:, :, c, bi]                          # [R, A]
            rz = noise_ceiling.standardize_rows(r)
            frac.append(noise_ceiling.sahani_linden_power(r)["explainable_fraction"])
            frac_std.append(
                noise_ceiling.sahani_linden_power(rz)["explainable_fraction"])
            rho_sh.append(
                noise_ceiling.split_half_reliability(rz, n_splits, seed)["rho1_corrected"])
        frac = np.asarray(frac)
        frac_std, rho_sh = np.asarray(frac_std), np.asarray(rho_sh)
        # Rank on the standardised estimate: the pipeline normalises per
        # recording, so per-subject amplitude is not information the encoder
        # can use, and the scale-invariant number is the relevant ceiling.
        best = int(np.nanargmax(frac_std))
        sl = noise_ceiling.sahani_linden_power(
            noise_ceiling.standardize_rows(power[:, :, best, bi]))
        per_band[band] = {
            "sahani_linden_raw_best_channel": float(frac[best]),
            "sahani_linden_raw_mean_over_channels": float(np.nanmean(frac)),
            "sahani_linden_mean_over_channels": float(np.nanmean(frac_std)),
            "sahani_linden_best_channel": float(frac_std[best]),
            "split_half_best_channel": float(rho_sh[best]),
            "best_channel_index": best,
            "cc_max_by_k": {
                str(k): noise_ceiling.cc_max(sl["SP"], sl["NP"], k)
                for k in (1, 2, 5, 10, 20, 50, 100)
            },
            "cc_norm_at_k1": {
                name: noise_ceiling.cc_norm(v, sl["SP"], sl["NP"], 1)
                for name, v in OBSERVED_PROBE_R.items()
            },
        }
    out["bands"] = per_band

    # (2) Held-out CorrCA components -- the multivariate readout a ridge probe
    #     on a 512-d embedding can actually exploit.
    ho = corrca.get("held_out_waveform_isc") or []
    if ho:
        snr = sum(v / (1.0 - v) for v in ho if 0.0 < v < 1.0)
        combined = snr / (1.0 + snr) if snr > 0 else float("nan")
        out["corrca_combined"] = {
            "reliability": float(combined),
            "cc_max_by_k": {
                str(k): ceiling(combined, k)
                for k in (1, 2, 5, 10, 20, 50, 100)
            },
            "cc_norm_at_k1": {
                name: (v / ceiling(combined, 1)) if combined > 0 else float("nan")
                for name, v in OBSERVED_PROBE_R.items()
            },
        }
    return out


def spearman_brown_table(rho1_by_name: dict[str, float],
                         ks=(1, 2, 5, 10, 20, 50, 100)) -> dict:
    return {
        name: {
            "rho1": rho,
            "ceiling_by_k": {str(k): ceiling(rho, k) for k in ks},
        }
        for name, rho in rho1_by_name.items()
        if math.isfinite(rho)
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--task", default="ThePresent")
    p.add_argument("--window-size-seconds", type=float, default=2.0)
    p.add_argument("--preprocessed-dir", default=None)
    p.add_argument("--max-recordings", type=int, default=None,
                   help="Cap recordings to bound memory (~21 MB each at 2 s / 101 anchors).")
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n-splits", type=int, default=50,
                   help="Random splits for the Hsu-style split-half estimator.")
    p.add_argument("--min-channel-std-ratio", type=float, default=1e-6,
                   help="Drop channels whose median std is below this ratio of "
                        "the across-channel median (catches flat reference "
                        "channels such as EGI 'Cz').")
    p.add_argument("--output", default="experiments/snr_scaling/isc.json")
    args = p.parse_args()

    x, subjects, anchors, sfreq, dropped = load_aligned(
        args.split, args.task, args.window_size_seconds,
        args.preprocessed_dir, args.max_recordings,
        args.min_channel_std_ratio,
    )
    logger.info("Aligned array %s (%.2f GB), sfreq=%.1f",
                x.shape, x.nbytes / 1e9, sfreq)

    logger.info("Waveform ISC...")
    waveform = analyse_waveform(x)
    logger.info("Band-power ISC...")
    bands = analyse_bands(x, sfreq)
    logger.info("CorrCA (cross-validated)...")
    corrca = analyse_corrca(x, subjects, sfreq, args.n_components, args.seed)
    logger.info("Noise ceilings (Sahani-Linden / split-half / Schoppe)...")
    ceilings = analyse_noise_ceiling(x, sfreq, corrca, args.seed, args.n_splits)

    # The headline rho1 candidates, in increasing order of what a probe can
    # exploit. The CorrCA held-out number is the one the paper should quote.
    rho1_by_name = {
        "waveform_mean_over_channels": waveform["mean_over_channels"],
        "waveform_best_channel": waveform["max_channel_isc"],
        "band_delta_theta_best_channel": bands["delta_theta"]["max_channel_isc"],
        "band_alpha_best_channel": bands["alpha"]["max_channel_isc"],
        "corrca_held_out_max_component": corrca.get(
            "held_out_max_component_isc", float("nan")),
        # The number the paper should quote: what a ridge probe reading the
        # whole embedding can exploit.
        "corrca_held_out_combined": corrca.get(
            "held_out_combined_reliability", float("nan")),
    }

    result = {
        "split": args.split,
        "task": args.task,
        "window_size_seconds": args.window_size_seconds,
        "sfreq": sfreq,
        "n_recordings": int(x.shape[0]),
        "n_unique_subjects": len(set(subjects)),
        "n_anchors": len(anchors),
        "n_chans": int(x.shape[2]),
        "dropped_flat_channels": dropped,
        "waveform": waveform,
        "bands": bands,
        "corrca": corrca,
        "rho1_candidates": rho1_by_name,
        "spearman_brown": spearman_brown_table(rho1_by_name),
        "noise_ceiling": ceilings,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print(f"rho1 estimates -- {args.task} {args.split} "
          f"({x.shape[0]} recordings, {len(set(subjects))} subjects, "
          f"{len(anchors)} anchors)")
    print("=" * 70)
    for name, rho in rho1_by_name.items():
        if not math.isfinite(rho):
            print(f"  {name:<34} n/a")
            continue
        print(f"  {name:<34} rho1 = {rho:7.4f}   "
              f"K=1 ceiling {ceiling(rho, 1):.3f}   "
              f"K=10 {ceiling(rho, 10):.3f}   K=50 {ceiling(rho, 50):.3f}")
    print("\n  Band ISC (window-level log power, mean over channels):")
    for b, v in bands.items():
        print(f"    {b:<14} {v['mean_over_channels']:7.4f}   "
              f"(best channel {v['max_channel_isc']:.4f})")
    if "error" not in corrca:
        print(f"\n  CorrCA in-sample eigenvalues:  "
              f"{[round(v, 4) for v in corrca['in_sample_eigenvalues']]}")
        print(f"  CorrCA held-out component ISC: "
              f"{[round(v, 4) for v in corrca['held_out_waveform_isc']]}")
        print("  (the optimism gap between these two is why the split exists)")

    print("\n" + "-" * 70)
    print("  Literature noise ceilings (subjects as the repeat axis)")
    print("-" * 70)
    print("  Best channel per band. SL(std) and split-half are scale-invariant")
    print("  and should match the pairwise ISC above; SL(raw) is deflated by")
    print("  between-subject amplitude heterogeneity.")
    print(f"  {'band':<13}{'SL(raw)':>9}{'SL(std)':>9}{'split-half':>12}"
          f"{'ISC':>9}{'CC_max K=1':>12}{'K=10':>8}")
    for b, v in ceilings["bands"].items():
        print(f"  {b:<13}{v['sahani_linden_raw_best_channel']:9.4f}"
              f"{v['sahani_linden_best_channel']:9.4f}"
              f"{v['split_half_best_channel']:12.4f}"
              f"{bands[b]['max_channel_isc']:9.4f}"
              f"{v['cc_max_by_k']['1']:12.3f}{v['cc_max_by_k']['10']:8.3f}")
    cc = ceilings.get("corrca_combined")
    if cc:
        print(f"\n  CorrCA combined reliability {cc['reliability']:.4f} "
              f"-> CC_max K=1 {cc['cc_max_by_k']['1']:.3f}, "
              f"K=10 {cc['cc_max_by_k']['10']:.3f}, "
              f"K=50 {cc['cc_max_by_k']['50']:.3f}")
        print("  CC_norm at K=1 (1.0 = at the ceiling):")
        for name, v in cc["cc_norm_at_k1"].items():
            print(f"    {name:<34}{v:6.3f}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
