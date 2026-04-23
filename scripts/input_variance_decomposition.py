"""Input-space variance decomposition (before the encoder).

Applies the same nested-ANOVA decomposition that scripts/variance_decomposition.py
runs on JEPA embeddings, but on simple per-clip summary features of the
*input* EEG. Tests whether the variance structure we observe in learned
representations is inherited from the input variance structure or actually
shaped by the encoder/regularizer.

Four conditions, mirroring the training arms:

| condition       | norm_mode       | CorrCA | notes                              |
|-----------------|-----------------|--------|------------------------------------|
| raw_global      | global          | no     | matches SIGReg/VICReg baselines    |
| per_rec         | per_recording   | no     | matches retrain_perrec arm         |
| corrca_global   | global          | yes    | ablation (CorrCA w/o per-rec norm) |
| corrca_per_rec  | per_recording   | yes    | matches CorrCA training            |

Feature per clip (default): per-channel RMS averaged across windows. The
resulting vector [C] captures channel-wise amplitude, which is where
subject-baseline differences live in EEG (each subject's channels have
characteristic amplitudes). Under per-rec norm, per-channel RMS gets
flattened toward 1 within a recording → prediction: η²_subj collapses
in the per_rec condition.

Output schema matches scripts/variance_decomposition.py so
`--aggregate_dir` / `--reanalyze_dir` work the same way.

Usage
-----
# All 4 conditions in one call, K=32, val split:
python scripts/input_variance_decomposition.py \\
    --output_dir=outputs/input_variance_decomp \\
    --n_windows=4 --window_size_seconds=4 \\
    --n_clips_per_rec=32 \\
    --corrca_filters=/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz

# Self-test (synthetic data, no dataset needed):
python scripts/input_variance_decomposition.py --selftest
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path

import fire
import numpy as np

from scripts.variance_decomposition import (
    DecompStats,
    _aggregate,
    _meta_arrays,
    _reanalyze,
    _selftest as _decomp_selftest,
    decompose,
)

logger = logging.getLogger("input_variance_decomp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Feature extraction per clip
# ---------------------------------------------------------------------------


def _rms_per_channel(eeg):
    """Per-channel RMS averaged across windows and timepoints.

    eeg: torch.Tensor [n_windows, C, T]
    returns: np.ndarray [C]
    """
    import torch
    return torch.sqrt(torch.mean(eeg ** 2, dim=(0, 2))).cpu().numpy()


def _log_power_per_channel(eeg):
    """log(mean power + eps) per channel. More Gaussian-like than RMS."""
    import torch
    power = torch.mean(eeg ** 2, dim=(0, 2))
    return torch.log(power + 1e-8).cpu().numpy()


def _flat(eeg):
    """Flatten [n_windows, C, T] → [n_windows*C*T]. High-dim; costly but lossless."""
    return eeg.cpu().numpy().reshape(-1)


FEATURES = {
    "rms": _rms_per_channel,
    "log_power": _log_power_per_channel,
    "flat": _flat,
}


# ---------------------------------------------------------------------------
# Dataset construction per condition
# ---------------------------------------------------------------------------


CONDITIONS = {
    "raw_global":       dict(norm_mode="global",         corrca=False),
    "per_rec":          dict(norm_mode="per_recording",  corrca=False),
    "corrca_global":    dict(norm_mode="global",         corrca=True),
    "corrca_per_rec":   dict(norm_mode="per_recording",  corrca=True),
}


def _build_dataset(n_windows, window_size_seconds, split, norm_mode, corrca_path,
                   cfg_fname, batch_size, num_workers):
    """Build a JEPAMovieDataset with the requested preprocessing config."""
    from eb_jepa.datasets.hbn import JEPAMovieDataset
    from eb_jepa.training_utils import load_config
    from experiments.eeg_jepa.main import resolve_preprocessed_dir

    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "data.norm_mode": norm_mode,
    }
    if corrca_path:
        overrides["data.corrca_filters"] = corrca_path

    cfg = load_config(cfg_fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feature_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))

    logger.info("Loading train set (for norm stats)...")
    train_set = JEPAMovieDataset(
        split="train", n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feature_names, cfg=cfg.data,
        preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )
    logger.info("Loading %s set...", split)
    dataset = JEPAMovieDataset(
        split=split, n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feature_names, eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data,
        preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )
    return dataset


# ---------------------------------------------------------------------------
# Per-clip feature extraction
# ---------------------------------------------------------------------------


def _features_per_clip(dataset, n_clips_per_rec, feature_fn):
    """Return Z [S, K, D] of per-clip features and recording metadata.

    Mirrors scripts.variance_decomposition._embed_per_clip but replaces
    the encoder forward pass with a simple feature extraction.
    """
    import torch
    from eb_jepa.datasets.hbn import _read_raw_windows

    all_feats = []
    all_meta = []

    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
        n_clips = n_total - required + 1
        if n_clips < n_clips_per_rec:
            continue

        starts = np.linspace(0, n_clips - 1, n_clips_per_rec, dtype=int)
        clip_feats = []
        for start in starts:
            indices = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
            eeg = torch.from_numpy(eeg_np)  # [n_windows, C, T]

            # Replicate __getitem__ preprocessing exactly (same as the
            # encoder-side script):
            if dataset._norm_mode == "per_recording":
                rec_mean = eeg.mean(dim=(0, 2), keepdim=True)
                rec_std = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
                eeg = (eeg - rec_mean) / rec_std
            else:
                eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
            if dataset._add_envelope:
                eeg = dataset._append_lowfreq_envelope(eeg)
            if getattr(dataset, "_corrca_W", None) is not None:
                eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)

            clip_feats.append(feature_fn(eeg))

        all_feats.append(np.stack(clip_feats))  # [K, D]
        all_meta.append(dataset._recording_metadata[rec_idx])

        if (rec_idx + 1) % 50 == 0:
            logger.info("  extracted %d/%d recordings", rec_idx + 1, len(dataset))

    Z = np.stack(all_feats)  # [S, K, D]
    return Z, all_meta


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------


def _run_condition(condition_name, cond_cfg, output_dir, n_windows,
                   window_size_seconds, n_clips_per_rec, split, feature,
                   corrca_path, cfg_fname, batch_size, num_workers):
    """Build dataset, extract features, run decomposition, save results."""
    corrca_arg = corrca_path if cond_cfg["corrca"] else ""
    if cond_cfg["corrca"] and not corrca_path:
        raise ValueError(
            f"Condition {condition_name!r} needs --corrca_filters=<path>"
        )

    dataset = _build_dataset(
        n_windows, window_size_seconds, split,
        norm_mode=cond_cfg["norm_mode"], corrca_path=corrca_arg,
        cfg_fname=cfg_fname, batch_size=batch_size, num_workers=num_workers,
    )
    feature_fn = FEATURES[feature]

    logger.info("[%s] extracting features (K=%d, feature=%s)",
                condition_name, n_clips_per_rec, feature)
    Z, meta_list = _features_per_clip(dataset, n_clips_per_rec, feature_fn)
    logger.info("[%s] Z shape = %s", condition_name, Z.shape)

    stats, _raw = decompose(Z)
    logger.info(
        "[%s] η²_subj=%.4f η²_stim=%.4f stim/within=%.4f  eff_rank(subj/within)=%.2f/%.2f",
        condition_name, stats.eta_sq, stats.eta_sq_stimulus,
        stats.stim_frac_of_within, stats.eff_rank_subject, stats.eff_rank_within,
    )

    subject_ids, ages, sexes = _meta_arrays(meta_list)

    run_dir = Path(output_dir) / condition_name
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        run_dir / "embeddings.npz",
        Z=Z, subject_ids=subject_ids, ages=ages, sexes=sexes,
    )
    stats_dict = asdict(stats)
    stats_dict.update({
        "run_name": condition_name,
        "condition": condition_name,
        "split": split,
        "n_windows": n_windows,
        "window_size_seconds": window_size_seconds,
        "n_clips_per_rec": n_clips_per_rec,
        "feature": feature,
        "norm_mode": cond_cfg["norm_mode"],
        "corrca": cond_cfg["corrca"],
        "embed_dim": Z.shape[-1],
    })
    with open(run_dir / "stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)
    logger.info("[%s] wrote %s", condition_name, run_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    output_dir: str = "outputs/input_variance_decomp",
    n_windows: int = 4,
    window_size_seconds: int = 4,
    n_clips_per_rec: int = 32,
    split: str = "val",
    feature: str = "rms",
    conditions: str = "all",  # comma-separated subset of CONDITIONS or "all"
    corrca_filters: str = "",
    batch_size: int = 64,
    num_workers: int = 4,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    seed: int = 2025,
    # Compat modes with scripts/variance_decomposition.py
    aggregate_dir: str = "",
    reanalyze_dir: str = "",
    selftest: bool = False,
):
    """Run input-space variance decomposition for one or more conditions."""
    if selftest:
        _decomp_selftest()
        return

    if reanalyze_dir:
        _reanalyze(Path(reanalyze_dir))
        _aggregate(Path(reanalyze_dir))
        return

    if aggregate_dir:
        _aggregate(Path(aggregate_dir))
        return

    if conditions == "all":
        wanted = list(CONDITIONS.keys())
    else:
        wanted = [c.strip() for c in conditions.split(",") if c.strip()]
        for c in wanted:
            if c not in CONDITIONS:
                raise ValueError(f"Unknown condition {c!r}; valid: {list(CONDITIONS)}")

    if feature not in FEATURES:
        raise ValueError(f"Unknown feature {feature!r}; valid: {list(FEATURES)}")

    needs_corrca = any(CONDITIONS[c]["corrca"] for c in wanted)
    if needs_corrca and not corrca_filters:
        raise ValueError("At least one condition needs --corrca_filters=<path>")

    from eb_jepa.training_utils import setup_seed
    setup_seed(seed)

    logger.info("Running %d condition(s): %s", len(wanted), ", ".join(wanted))
    for cond_name in wanted:
        _run_condition(
            cond_name, CONDITIONS[cond_name], output_dir,
            n_windows, window_size_seconds, n_clips_per_rec, split,
            feature, corrca_filters, fname, batch_size, num_workers,
        )

    _aggregate(Path(output_dir))
    logger.info("All done.")


if __name__ == "__main__":
    fire.Fire(run)
