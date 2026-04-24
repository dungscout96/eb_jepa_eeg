"""Input-space variance + predictability decomposition (pre-encoder).

Tests Littwin's JEPA claim: representation variance tracks the fraction of
input variance that is *predictable from context to target* under the
masked-prediction task, not merely total input variance per source. The
two quantities coincide in many cases but diverge in informative ones
(e.g. CorrCA may reduce subject variance while leaving within-recording
constancy — and therefore subject predictability — intact).

Procedure per clip
------------------
Each clip is read as `[n_windows, C, T]` float EEG. We compute three
per-channel RMS feature vectors:

- `X_full`: RMS across the whole clip → for static variance decomposition.
- `X_ctx`: RMS of the first half along time → context features.
- `X_tgt`: RMS of the second half → target features.

Then:

(1) **Static** decomposition on `X_full`: subject / stimulus / residual
    variance components. Tells us what's IN the input.

(2) **Predictability** decomposition via OLS `X_tgt ≈ W · X_ctx + b`:
    residual ε = X_tgt − (W·X_ctx + b), decomposed the same way. R² per
    source = 1 − Var_source(ε) / Var_source(X_tgt). Tells us how much of
    each variance component a linear context→target regression can
    recover.

The strongest Littwin test is comparing (2) — not (1) — to the encoder's
representation variance from scripts/variance_decomposition.py. Per-rec
norm and CorrCA affect variance and predictability differently; the
discrepancy is the diagnostic.

Four conditions, mirroring training arms:

| condition       | norm_mode     | CorrCA | matches                              |
|-----------------|---------------|--------|--------------------------------------|
| raw_global      | global        | no     | SIGReg/VICReg baselines              |
| per_rec         | per_recording | no     | retrain_perrec arm                   |
| corrca_global   | global        | yes    | ablation (CorrCA w/o per-rec norm)   |
| corrca_per_rec  | per_recording | yes    | CorrCA training                      |

Usage
-----
# All 4 conditions in one call, K=32, val split:
python scripts/input_variance_decomposition.py \\
    --output_dir=outputs/input_predictability_decomp \\
    --n_windows=4 --window_size_seconds=4 --n_clips_per_rec=32 \\
    --corrca_filters=/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz

# Selftest (no dataset needed, synthetic toy cases):
python scripts/input_variance_decomposition.py --selftest
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import fire
import numpy as np

from scripts.variance_decomposition import (
    _aggregate,
    _meta_arrays,
    _reanalyze,
    decompose,
)

logger = logging.getLogger("input_predictability")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Context/target feature extraction per clip
# ---------------------------------------------------------------------------


def _full_ctx_tgt_rms(eeg):
    """Return (rms_full, rms_ctx, rms_tgt) per channel for one clip.

    eeg: torch.Tensor [n_windows, C, T]
    returns: (np.ndarray [C], np.ndarray [C], np.ndarray [C])
    - rms_full: RMS across the entire clip (used for static variance decomp)
    - rms_ctx:  RMS across the first half (context)
    - rms_tgt:  RMS across the second half (target)
    """
    import torch
    nw, C, T = eeg.shape
    # Flatten time across windows so the split is contiguous, not per-window.
    flat = eeg.permute(1, 0, 2).reshape(C, nw * T)   # [C, nw*T]
    mid = (nw * T) // 2
    full_rms = torch.sqrt(torch.mean(flat ** 2, dim=1))
    ctx_rms = torch.sqrt(torch.mean(flat[:, :mid] ** 2, dim=1))
    tgt_rms = torch.sqrt(torch.mean(flat[:, mid:] ** 2, dim=1))
    return full_rms.cpu().numpy(), ctx_rms.cpu().numpy(), tgt_rms.cpu().numpy()


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


CONDITIONS = {
    "raw_global":     dict(norm_mode="global",        corrca=False),
    "per_rec":        dict(norm_mode="per_recording", corrca=False),
    "corrca_global":  dict(norm_mode="global",        corrca=True),
    "corrca_per_rec": dict(norm_mode="per_recording", corrca=True),
}


def _build_dataset(n_windows, window_size_seconds, split, norm_mode, corrca_path,
                   cfg_fname, batch_size, num_workers):
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
# Per-clip extraction across a dataset
# ---------------------------------------------------------------------------


def _ctx_tgt_per_clip(dataset, n_clips_per_rec):
    """Return X_full, X_ctx, X_tgt of shape [S, K, C] and per-recording metadata."""
    import torch
    from eb_jepa.datasets.hbn import _read_raw_windows

    all_full, all_ctx, all_tgt, all_meta = [], [], [], []
    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
        n_clips = n_total - required + 1
        if n_clips < n_clips_per_rec:
            continue

        starts = np.linspace(0, n_clips - 1, n_clips_per_rec, dtype=int)
        full_feats, ctx_feats, tgt_feats = [], [], []
        for start in starts:
            indices = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
            eeg = torch.from_numpy(eeg_np)  # [n_windows, C, T]

            # Replicate __getitem__ preprocessing exactly.
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

            full, ctx, tgt = _full_ctx_tgt_rms(eeg)
            full_feats.append(full); ctx_feats.append(ctx); tgt_feats.append(tgt)

        all_full.append(np.stack(full_feats))  # [K, C]
        all_ctx.append(np.stack(ctx_feats))
        all_tgt.append(np.stack(tgt_feats))
        all_meta.append(dataset._recording_metadata[rec_idx])

        if (rec_idx + 1) % 50 == 0:
            logger.info("  extracted %d/%d recordings", rec_idx + 1, len(dataset))

    return np.stack(all_full), np.stack(all_ctx), np.stack(all_tgt), all_meta


# ---------------------------------------------------------------------------
# Predictability decomposition
# ---------------------------------------------------------------------------


def predictability_decompose(X_ctx, X_tgt):
    """Fit OLS target = W·context + b, then decompose target, prediction,
    and residual by variance source (subject × stimulus × residual).

    Two predictability measures are reported per source k:

    1. predictability_k = Var_k(ŷ) / Var_k(X_tgt)   ← primary (Littwin)
       "Fraction of source-k variance captured by the prediction ŷ."

    2. r2_k = 1 − Var_k(ε) / Var_k(X_tgt)           ← sanity check
       "Fraction of source-k variance not left in the residual ε."

    The two coincide iff OLS orthogonality extends to the source-decomposed
    level (i.e. Cov_k(ŷ, ε) = 0). Reporting both exposes where it doesn't.

    Inputs: X_ctx, X_tgt shape [S, K, C]. Returns dict with:
      - stats_tgt:   decomposition of X_tgt   (DecompStats)
      - stats_pred:  decomposition of ŷ       (DecompStats)
      - stats_eps:   decomposition of ε       (DecompStats)
      - predictability_{total, subject, stimulus, residual}
      - r2_{total, subject, stimulus, residual}
      - weight_fro, bias_norm
    """
    S, K, C = X_tgt.shape
    N = S * K
    X_ctx_flat = X_ctx.reshape(N, C)
    X_tgt_flat = X_tgt.reshape(N, C)
    X_aug = np.hstack([X_ctx_flat, np.ones((N, 1), dtype=X_ctx_flat.dtype)])
    coef, _, _, _ = np.linalg.lstsq(X_aug, X_tgt_flat, rcond=None)
    W, b = coef[:-1], coef[-1]

    Y_pred_flat = X_ctx_flat @ W + b      # [N, C]
    eps_flat = X_tgt_flat - Y_pred_flat   # [N, C]
    Y_pred = Y_pred_flat.reshape(S, K, C)
    eps = eps_flat.reshape(S, K, C)

    stats_tgt,  _ = decompose(X_tgt)
    stats_pred, _ = decompose(Y_pred)
    stats_eps,  _ = decompose(eps)

    def _safe(num, denom):
        return float(num / denom) if denom > 0 else float("nan")

    return {
        "stats_tgt":  stats_tgt,
        "stats_pred": stats_pred,
        "stats_eps":  stats_eps,
        # Primary — Var_k(ŷ) / Var_k(X_tgt).
        "predictability_total":     _safe(stats_pred.var_total,     stats_tgt.var_total),
        "predictability_subject":   _safe(stats_pred.var_subject,   stats_tgt.var_subject),
        "predictability_stimulus":  _safe(stats_pred.var_stimulus,  stats_tgt.var_stimulus),
        "predictability_residual":  _safe(stats_pred.var_residual,  stats_tgt.var_residual),
        # Sanity check — 1 − Var_k(ε) / Var_k(X_tgt).
        "r2_total":     1.0 - _safe(stats_eps.var_total,     stats_tgt.var_total),
        "r2_subject":   1.0 - _safe(stats_eps.var_subject,   stats_tgt.var_subject),
        "r2_stimulus":  1.0 - _safe(stats_eps.var_stimulus,  stats_tgt.var_stimulus),
        "r2_residual":  1.0 - _safe(stats_eps.var_residual,  stats_tgt.var_residual),
        "weight_fro":   float(np.linalg.norm(W, "fro")),
        "bias_norm":    float(np.linalg.norm(b)),
    }


# ---------------------------------------------------------------------------
# Per-condition runner
# ---------------------------------------------------------------------------


def _run_condition(condition_name, cond_cfg, output_dir, n_windows,
                   window_size_seconds, n_clips_per_rec, split,
                   corrca_path, cfg_fname, batch_size, num_workers):
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

    logger.info("[%s] extracting full/context/target features (K=%d)",
                condition_name, n_clips_per_rec)
    X_full, X_ctx, X_tgt, meta_list = _ctx_tgt_per_clip(dataset, n_clips_per_rec)
    logger.info("[%s] X_full/ctx/tgt shapes all %s", condition_name, X_full.shape)

    # (1) Static variance decomposition on the full-clip features.
    stats_full, _ = decompose(X_full)
    logger.info(
        "[%s] [STATIC full-clip] η²_subj=%.4f η²_stim=%.4f stim/within=%.4f  "
        "Var_total=%.4f",
        condition_name, stats_full.eta_sq, stats_full.eta_sq_stimulus,
        stats_full.stim_frac_of_within, stats_full.var_total,
    )

    # (2) Predictability decomposition on context/target halves.
    res = predictability_decompose(X_ctx, X_tgt)
    logger.info(
        "[%s] [PREDICT ctx→tgt] predictability: total=%.4f subj=%.4f stim=%.4f res=%.4f",
        condition_name,
        res["predictability_total"], res["predictability_subject"],
        res["predictability_stimulus"], res["predictability_residual"],
    )
    logger.info(
        "[%s] [PREDICT ctx→tgt]  (R² sanity): total=%.4f subj=%.4f stim=%.4f res=%.4f",
        condition_name,
        res["r2_total"], res["r2_subject"], res["r2_stimulus"], res["r2_residual"],
    )

    subject_ids, ages, sexes = _meta_arrays(meta_list)

    run_dir = Path(output_dir) / condition_name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "features.npz",
        X_full=X_full, X_ctx=X_ctx, X_tgt=X_tgt,
        subject_ids=subject_ids, ages=ages, sexes=sexes,
    )

    # JSON — both the static variance decomposition (X_full) and the
    # predictability decomposition (ε from ctx→tgt OLS). The cleanest
    # Littwin test: compare stats_full's η²_source against r2_source.
    out = {
        "run_name":         condition_name,
        "condition":        condition_name,
        "split":            split,
        "n_windows":        n_windows,
        "window_size_seconds": window_size_seconds,
        "n_clips_per_rec":  n_clips_per_rec,
        "feature":          "channel_rms",
        "norm_mode":        cond_cfg["norm_mode"],
        "corrca":           cond_cfg["corrca"],
        "embed_dim":        X_full.shape[-1],
        # (1) Static: what's IN the input, decomposed.
        "stats_full_clip":  asdict(stats_full),
        # (2) Predictability: how much of each source is recoverable via OLS.
        "stats_target":              asdict(res["stats_tgt"]),
        "stats_predicted":           asdict(res["stats_pred"]),
        "stats_residual":            asdict(res["stats_eps"]),
        "predictability_total":      res["predictability_total"],
        "predictability_subject":    res["predictability_subject"],
        "predictability_stimulus":   res["predictability_stimulus"],
        "predictability_residual":   res["predictability_residual"],
        "r2_total":                  res["r2_total"],
        "r2_subject":                res["r2_subject"],
        "r2_stimulus":               res["r2_stimulus"],
        "r2_residual":               res["r2_residual"],
        "weight_fro":                res["weight_fro"],
        "bias_norm":                 res["bias_norm"],
    }
    with open(run_dir / "stats.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("[%s] wrote %s", condition_name, run_dir)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest():
    """Toy cases: subject DC-only, stimulus-only, and pure noise.

    Predictability claim: subject DC should give R²_subj ≈ 1.0 (constants
    are trivially recovered); stimulus-shared between ctx and tgt gives
    R²_stim ≈ 1.0; pure noise gives R² ≈ 0.
    """
    rng = np.random.default_rng(0)
    S, K, C = 60, 20, 16

    def _fmt(r):
        return (f"pred: tot={r['predictability_total']:+.3f} "
                f"subj={r['predictability_subject']:+.3f} "
                f"stim={r['predictability_stimulus']:+.3f}  | "
                f"R²: tot={r['r2_total']:+.3f} "
                f"subj={r['r2_subject']:+.3f} stim={r['r2_stimulus']:+.3f}")

    # Case A: pure noise
    X_ctx = rng.standard_normal((S, K, C))
    X_tgt = rng.standard_normal((S, K, C))
    print(f"[noise]   {_fmt(predictability_decompose(X_ctx, X_tgt))}")

    # Case B: subject DC offset shared across ctx/tgt
    offsets = rng.standard_normal((S, 1, C)) * 3.0
    X_ctx = rng.standard_normal((S, K, C)) + offsets
    X_tgt = rng.standard_normal((S, K, C)) + offsets
    print(f"[subjDC]  {_fmt(predictability_decompose(X_ctx, X_tgt))}  (expect subj ≫ 0)")

    # Case C: shared stimulus (same clip-pos signal across ctx/tgt)
    stim = rng.standard_normal((1, K, C)) * 3.0
    X_ctx = rng.standard_normal((S, K, C)) + stim
    X_tgt = rng.standard_normal((S, K, C)) + stim
    print(f"[stim]    {_fmt(predictability_decompose(X_ctx, X_tgt))}  (expect stim ≫ 0)")

    # Case D: both
    X_ctx = rng.standard_normal((S, K, C)) + offsets + stim
    X_tgt = rng.standard_normal((S, K, C)) + offsets + stim
    print(f"[both]    {_fmt(predictability_decompose(X_ctx, X_tgt))}  (expect both ≫ 0)")
    print("selftest OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    output_dir: str = "outputs/input_predictability_decomp",
    n_windows: int = 4,
    window_size_seconds: int = 4,
    n_clips_per_rec: int = 32,
    split: str = "val",
    conditions: str = "all",
    corrca_filters: str = "",
    batch_size: int = 64,
    num_workers: int = 4,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    seed: int = 2025,
    aggregate_dir: str = "",
    reanalyze_dir: str = "",
    selftest: bool = False,
):
    """Run input-space predictability decomposition for one or more conditions."""
    if selftest:
        _selftest()
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

    needs_corrca = any(CONDITIONS[c]["corrca"] for c in wanted)
    if needs_corrca and not corrca_filters:
        raise ValueError("At least one condition needs --corrca_filters=<path>")

    from eb_jepa.training_utils import setup_seed
    setup_seed(seed)

    logger.info("Running %d condition(s): %s", len(wanted), ", ".join(wanted))
    for cond in wanted:
        _run_condition(
            cond, CONDITIONS[cond], output_dir,
            n_windows, window_size_seconds, n_clips_per_rec, split,
            corrca_filters, fname, batch_size, num_workers,
        )
    logger.info("All done.")


if __name__ == "__main__":
    fire.Fire(run)
