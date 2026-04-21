"""Variance decomposition of frozen EEG JEPA embeddings: between- vs within-subject.

Each recording in HBN = one subject watching ThePresent, so recording index ≡
subject index. For each checkpoint we extract K clips per subject, mean-pool
the encoder tokens into one D-dim embedding per clip, and stack into Z[S,K,D].

Decomposition (law of total variance on embedding vectors):

    grand_mean     = mean over all S*K embeddings                      [D]
    subject_means  = per-subject mean over K clips                     [S, D]
    Var_total      = (1 / (S*K)) Σ_{s,k} ||z_{s,k} - grand_mean||²
    Var_subject    = (K / (S*K)) Σ_s ||μ_s - grand_mean||²
    Var_within     = Var_total - Var_subject
    η²             = Var_subject / Var_total

And the matrix form:

    C_total   = cov of all [S*K, D] rows      (bessel-corrected)
    C_subject = K * cov of [S, D] subject means   (so trace adds up)
    C_within  = C_total - C_subject
    principal_angles(top-k eigvecs(C_subject), top-k eigvecs(C_within))

K=4 caveat: Var_within is an unbiased population estimate but C_subject is
noisier with small K. All checkpoints under comparison share the same K, so
relative differences are still interpretable.

Usage
-----
Single checkpoint:

    PYTHONPATH=. uv run --group eeg python scripts/variance_decomposition.py \\
        --checkpoint=/path/to/latest.pth.tar \\
        --n_windows=1 --window_size_seconds=1 \\
        --n_clips_per_rec=4 --split=val \\
        --output_dir=outputs/variance_decomp

Aggregate over all per-checkpoint outputs:

    python scripts/variance_decomposition.py \\
        --aggregate_dir=outputs/variance_decomp

Self-test (random tensors, no checkpoint needed):

    python scripts/variance_decomposition.py --selftest
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import fire
import numpy as np

import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Encoder rebuild + checkpoint load (mirrors probe_eval.py)
# ---------------------------------------------------------------------------


def _build_and_load(checkpoint_path: Path, cfg_fname: str, n_windows: int,
                    window_size_seconds: int, split: str, device,
                    batch_size: int, num_workers: int,
                    norm_mode: str = "", add_envelope: bool = False,
                    corrca_filters: str = ""):
    """Reconstruct MaskedJEPA and load weights. Returns (jepa, dataset, cfg)."""
    import torch
    from eb_jepa.architectures import EEGEncoderTokens, MaskedPredictor, Projector
    from eb_jepa.datasets.hbn import JEPAMovieDataset
    from eb_jepa.jepa import MaskedJEPA
    from eb_jepa.losses import VCLoss
    from eb_jepa.masking import MultiBlockMaskCollator
    from eb_jepa.training_utils import load_checkpoint, load_config
    from experiments.eeg_jepa.main import resolve_preprocessed_dir

    ckpt_sd = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    ).get("model_state_dict", {})

    depth = max(
        int(k.split(".")[3]) + 1
        for k in ckpt_sd
        if k.startswith("context_encoder.transformer.layers.")
    )
    if "predictor.input_proj.weight" in ckpt_sd:
        pred_dim = int(ckpt_sd["predictor.input_proj.weight"].shape[0])
    elif "predictor.output_proj.weight" in ckpt_sd:
        pred_dim = int(ckpt_sd["predictor.output_proj.weight"].shape[0])
    else:
        pred_dim = None

    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "model.encoder_depth": depth,
        "model.predictor_embed_dim": pred_dim,
    }
    if norm_mode:
        overrides["data.norm_mode"] = norm_mode
    if add_envelope:
        overrides["data.add_envelope"] = True
    if corrca_filters:
        overrides["data.corrca_filters"] = corrca_filters
    cfg = load_config(cfg_fname, overrides)

    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feature_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))

    # Train set is needed only for normalization stats (we pass them into the
    # eval-split dataset so z-score matches training).
    logger.info("Loading train set (for norm stats)...")
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )

    logger.info("Loading %s set...", split)
    dataset = JEPAMovieDataset(
        split=split,
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )

    n_chans = dataset.n_chans
    n_times = dataset.n_times
    embed_dim = cfg.model.encoder_embed_dim
    chs_info = dataset.get_chs_info()
    masking_cfg = cfg.get("masking", {})

    encoder = EEGEncoderTokens(
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=n_windows,
        patch_size=cfg.model.get("patch_size", 200),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=chs_info,
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
    )
    target_encoder = copy.deepcopy(encoder)
    predictor = MaskedPredictor(
        embed_dim=embed_dim,
        depth=cfg.model.get("predictor_depth", 2),
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
        predictor_dim=pred_dim,
    )
    mask_collator = MultiBlockMaskCollator(
        n_channels=n_chans,
        n_windows=n_windows,
        n_patches_per_window=encoder.n_patches_per_window,
        n_pred_masks_short=masking_cfg.get("n_pred_masks_short", 2),
        n_pred_masks_long=masking_cfg.get("n_pred_masks_long", 2),
        short_channel_scale=tuple(masking_cfg.get("short_channel_scale", [0.08, 0.15])),
        short_patch_scale=tuple(masking_cfg.get("short_patch_scale", [0.3, 0.6])),
        long_channel_scale=tuple(masking_cfg.get("long_channel_scale", [0.15, 0.35])),
        long_patch_scale=tuple(masking_cfg.get("long_patch_scale", [0.5, 1.0])),
        min_context_fraction=masking_cfg.get("min_context_fraction", 0.15),
    )

    regularizer = None
    if any(k.startswith("regularizer.") for k in ckpt_sd):
        projector = Projector(f"{embed_dim}-{embed_dim * 4}-{embed_dim * 4}")
        regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)

    jepa = MaskedJEPA(
        encoder, target_encoder, predictor, mask_collator, regularizer,
    ).to(device)
    info = load_checkpoint(checkpoint_path, jepa, optimizer=None, device=device, strict=False)
    logger.info("Loaded checkpoint at epoch %s", info.get("epoch", "?"))
    for p in jepa.parameters():
        p.requires_grad_(False)
    jepa.eval()
    return jepa, dataset, cfg


# ---------------------------------------------------------------------------
# Per-clip embedding extraction
# ---------------------------------------------------------------------------


def _embed_per_clip(dataset, jepa, device, n_clips_per_rec: int):
    """Return Z [S, K, D] embeddings and per-recording metadata.

    Recordings with too few frames to yield K clips are dropped (rare).
    """
    import torch
    from eb_jepa.datasets.hbn import _read_raw_windows
    jepa.eval()
    all_embs = []
    all_meta = []

    with torch.no_grad():
        for rec_idx in range(len(dataset)):
            crop_inds = dataset._crop_inds[rec_idx]
            n_total = len(crop_inds)
            required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
            n_clips = n_total - required + 1
            if n_clips < n_clips_per_rec:
                continue

            starts = np.linspace(0, n_clips - 1, n_clips_per_rec, dtype=int)
            clip_embs = []
            for start in starts:
                indices = list(range(start, start + required, dataset.temporal_stride))
                eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
                eeg = torch.from_numpy(eeg_np)  # [n_windows, C, T]

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

                eeg = eeg.unsqueeze(0).to(device)  # [1, n_windows, C, T]
                tokens = jepa.context_encoder.encode_tokens(eeg, mask=None)
                emb = tokens.mean(dim=1).squeeze(0).cpu().numpy()  # [D]
                clip_embs.append(emb)

            all_embs.append(np.stack(clip_embs))  # [K, D]
            all_meta.append(dataset._recording_metadata[rec_idx])

            if (rec_idx + 1) % 50 == 0:
                logger.info("  embedded %d/%d recordings", rec_idx + 1, len(dataset))

    Z = np.stack(all_embs)  # [S, K, D]
    return Z, all_meta


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------


@dataclass
class DecompStats:
    S: int
    K: int
    D: int
    # Two-way (subject vs within) — what the script computed originally.
    var_total: float
    var_subject: float
    var_within: float
    eta_sq: float
    # Three-way (subject × clip-position × residual). Valid only when clip
    # index k is time-aligned across subjects (linspace sampling of the same
    # stimulus, which is what _embed_per_clip does). Var_within decomposes
    # further as var_within = var_stimulus + var_residual.
    #   var_stimulus  = S * mean_k ||μ_k − grand||²  (between-clip, pooled over subj)
    #   var_residual  = everything orthogonal to subject-marginal and clip-marginal
    # Null for iid Gaussian: var_stimulus/var_total ≈ 1/S, so anything
    # meaningfully above that is real stimulus-locked signal.
    var_stimulus: float
    var_residual: float
    eta_sq_stimulus: float          # var_stimulus / var_total
    stim_frac_of_within: float      # var_stimulus / var_within
    trace_identity_residual: float  # |trace(C_s)+trace(C_w)-trace(C_t)| / trace(C_t)
    eigvals_total: list  # top-min(D,40)
    eigvals_subject: list
    eigvals_within: list
    eff_rank_total: float
    eff_rank_subject: float
    eff_rank_within: float
    principal_angles_deg: dict  # {"k=1": [..], "k=2": [..], ...}


def _effective_rank(eigvals: np.ndarray) -> float:
    w = np.clip(eigvals, a_min=0.0, a_max=None)
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    # 0 * log(0) := 0
    p_nz = p[p > 0]
    H = -(p_nz * np.log(p_nz)).sum()
    return float(math.exp(H))


def decompose(Z: np.ndarray) -> tuple[DecompStats, dict]:
    """Run the full decomposition. Returns (stats, raw_arrays)."""
    S, K, D = Z.shape
    Z_flat = Z.reshape(S * K, D)

    grand_mean = Z_flat.mean(axis=0)
    subj_mean = Z.mean(axis=1)  # [S, D]

    dev_total = Z_flat - grand_mean
    dev_subj = subj_mean - grand_mean

    var_total = float((dev_total ** 2).sum() / (S * K))
    var_subject = float(K * (dev_subj ** 2).sum() / (S * K))
    var_within = var_total - var_subject
    eta_sq = var_subject / var_total if var_total > 0 else float("nan")

    # Three-way split: subject × clip-position × residual.
    # Clips are sampled via np.linspace(0, n_clips-1, K) per recording, so
    # clip index k corresponds to roughly the same movie position across
    # subjects — the between-clip-position variance is stimulus-locked.
    stim_mean = Z.mean(axis=0)  # [K, D]  mean over subjects at each clip-position
    dev_stim = stim_mean - grand_mean
    var_stimulus = float(S * (dev_stim ** 2).sum() / (S * K))
    var_residual = var_within - var_stimulus
    eta_sq_stimulus = var_stimulus / var_total if var_total > 0 else float("nan")
    stim_frac_of_within = var_stimulus / var_within if var_within > 0 else float("nan")

    # Covariance (bessel-corrected; use same estimator for all three so trace
    # identity holds approximately).
    C_total = np.cov(Z_flat, rowvar=False)                  # (S*K - 1) divisor
    C_subject_raw = np.cov(subj_mean, rowvar=False)         # (S - 1) divisor
    # Scale: sum-of-squared-deviations / (S*K - 1), same denominator as C_total,
    # matches the scalar Var_subject under large S,K.
    C_subject = (K * (S - 1) / (S * K - 1)) * C_subject_raw
    C_within = C_total - C_subject

    eig_t = np.linalg.eigvalsh(C_total)[::-1]
    eig_s = np.linalg.eigvalsh(C_subject)[::-1]
    eig_w = np.linalg.eigvalsh(C_within)[::-1]

    trace_id_res = abs(eig_s.sum() + eig_w.sum() - eig_t.sum()) / max(abs(eig_t.sum()), 1e-12)

    # Full eigendecomposition for eigenvectors (needed for principal angles).
    w_s, V_s = np.linalg.eigh(C_subject)
    w_w, V_w = np.linalg.eigh(C_within)
    V_s = V_s[:, ::-1]
    V_w = V_w[:, ::-1]

    from scipy.linalg import subspace_angles
    angles = {}
    for k in (1, 2, 5, 10):
        if k <= D:
            angs = subspace_angles(V_s[:, :k], V_w[:, :k])
            angles[f"k={k}"] = np.rad2deg(angs).tolist()

    top_n = min(D, 40)
    stats = DecompStats(
        S=S, K=K, D=D,
        var_total=var_total,
        var_subject=var_subject,
        var_within=var_within,
        eta_sq=eta_sq,
        var_stimulus=var_stimulus,
        var_residual=var_residual,
        eta_sq_stimulus=eta_sq_stimulus,
        stim_frac_of_within=stim_frac_of_within,
        trace_identity_residual=float(trace_id_res),
        eigvals_total=eig_t[:top_n].tolist(),
        eigvals_subject=eig_s[:top_n].tolist(),
        eigvals_within=eig_w[:top_n].tolist(),
        eff_rank_total=_effective_rank(eig_t),
        eff_rank_subject=_effective_rank(eig_s),
        eff_rank_within=_effective_rank(eig_w),
        principal_angles_deg=angles,
    )
    raw = {"C_total": C_total, "C_subject": C_subject, "C_within": C_within,
           "V_subject": V_s, "V_within": V_w}
    return stats, raw


# ---------------------------------------------------------------------------
# Metadata parsing (light; ages/sexes saved for downstream plots if desired)
# ---------------------------------------------------------------------------


def _meta_arrays(meta_list):
    S = len(meta_list)
    subject_ids = np.array([str(m.get("subject", m.get("participant_id", f"rec{i}")))
                            for i, m in enumerate(meta_list)])
    ages = np.full(S, np.nan)
    sexes = np.full(S, np.nan)
    for i, m in enumerate(meta_list):
        a = m.get("age", None)
        if a is not None:
            try:
                ages[i] = float(a)
            except (TypeError, ValueError):
                pass
        sx = str(m.get("sex", m.get("gender", ""))).strip().lower()
        if sx in ("m", "male"):
            sexes[i] = 1.0
        elif sx in ("f", "female"):
            sexes[i] = 0.0
    return subject_ids, ages, sexes


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _run_name_from_ckpt(checkpoint_path: Path) -> str:
    # .../<run_name>/latest.pth.tar  →  <run_name>
    return checkpoint_path.parent.name


def run(
    checkpoint: str = "",
    n_windows: int = 1,
    window_size_seconds: int = 1,
    n_clips_per_rec: int = 4,
    split: str = "val",
    batch_size: int = 64,
    num_workers: int = 4,
    output_dir: str = "outputs/variance_decomp",
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    seed: int = 2025,
    # Data preprocessing overrides (must match training config)
    norm_mode: str = "",
    add_envelope: bool = False,
    corrca_filters: str = "",
    # Aggregation mode
    aggregate_dir: str = "",
    # Reanalyze mode: recompute stats.json from saved embeddings.npz (no re-embedding).
    reanalyze_dir: str = "",
    # Self-test
    selftest: bool = False,
):
    """Single-checkpoint variance decomposition, aggregation, or selftest."""
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

    assert checkpoint, "Pass --checkpoint=/path/to/latest.pth.tar (or --selftest / --aggregate_dir)"
    from eb_jepa.training_utils import setup_device, setup_seed
    setup_seed(seed)
    device = setup_device("auto")

    ckpt_path = Path(checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    jepa, dataset, cfg = _build_and_load(
        ckpt_path, fname, n_windows, window_size_seconds, split,
        device, batch_size, num_workers,
        norm_mode=norm_mode, add_envelope=add_envelope, corrca_filters=corrca_filters,
    )

    logger.info("Extracting per-clip embeddings (K=%d clips/rec)", n_clips_per_rec)
    Z, meta_list = _embed_per_clip(dataset, jepa, device, n_clips_per_rec)
    logger.info("Embeddings: Z shape = %s", Z.shape)

    stats, _raw = decompose(Z)
    logger.info("η² = %.4f   var_total=%.4f   var_subject=%.4f   var_within=%.4f",
                stats.eta_sq, stats.var_total, stats.var_subject, stats.var_within)

    subject_ids, ages, sexes = _meta_arrays(meta_list)

    run_name = _run_name_from_ckpt(ckpt_path)
    out = Path(output_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "embeddings.npz",
        Z=Z, subject_ids=subject_ids, ages=ages, sexes=sexes,
    )

    stats_dict = asdict(stats)
    stats_dict.update({
        "checkpoint": str(ckpt_path),
        "run_name": run_name,
        "split": split,
        "n_windows": n_windows,
        "window_size_seconds": window_size_seconds,
        "n_clips_per_rec": n_clips_per_rec,
        "embed_dim": cfg.model.encoder_embed_dim,
        "encoder_depth": cfg.model.encoder_depth,
    })
    with open(out / "stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    logger.info("Wrote %s and %s", out / "embeddings.npz", out / "stats.json")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _parse_run_name(run_name: str) -> dict:
    """Best-effort tagging from run names like
    'eeg_jepa_bs64_lr0.0005_sigreg0.1_nw1_ws1s_seed2025' or
    'eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed2025'.
    """
    tags = {"regularizer": "unknown", "coeff": None, "nw": None, "ws": None,
            "seed": None}
    parts = run_name.split("_")
    for p in parts:
        if p.startswith("sigreg"):
            tags["regularizer"] = "sigreg"
            try:
                tags["coeff"] = float(p[len("sigreg"):])
            except ValueError:
                pass
        elif p.startswith("std"):
            tags["regularizer"] = "vicreg"
        elif p.startswith("nw") and p[2:].isdigit():
            tags["nw"] = int(p[2:])
        elif p.startswith("ws") and p[2:].rstrip("s").isdigit():
            tags["ws"] = int(p[2:].rstrip("s"))
        elif p.startswith("seed"):
            try:
                tags["seed"] = int(p[len("seed"):])
            except ValueError:
                pass
    tags["config"] = (f"nw{tags['nw']}_ws{tags['ws']}"
                      if tags["nw"] and tags["ws"] else "unknown")
    return tags


def _aggregate(agg_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for stats_path in sorted(agg_dir.glob("*/stats.json")):
        with open(stats_path) as f:
            s = json.load(f)
        tags = _parse_run_name(s["run_name"])
        rows.append({**s, **tags})
    if not rows:
        logger.warning("No stats.json files under %s", agg_dir)
        return

    figs = agg_dir / "figures"
    figs.mkdir(exist_ok=True)

    # η² bar plot: one bar per run, grouped by regularizer/coeff
    labels = [f"{r['regularizer']}"
              + (f"({r['coeff']})" if r["coeff"] is not None else "")
              + f"\n{r['config']} s{r['seed']}" for r in rows]
    eta = [r["eta_sq"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rows)), 4))
    bars = ax.bar(range(len(rows)), eta)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("η² = Var_subject / Var_total")
    ax.set_title("Fraction of representation variance explained by subject identity")
    ax.axhline(1 / rows[0]["n_clips_per_rec"], color="gray", linestyle="--",
               label=f"null (iid, 1/K={1/rows[0]['n_clips_per_rec']:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "eta_sq_by_condition.png", dpi=150)
    plt.close(fig)

    # Eigenspectra: log plot of C_subject vs C_within per config
    configs = sorted({r["config"] for r in rows})
    for cfg_name in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg_name]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for r in cfg_rows:
            lbl = f"{r['regularizer']}"
            if r["coeff"] is not None:
                lbl += f"({r['coeff']})"
            lbl += f" s{r['seed']}"
            axes[0].plot(r["eigvals_subject"], label=lbl, alpha=0.8)
            axes[1].plot(r["eigvals_within"], label=lbl, alpha=0.8)
        for ax, title in zip(axes, ("C_subject", "C_within")):
            ax.set_yscale("log")
            ax.set_xlabel("component index")
            ax.set_title(title)
        axes[0].set_ylabel("eigenvalue")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Eigenspectra — {cfg_name}")
        fig.tight_layout()
        fig.savefig(figs / f"eigenspectra_{cfg_name}.png", dpi=150)
        plt.close(fig)

    # Principal angles at k=5
    k_key = "k=5"
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rows)), 4))
    mean_angles = [np.mean(r["principal_angles_deg"].get(k_key, [np.nan]))
                   for r in rows]
    ax.bar(range(len(rows)), mean_angles)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean principal angle (deg)")
    ax.set_title(f"Subject vs within-subject subspace overlap (top-5, {k_key})")
    ax.axhline(90, color="gray", linestyle="--", label="orthogonal")
    ax.set_ylim(0, 95)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "principal_angles_k5.png", dpi=150)
    plt.close(fig)

    # Stimulus fraction plot — only renders if the field is present (older
    # stats.json files lack it; they'll show 0 until reanalyze is run).
    stim_frac = [r.get("stim_frac_of_within", 0.0) for r in rows]
    stim_null = 1.0 / rows[0]["S"] if rows else 0.0
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(rows)), 4))
    ax.bar(range(len(rows)), stim_frac)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Var_stimulus / Var_within")
    ax.set_title("Stimulus-locked share of within-subject variance")
    ax.axhline(stim_null, color="gray", linestyle="--",
               label=f"null (iid, ≈1/S={stim_null:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "stimulus_fraction.png", dpi=150)
    plt.close(fig)

    # Summary markdown
    md_lines = [
        "# Variance Decomposition Summary",
        "",
        "| run | regularizer | coeff | config | seed | S | K | D "
        "| η²_subj | η²_stim | stim/within | eff_rank(C_subj) | eff_rank(C_within) | angle@k=5 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        angle = np.mean(r["principal_angles_deg"].get(k_key, [np.nan]))
        md_lines.append(
            f"| {r['run_name']} | {r['regularizer']} | {r['coeff']} | "
            f"{r['config']} | {r['seed']} | {r['S']} | {r['K']} | {r['D']} | "
            f"{r['eta_sq']:.3f} | "
            f"{r.get('eta_sq_stimulus', float('nan')):.4f} | "
            f"{r.get('stim_frac_of_within', float('nan')):.4f} | "
            f"{r['eff_rank_subject']:.2f} | {r['eff_rank_within']:.2f} | "
            f"{angle:.1f} |"
        )
    (agg_dir / "summary.md").write_text("\n".join(md_lines) + "\n")
    logger.info("Wrote %s and figures/ under %s", agg_dir / "summary.md", agg_dir)


# ---------------------------------------------------------------------------
# Reanalyze: recompute stats.json from saved embeddings.npz files
# ---------------------------------------------------------------------------


def _reanalyze(dir_path: Path):
    """Walk dir_path/*/embeddings.npz, recompute stats.json with latest fields."""
    import json as _json
    n = 0
    for npz_path in sorted(dir_path.glob("*/embeddings.npz")):
        run_dir = npz_path.parent
        data = np.load(npz_path, allow_pickle=True)
        Z = data["Z"]
        stats, _raw = decompose(Z)

        # Preserve external metadata from the old stats.json if present.
        old = {}
        stats_path = run_dir / "stats.json"
        if stats_path.exists():
            old = _json.load(open(stats_path))
        new = asdict(stats)
        for k in ("checkpoint", "run_name", "split", "n_windows",
                  "window_size_seconds", "n_clips_per_rec", "embed_dim",
                  "encoder_depth"):
            if k in old:
                new[k] = old[k]
        new.setdefault("run_name", run_dir.name)
        with open(stats_path, "w") as f:
            _json.dump(new, f, indent=2)
        n += 1
        logger.info("  reanalyzed %s: η²_subj=%.3f  η²_stim=%.4f  stim/within=%.4f",
                    run_dir.name, stats.eta_sq,
                    stats.eta_sq_stimulus, stats.stim_frac_of_within)
    logger.info("Reanalyzed %d runs under %s", n, dir_path)


# ---------------------------------------------------------------------------
# Self-test (no checkpoint required)
# ---------------------------------------------------------------------------


def _selftest():
    rng = np.random.default_rng(0)
    S, K, D = 50, 10, 16

    # Case A: iid Gaussian — η²_subj ≈ 1/K, η²_stim ≈ 1/S.
    Z = rng.standard_normal((S, K, D))
    stats, _ = decompose(Z)
    print("[iid] η²_subj=%.3f (≈1/K=%.3f)  η²_stim=%.4f (≈1/S=%.4f)"
          % (stats.eta_sq, 1 / K, stats.eta_sq_stimulus, 1 / S))
    print("[iid] trace identity residual: %.2e" % stats.trace_identity_residual)
    print("[iid] var_total=%.4f  subj+within=%.4f  subj+stim+res=%.4f"
          % (stats.var_total,
             stats.var_subject + stats.var_within,
             stats.var_subject + stats.var_stimulus + stats.var_residual))

    # Case B: strong subject structure.
    offsets = rng.standard_normal((S, 1, D)) * 5.0
    Z2 = rng.standard_normal((S, K, D)) + offsets
    stats2, _ = decompose(Z2)
    print("[subj] η²_subj=%.3f (expected high)  η²_stim=%.4f (expected ≈1/S)"
          % (stats2.eta_sq, stats2.eta_sq_stimulus))

    # Case C: strong clip-position (stimulus) structure, shared across subjects.
    stim = rng.standard_normal((1, K, D)) * 5.0
    Z3 = rng.standard_normal((S, K, D)) + stim
    stats3, _ = decompose(Z3)
    print("[stim] η²_subj=%.3f (expected ≈1/K)  η²_stim=%.3f (expected high)"
          % (stats3.eta_sq, stats3.eta_sq_stimulus))
    print("[stim] stim/within=%.3f" % stats3.stim_frac_of_within)

    assert abs(stats.var_total - (stats.var_subject + stats.var_within)) < 1e-8
    assert abs(stats.var_within - (stats.var_stimulus + stats.var_residual)) < 1e-8
    assert stats.trace_identity_residual < 1e-8
    assert stats2.eta_sq > 0.5
    assert stats3.eta_sq_stimulus > 0.5
    print("selftest OK")


if __name__ == "__main__":
    fire.Fire(run)
