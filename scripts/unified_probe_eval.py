"""Unified canonical probe eval — one script, five probe families.

Implements the standard evaluation protocol specified in
``docs/evaluation_guide.md``:

  - Stim regression (4 features × {Pearson r, R²}):     Ridge(α=1.0)
  - Stim classification (4 features × {AUC, bal_acc}):  LogReg(C=1, lbfgs)
  - Subject age (continuous):                           Ridge(α=1.0)
  - Subject sex (binary):                               LogReg(C=1, lbfgs)
  - Movie ID (20-bin top-1, top-5):                     LogReg(C=1, multinomial, lbfgs)

All on kc-pool features (5 channels × 64 dim = 320-d per clip), n_passes
random clip draws averaged per recording. Closed-form / LBFGS solvers
throughout — deterministic given (encoder, probe_seed).

Output:
  - results/<exp_tag>/<seed>/metrics.json — all 18 headline numbers
  - predictions/<exp_tag>/<seed>/test_seed{seed}.npz — per-recording preds
    for bootstrap (B=2000 recording-level)

Usage on Delta:
  PYTHONPATH=. uv run --group eeg python scripts/unified_probe_eval.py \\
      --checkpoint=/path/to/latest.pth.tar \\
      --n_windows=2 --window_size_seconds=4 \\
      --norm_mode=per_recording --corrca_filters=corrca_filters.npz \\
      --n_passes=20 --seed=42 \\
      --output_json=results/unified/<tag>_seed42.json \\
      --save_predictions_dir=predictions/unified/<tag>_seed42

Reference: ``docs/evaluation_guide.md`` (canonical protocol).
"""

import json
import math
import os
from pathlib import Path

import fire
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_seed
from experiments.eeg_jepa.main import resolve_preprocessed_dir

logger = get_logger(__name__)


# ============================================================
# Feature extraction — pluggable via `--feature_source` flag.
# All sources return a 1D np.ndarray per clip; kc-pool semantics where applicable.
# ============================================================

BAND_EDGES = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def _load_encoder(checkpoint, train_set, cfg, n_windows, random_init=False, variant=None):
    """Load (or randomly init) the EEGEncoderTokens context encoder.

    variant: optional surgical modification applied AFTER weight load/init.
        - "no_transformer": replace transformer with identity (returns input).
            Tests: contribution of attention+MLP on top of patch_embed+pos.
        - "no_attn": zero out each attn.to_out.weight so attn(x)=0 (residual passes
            through). Tests: attention vs MLP-only mixing at random init.
        - "no_pos": replace positional embedding with zeros. Tests: contribution of
            clip-invariant 4D Fourier pos signal (expected ~0 after Ridge std).
    """
    from eb_jepa.architectures import EEGEncoderTokens
    from eb_jepa.training_utils import setup_device

    device = setup_device("auto")
    encoder = EEGEncoderTokens(
        n_chans=train_set.n_chans,
        n_times=train_set.n_times,
        embed_dim=cfg.model.encoder_embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=n_windows,
        patch_size=cfg.model.get("patch_size", 50),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=train_set.get_chs_info(),
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
    ).to(device)
    if not random_init:
        sd_dict = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = sd_dict.get("model_state_dict", sd_dict)
        ce_sd = {
            k[len("context_encoder.") :]: v
            for k, v in sd.items()
            if k.startswith("context_encoder.")
        }
        encoder.load_state_dict(ce_sd, strict=False)

    if variant == "no_transformer":
        encoder.transformer = nn.Identity()
    elif variant == "no_attn":
        for attn, _ff in encoder.transformer.layers:
            attn.to_out.weight.data.zero_()
    elif variant == "no_pos":
        n_tok = (
            encoder.n_chans * encoder.n_windows * encoder.n_patches_per_window
        )
        zero_pos = torch.zeros(1, n_tok, encoder.embed_dim, device=device)
        encoder._compute_pos_embed = lambda B, dev, _z=zero_pos: _z.expand(B, -1, -1)
    elif variant is not None:
        raise ValueError(f"unknown encoder variant: {variant}")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, device


def _kc_features(eeg, encoder, device):
    """kc-pool: C ch × D d per clip, mean across windows. eeg: [nw, C, T_samp]."""
    with torch.no_grad():
        x = eeg.unsqueeze(0).to(device)
        tokens = encoder.encode_tokens(x, mask=None)
        B = tokens.shape[0]
        C = encoder.n_chans
        T = encoder.n_windows
        P = encoder.n_patches_per_window
        D = encoder.embed_dim
        x_tok = tokens.view(B, C, T, P, D)
        pooled = x_tok.mean(dim=3)  # [B, C, T, D]
        pooled = pooled.permute(0, 2, 1, 3).reshape(B, T, C * D)
        emb = pooled.mean(dim=1).squeeze(0).cpu().numpy()  # [C*D]
    return emb


def _raw_corrca_features(eeg, downsample_to=100):
    """Box-mean-pool CorrCA-projected EEG to fixed sample count, mean over windows.

    eeg: [nw, C, T_samp] (CorrCA-projected by dataset). Returns [C * downsample_to].
    """
    nw, C, Ts = eeg.shape
    factor = max(Ts // downsample_to, 1)
    Ts_trim = factor * downsample_to
    x = eeg[..., :Ts_trim].reshape(nw, C, downsample_to, factor).mean(-1)  # [nw, C, ds]
    return x.mean(0).flatten().numpy().astype(np.float32)  # [C * ds]


def _psd_band_features(eeg, sfreq):
    """Welch PSD per channel × 5 bands, log-pow, mean over windows.

    eeg: [nw, C, T_samp] (no CorrCA). Returns [C * 5].
    """
    from scipy.signal import welch

    nw, C, Ts = eeg.shape
    flat = eeg.numpy().astype(np.float64).reshape(nw * C, Ts)
    nperseg = min(int(sfreq), 256, Ts)
    f, P = welch(flat, fs=float(sfreq), nperseg=nperseg, axis=-1)
    feats = np.zeros((nw * C, len(BAND_EDGES)), dtype=np.float32)
    for i, (lo, hi) in enumerate(BAND_EDGES.values()):
        mask = (f >= lo) & (f < hi)
        if mask.sum() == 0:
            continue
        bp = P[:, mask].mean(axis=-1)
        feats[:, i] = np.log10(bp + 1e-12).astype(np.float32)
    feats = feats.reshape(nw, C, len(BAND_EDGES))  # [nw, C, 5]
    return feats.mean(0).flatten().astype(np.float32)  # [C * 5]


def _stats7_features(eeg, sfreq, chan_slice=None, pooled=False):
    """Per-channel 7 summary stats: mean, std, log-power in {δ, θ, α, β, γ}.

    Matches paper's "Trivial CorrCA stats" (5×7=35) and "Trivial raw 129-ch stats"
    (129×7=903) baselines.

    eeg: [nw, C, T_samp].
    chan_slice: list of channel indices to keep (e.g. [0] for chan1_only D=7).
    pooled: if True, average stats across channels then tile back to keep D unchanged
            (matches the "_pooled" variants in trivial_ridge_baseline.py — feature
            dimension stays the same but rank collapses to 7).
    Returns [C_kept * 7] (or [C_kept * 7] tiled if pooled=True).
    """
    from scipy.signal import welch

    nw, C, Ts = eeg.shape
    eeg_np = eeg.numpy().astype(np.float64)
    if chan_slice is not None:
        eeg_np = eeg_np[:, chan_slice, :]
        C = eeg_np.shape[1]
    means = eeg_np.mean(axis=(0, 2))
    stds = eeg_np.std(axis=(0, 2))
    flat = eeg_np.reshape(nw * C, Ts)
    nperseg = min(int(sfreq), 256, Ts)
    f, P = welch(flat, fs=float(sfreq), nperseg=nperseg, axis=-1)
    band_pow = np.zeros((nw * C, len(BAND_EDGES)), dtype=np.float64)
    for i, (lo, hi) in enumerate(BAND_EDGES.values()):
        mask = (f >= lo) & (f < hi)
        if mask.sum() == 0:
            continue
        band_pow[:, i] = np.log10(P[:, mask].mean(axis=-1) + 1e-12)
    band_pow = band_pow.reshape(nw, C, len(BAND_EDGES)).mean(axis=0)
    feats = np.concatenate([means[:, None], stds[:, None], band_pow], axis=1).astype(
        np.float32
    )  # [C, 7]
    if pooled:
        pooled_stats = feats.mean(axis=0, keepdims=True)
        feats = np.broadcast_to(pooled_stats, (C, 7)).copy()
    return feats.flatten()  # [C * 7]


def _make_feature_fn(feature_source, checkpoint, train_set, cfg, n_windows, sfreq):
    """Build the per-clip feature_fn(eeg) -> 1D np.ndarray for the requested source."""
    if feature_source == "jepa":
        encoder, device = _load_encoder(checkpoint, train_set, cfg, n_windows)
        return (
            lambda eeg: _kc_features(eeg, encoder, device),
            encoder.embed_dim * encoder.n_chans,
        )
    if feature_source == "random_init":
        encoder, device = _load_encoder(
            None, train_set, cfg, n_windows, random_init=True
        )
        return (
            lambda eeg: _kc_features(eeg, encoder, device),
            encoder.embed_dim * encoder.n_chans,
        )
    if feature_source == "random_no_transformer":
        # Ablation: random patch_embed + 4D Fourier pos, transformer = identity.
        # Isolates the random-projection contribution (no attn/MLP mixing).
        encoder, device = _load_encoder(
            None, train_set, cfg, n_windows, random_init=True, variant="no_transformer"
        )
        return (
            lambda eeg: _kc_features(eeg, encoder, device),
            encoder.embed_dim * encoder.n_chans,
        )
    if feature_source == "random_no_attn":
        # Ablation: random patch_embed + pos + GEGLU MLP only (attention zeroed).
        # Tests whether random softmax attention helps or hurts at init.
        encoder, device = _load_encoder(
            None, train_set, cfg, n_windows, random_init=True, variant="no_attn"
        )
        return (
            lambda eeg: _kc_features(eeg, encoder, device),
            encoder.embed_dim * encoder.n_chans,
        )
    if feature_source == "random_no_pos":
        # Ablation: full random encoder with positional embedding zeroed.
        # Sanity: pos is clip-invariant so should be killed by Ridge std anyway → ~0 Δ.
        encoder, device = _load_encoder(
            None, train_set, cfg, n_windows, random_init=True, variant="no_pos"
        )
        return (
            lambda eeg: _kc_features(eeg, encoder, device),
            encoder.embed_dim * encoder.n_chans,
        )
    if feature_source == "raw_corrca":
        # eeg comes from dataset already CorrCA-projected (when corrca_filters set)
        ds = 100
        return (
            lambda eeg: _raw_corrca_features(eeg, downsample_to=ds),
            train_set.n_chans * ds,
        )
    if feature_source == "raw_corrca_64":
        # Matched-D linear ceiling: 5 chans × 64 ds_samples = 320-d (matches JEPA kc-pool)
        return (
            lambda eeg: _raw_corrca_features(eeg, downsample_to=64),
            train_set.n_chans * 64,
        )
    if feature_source == "raw_corrca_pca":
        # Wide extraction (5 × 200 = 1000-d); PCA-projected per-channel to 64 in run() → 320-d.
        return (
            lambda eeg: _raw_corrca_features(eeg, downsample_to=200),
            train_set.n_chans * 200,
        )
    if feature_source == "psd_band":
        # PSD-only on UNPROJECTED EEG (no mean/std).
        return lambda eeg: _psd_band_features(
            eeg, sfreq=sfreq
        ), train_set.n_chans * len(BAND_EDGES)
    if feature_source == "corrca_stats":
        # Paper baseline: 7 stats × 5 CorrCA chans = 35-d. Caller MUST pass corrca_filters.
        return lambda eeg: _stats7_features(eeg, sfreq=sfreq), train_set.n_chans * 7
    if feature_source == "corrca_stats_chan1":
        # Diagnostic: stats from first CorrCA channel only (D=7).
        return lambda eeg: _stats7_features(eeg, sfreq=sfreq, chan_slice=[0]), 7
    if feature_source == "corrca_stats_pooled":
        # Diagnostic: 5×7=35 dims but pooled across channels (rank-7).
        return (
            lambda eeg: _stats7_features(eeg, sfreq=sfreq, pooled=True),
            train_set.n_chans * 7,
        )
    if feature_source == "raw_stats":
        # Paper baseline: 7 stats × 129 raw chans = 903-d. Caller MUST set corrca_filters="".
        return lambda eeg: _stats7_features(eeg, sfreq=sfreq), train_set.n_chans * 7
    if feature_source == "raw_stats_pooled":
        # Diagnostic: 129×7=903 dims but pooled across channels (rank-7). No CorrCA.
        return (
            lambda eeg: _stats7_features(eeg, sfreq=sfreq, pooled=True),
            train_set.n_chans * 7,
        )
    if feature_source == "trf_corrca5":
        # TRF-style backward decoder on CorrCA-5: high-resolution time-stacked
        # EEG (5 chans × 100 time-bins per window-mean = 500-d). Box-mean to 100
        # samples preserves multi-lag structure that Ridge can find weights for
        # — closer to canonical mTRF backward decoder (Crosse 2016) than to the
        # 7-stat summary in corrca_stats. Caller MUST pass corrca_filters.
        return (
            lambda eeg: _raw_corrca_features(eeg, downsample_to=100),
            train_set.n_chans * 100,
        )
    if feature_source == "trf_raw129":
        # TRF backward decoder on raw 129-channel EEG: 129 × 50 time-bins =
        # 6450-d. Aggressive downsample (50 vs 100) keeps Ridge tractable; 25 Hz
        # effective sampling rate is sufficient for canonical TRF lag patterns
        # on a stim envelope at < 12.5 Hz. Caller MUST set corrca_filters="".
        return (
            lambda eeg: _raw_corrca_features(eeg, downsample_to=50),
            train_set.n_chans * 50,
        )
    raise ValueError(f"unsupported feature_source: {feature_source}")


def _extract(dataset, n_passes, feature_fn, seed, train_order=False):
    """Per-recording: n_passes random clips → per-clip features + per-clip labels.

    feature_fn: callable (eeg [nw,C,T]) -> 1D np.ndarray feature vector.
    Returns features [n_rec, n_passes, D] and labels [n_rec, n_passes, n_features].
    """
    rng = torch.Generator().manual_seed(seed)
    n_rec = len(dataset)
    if train_order:
        feats_buckets = [[None] * n_passes for _ in range(n_rec)]
        labels_buckets = [[None] * n_passes for _ in range(n_rec)]
        for p in range(n_passes):
            for rec_idx in torch.randperm(n_rec, generator=rng).tolist():
                eeg, feats, _ = dataset[rec_idx]
                feats_buckets[rec_idx][p] = feature_fn(eeg)
                labels_buckets[rec_idx][p] = feats.mean(dim=0).numpy()
        feats_arr = [np.stack(fb) for fb in feats_buckets]
        labels_arr = [np.stack(lb) for lb in labels_buckets]
    else:
        feats_arr = []
        labels_arr = []
        for rec_idx in range(n_rec):
            f_passes = []
            l_passes = []
            for _ in range(n_passes):
                eeg, feats, _ = dataset[rec_idx]
                f_passes.append(feature_fn(eeg))
                l_passes.append(feats.mean(dim=0).numpy())
            feats_arr.append(np.stack(f_passes))
            labels_arr.append(np.stack(l_passes))
    X = np.stack(feats_arr)  # [n_rec, n_passes, D]
    Y = np.stack(labels_arr)  # [n_rec, n_passes, n_features]
    return X, Y


def _subject_labels(dataset):
    """Extract age (float) and sex (0/1, NaN if missing) per recording."""
    n = len(dataset)
    ages = np.full(n, np.nan, dtype=np.float32)
    sexes = np.full(n, np.nan, dtype=np.float32)
    for i, m in enumerate(dataset._recording_metadata):
        a = m.get("age", None)
        if a is not None and not (isinstance(a, float) and math.isnan(a)):
            try:
                ages[i] = float(a)
            except (TypeError, ValueError):
                pass
        s = m.get("sex", m.get("gender", ""))
        if isinstance(s, str):
            s = s.strip().lower()
            if s in ("m", "male", "1", "1.0"):
                sexes[i] = 1.0
            elif s in ("f", "female", "0", "0.0"):
                sexes[i] = 0.0
    return ages, sexes


# ============================================================
# Probe families (canonical heads per evaluation_guide.md).
# ============================================================


def _pearson_safe_local(p, y):
    if np.std(p) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(pearsonr(p, y).statistic)


def _r2_local(y, pred):
    return float(r2_score(y, pred)) if len(y) >= 2 else float("nan")


def _ridge_reg(Xtr, ytr, Xev, yev):
    """Per-recording Ridge regression. Returns Pearson r + R² + per-rec preds."""
    valid_tr = ~np.isnan(ytr)
    valid_ev = ~np.isnan(yev)
    if valid_tr.sum() < 2 or valid_ev.sum() < 2:
        return float("nan"), float("nan"), np.full(len(yev), np.nan)
    ym, ys = ytr[valid_tr].mean(), ytr[valid_tr].std() + 1e-8
    probe = Ridge(alpha=1.0).fit(Xtr[valid_tr], (ytr[valid_tr] - ym) / ys)
    pred_norm = probe.predict(Xev)
    pred = pred_norm * ys + ym
    r = pearsonr(pred[valid_ev], yev[valid_ev]).statistic
    r2 = r2_score(yev[valid_ev], pred[valid_ev])
    return float(r), float(r2), pred


def _logreg_bin(Xtr, ytr, Xev, yev):
    """Binary LogReg (median-split for continuous targets via caller)."""
    valid_tr = ~np.isnan(ytr)
    valid_ev = ~np.isnan(yev)
    if valid_tr.sum() < 4 or valid_ev.sum() < 4 or len(np.unique(ytr[valid_tr])) < 2:
        return float("nan"), float("nan"), np.full(len(yev), np.nan)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(Xtr[valid_tr], ytr[valid_tr].astype(int))
    proba = clf.predict_proba(Xev)[:, 1]
    pred = (proba > 0.5).astype(int)
    auc = (
        roc_auc_score(yev[valid_ev], proba[valid_ev])
        if len(np.unique(yev[valid_ev])) > 1
        else float("nan")
    )
    bal = balanced_accuracy_score(yev[valid_ev], pred[valid_ev])
    return float(auc), float(bal), proba


def _logreg_multi(Xtr, ytr, Xev, yev, n_classes):
    """Multinomial LogReg for movie_id top-1 / top-5.

    Note: scikit-learn dropped the `multi_class` kwarg; lbfgs solver
    defaults to multinomial when n_classes > 2."""
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    clf.fit(Xtr, ytr.astype(int))
    proba = clf.predict_proba(Xev)
    # top-1 / top-5
    topk = min(5, n_classes)
    top1 = (proba.argmax(axis=1) == yev.astype(int)).mean()
    top5_idx = np.argpartition(-proba, kth=min(topk - 1, proba.shape[1] - 1), axis=1)[
        :, :topk
    ]
    top5 = np.array([yev[i].astype(int) in top5_idx[i] for i in range(len(yev))]).mean()
    return float(top1), float(top5), proba


# ============================================================
# Main entry — orchestrates all 5 probe families.
# ============================================================


def run(
    checkpoint: str = "",
    feature_source: str = "jepa",
    n_windows: int = 4,
    window_size_seconds: int = 2,
    norm_mode: str = "per_recording",
    corrca_filters: str = "",
    n_passes: int = 20,
    seed: int = 42,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    output_json: str = "",
    save_predictions_dir: str = "",
    movie_id_n_bins: int = 20,
    external_npz_dir: str = "",
):
    """Unified canonical probe eval entry point.

    feature_source="external_npz" path: skip encoder/feature_fn entirely and
    load pre-extracted features + labels from
        {external_npz_dir}/train_seed{seed}.npz
        {external_npz_dir}/val_seed{seed+1}.npz
        {external_npz_dir}/test_seed{seed+2}.npz
    Each NPZ must contain `embs [n_rec, n_passes, D]` and
    `labels [n_rec, n_passes, n_features]`. Used to plug Tier 3 / Tier 2
    forward-passes (which run in their own scripts) into the canonical
    Ridge/LogReg/bootstrap pipeline. Match scale/density/seeds via:
        n_passes=20, seeds {42,123,456,789,2025}, train extraction with
        train_order=True (outer pass × inner randperm rec), val/test with
        sequential rec × passes.
    """
    if feature_source == "jepa" and not checkpoint:
        raise ValueError("--checkpoint is required when feature_source=jepa")
    if feature_source == "external_npz" and not external_npz_dir:
        raise ValueError(
            "--external_npz_dir is required when feature_source=external_npz"
        )
    setup_seed(seed)
    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.norm_mode": norm_mode,
    }
    if corrca_filters:
        overrides["data.corrca_filters"] = corrca_filters
    cfg = load_config(fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feature_names = list(
        cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES)
    )

    logger.info("Loading datasets (n_passes=%d) ...", n_passes)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    eval_sets = {}
    for split in ("val", "test"):
        eval_sets[split] = JEPAMovieDataset(
            split=split,
            n_windows=n_windows,
            window_size_seconds=window_size_seconds,
            feature_names=feature_names,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
        )
    logger.info(
        "n_chans=%d, n_train=%d, n_val=%d, n_test=%d",
        train_set.n_chans,
        len(train_set),
        len(eval_sets["val"]),
        len(eval_sets["test"]),
    )

    sfreq = train_set.n_times / window_size_seconds

    if feature_source == "external_npz":
        npz_dir = Path(external_npz_dir)
        def _load_external(p: Path):
            d = np.load(p)
            X = np.asarray(d["embs"], dtype=np.float32)
            Y = np.asarray(d["labels"], dtype=np.float32)
            assert X.ndim == 3 and Y.ndim == 3, \
                f"{p}: expected embs/labels of shape [n_rec, n_passes, D]; got {X.shape}, {Y.shape}"
            assert X.shape[0] == Y.shape[0] and X.shape[1] == Y.shape[1], \
                f"{p}: mismatched (n_rec, n_passes) between embs and labels"
            return X, Y
        Xtr_g, Ytr_g = _load_external(npz_dir / f"train_seed{seed}.npz")
        Xv_g, Yv_g = _load_external(npz_dir / f"val_seed{seed + 1}.npz")
        Xt_g, Yt_g = _load_external(npz_dir / f"test_seed{seed + 2}.npz")
        # Audit: enforce same n_passes as the canonical protocol.
        if Xtr_g.shape[1] != n_passes:
            raise ValueError(
                f"External train NPZ has n_passes={Xtr_g.shape[1]} but "
                f"--n_passes={n_passes}. Re-extract with matching density."
            )
        # Audit: train rec count must match the dataset's train split.
        if Xtr_g.shape[0] != len(train_set):
            raise ValueError(
                f"External train NPZ has n_rec={Xtr_g.shape[0]} but "
                f"len(train_set)={len(train_set)}. Re-extract with matching split."
            )
        expected_d = Xtr_g.shape[-1]
        logger.info(
            "feature_source=external_npz, D=%d; loaded train %s, val %s, test %s",
            expected_d, Xtr_g.shape, Xv_g.shape, Xt_g.shape,
        )
    else:
        feature_fn, expected_d = _make_feature_fn(
            feature_source, checkpoint, train_set, cfg, n_windows, sfreq
        )
        logger.info(
            "feature_source=%s, expected D=%d; extracting features ...",
            feature_source, expected_d,
        )
        Xtr_g, Ytr_g = _extract(train_set, n_passes, feature_fn, seed, train_order=True)
        Xv_g, Yv_g = _extract(eval_sets["val"], n_passes, feature_fn, seed + 1)
        Xt_g, Yt_g = _extract(eval_sets["test"], n_passes, feature_fn, seed + 2)
    logger.info(
        "Train: X=%s Y=%s; Val: X=%s; Test: X=%s",
        Xtr_g.shape,
        Ytr_g.shape,
        Xv_g.shape,
        Xt_g.shape,
    )

    # Per-channel PCA reduction for raw_corrca_pca: 5×200=1000-d → 5×64=320-d
    # PCA is fit on training set per channel and applied to all splits.
    if feature_source == "raw_corrca_pca":
        from sklearn.decomposition import PCA

        n_chans_pca = train_set.n_chans
        per_chan_in = Xtr_g.shape[-1] // n_chans_pca
        n_comp = 64
        logger.info(
            "Fitting per-channel PCA: %d chans × %d→%d",
            n_chans_pca,
            per_chan_in,
            n_comp,
        )

        def _per_chan_pca(X_g):
            n_rec, n_p, D = X_g.shape
            return X_g.reshape(n_rec, n_p, n_chans_pca, per_chan_in)

        Xtr_pc = _per_chan_pca(Xtr_g)
        Xv_pc = _per_chan_pca(Xv_g)
        Xt_pc = _per_chan_pca(Xt_g)
        Xtr_red, Xv_red, Xt_red = [], [], []
        for c in range(n_chans_pca):
            tr_flat = Xtr_pc[..., c, :].reshape(-1, per_chan_in)
            v_flat = Xv_pc[..., c, :].reshape(-1, per_chan_in)
            t_flat = Xt_pc[..., c, :].reshape(-1, per_chan_in)
            pca = PCA(n_components=n_comp, random_state=seed).fit(tr_flat)
            Xtr_red.append(
                pca.transform(tr_flat).reshape(Xtr_g.shape[0], Xtr_g.shape[1], n_comp)
            )
            Xv_red.append(
                pca.transform(v_flat).reshape(Xv_g.shape[0], Xv_g.shape[1], n_comp)
            )
            Xt_red.append(
                pca.transform(t_flat).reshape(Xt_g.shape[0], Xt_g.shape[1], n_comp)
            )
        Xtr_g = np.concatenate(Xtr_red, axis=-1)
        Xv_g = np.concatenate(Xv_red, axis=-1)
        Xt_g = np.concatenate(Xt_red, axis=-1)
        logger.info(
            "After PCA: Xtr=%s Xv=%s Xt=%s", Xtr_g.shape, Xv_g.shape, Xt_g.shape
        )

    # Standardize features (per-feature train stats)
    Xtr_flat = Xtr_g.reshape(-1, Xtr_g.shape[-1])
    mu = Xtr_flat.mean(axis=0, keepdims=True)
    sd = Xtr_flat.std(axis=0, keepdims=True) + 1e-8

    def _stdz(X):
        return (X - mu) / sd

    Xtr_g = _stdz(Xtr_g)
    Xv_g = _stdz(Xv_g)
    Xt_g = _stdz(Xt_g)

    # ===== Stim regression / classification: per-clip semantics (PR #15 protocol)
    # Flatten (n_rec, n_passes) → (n_rec × n_passes,) for Ridge / LogReg fitting
    # so the probe sees ALL clips, not per-recording means. Matches
    # trivial_ridge_baseline.py exactly.
    Xtr_flat_clips = Xtr_g.reshape(-1, Xtr_g.shape[-1])  # [n_train_clips, D]
    Xv_flat_clips = Xv_g.reshape(-1, Xv_g.shape[-1])
    Xt_flat_clips = Xt_g.reshape(-1, Xt_g.shape[-1])
    Ytr_flat_clips = Ytr_g.reshape(-1, Ytr_g.shape[-1])  # [n_train_clips, n_features]
    Yv_flat_clips = Yv_g.reshape(-1, Yv_g.shape[-1])
    Yt_flat_clips = Yt_g.reshape(-1, Yt_g.shape[-1])

    # ===== Subject + movie_id: per-recording semantics
    # These are recording-level labels; can't be flattened without changing meaning.
    age_tr, sex_tr = _subject_labels(train_set)
    age_v, sex_v = _subject_labels(eval_sets["val"])
    age_t, sex_t = _subject_labels(eval_sets["test"])
    Ytr_rec = Ytr_g.mean(axis=1)
    Yv_rec = Yv_g.mean(axis=1)
    Yt_rec = Yt_g.mean(axis=1)
    Xtr_rec = Xtr_g.mean(axis=1)  # [n_rec, D]
    Xv_rec = Xv_g.mean(axis=1)
    Xt_rec = Xt_g.mean(axis=1)
    pos_idx = (
        feature_names.index("position_in_movie")
        if "position_in_movie" in feature_names
        else None
    )
    if pos_idx is not None:
        pos_tr_rec = Ytr_rec[:, pos_idx]
        pos_v_rec = Yv_rec[:, pos_idx]
        pos_t_rec = Yt_rec[:, pos_idx]
        pos_min = pos_tr_rec.min()
        pos_max = pos_tr_rec.max() + 1e-8
        edges = np.linspace(pos_min, pos_max, movie_id_n_bins + 1)
        bin_tr = np.clip(np.digitize(pos_tr_rec, edges) - 1, 0, movie_id_n_bins - 1)
        bin_v = np.clip(np.digitize(pos_v_rec, edges) - 1, 0, movie_id_n_bins - 1)
        bin_t = np.clip(np.digitize(pos_t_rec, edges) - 1, 0, movie_id_n_bins - 1)

    metrics = {}
    preds_npz = {"feature_names": np.array(feature_names, dtype="<U24")}

    # ---- Stim regression + classification (4 features each) — per-clip Ridge/LogReg
    # CANONICAL PROTOCOL B:
    #   - TRAIN: per-clip flat (n_train_rec × n_passes = ~14k samples). n_passes=20
    #     acts as data augmentation; Ridge fit on flat clips.
    #   - TEST/VAL: predict on per-clip flat (n × n_passes = ~2160 samples). Pearson
    #     r and AUC computed directly on the flat arrays. NPZ saves flat per-clip
    #     predictions; bootstrap_unified.py reshapes to (n_rec, n_passes) and
    #     resamples recordings (axis 0), then recomputes the metric on the
    #     flattened resampled array.
    for fi, fname_feat in enumerate(feature_names):
        ytr = Ytr_flat_clips[:, fi]
        yv = Yv_flat_clips[:, fi]
        yt = Yt_flat_clips[:, fi]
        # Regression
        for split, X, y, tag in [
            ("val", Xv_flat_clips, yv, "val"),
            ("test", Xt_flat_clips, yt, "test"),
        ]:
            r, r2, pred = _ridge_reg(Xtr_flat_clips, ytr, X, y)
            metrics[f"{tag}/reg_{fname_feat}_corr"] = r
            metrics[f"{tag}/reg_{fname_feat}_r2"] = r2
            preds_npz[f"{tag}_reg_{fname_feat}_pred"] = pred.astype(np.float32)
            preds_npz[f"{tag}_reg_{fname_feat}_target"] = y.astype(np.float32)
        # Classification (binarize at train median over flattened train labels)
        med = np.nanmedian(ytr)
        ytr_bin = (ytr > med).astype(np.float32)
        for split, X, y, tag in [
            ("val", Xv_flat_clips, yv, "val"),
            ("test", Xt_flat_clips, yt, "test"),
        ]:
            y_bin = (y > med).astype(np.float32)
            auc, bal, proba = _logreg_bin(Xtr_flat_clips, ytr_bin, X, y_bin)
            metrics[f"{tag}/cls_{fname_feat}_auc"] = auc
            metrics[f"{tag}/cls_{fname_feat}_bal_acc"] = bal
            preds_npz[f"{tag}_cls_{fname_feat}_proba"] = proba.astype(np.float32)
            preds_npz[f"{tag}_cls_{fname_feat}_target"] = y_bin.astype(np.float32)

    # ---- Subject — age regression ----
    for split, X, y, tag in [
        ("val", Xv_rec, age_v, "val"),
        ("test", Xt_rec, age_t, "test"),
    ]:
        r, r2, pred = _ridge_reg(Xtr_rec, age_tr, X, y)
        metrics[f"{tag}/subject/age_reg/corr"] = r
        metrics[f"{tag}/subject/age_reg/r2"] = r2
        preds_npz[f"{tag}_age_pred"] = pred.astype(np.float32)
        preds_npz[f"{tag}_age_target"] = y.astype(np.float32)

    # ---- Subject — sex classification ----
    for split, X, y, tag in [
        ("val", Xv_rec, sex_v, "val"),
        ("test", Xt_rec, sex_t, "test"),
    ]:
        auc, bal, proba = _logreg_bin(Xtr_rec, sex_tr, X, y)
        metrics[f"{tag}/subject/sex/auc"] = auc
        metrics[f"{tag}/subject/sex/bal_acc"] = bal
        preds_npz[f"{tag}_sex_proba"] = proba.astype(np.float32)
        preds_npz[f"{tag}_sex_target"] = y.astype(np.float32)

    # ---- Movie-ID 20-class top-1 / top-5 ----
    # Wrapped in try/except so a failure here doesn't drop the other 16 metrics.
    if pos_idx is not None:
        for split, X, y, tag in [
            ("val", Xv_rec, bin_v, "val"),
            ("test", Xt_rec, bin_t, "test"),
        ]:
            try:
                top1, top5, proba = _logreg_multi(
                    Xtr_rec, bin_tr, X, y, movie_id_n_bins
                )
                metrics[f"{tag}/movie_id/top1"] = top1
                metrics[f"{tag}/movie_id/top5"] = top5
                preds_npz[f"{tag}_movie_id_proba"] = proba.astype(np.float32)
                preds_npz[f"{tag}_movie_id_target"] = y.astype(np.int32)
            except Exception as e:
                logger.warning("movie_id %s probe failed: %s", tag, e)
                metrics[f"{tag}/movie_id/top1"] = float("nan")
                metrics[f"{tag}/movie_id/top5"] = float("nan")

    # rec_ids for bootstrap
    preds_npz["val_rec_ids"] = np.arange(len(eval_sets["val"]), dtype=np.int64)
    preds_npz["test_rec_ids"] = np.arange(len(eval_sets["test"]), dtype=np.int64)

    # ---- Persist ----
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(
                {
                    "checkpoint": checkpoint,
                    "n_windows": n_windows,
                    "window_size_seconds": window_size_seconds,
                    "n_passes": n_passes,
                    "seed": seed,
                    "metrics": metrics,
                    "protocol": "unified_probe_eval — kc-pool + Ridge(α=1) + LogReg(C=1, lbfgs); see docs/evaluation_guide.md",
                },
                f,
                indent=2,
            )
        logger.info("Wrote %s", output_json)

    if save_predictions_dir:
        Path(save_predictions_dir).mkdir(parents=True, exist_ok=True)
        npz_path = os.path.join(save_predictions_dir, f"test_seed{seed}.npz")
        np.savez_compressed(npz_path, **preds_npz)
        logger.info("Wrote per-rec predictions: %s", npz_path)

    # ---- Print headline (test split) ----
    logger.info("=== Headline metrics (test split) ===")
    for k in sorted(metrics):
        if k.startswith("test/"):
            logger.info("  %-50s %+.4f", k, metrics[k])


if __name__ == "__main__":
    fire.Fire(run)
