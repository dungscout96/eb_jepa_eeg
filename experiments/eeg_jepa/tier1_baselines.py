"""Tier 1 baselines for EEG-JEPA stimulus-feature probing.

Three feature sources are evaluated under the *same* probe-head + same data
splits + same per-recording z-norm pipeline as Exp 6, so all numbers land on
the same metric scale as ``probe_eval.py``:

  1. ``raw_corrca``    — CorrCA-5 projected EEG, downsampled per window.
                         Tests how much linear stimulus signal is in the
                         input *before* any encoder.
  2. ``psd_band``      — Welch PSD per channel × 5 bands (delta/theta/alpha/beta/gamma).
                         Tests whether band-power features alone match Exp 6.
  3. ``random_init``   — Exp 6 architecture (CorrCA-5 + EEGEncoderTokens
                         depth=2, embed_dim=64) with random weights, tokens
                         mean-pooled. Tests whether SSL pretraining matters
                         vs. just the architecture + linear probes.

Probe heads (identical to ``probe_eval.py``):
  * Movie-feature regression / classification — ``MovieFeatureHead`` (MLP)
    applied per window. Reports per-feature Pearson r and AUC on val/test.
  * Subject-trait probes — single ``nn.Linear`` on per-recording mean
    embedding for age (regression + median classification) and sex.
  * Movie identity — ``nn.Linear`` over 20 temporal bins of position_in_movie.

Usage
-----
    PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/tier1_baselines.py \\
        --baseline=raw_corrca --seed=42 --splits=val,test \\
        --corrca_filters=corrca_filters.npz \\
        --output_json=/abs/path/to/tier1_raw_corrca_seed42.json

The output JSON mirrors the metric keys produced by ``probe_eval.py``.
"""

import json
import math
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.signal import welch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from scipy.stats import pearsonr
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import EEGEncoderTokens, MovieFeatureHead
from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_device, setup_seed
from experiments.eeg_jepa.main import (
    ClassificationLoss,
    RegressionLoss,
    resolve_preprocessed_dir,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Frequency band definitions for PSD features (Hz)
# ---------------------------------------------------------------------------
BAND_EDGES = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


# ---------------------------------------------------------------------------
# Feature extractors — each returns [B, D, T_windows, 1, 1] to feed
# MovieFeatureHead with no further reshaping.
# ---------------------------------------------------------------------------


class RawCorrCAExtractor:
    """Flatten + downsample CorrCA-5 EEG to a per-window feature vector."""

    def __init__(self, downsample_to: int = 100):
        self.downsample_to = downsample_to
        self.feat_dim = None  # set on first call

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: [B, T_win, C=5, T_samp] (CorrCA already applied by dataset)
        B, T, C, Ts = eeg.shape
        if Ts % self.downsample_to != 0:
            # Trim to nearest multiple
            Ts = (Ts // self.downsample_to) * self.downsample_to
            eeg = eeg[..., :Ts]
        factor = Ts // self.downsample_to
        # Box-mean pool: [B, T, C, downsample_to]
        x = eeg.reshape(B, T, C, self.downsample_to, factor).mean(dim=-1)
        # Per-window feature: flatten C × downsample_to
        x = x.reshape(B, T, C * self.downsample_to)  # [B, T, D]
        D = x.shape[-1]
        self.feat_dim = D
        # MovieFeatureHead expects [B, D, T, 1, 1]
        return x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)


class PSDBandExtractor:
    """Welch PSD integrated into 5 frequency bands per channel per window."""

    def __init__(self, sfreq: float, n_chans_in: int):
        self.sfreq = float(sfreq)
        self.feat_dim = n_chans_in * len(BAND_EDGES)
        # Welch window: 1 s with 50% overlap → freq resolution 1 Hz
        self.nperseg = min(int(self.sfreq), 256)

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: [B, T_win, C, T_samp] (no CorrCA)
        B, T, C, Ts = eeg.shape
        x = eeg.detach().cpu().numpy().astype(np.float64)
        # Reshape so welch can run on the time axis: [B*T*C, T_samp]
        flat = x.reshape(B * T * C, Ts)
        f, P = welch(flat, fs=self.sfreq, nperseg=min(self.nperseg, Ts), axis=-1)
        # Integrate into bands
        n_bands = len(BAND_EDGES)
        feats = np.zeros((B * T * C, n_bands), dtype=np.float32)
        for i, (lo, hi) in enumerate(BAND_EDGES.values()):
            mask = (f >= lo) & (f < hi)
            if mask.sum() == 0:
                continue
            # Average power in band, then log to compress dynamic range
            band_power = P[:, mask].mean(axis=-1)
            feats[:, i] = np.log10(band_power + 1e-12).astype(np.float32)
        feats = feats.reshape(B, T, C * n_bands)  # [B, T, D]
        out = torch.from_numpy(feats).to(eeg.device)
        return out.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)


class RandomInitEncoderExtractor:
    """Random-init EEGEncoderTokens; mean-pool tokens → per-window embedding."""

    def __init__(self, encoder: EEGEncoderTokens):
        self.encoder = encoder
        self.feat_dim = encoder.embed_dim
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: [B, T_win, C, T_samp]
        with torch.no_grad():
            tokens = self.encoder.encode_tokens(eeg, mask=None)  # [B, C*T*P, D]
            x = self.encoder.pool_to_windows(tokens)  # [B, D, T, 1, 1]
        return x


# ---------------------------------------------------------------------------
# Per-clip embedding + per-recording embedding
# ---------------------------------------------------------------------------


def _features_loader(loader, extractor, device, pos_idx):
    """Run extractor over a loader, return per-clip features, targets, positions."""
    all_feats, all_targets, all_positions = [], [], []
    for eeg, features, _ in tqdm(loader, desc="extracting", leave=False):
        eeg = eeg.to(device, non_blocking=True)
        emb = extractor(eeg)  # [B, D, T, 1, 1]
        all_feats.append(emb.detach().cpu())
        all_targets.append(features.cpu())
        if pos_idx is not None:
            all_positions.append(features[:, :, pos_idx].mean(dim=1).numpy())
    feats = torch.cat(all_feats, dim=0)  # [N, D, T, 1, 1]
    targets = torch.cat(all_targets, dim=0)  # [N, T, n_features]
    positions = np.concatenate(all_positions) if all_positions else None
    return feats, targets, positions


def _features_per_recording(dataset, extractor, device, max_clips_per_rec=4):
    """Mean-pool extractor output across clips × windows → [N_rec, D]."""
    rec_embs, rec_meta = [], []
    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        n_sample = min(max_clips_per_rec, n_clips)
        starts = np.linspace(0, n_clips - 1, n_sample, dtype=int)

        clip_embs = []
        for start in starts:
            indices = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
            eeg = torch.from_numpy(eeg_np)
            if dataset._norm_mode == "per_recording":
                rec_mean = eeg.mean(dim=(0, 2), keepdim=True)
                rec_std = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
                eeg = (eeg - rec_mean) / rec_std
            else:
                eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
            if dataset._add_envelope:
                eeg = dataset._append_lowfreq_envelope(eeg)
            if dataset._corrca_W is not None:
                eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)
            eeg = eeg.unsqueeze(0).to(device)  # [1, T_win, C, T_samp]
            with torch.no_grad():
                emb = extractor(eeg)  # [1, D, T, 1, 1]
            # Mean over windows → [D]
            clip_embs.append(emb.squeeze(0).mean(dim=(1, 2, 3)).cpu())

        if clip_embs:
            rec_emb = torch.stack(clip_embs).mean(dim=0)  # [D]
            rec_embs.append(rec_emb.numpy())
            rec_meta.append(dataset._recording_metadata[rec_idx])
    return np.stack(rec_embs), rec_meta


# ---------------------------------------------------------------------------
# Movie-feature probe (matches probe_eval.py: MovieFeatureHead MLP)
# ---------------------------------------------------------------------------


def _train_movie_probes(train_feats, train_targets, n_features, hdec, device,
                        feature_stats, feature_median, probe_epochs, probe_lr,
                        batch_size=256):
    """Train MovieFeatureHead regression + classification probes."""
    D = train_feats.shape[1]
    reg_head = MovieFeatureHead(D, hdec, n_features).to(device)
    cls_head = MovieFeatureHead(D, hdec, n_features).to(device)
    reg_loss_fn = RegressionLoss(
        feature_stats["mean"].to(device), feature_stats["std"].to(device)
    )
    cls_loss_fn = ClassificationLoss(feature_median.to(device))

    opt = Adam(
        list(reg_head.parameters()) + list(cls_head.parameters()), lr=probe_lr
    )

    n = train_feats.shape[0]
    for epoch in range(probe_epochs):
        reg_head.train(); cls_head.train()
        perm = torch.randperm(n)
        reg_total = cls_total = 0.0
        n_batches = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            x = train_feats[idx].to(device)  # [B, D, T, 1, 1]
            y = train_targets[idx].to(device)  # [B, T, n_features]
            opt.zero_grad()
            reg_pred = reg_head(x)
            cls_pred = cls_head(x)
            reg_loss = reg_loss_fn(reg_pred, y)
            cls_loss = cls_loss_fn(cls_pred, y)
            (reg_loss + cls_loss).backward()
            opt.step()
            reg_total += reg_loss.item()
            cls_total += cls_loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == probe_epochs - 1:
            logger.info(
                "Movie probe ep %d/%d  reg=%.4f cls=%.4f",
                epoch + 1, probe_epochs,
                reg_total / max(n_batches, 1), cls_total / max(n_batches, 1),
            )
    return reg_head, cls_head


@torch.inference_mode()
def _eval_movie_probes(reg_head, cls_head, eval_feats, eval_targets,
                       feature_stats, feature_median, feature_names, device,
                       batch_size=512):
    reg_head.eval(); cls_head.eval()
    n = eval_feats.shape[0]
    reg_preds, cls_preds = [], []
    for s in range(0, n, batch_size):
        x = eval_feats[s:s + batch_size].to(device)
        reg_preds.append(reg_head(x).cpu())
        cls_preds.append(cls_head(x).cpu())
    reg_preds = torch.cat(reg_preds, dim=0).flatten(0, 1).numpy()
    cls_preds = torch.cat(cls_preds, dim=0).flatten(0, 1)
    targets = eval_targets.flatten(0, 1).numpy()

    mean = feature_stats["mean"].numpy()
    std = feature_stats["std"].numpy()
    reg_preds_un = reg_preds * (std + 1e-8) + mean

    metrics = {}
    for i, name in enumerate(feature_names):
        pred = reg_preds_un[:, i]
        targ = targets[:, i]
        if np.std(targ) > 1e-10 and np.std(pred) > 1e-10:
            metrics[f"reg_{name}_corr"] = float(pearsonr(pred, targ).statistic)
        else:
            metrics[f"reg_{name}_corr"] = 0.0

    cls_probs = torch.sigmoid(cls_preds).numpy()
    cls_labels = (cls_probs > 0.5).astype(int)
    median_np = feature_median.numpy()
    binary_targets = (targets > median_np).astype(int)
    for i, name in enumerate(feature_names):
        pred_label = cls_labels[:, i]
        true_label = binary_targets[:, i]
        prob = cls_probs[:, i]
        metrics[f"cls_{name}_acc"] = float(accuracy_score(true_label, pred_label))
        metrics[f"cls_{name}_bal_acc"] = float(
            balanced_accuracy_score(true_label, pred_label)
        )
        try:
            metrics[f"cls_{name}_auc"] = float(roc_auc_score(true_label, prob))
        except ValueError:
            metrics[f"cls_{name}_auc"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Movie identity probe (per-clip, 20 bins)
# ---------------------------------------------------------------------------


def _train_movie_id_probe(clip_embs, positions, device, n_bins, probe_epochs,
                          probe_lr):
    bin_edges = np.linspace(positions.min(), positions.max() + 1e-8, n_bins + 1)
    bin_labels = np.clip(np.digitize(positions, bin_edges) - 1, 0, n_bins - 1)
    X = torch.from_numpy(clip_embs).float().to(device)
    y = torch.from_numpy(bin_labels).long().to(device)
    D = X.shape[1]
    probe = nn.Linear(D, n_bins).to(device)
    opt = Adam(probe.parameters(), lr=probe_lr)
    probe.train()
    for _ in range(probe_epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(probe(X), y)
        loss.backward()
        opt.step()
    return probe, bin_edges


@torch.inference_mode()
def _eval_movie_id_probe(probe, clip_embs, positions, device, bin_edges):
    n_bins = len(bin_edges) - 1
    bin_labels = np.clip(np.digitize(positions, bin_edges) - 1, 0, n_bins - 1)
    X = torch.from_numpy(clip_embs).float().to(device)
    y_true = torch.from_numpy(bin_labels).long()
    logits = probe(X).cpu()
    preds = logits.argmax(dim=1)
    top1 = float((preds == y_true).float().mean())
    k = min(5, n_bins)
    _, top_k = logits.topk(k, dim=1)
    top5 = float((top_k == y_true.unsqueeze(1)).any(dim=1).float().mean())
    return {"top1_acc": top1, "top5_acc": top5, "chance": 1.0 / n_bins,
            "n_bins": n_bins}


# ---------------------------------------------------------------------------
# Subject-trait probes (per-recording linear)
# ---------------------------------------------------------------------------


def _extract_subject_labels(metadata_list, median_age=None):
    n = len(metadata_list)
    ages = np.full(n, np.nan)
    sexes = np.full(n, np.nan)
    for i, m in enumerate(metadata_list):
        if "age" in m:
            try:
                ages[i] = float(m["age"])
            except (ValueError, TypeError):
                pass
        sex_val = m.get("sex", m.get("gender", ""))
        if isinstance(sex_val, str):
            s = sex_val.strip().lower()
            if s in ("m", "male"):
                sexes[i] = 1.0
            elif s in ("f", "female"):
                sexes[i] = 0.0
    labels = {}
    valid_ages = ages[~np.isnan(ages)]
    if len(valid_ages) >= 10:
        labels["age_reg"] = ages
        if median_age is None:
            median_age = float(np.median(valid_ages))
        labels["age_cls"] = np.where(np.isnan(ages), np.nan,
                                     (ages > median_age).astype(float))
    valid_sex = sexes[~np.isnan(sexes)]
    if len(valid_sex) >= 10:
        labels["sex"] = sexes
    return labels


def _train_cls_probe(train_embs, labels, device, probe_epochs, probe_lr):
    valid = ~np.isnan(labels)
    X = torch.from_numpy(train_embs[valid]).float().to(device)
    y = torch.from_numpy(labels[valid]).float().to(device)
    probe = nn.Linear(X.shape[1], 1).to(device)
    opt = Adam(probe.parameters(), lr=probe_lr)
    probe.train()
    for _ in range(probe_epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            probe(X).squeeze(-1), y
        )
        loss.backward()
        opt.step()
    return probe


def _train_reg_probe(train_embs, labels, device, probe_epochs, probe_lr):
    valid = ~np.isnan(labels)
    X = torch.from_numpy(train_embs[valid]).float().to(device)
    y = torch.from_numpy(labels[valid]).float().to(device)
    y_mean, y_std = y.mean(), y.std().clamp(min=1e-6)
    y_norm = (y - y_mean) / y_std
    probe = nn.Linear(X.shape[1], 1).to(device)
    opt = Adam(probe.parameters(), lr=probe_lr)
    probe.train()
    for _ in range(probe_epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(probe(X).squeeze(-1), y_norm)
        loss.backward()
        opt.step()
    return probe, y_mean.item(), y_std.item()


@torch.inference_mode()
def _eval_cls_probe(probe, embs, labels, device):
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return {"bal_acc": float("nan"), "auc": float("nan")}
    X = torch.from_numpy(embs[valid]).float().to(device)
    logits = probe(X).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y_true = labels[valid].astype(int)
    bal_acc = balanced_accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    return {"bal_acc": float(bal_acc), "auc": float(auc)}


@torch.inference_mode()
def _eval_reg_probe(probe, embs, labels, device, y_mean, y_std):
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return {"mae": float("nan"), "corr": float("nan"), "r2": float("nan")}
    X = torch.from_numpy(embs[valid]).float().to(device)
    y_true = labels[valid]
    y_pred = probe(X).squeeze(-1).cpu().numpy() * y_std + y_mean
    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "corr": corr, "r2": r2}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    baseline: str,
    seed: int = 42,
    # Data — match Exp 6 defaults
    n_windows: int = 4,
    window_size_seconds: int = 2,
    batch_size: int = 64,
    num_workers: int = 4,
    norm_mode: str = "per_recording",
    corrca_filters: str = "corrca_filters.npz",
    # Probe-head capacity (matches default.yaml: hdec=64)
    hdec: int = 64,
    # Probe training
    probe_epochs: int = 20,
    probe_lr: float = 1e-3,
    subject_probe_epochs: int = 100,
    subject_probe_lr: float = 1e-3,
    # PSD baseline
    psd_downsample_to: int = 100,
    # Random-init encoder shape (defaults match Exp 6, not default.yaml)
    encoder_depth: int = 2,
    encoder_embed_dim: int = 64,
    encoder_heads: int = 4,
    encoder_head_dim: int = 16,
    # Eval splits
    splits: str = "val,test",
    # Output
    output_json: str = "",
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
):
    """Run a Tier 1 baseline.

    Args:
        baseline: One of {'raw_corrca', 'psd_band', 'random_init'}.
        seed: Random seed (controls extractor init for random_init + probe init).
        n_windows: Windows per clip (must match Exp 6 = 4 for comparability).
        window_size_seconds: Window length seconds (must match Exp 6 = 2).
        norm_mode: EEG normalization mode (default 'per_recording', as Exp 6).
        corrca_filters: Path to CorrCA .npz. Used by raw_corrca + random_init.
                        Set to '' to disable (required for psd_band, which
                        operates on raw 129-ch EEG).
        hdec: MovieFeatureHead hidden dim (matches default.yaml hdec=64).
        psd_downsample_to: Unused by raw_corrca (kept for arg parity).
        output_json: If set, dump all metrics to this path.
    """
    assert baseline in ("raw_corrca", "psd_band", "random_init"), baseline
    setup_seed(seed)
    device = setup_device("auto")

    # Build config; PSD baseline must NOT apply CorrCA (it works on raw 129-ch).
    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "data.norm_mode": norm_mode,
        "model.encoder_depth": encoder_depth,
        "model.encoder_embed_dim": encoder_embed_dim,
        "model.encoder_heads": encoder_heads,
        "model.encoder_head_dim": encoder_head_dim,
    }
    if baseline == "psd_band":
        overrides["data.corrca_filters"] = None
    else:
        overrides["data.corrca_filters"] = corrca_filters or None

    cfg = load_config(fname, overrides)

    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feature_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))
    n_features = len(feature_names)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading train set (baseline=%s, seed=%d)...", baseline, seed)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feature_names,
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    feature_stats = train_set.compute_feature_stats()
    feature_median = train_set.compute_feature_median()

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    eval_sets, eval_loaders = {}, {}
    splits_list = [s.strip() for s in splits.split(",")]
    for split in splits_list:
        ds = JEPAMovieDataset(
            split=split,
            n_windows=n_windows,
            window_size_seconds=window_size_seconds,
            feature_names=feature_names,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
        )
        eval_sets[split] = ds
        eval_loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    n_chans_in = train_set.n_chans  # post-CorrCA channel count, or 129 if disabled
    n_times = train_set.n_times
    sfreq = float(train_set.sfreq)

    # ------------------------------------------------------------------
    # Build feature extractor
    # ------------------------------------------------------------------
    logger.info("Building extractor: %s (n_chans_in=%d, n_times=%d, sfreq=%.1f)",
                baseline, n_chans_in, n_times, sfreq)
    if baseline == "raw_corrca":
        assert train_set._corrca_W is not None, \
            "raw_corrca requires --corrca_filters to be set"
        extractor = RawCorrCAExtractor(downsample_to=psd_downsample_to)
    elif baseline == "psd_band":
        assert train_set._corrca_W is None, \
            "psd_band must run on raw EEG; do not pass --corrca_filters"
        extractor = PSDBandExtractor(sfreq=sfreq, n_chans_in=n_chans_in)
    elif baseline == "random_init":
        encoder = EEGEncoderTokens(
            n_chans=n_chans_in,
            n_times=n_times,
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
        extractor = RandomInitEncoderExtractor(encoder)

    # ------------------------------------------------------------------
    # Per-clip features (movie probes + movie ID)
    # ------------------------------------------------------------------
    pos_idx = (
        feature_names.index("position_in_movie")
        if "position_in_movie" in feature_names else None
    )

    logger.info("Extracting train per-clip features...")
    train_feats, train_targets, train_positions = _features_loader(
        train_loader, extractor, device, pos_idx
    )
    logger.info(
        "Train per-clip: %d clips, feat_dim=%d", train_feats.shape[0], train_feats.shape[1]
    )

    eval_feats, eval_targets, eval_positions = {}, {}, {}
    for split in splits_list:
        f, t, p = _features_loader(eval_loaders[split], extractor, device, pos_idx)
        eval_feats[split] = f
        eval_targets[split] = t
        eval_positions[split] = p
        logger.info("%s per-clip: %d clips", split, f.shape[0])

    # ------------------------------------------------------------------
    # Train movie-feature probes
    # ------------------------------------------------------------------
    logger.info("Training movie-feature probes (%d epochs)...", probe_epochs)
    reg_head, cls_head = _train_movie_probes(
        train_feats, train_targets, n_features, hdec, device,
        feature_stats, feature_median, probe_epochs, probe_lr,
    )

    # ------------------------------------------------------------------
    # Train movie-ID probe (per-clip, on mean-pooled embedding)
    # ------------------------------------------------------------------
    movie_id_probe = movie_id_bin_edges = None
    if pos_idx is not None:
        # Mean across windows → [N, D]
        train_clip_embs = train_feats.mean(dim=(2, 3, 4)).numpy()
        movie_id_probe, movie_id_bin_edges = _train_movie_id_probe(
            train_clip_embs, train_positions, device, n_bins=20,
            probe_epochs=probe_epochs, probe_lr=probe_lr,
        )

    # ------------------------------------------------------------------
    # Per-recording embeddings + subject probes
    # ------------------------------------------------------------------
    logger.info("Computing train per-recording embeddings...")
    train_rec_embs, train_meta = _features_per_recording(
        train_set, extractor, device,
    )
    train_labels = _extract_subject_labels(train_meta)
    train_ages = np.array([float(m["age"]) for m in train_meta if "age" in m])
    train_median_age = float(np.median(train_ages)) if len(train_ages) >= 10 else None
    logger.info("Train rec: %d, labels: %s", len(train_rec_embs), list(train_labels.keys()))

    subject_probes = {}
    for label_name, labels in train_labels.items():
        if label_name == "age_reg":
            probe, ym, ys = _train_reg_probe(
                train_rec_embs, labels, device,
                subject_probe_epochs, subject_probe_lr,
            )
            subject_probes[label_name] = ("reg", probe, ym, ys)
        else:
            probe = _train_cls_probe(
                train_rec_embs, labels, device,
                subject_probe_epochs, subject_probe_lr,
            )
            subject_probes[label_name] = ("cls", probe)

    # ------------------------------------------------------------------
    # Eval on each split
    # ------------------------------------------------------------------
    all_metrics = {}

    for split in splits_list:
        # Movie probes
        m = _eval_movie_probes(
            reg_head, cls_head, eval_feats[split], eval_targets[split],
            feature_stats, feature_median, feature_names, device,
        )
        for k, v in m.items():
            all_metrics[f"{split}/{k}"] = v

        # Movie ID
        if movie_id_probe is not None:
            clip_embs = eval_feats[split].mean(dim=(2, 3, 4)).numpy()
            mid = _eval_movie_id_probe(
                movie_id_probe, clip_embs, eval_positions[split],
                device, movie_id_bin_edges,
            )
            for k, v in mid.items():
                all_metrics[f"{split}/movie_id/{k}"] = v

        # Subject probes
        if subject_probes:
            rec_embs, rec_meta = _features_per_recording(
                eval_sets[split], extractor, device,
            )
            eval_labels = _extract_subject_labels(rec_meta, median_age=train_median_age)
            for label_name, info in subject_probes.items():
                ev = eval_labels.get(label_name)
                if ev is None:
                    continue
                if info[0] == "cls":
                    metrics = _eval_cls_probe(info[1], rec_embs, ev, device)
                    for k, v in metrics.items():
                        all_metrics[f"{split}/subject/{label_name}/{k}"] = v
                else:
                    _, probe, ym, ys = info
                    metrics = _eval_reg_probe(probe, rec_embs, ev, device, ym, ys)
                    for k, v in metrics.items():
                        all_metrics[f"{split}/subject/{label_name}/{k}"] = v

    # ------------------------------------------------------------------
    # Print + dump
    # ------------------------------------------------------------------
    print(f"\n=== Tier 1 baseline: {baseline} (seed={seed}) ===")
    for k in sorted(all_metrics.keys()):
        v = all_metrics[k]
        vs = f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v)
        print(f"  {k}: {vs}")

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_metrics_meta = {
            "baseline": baseline,
            "seed": seed,
            "n_windows": n_windows,
            "window_size_seconds": window_size_seconds,
            "norm_mode": norm_mode,
            "corrca_filters": corrca_filters if baseline != "psd_band" else None,
            "n_chans_in": n_chans_in,
            "feat_dim": int(extractor.feat_dim),
            "metrics": all_metrics,
        }
        out_path.write_text(json.dumps(all_metrics_meta, indent=2))
        logger.info("Wrote metrics to %s", out_path)

    return all_metrics


if __name__ == "__main__":
    fire.Fire(run)
