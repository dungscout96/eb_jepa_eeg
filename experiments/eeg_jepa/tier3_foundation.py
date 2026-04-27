"""Tier 3 — frozen EEG foundation models + Tier-1-style probes.

Loads pretrained EEG foundation models from braindecode/HF, applies the
**model-appropriate preprocessing** for each, freezes the encoder, extracts
per-clip embeddings, and trains the same MovieFeatureHead probe + per-recording
linear probes used in Tier 1 / Tier 2.

Models (all via braindecode.models.*.from_pretrained):
  - labram   : 200 Hz, 0.5-44.5 Hz BP, 19-ch 10-20, scale x0.01 (LaBraM convention)
  - eegpt    : 256 Hz, 0-38 Hz LP,    19-ch 10-20, EMA standardize, 4-s window
  - biot     : 200 Hz, no BP,         19-ch 10-20, 95th-percentile per-channel norm
  - cbramod  : 200 Hz, 0.3-50 Hz BP, 19-ch 10-20, per-channel mean removal

HBN channel selection: 19 standard 10-20 electrodes mapped from GSN-HydroCel-129
E# numbering. All mappings present (verified Cz + 18 lateral electrodes).

Usage
-----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/tier3_foundation.py \\
    --model=labram --seed=42 --output_json=/abs/out.json --save_embeddings=/abs/dir
"""

import json
import math
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from scipy.stats import pearsonr
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from braindecode.models import BIOT, CBraMod, EEGPT, Labram

from eb_jepa.architectures import MovieFeatureHead
from experiments.eeg_jepa.external.luna.LUNA import LUNA as _LUNAModel
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
# HBN GSN-HydroCel-129 → standard 10-20 (19 channels)
# ---------------------------------------------------------------------------

EGI_TO_1020 = {
    "Fp1": "E22", "Fp2": "E9",
    "F7":  "E33", "F3":  "E24", "Fz": "E11", "F4":  "E124", "F8": "E122",
    "T7":  "E45", "C3":  "E36", "Cz": "Cz",  "C4":  "E104", "T8": "E108",
    "P7":  "E58", "P3":  "E52", "Pz": "E62", "P4":  "E92",  "P8": "E96",
    "O1":  "E70", "O2":  "E83",
}
TEN_TWENTY_NAMES = list(EGI_TO_1020.keys())  # length 19


def _hbn_to_1020_indices(hbn_chs_info, subset: list[str] | None = None) -> list[int]:
    """Map HBN chs_info entries to indices of the requested 10-20 channel subset."""
    name_to_idx = {ch["ch_name"]: i for i, ch in enumerate(hbn_chs_info)}
    names = subset or TEN_TWENTY_NAMES
    return [name_to_idx[EGI_TO_1020[name]] for name in names]


# ---------------------------------------------------------------------------
# Per-model preprocessing wrappers
# ---------------------------------------------------------------------------


class ModelSpec:
    """Per-FM preprocessing + load spec."""

    def __init__(self, name: str):
        self.name = name
        if name == "biot":
            # BIOT: pretrained on 16-ch, 200 Hz; channel-agnostic via embedding.
            # No specific BP; per-channel 95-pctile normalization.
            self.target_sfreq = 200.0
            self.bandpass = (None, None)
            self.window_seconds = 2.0
            self.norm_mode = "p95"
            self.n_chans = 16  # subset of our 19 (drop Cz, T7, T8 to keep symmetry)
            self.n_outputs_pretrain = 2
            self.channel_subset = [
                "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                "C3", "C4",
                "P7", "P3", "Pz", "P4", "P8",
                "O1", "O2",
            ]  # 16 of our 19, drop Cz, T7, T8
        elif name == "cbramod":
            # CBraMod: 19-ch 10-20, 200 Hz, 0.3-50 Hz BP, per-channel mean removal.
            self.target_sfreq = 200.0
            self.bandpass = (0.3, 50.0)
            self.window_seconds = 2.0
            self.norm_mode = "mean_remove"
            self.n_chans = 19
            self.n_outputs_pretrain = 2
            self.channel_subset = TEN_TWENTY_NAMES
        elif name == "luna":
            # LUNA (NeurIPS 2025) — topology-agnostic; takes ARBITRARY channels
            # via 3D coordinate positional encoding. We use the full 129-ch
            # HBN montage with the native 200 Hz sample rate (no resample) so
            # the per-window length stays 400 samples (10 patches at default
            # patch_size=40), matching Tier 1/2's 4 x 2-s clip context.
            # LUNA was pretrained at 256 Hz on TUEG; feeding 200 Hz changes
            # absolute-frequency semantics in the FrequencyFeatureEmbedder
            # but the paper claims topology- and rate-flexible inference.
            self.target_sfreq = 200.0
            self.bandpass = (None, None)
            self.window_seconds = 2.0
            self.norm_mode = "ems"  # per-window per-channel z-score
            self.n_chans = 129
            self.n_outputs_pretrain = 0  # placeholder; loader uses num_classes=4 head
            self.channel_subset = None  # no selection — use all HBN channels
        elif name == "labram":
            # LaBraM: 200 Hz, 0.5-44.5 Hz BP; pretrain expects ~62-ch and
            # 16 patches × 200 samples = 16 s. Channel naming is required
            # via forward(ch_names=...). Integration deferred — this raises.
            raise NotImplementedError(
                "LaBraM tier3 integration is deferred — requires "
                "patch-position match (16 patches at 200 Hz = 16 s) "
                "and ch_names-aware forward call. Run BIOT/CBraMod first."
            )
        elif name == "eegpt":
            # EEGPT: 256 Hz, 4-s window, 62-ch with hardcoded chans_id [1,62].
            # Need exact 62-ch montage match — deferred.
            raise NotImplementedError(
                "EEGPT tier3 integration is deferred — chans_id is fixed "
                "to 62 specific channel positions in the pretrained "
                "checkpoint. Run BIOT/CBraMod first."
            )
        else:
            raise ValueError(name)
        self.n_times = int(round(self.target_sfreq * self.window_seconds))
        self.scale = 1.0  # legacy field


def _resample_torch(x: torch.Tensor, src_sfreq: float, tgt_sfreq: float) -> torch.Tensor:
    if abs(src_sfreq - tgt_sfreq) < 1e-6:
        return x
    # Linear interpolation has no anti-alias filter; safe only for upsampling
    # or near-no-op rates. All current FMs target 200 Hz (HBN source) so this
    # branch should not trigger; assert to catch silent regressions.
    raise NotImplementedError(
        f"_resample_torch hit non-trivial branch (src={src_sfreq}, tgt={tgt_sfreq}); "
        "use a Kaiser-windowed resampler (torchaudio.functional.resample) before enabling."
    )


def _bandpass_fft(x: torch.Tensor, sfreq: float, lo, hi) -> torch.Tensor:
    if lo is None and hi is None:
        return x
    T = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / sfreq).to(x.device)
    mask = torch.ones_like(freqs, dtype=torch.bool)
    if lo is not None:
        mask &= freqs >= lo
    if hi is not None:
        mask &= freqs <= hi
    return torch.fft.irfft(X * mask, n=T, dim=-1)


def _normalize(x: torch.Tensor, mode: str, scale: float) -> torch.Tensor:
    if mode == "scale":
        return x * scale
    if mode == "ems":
        # Trial-level z-score (offline approximation of EMA standardization)
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (x - m) / s
    if mode == "p95":
        absx = x.abs()
        p95 = torch.quantile(absx, 0.95, dim=-1, keepdim=True).clamp(min=1e-8)
        return x / p95
    if mode == "mean_remove":
        return x - x.mean(dim=-1, keepdim=True)
    return x


# ---------------------------------------------------------------------------
# Model loading (via braindecode HF integration)
# ---------------------------------------------------------------------------

HF_REPOS = {
    "biot":    "braindecode/biot-pretrained-prest-16chs",
    "cbramod": "braindecode/cbramod-pretrained",
    "luna":    "thorir/LUNA",
}

MODEL_CLS = {
    "biot":    BIOT,
    "cbramod": CBraMod,
    "luna":    _LUNAModel,
}


def _load_pretrained(name: str, spec: ModelSpec, chs_info_subset):
    """Per-model loading.

    BIOT: from_pretrained works directly (n_outputs=2 to match pretrain).
    CBraMod: has lazy parameters; needs build → dummy forward → manual
             safetensor load.
    """
    cls = MODEL_CLS[name]
    repo = HF_REPOS[name]
    logger.info("Loading pretrained %s from %s", name, repo)

    if name == "biot":
        return cls.from_pretrained(
            repo,
            n_chans=spec.n_chans,
            n_times=spec.n_times,
            sfreq=spec.target_sfreq,
            chs_info=chs_info_subset,
            n_outputs=spec.n_outputs_pretrain,
        )

    if name == "cbramod":
        model = cls(
            n_outputs=spec.n_outputs_pretrain,
            n_chans=spec.n_chans,
            n_times=spec.n_times,
            sfreq=spec.target_sfreq,
            chs_info=chs_info_subset,
        )
        with torch.no_grad():
            _ = model(torch.randn(1, spec.n_chans, spec.n_times))
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = hf_hub_download(repo_id=repo, filename="model.safetensors")
        sd = load_file(path)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            "CBraMod loaded: %d missing, %d unexpected (final layer is missing by design)",
            len(missing), len(unexpected),
        )
        return model

    if name == "luna":
        # LUNA in classification mode (num_classes=4 — arbitrary, head is unused).
        # Loaded weights bring encoder; classifier head stays randomly init'd
        # (we hook penultimate, not the head output).
        model = cls(num_classes=4)
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        path = hf_hub_download(repo_id=repo, filename="LUNA_base.safetensors")
        sd = load_file(path)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(
            "LUNA loaded: %d missing (classifier head, expected), %d unexpected",
            len(missing), len(unexpected),
        )
        return model

    raise ValueError(f"unknown model {name}")


def _embedding(model: nn.Module, x: torch.Tensor,
               channel_locations: torch.Tensor | None = None) -> torch.Tensor:
    """Capture the pretrained encoder's output as the per-clip embedding.

    BIOT / CBraMod: forward = ``model(x)``; the last ``Linear/Conv`` is the
    classification head, so hooking its input gives the encoder output.

    LUNA: the last ``Linear`` is inside the (randomly initialized)
    ``ClassificationHeadWithQueries.decoder_ffn``. Hooking that returns a
    partially-random tensor. The true encoder output is ``model.norm`` (the
    ``LayerNorm`` after ``model.blocks``), so we hook that explicitly and
    mean-pool over the patch dimension to get a single per-clip vector.
    """
    captured = {}

    is_luna = channel_locations is not None
    if is_luna:
        def _hook(_m, _inp, out):
            # out: (B, N, D) — mean-pool over patches for a fixed-D embedding
            captured["x"] = out.detach().mean(dim=1)
        target = model.norm
    else:
        def _hook(_m, inp, _out):
            x_in = inp[0]
            captured["x"] = x_in.detach().reshape(x_in.size(0), -1)
        target = None
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                target = m

    h = target.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            if is_luna:
                model(x, mask=None, channel_locations=channel_locations)
            else:
                model(x)
    finally:
        h.remove()
    return captured["x"]


# ---------------------------------------------------------------------------
# Subject probe helpers (mirror Tier 1 / Tier 2)
# ---------------------------------------------------------------------------


def _features_per_recording(dataset, fm, spec, ch_idx, src_sfreq, device,
                            channel_locations: torch.Tensor | None = None,
                            max_clips_per_rec: int = 4):
    rec_embs, rec_meta = [], []
    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        # Non-overlapping clip starts (stride = required) so per-rec averaging
        # combines genuinely independent samples instead of duplicates.
        max_non_overlap = (n_clips - 1) // required + 1
        n_sample = min(max_clips_per_rec, max_non_overlap)
        starts = (np.arange(n_sample) * required).astype(int)
        clip_embs = []
        for start in starts:
            indices = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
            eeg = torch.from_numpy(eeg_np)
            if dataset._norm_mode == "per_recording":
                rm = eeg.mean(dim=(0, 2), keepdim=True)
                rs = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
                eeg = (eeg - rm) / rs
            elif dataset._norm_mode == "none":
                pass  # raw µV — let the FM-spec normalization be the only one
            else:
                eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
            # eeg: [n_windows, 129, T_src]; select channels for FMs that need it
            if ch_idx is not None:
                eeg = eeg[:, ch_idx, :]
            eeg = eeg.to(device)
            eeg = _resample_torch(eeg, src_sfreq, spec.target_sfreq)
            eeg = _bandpass_fft(eeg, spec.target_sfreq, *spec.bandpass)
            eeg = _normalize(eeg, spec.norm_mode, spec.scale)
            ch_locs = None
            if channel_locations is not None:
                ch_locs = channel_locations.expand(eeg.size(0), -1, -1).to(device)
            emb = _embedding(fm, eeg, ch_locs)
            clip_embs.append(emb.mean(dim=0).cpu())
        if clip_embs:
            rec_embs.append(torch.stack(clip_embs).mean(dim=0).numpy())
            rec_meta.append(dataset._recording_metadata[rec_idx])
    return np.stack(rec_embs), rec_meta


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


def _train_cls_probe(train_embs, labels, device, epochs, lr):
    valid = ~np.isnan(labels)
    X = torch.from_numpy(train_embs[valid]).float().to(device)
    y = torch.from_numpy(labels[valid]).float().to(device)
    probe = nn.Linear(X.shape[1], 1).to(device)
    opt = Adam(probe.parameters(), lr=lr)
    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(probe(X).squeeze(-1), y)
        loss.backward(); opt.step()
    return probe


def _train_reg_probe(train_embs, labels, device, epochs, lr):
    valid = ~np.isnan(labels)
    X = torch.from_numpy(train_embs[valid]).float().to(device)
    y = torch.from_numpy(labels[valid]).float().to(device)
    ym, ys = y.mean(), y.std().clamp(min=1e-6)
    yn = (y - ym) / ys
    probe = nn.Linear(X.shape[1], 1).to(device)
    opt = Adam(probe.parameters(), lr=lr)
    probe.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(probe(X).squeeze(-1), yn)
        loss.backward(); opt.step()
    return probe, ym.item(), ys.item()


@torch.inference_mode()
def _eval_cls(probe, embs, labels, device):
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return {"bal_acc": float("nan"), "auc": float("nan")}
    X = torch.from_numpy(embs[valid]).float().to(device)
    logits = probe(X).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y_true = labels[valid].astype(int)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    return {"bal_acc": float(balanced_accuracy_score(y_true, preds)), "auc": float(auc)}


@torch.inference_mode()
def _eval_reg(probe, embs, labels, device, ym, ys):
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return {"mae": float("nan"), "corr": float("nan"), "r2": float("nan")}
    X = torch.from_numpy(embs[valid]).float().to(device)
    y_true = labels[valid]
    y_pred = probe(X).squeeze(-1).cpu().numpy() * ys + ym
    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "corr": corr, "r2": r2}


# ---------------------------------------------------------------------------
# Per-clip feature extraction & probe training
# ---------------------------------------------------------------------------


def _features_loader(loader, fm, spec, ch_idx, src_sfreq, device, pos_idx,
                     channel_locations: torch.Tensor | None = None):
    all_embs, all_targets, all_positions = [], [], []
    for eeg, features, _ in tqdm(loader, desc="extracting", leave=False):
        # eeg: [B, n_windows, 129, T_src]; with --norm_mode=none the dataset
        # leaves raw µV and we let the FM-spec _normalize() be the sole norm.
        B, T, C, Ts = eeg.shape
        if ch_idx is not None:
            eeg = eeg[:, :, ch_idx, :]
            eeg = eeg.reshape(B * T, len(ch_idx), Ts)
        else:
            eeg = eeg.reshape(B * T, C, Ts)
        eeg = eeg.to(device)
        eeg = _resample_torch(eeg, src_sfreq, spec.target_sfreq)
        eeg = _bandpass_fft(eeg, spec.target_sfreq, *spec.bandpass)
        eeg = _normalize(eeg, spec.norm_mode, spec.scale)
        ch_locs = None
        if channel_locations is not None:
            ch_locs = channel_locations.expand(eeg.size(0), -1, -1).to(device)
        emb = _embedding(fm, eeg, ch_locs)              # [B*T, D]
        D = emb.shape[1]
        emb = emb.view(B, T, D).mean(dim=1)             # pool over windows
        all_embs.append(emb.detach().cpu())
        all_targets.append(features.cpu())
        if pos_idx is not None:
            all_positions.append(features[:, :, pos_idx].mean(dim=1).numpy())
    feats = torch.cat(all_embs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    positions = np.concatenate(all_positions) if all_positions else None
    return feats, targets, positions


def _train_movie_probes(X, y_per_window, n_features, hdec, device,
                        feat_mean, feat_std, feat_median,
                        epochs, lr, batch_size=256):
    """Probe head on per-clip mean-pooled embeddings, predicting per-window
    targets through MovieFeatureHead with T=1 (clip-level prediction)."""
    D = X.shape[1]
    reg_head = MovieFeatureHead(D, hdec, n_features).to(device)
    cls_head = MovieFeatureHead(D, hdec, n_features).to(device)
    reg_loss_fn = RegressionLoss(feat_mean.to(device), feat_std.to(device))
    cls_loss_fn = ClassificationLoss(feat_median.to(device))

    # Use clip-mean targets
    y = y_per_window.mean(dim=1)  # [N, n_features]

    opt = Adam(list(reg_head.parameters()) + list(cls_head.parameters()), lr=lr)
    n = X.shape[0]
    for epoch in range(epochs):
        reg_head.train(); cls_head.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            x = X[idx].to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1, 1]
            yb = y[idx].to(device).unsqueeze(1)  # [B, 1, n_features]
            opt.zero_grad()
            r = reg_head(x); c = cls_head(x)
            (reg_loss_fn(r, yb) + cls_loss_fn(c, yb)).backward()
            opt.step()
    return reg_head, cls_head


@torch.inference_mode()
def _eval_movie_probes(reg_head, cls_head, X, y_per_window,
                       feat_mean, feat_std, feat_median, feat_names, device,
                       batch_size=512):
    reg_head.eval(); cls_head.eval()
    n = X.shape[0]
    reg_preds, cls_preds = [], []
    for s in range(0, n, batch_size):
        x = X[s:s + batch_size].to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        reg_preds.append(reg_head(x).cpu())
        cls_preds.append(cls_head(x).cpu())
    reg_p = torch.cat(reg_preds, dim=0).squeeze(1).numpy()
    cls_p = torch.cat(cls_preds, dim=0).squeeze(1)
    targ = y_per_window.mean(dim=1).numpy()

    fmean = feat_mean.numpy(); fstd = feat_std.numpy()
    reg_p_un = reg_p * (fstd + 1e-8) + fmean
    metrics = {}
    for i, name in enumerate(feat_names):
        if np.std(targ[:, i]) > 1e-10 and np.std(reg_p_un[:, i]) > 1e-10:
            metrics[f"reg_{name}_corr"] = float(pearsonr(reg_p_un[:, i], targ[:, i]).statistic)
        else:
            metrics[f"reg_{name}_corr"] = 0.0
    cls_probs = torch.sigmoid(cls_p).numpy()
    cls_labels = (cls_probs > 0.5).astype(int)
    binary = (targ > feat_median.numpy()).astype(int)
    for i, name in enumerate(feat_names):
        metrics[f"cls_{name}_acc"] = float(accuracy_score(binary[:, i], cls_labels[:, i]))
        metrics[f"cls_{name}_bal_acc"] = float(
            balanced_accuracy_score(binary[:, i], cls_labels[:, i])
        )
        try:
            metrics[f"cls_{name}_auc"] = float(roc_auc_score(binary[:, i], cls_probs[:, i]))
        except ValueError:
            metrics[f"cls_{name}_auc"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    model: str,
    seed: int = 42,
    n_windows: int = 4,
    window_size_seconds: int = 2,
    batch_size: int = 32,
    num_workers: int = 4,
    norm_mode: str = "none",
    hdec: int = 64,
    probe_epochs: int = 20,
    probe_lr: float = 1e-3,
    subject_probe_epochs: int = 100,
    subject_probe_lr: float = 1e-3,
    splits: str = "val,test",
    output_json: str = "",
    save_embeddings: str = "",
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
):
    assert model in HF_REPOS, f"unknown model {model}; choose from {list(HF_REPOS)}"
    setup_seed(seed)
    device = setup_device("auto")
    spec = ModelSpec(model)

    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "data.norm_mode": norm_mode,
        "data.corrca_filters": None,
    }
    cfg = load_config(fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feat_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))
    n_features = len(feat_names)

    logger.info("Loading train (FM=%s seed=%d, target_sfreq=%.1f, win=%.1fs)",
                model, seed, spec.target_sfreq, spec.window_seconds)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feat_names, cfg=cfg.data,
        preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )
    feat_stats = train_set.compute_feature_stats()
    feat_median = train_set.compute_feature_median()

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    if isinstance(splits, str):
        splits_list = [s.strip() for s in splits.split(",")]
    else:
        splits_list = [str(s).strip() for s in splits]
    eval_sets, eval_loaders = {}, {}
    for split in splits_list:
        ds = JEPAMovieDataset(
            split=split,
            n_windows=n_windows, window_size_seconds=window_size_seconds,
            feature_names=feat_names,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data,
            preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
        )
        eval_sets[split] = ds
        eval_loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    src_sfreq = float(train_set.sfreq)
    chs_info_full = train_set.get_chs_info()
    if spec.channel_subset is None:
        # LUNA: use all 129 HBN channels with 3D coordinates as positional info
        ch_idx = None
        chs_info_subset = chs_info_full
        channel_locations_t = torch.tensor(
            np.array([ch["loc"][:3] for ch in chs_info_full]), dtype=torch.float32,
        ).unsqueeze(0)  # (1, 129, 3)
        logger.info("Using all %d HBN channels (LUNA topology-agnostic)", len(chs_info_full))
    else:
        ch_idx = _hbn_to_1020_indices(chs_info_full, subset=spec.channel_subset)
        chs_info_subset = []
        for src_idx, ten20_name in zip(ch_idx, spec.channel_subset):
            ch = dict(chs_info_full[src_idx])
            ch["ch_name"] = ten20_name
            chs_info_subset.append(ch)
        channel_locations_t = None
        logger.info("Selected %d channels for %s: %s",
                    len(spec.channel_subset), model, spec.channel_subset)

    fm = _load_pretrained(model, spec, chs_info_subset).to(device)
    fm.eval()
    for p in fm.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in fm.parameters())
    logger.info("FM %s loaded: %d params (frozen)", model, n_params)

    # Per-clip features
    pos_idx = (
        feat_names.index("position_in_movie")
        if "position_in_movie" in feat_names else None
    )
    logger.info("Extracting train per-clip features...")
    train_X, train_y, train_pos = _features_loader(
        train_loader, fm, spec, ch_idx, src_sfreq, device, pos_idx,
        channel_locations=channel_locations_t,
    )
    logger.info("Train embeddings shape: %s", tuple(train_X.shape))

    eval_X, eval_y, eval_pos = {}, {}, {}
    for split in splits_list:
        X, y, p = _features_loader(
            eval_loaders[split], fm, spec, ch_idx, src_sfreq, device, pos_idx,
            channel_locations=channel_locations_t,
        )
        eval_X[split] = X; eval_y[split] = y; eval_pos[split] = p
        logger.info("%s embeddings shape: %s", split, tuple(X.shape))

    if save_embeddings:
        emb_dir = Path(save_embeddings)
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            emb_dir / f"tier3_{model}_seed{seed}_train.npz",
            embs=train_X.numpy(), targets=train_y.numpy(),
            positions=train_pos if train_pos is not None else np.array([]),
        )
        for split in splits_list:
            np.savez(
                emb_dir / f"tier3_{model}_seed{seed}_{split}.npz",
                embs=eval_X[split].numpy(), targets=eval_y[split].numpy(),
                positions=eval_pos[split] if eval_pos[split] is not None else np.array([]),
            )

    # Probes
    logger.info("Training movie-feature probes (%d epochs)...", probe_epochs)
    reg_head, cls_head = _train_movie_probes(
        train_X, train_y, n_features, hdec, device,
        feat_stats["mean"], feat_stats["std"], feat_median,
        probe_epochs, probe_lr,
    )

    all_metrics = {}
    for split in splits_list:
        m = _eval_movie_probes(
            reg_head, cls_head, eval_X[split], eval_y[split],
            feat_stats["mean"], feat_stats["std"], feat_median,
            feat_names, device,
        )
        for k, v in m.items():
            all_metrics[f"{split}/{k}"] = v

    # Per-recording subject probes
    logger.info("Per-recording embeddings (train)...")
    train_rec_embs, train_meta = _features_per_recording(
        train_set, fm, spec, ch_idx, src_sfreq, device,
        channel_locations=channel_locations_t,
    )
    train_labels = _extract_subject_labels(train_meta)
    train_ages = np.array([float(m["age"]) for m in train_meta if "age" in m])
    train_median_age = float(np.median(train_ages)) if len(train_ages) >= 10 else None

    subject_probes = {}
    for label_name, labels in train_labels.items():
        if label_name == "age_reg":
            probe, ym, ys = _train_reg_probe(
                train_rec_embs, labels, device, subject_probe_epochs, subject_probe_lr,
            )
            subject_probes[label_name] = ("reg", probe, ym, ys)
        else:
            probe = _train_cls_probe(
                train_rec_embs, labels, device, subject_probe_epochs, subject_probe_lr,
            )
            subject_probes[label_name] = ("cls", probe)

    for split in splits_list:
        rec_embs, rec_meta = _features_per_recording(
            eval_sets[split], fm, spec, ch_idx, src_sfreq, device,
            channel_locations=channel_locations_t,
        )
        eval_labels = _extract_subject_labels(rec_meta, median_age=train_median_age)
        for label_name, info in subject_probes.items():
            ev = eval_labels.get(label_name)
            if ev is None:
                continue
            if info[0] == "cls":
                m = _eval_cls(info[1], rec_embs, ev, device)
            else:
                _, probe, ym, ys = info
                m = _eval_reg(probe, rec_embs, ev, device, ym, ys)
            for k, v in m.items():
                all_metrics[f"{split}/subject/{label_name}/{k}"] = v

    print(f"\n=== Tier 3 FM: {model} (seed={seed}, params={n_params}) ===")
    for k in sorted(all_metrics.keys()):
        v = all_metrics[k]
        vs = f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v)
        print(f"  {k}: {vs}")

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": model,
            "seed": seed,
            "n_chans": spec.n_chans,
            "n_times": spec.n_times,
            "target_sfreq": spec.target_sfreq,
            "n_params": n_params,
            "metrics": all_metrics,
        }, indent=2))
        logger.info("wrote metrics to %s", out_path)

    return all_metrics


if __name__ == "__main__":
    fire.Fire(run)
