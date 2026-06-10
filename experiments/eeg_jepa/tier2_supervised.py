"""Tier 2 — supervised end-to-end braindecode baselines.

Trains ShallowFBCSPNet / Deep4Net / EEGNetv4 / EEGNeX end-to-end on the
4 stimulus probe targets jointly (multi-output regression head + parallel
classification head), using the *same* per-window-targeted setup as Exp 6 /
Tier 1 so numbers are directly comparable.

Per window (2-s, 129 channels, per-recording z-norm, no CorrCA):
  - Backbone produces a logit vector of length n_outputs=4 (regression).
    A second copy with the same backbone produces 4 logits for classification.
  - Train end-to-end with MSE + BCE on per-window features + median splits.
  - Early-stopping on val mean regression corr (mean of 4 features).

Per-recording subject probes (age regression, age binary, sex):
  - Extract pre-final-layer activations on a sub-sample of clips per recording,
    average → per-recording embedding [N_rec, D]. Train fresh nn.Linear probes.

Usage
-----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/tier2_supervised.py \\
    --model=shallow --seed=42 --output_json=/abs/out.json \\
    --save_embeddings=/abs/emb_dir
"""

import json
import math
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from scipy.stats import pearsonr
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from braindecode.models import (
    Deep4Net,
    EEGNetv4,
    EEGNeX,
    ShallowFBCSPNet,
)

from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_device, setup_seed
from experiments.eeg_jepa.main import resolve_preprocessed_dir

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "shallow": ShallowFBCSPNet,
    "deep4": Deep4Net,
    "eegnet": EEGNetv4,
    "eegnex": EEGNeX,
}


def _build_model(name: str, n_chans: int, n_times: int,
                 n_outputs: int, sfreq: float, chs_info=None) -> nn.Module:
    cls = MODEL_REGISTRY[name]
    if name == "eegnex":
        return cls(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
            chs_info=chs_info, sfreq=sfreq,
        )
    if name in ("shallow", "deep4"):
        return cls(
            n_chans=n_chans, n_outputs=n_outputs, n_times=n_times,
            final_conv_length="auto",
        )
    return cls(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)


class NativePreprocWrapper(nn.Module):
    """In-graph resample + bandpass + per-window standardization.

    For Deep4 / Shallow native preprocessing: HBN comes at 200 Hz, the original
    Schirrmeister-2017 designs were tuned for 250 Hz with 4-38 Hz bandpass.

    Resamples by ``torch.nn.functional.interpolate(mode='linear')`` and applies
    a zero-phase bandpass via FFT mask (cheap, no filter design needed).
    """

    def __init__(self, model: nn.Module, source_sfreq: float,
                 target_sfreq: float, lowcut: float, highcut: float):
        super().__init__()
        self.model = model
        self.source_sfreq = float(source_sfreq)
        self.target_sfreq = float(target_sfreq)
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        if abs(self.target_sfreq - self.source_sfreq) < 1e-6:
            return x
        new_T = int(round(x.shape[-1] * self.target_sfreq / self.source_sfreq))
        return F.interpolate(x, size=new_T, mode="linear", align_corners=False)

    def _bandpass(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] real → FFT mask out-of-band, IFFT
        T = x.shape[-1]
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.target_sfreq).to(x.device)
        mask = (freqs >= self.lowcut) & (freqs <= self.highcut)
        return torch.fft.irfft(X * mask, n=T, dim=-1)

    def _per_window_standardize(self, x: torch.Tensor) -> torch.Tensor:
        # Per-channel z-score within each window (Schirrmeister "exponential
        # moving standardization" approximation; full EMA needs sequential
        # state, this trial-level z is the standard offline equivalent).
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (x - m) / s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resample(x)
        x = self._bandpass(x)
        x = self._per_window_standardize(x)
        return self.model(x)


# ---------------------------------------------------------------------------
# Wrappers: stack of two backbones (regression + classification heads)
# ---------------------------------------------------------------------------


class DualHeadModel(nn.Module):
    """One backbone → two parallel n_outputs-dim heads (reg + cls).

    Re-uses a single shared backbone because braindecode models include the
    head internally; cheaper to instantiate two separate copies than to
    surgery the heads. ~2x params, but still tiny (<2M).
    """

    def __init__(self, name: str, n_chans: int, n_times: int, n_features: int,
                 sfreq: float, chs_info=None,
                 native_preproc: bool = False,
                 source_sfreq: float | None = None):
        super().__init__()
        # When native_preproc is on, the inner model expects the *resampled*
        # n_times (computed at the call site). The wrapper handles resample
        # + bandpass + standardize on the fly.
        reg_model = _build_model(name, n_chans, n_times, n_features, sfreq, chs_info)
        cls_model = _build_model(name, n_chans, n_times, n_features, sfreq, chs_info)
        if native_preproc:
            assert source_sfreq is not None
            # 4-38 Hz bandpass per Schirrmeister 2017
            self.reg = NativePreprocWrapper(reg_model, source_sfreq, sfreq, 4.0, 38.0)
            self.cls = NativePreprocWrapper(cls_model, source_sfreq, sfreq, 4.0, 38.0)
        else:
            self.reg = reg_model
            self.cls = cls_model

    def forward(self, x: torch.Tensor):
        return self.reg(x), self.cls(x)


def _embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return penultimate activation of the regression head as embedding.

    Hooks the *input* to the model's final linear/conv module and returns the
    flattened activation. Works for all four braindecode architectures we use.
    """
    backbone = model.reg if isinstance(model, DualHeadModel) else model
    captured = {}

    def _hook(_m, inp, _out):
        x_in = inp[0]
        captured["x"] = x_in.detach().reshape(x_in.size(0), -1)

    final = None
    for m in backbone.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            final = m
    h = final.register_forward_hook(_hook)
    with torch.no_grad():
        backbone(x)
    h.remove()
    return captured["x"]


# ---------------------------------------------------------------------------
# Per-recording subject helpers (mirror tier1)
# ---------------------------------------------------------------------------


def _features_per_recording(dataset, model, device, n_chans_in,
                            max_clips_per_rec: int = 4):
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
                rm = eeg.mean(dim=(0, 2), keepdim=True)
                rs = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
                eeg = (eeg - rm) / rs
            else:
                eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
            if dataset._add_envelope:
                eeg = dataset._append_lowfreq_envelope(eeg)
            if dataset._corrca_W is not None:
                eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)
            # eeg: [n_windows, C, T] → flatten windows into batch dim
            x = eeg.to(device)
            emb = _embedding(model, x)  # [n_windows, D]
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
        loss.backward()
        opt.step()
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
        loss.backward()
        opt.step()
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
    bal_acc = balanced_accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    return {"bal_acc": float(bal_acc), "auc": float(auc)}


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
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "corr": corr, "r2": r2}


# ---------------------------------------------------------------------------
# Train + eval for the supervised end-to-end model
# ---------------------------------------------------------------------------


def _epoch(model, loader, opt, feat_mean, feat_std, feat_median, device,
           train: bool):
    model.train(train)
    reg_loss_sum = cls_loss_sum = 0.0
    n_batches = 0
    all_reg_pred, all_cls_pred, all_targets = [], [], []
    for eeg, features, _ in loader:
        # eeg: [B, T_win, C, T_samp]; features: [B, T_win, n_features]
        B, T, C, Ts = eeg.shape
        eeg = eeg.reshape(B * T, C, Ts).to(device, non_blocking=True)
        targets = features.reshape(B * T, -1).to(device, non_blocking=True)
        targets_norm = (targets - feat_mean) / (feat_std + 1e-8)
        binary = (targets > feat_median).float()

        if train:
            opt.zero_grad()
        reg_pred, cls_pred = model(eeg)
        reg_loss = F.mse_loss(reg_pred, targets_norm)
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred, binary)
        loss = reg_loss + cls_loss
        if train:
            loss.backward()
            opt.step()

        reg_loss_sum += float(reg_loss.item())
        cls_loss_sum += float(cls_loss.item())
        n_batches += 1
        if not train:
            all_reg_pred.append(reg_pred.detach().cpu())
            all_cls_pred.append(cls_pred.detach().cpu())
            all_targets.append(targets.cpu())
    out = {
        "reg_loss": reg_loss_sum / max(n_batches, 1),
        "cls_loss": cls_loss_sum / max(n_batches, 1),
    }
    if not train:
        out["reg_pred"] = torch.cat(all_reg_pred, dim=0)
        out["cls_pred"] = torch.cat(all_cls_pred, dim=0)
        out["targets"] = torch.cat(all_targets, dim=0)
    return out


def _metrics_from_eval(out: dict, feat_mean, feat_std, feat_median, feat_names):
    reg_p = out["reg_pred"].numpy() * feat_std.cpu().numpy() + feat_mean.cpu().numpy()
    cls_p = torch.sigmoid(out["cls_pred"]).numpy()
    targ = out["targets"].numpy()
    binary = (targ > feat_median.cpu().numpy()).astype(int)
    metrics = {}
    for i, name in enumerate(feat_names):
        if np.std(targ[:, i]) > 1e-10 and np.std(reg_p[:, i]) > 1e-10:
            metrics[f"reg_{name}_corr"] = float(pearsonr(reg_p[:, i], targ[:, i]).statistic)
        else:
            metrics[f"reg_{name}_corr"] = 0.0
        metrics[f"cls_{name}_acc"] = float(accuracy_score(binary[:, i], (cls_p[:, i] > 0.5).astype(int)))
        metrics[f"cls_{name}_bal_acc"] = float(
            balanced_accuracy_score(binary[:, i], (cls_p[:, i] > 0.5).astype(int))
        )
        try:
            metrics[f"cls_{name}_auc"] = float(roc_auc_score(binary[:, i], cls_p[:, i]))
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
    native_preproc: bool = False,
    target_sfreq: float = 250.0,
    batch_size: int = 64,
    num_workers: int = 4,
    norm_mode: str = "per_recording",
    epochs: int = 50,
    early_stop_patience: int = 8,
    lr: float = 1e-3,
    subject_probe_epochs: int = 100,
    subject_probe_lr: float = 1e-3,
    splits: str = "val,test",
    output_json: str = "",
    save_embeddings: str = "",
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
):
    assert model in MODEL_REGISTRY, f"unknown model {model}"
    setup_seed(seed)
    device = setup_device("auto")

    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "data.norm_mode": norm_mode,
        "data.corrca_filters": None,  # supervised models work on raw 129 ch
    }
    cfg = load_config(fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feat_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))
    n_features = len(feat_names)

    logger.info("Loading train (model=%s, seed=%d)...", model, seed)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
        feature_names=feat_names,
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    feat_stats = train_set.compute_feature_stats()
    feat_median = train_set.compute_feature_median()
    feat_mean = feat_stats["mean"].to(device)
    feat_std = feat_stats["std"].to(device)
    feat_med = feat_median.to(device)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )

    if isinstance(splits, str):
        splits_list = [s.strip() for s in splits.split(",")]
    else:
        splits_list = [str(s).strip() for s in splits]
    eval_sets, eval_loaders = {}, {}
    for split in splits_list:
        ds = JEPAMovieDataset(
            split=split,
            n_windows=n_windows,
            window_size_seconds=window_size_seconds,
            feature_names=feat_names,
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

    n_chans = train_set.n_chans
    n_times = train_set.n_times
    sfreq = float(train_set.sfreq)
    chs_info = train_set.get_chs_info()

    if native_preproc:
        # The wrapper resamples 200 -> target_sfreq before the inner net,
        # so the inner net must be sized for the post-resample length.
        inner_n_times = int(round(n_times * target_sfreq / sfreq))
        inner_sfreq = target_sfreq
    else:
        inner_n_times = n_times
        inner_sfreq = sfreq
    logger.info(
        "Build %s (n_chans=%d, n_times=%d, native_preproc=%s, inner_n_times=%d, inner_sfreq=%.1f)",
        model, n_chans, n_times, native_preproc, inner_n_times, inner_sfreq,
    )
    net = DualHeadModel(
        model, n_chans, inner_n_times, n_features, inner_sfreq, chs_info,
        native_preproc=native_preproc, source_sfreq=sfreq if native_preproc else None,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("model params=%d", n_params)

    opt = Adam(net.parameters(), lr=lr)

    # Train with early stopping on val mean reg corr
    best_val_corr = -math.inf
    best_state = None
    epochs_since_improve = 0
    val_loader = eval_loaders[splits_list[0]]
    for ep in range(1, epochs + 1):
        tr = _epoch(net, train_loader, opt, feat_mean, feat_std, feat_med,
                    device, train=True)
        vl = _epoch(net, val_loader, opt, feat_mean, feat_std, feat_med,
                    device, train=False)
        val_metrics = _metrics_from_eval(vl, feat_mean, feat_std, feat_med, feat_names)
        val_corr = float(np.mean([val_metrics[f"reg_{n}_corr"] for n in feat_names]))
        logger.info(
            "ep %d/%d  train reg=%.4f cls=%.4f | val reg=%.4f cls=%.4f mean_corr=%.4f",
            ep, epochs, tr["reg_loss"], tr["cls_loss"],
            vl["reg_loss"], vl["cls_loss"], val_corr,
        )
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= early_stop_patience:
                logger.info("early stop at ep %d (best val corr=%.4f)", ep, best_val_corr)
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # Test-set + per-clip predictions
    all_metrics = {}
    per_clip_records = {}
    for split in splits_list:
        out = _epoch(net, eval_loaders[split], opt, feat_mean, feat_std, feat_med,
                     device, train=False)
        m = _metrics_from_eval(out, feat_mean, feat_std, feat_med, feat_names)
        for k, v in m.items():
            all_metrics[f"{split}/{k}"] = v
        # Save per-clip-mean predictions for downstream bootstrap/permutation
        # reg_pred / targets here are per-window [N*T_win, n_features]; reshape.
        n_total = out["reg_pred"].shape[0]
        if n_total % n_windows == 0:
            n_clips = n_total // n_windows
            reg_p_clip = out["reg_pred"].view(n_clips, n_windows, n_features).mean(dim=1).numpy()
            targ_clip = out["targets"].view(n_clips, n_windows, n_features).mean(dim=1).numpy()
            per_clip_records[split] = {
                "reg_pred": reg_p_clip,
                "targets": targ_clip,
            }

    # ------------------------------------------------------------------
    # Per-recording subject probes
    # ------------------------------------------------------------------
    logger.info("Computing per-recording embeddings (train)...")
    train_rec_embs, train_meta = _features_per_recording(train_set, net, device, n_chans)
    train_labels = _extract_subject_labels(train_meta)
    train_ages = np.array([float(m["age"]) for m in train_meta if "age" in m])
    train_median_age = float(np.median(train_ages)) if len(train_ages) >= 10 else None

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

    for split in splits_list:
        rec_embs, rec_meta = _features_per_recording(
            eval_sets[split], net, device, n_chans,
        )
        eval_labels = _extract_subject_labels(rec_meta, median_age=train_median_age)
        for label_name, info in subject_probes.items():
            ev = eval_labels.get(label_name)
            if ev is None:
                continue
            if info[0] == "cls":
                metrics = _eval_cls(info[1], rec_embs, ev, device)
            else:
                _, probe, ym, ys = info
                metrics = _eval_reg(probe, rec_embs, ev, device, ym, ys)
            for k, v in metrics.items():
                all_metrics[f"{split}/subject/{label_name}/{k}"] = v

    # ------------------------------------------------------------------
    # Print + dump
    # ------------------------------------------------------------------
    print(f"\n=== Tier 2 supervised: {model} (seed={seed}, params={n_params}) ===")
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
            "n_chans": n_chans,
            "n_times": n_times,
            "n_params": n_params,
            "metrics": all_metrics,
            "best_val_mean_corr": best_val_corr,
        }, indent=2))
        logger.info("wrote metrics to %s", out_path)

    if save_embeddings:
        emb_dir = Path(save_embeddings)
        emb_dir.mkdir(parents=True, exist_ok=True)
        for split, recs in per_clip_records.items():
            np.savez(
                emb_dir / f"tier2_{model}_seed{seed}_{split}.npz",
                reg_pred=recs["reg_pred"],
                targets=recs["targets"],
            )
        logger.info("Saved per-clip predictions to %s", emb_dir)

    return all_metrics


if __name__ == "__main__":
    fire.Fire(run)
