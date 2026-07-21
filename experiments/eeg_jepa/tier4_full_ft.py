"""Tier 4 — full supervised fine-tuning of pretrained EEG foundation models.

Loads BIOT / CBraMod / LUNA from HF (same weights and per-FM preprocessing as
Tier 3), unfreezes the encoder, attaches a fresh ``Linear(D, 4)`` regression
head, and trains end-to-end on the 4 stimulus targets jointly using a negative
Pearson-correlation loss (literature-optimal for stimulus-EEG regression).

Differences from Tier 3 (frozen + linear probe):
- Encoder weights are trainable (full fine-tuning, no PEFT).
- Discriminative LRs: encoder 1e-5, head 1e-3 (AdamW, cosine + 5% warmup).
- Loss: ``-mean_k pearson_r(pred[:, k], target[:, k])`` summed across 4 targets.
- No parallel binary classification head — AUC is computed at eval time by
  ranking the continuous regression output (median-split labels).
- Per-recording subject probes still computed with frozen encoder *after* FT
  (post-hoc, on the now-finetuned representation).

Same data setup: 4×2-s clips, 129-ch HBN, splits R1-R4 / R5 / R6, 3 seeds.

Usage
-----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/tier4_full_ft.py \\
    --model=biot --seed=42 --output_json=/abs/out.json
"""

import json
import math
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_device, setup_seed
from experiments.eeg_jepa.main import resolve_preprocessed_dir
from experiments.eeg_jepa.tier3_foundation import (
    HF_REPOS,
    ModelSpec,
    TEN_TWENTY_NAMES,
    _bandpass_fft,
    _embedding,
    _eval_cls,
    _eval_reg,
    _extract_subject_labels,
    _features_per_recording,
    _hbn_to_1020_indices,
    _load_pretrained,
    _normalize,
    _train_cls_probe,
    _train_reg_probe,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loss + metric helpers
# ---------------------------------------------------------------------------


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``-mean_k r(pred[:, k], target[:, k])`` — VLAAI / ICASSP-2024 standard.

    Pearson r is the eval metric, so training to maximize it removes the
    train/eval objective mismatch that MSE introduces (MSE penalizes scale and
    offset errors that r ignores).
    """
    pc = pred - pred.mean(dim=0, keepdim=True)
    tc = target - target.mean(dim=0, keepdim=True)
    num = (pc * tc).sum(dim=0)
    denom = pc.norm(dim=0) * tc.norm(dim=0) + 1e-8
    r = num / denom
    return -r.mean()


def auc_from_continuous(pred: np.ndarray, target: np.ndarray,
                        median: float) -> float:
    """Median-split AUC computed from continuous regression output.

    Replaces the parallel CE head that Tier 2 used; the ranking is what AUC
    measures so the regression output is sufficient.
    """
    binary = (target > median).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return float("nan")
    try:
        return float(roc_auc_score(binary, pred))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Wrapped FT model: per-FM in-graph preproc + pretrained encoder + reg head
# ---------------------------------------------------------------------------


class FullFTModel(nn.Module):
    """Pretrained FM with gradient-tracked encoder + small reg head.

    The forward path is:
        raw µV (B, C, T_src)
            → resample-to-target (no-op for current FMs at 200 Hz)
            → FFT bandpass (FM-spec)
            → FM-spec normalize (p95 / mean_remove / ems)
            → pretrained encoder (BIOT / CBraMod / LUNA)
            → encoder embedding (B, D)
            → Linear(D, 4) reg head
    """

    def __init__(self, fm: nn.Module, spec: ModelSpec, src_sfreq: float,
                 emb_dim: int, n_outputs: int,
                 channel_locations: torch.Tensor | None = None):
        super().__init__()
        self.fm = fm
        self.spec = spec
        self.src_sfreq = float(src_sfreq)
        self.head = nn.Linear(emb_dim, n_outputs)
        self.channel_locations = channel_locations  # registered separately below
        if channel_locations is not None:
            self.register_buffer(
                "_channel_locations_buffer",
                channel_locations.detach().clone(),
                persistent=False,
            )

    def _preproc(self, x: torch.Tensor) -> torch.Tensor:
        # Currently no-op: src_sfreq == target_sfreq for all 3 active FMs.
        if abs(self.src_sfreq - self.spec.target_sfreq) > 1e-6:
            raise NotImplementedError("non-trivial resample not enabled")
        x = _bandpass_fft(x, self.spec.target_sfreq, *self.spec.bandpass)
        x = _normalize(x, self.spec.norm_mode, self.spec.scale)
        return x

    def _encoder_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Capture the pretrained encoder's output, gradient-tracked.

        Mirrors `_embedding` from tier3_foundation but does NOT call
        ``torch.no_grad`` and does NOT detach — we need gradients to flow back
        into the FM during fine-tuning.
        """
        captured = {}
        is_luna = self.channel_locations is not None
        if is_luna:
            def _hook(_m, _inp, out):
                captured["x"] = out.mean(dim=1)  # mean over N patches
            target = self.fm.norm
        else:
            def _hook(_m, inp, _out):
                captured["x"] = inp[0].reshape(inp[0].size(0), -1)
            target = None
            for mod in self.fm.modules():
                if isinstance(mod, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                    target = mod
        h = target.register_forward_hook(_hook)
        try:
            if is_luna:
                ch_locs = self._channel_locations_buffer.expand(x.size(0), -1, -1)
                self.fm(x, mask=None, channel_locations=ch_locs)
            else:
                self.fm(x)
        finally:
            h.remove()
        return captured["x"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preproc(x)
        emb = self._encoder_embedding(x)
        return self.head(emb)


def _infer_emb_dim(fm: nn.Module, spec: ModelSpec, n_chans: int,
                   channel_locations: torch.Tensor | None,
                   device: torch.device) -> int:
    """Run a single dummy forward to learn the encoder output width."""
    dummy = torch.zeros(1, n_chans, spec.n_times, device=device)
    is_luna = channel_locations is not None
    captured = {}
    if is_luna:
        def _hook(_m, _inp, out):
            captured["x"] = out.mean(dim=1)
        target = fm.norm
    else:
        def _hook(_m, inp, _out):
            captured["x"] = inp[0].reshape(inp[0].size(0), -1)
        target = None
        for mod in fm.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                target = mod
    h = target.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            if is_luna:
                ch_locs = channel_locations.expand(1, -1, -1).to(device)
                fm(dummy, mask=None, channel_locations=ch_locs)
            else:
                fm(dummy)
    finally:
        h.remove()
    return captured["x"].shape[1]


# ---------------------------------------------------------------------------
# Train / eval epoch
# ---------------------------------------------------------------------------


def _epoch(net: FullFTModel, loader: DataLoader, ch_idx, device,
           opt: AdamW | None = None, sched: LambdaLR | None = None,
           train: bool = False):
    net.train(train)
    loss_sum = 0.0
    n_batches = 0
    all_pred, all_target = [], []
    for eeg, features, _ in loader:
        # eeg: [B, T_win, 129, T_samp] (raw µV; norm_mode=none)
        B, T, C, Ts = eeg.shape
        if ch_idx is not None:
            eeg = eeg[:, :, ch_idx, :]
            eeg = eeg.reshape(B * T, len(ch_idx), Ts)
        else:
            eeg = eeg.reshape(B * T, C, Ts)
        eeg = eeg.to(device, non_blocking=True)
        # Average features over the n_windows windows of a clip (per-clip target);
        # but we predict per-window, so broadcast the per-window targets directly.
        targets = features.reshape(B * T, -1).to(device, non_blocking=True)

        if train:
            opt.zero_grad()
        pred = net(eeg)
        loss = neg_pearson_loss(pred, targets)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()

        loss_sum += float(loss.item())
        n_batches += 1
        if not train:
            all_pred.append(pred.detach().cpu())
            all_target.append(targets.cpu())
    out = {"loss": loss_sum / max(n_batches, 1)}
    if not train:
        out["pred"] = torch.cat(all_pred, dim=0)
        out["target"] = torch.cat(all_target, dim=0)
    return out


def _metrics_from_eval(out: dict, feat_median: torch.Tensor,
                       feat_names: list[str]) -> dict:
    pred = out["pred"].numpy()
    target = out["target"].numpy()
    median = feat_median.cpu().numpy()
    metrics = {}
    for i, name in enumerate(feat_names):
        if np.std(target[:, i]) > 1e-10 and np.std(pred[:, i]) > 1e-10:
            metrics[f"reg_{name}_corr"] = float(
                pearsonr(pred[:, i], target[:, i]).statistic
            )
        else:
            metrics[f"reg_{name}_corr"] = 0.0
        metrics[f"cls_{name}_auc"] = auc_from_continuous(
            pred[:, i], target[:, i], median[i]
        )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    model: str,
    seed: int = 42,
    n_windows: int = 4,
    window_size_seconds: int = 2,
    batch_size: int = 16,
    num_workers: int = 4,
    epochs: int = 30,
    early_stop_patience: int = 8,
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-2,
    warmup_frac: float = 0.05,
    subject_probe_epochs: int = 100,
    subject_probe_lr: float = 1e-3,
    splits: str = "val,test",
    output_json: str = "",
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
):
    assert model in HF_REPOS, f"unknown model {model}; choose from {list(HF_REPOS)}"
    setup_seed(seed)
    device = setup_device("auto")
    spec = ModelSpec(model)

    # `norm_mode=none` is critical: each FM applies its own normalization in
    # _preproc, so we must NOT pre-z-score upstream (Tier-3 double-norm bug).
    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "data.norm_mode": "none",
        "data.corrca_filters": None,
    }
    cfg = load_config(fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feat_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))
    n_features = len(feat_names)

    logger.info("Loading train (FT FM=%s seed=%d, target_sfreq=%.1f, win=%.1fs)",
                model, seed, spec.target_sfreq, spec.window_seconds)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feat_names, cfg=cfg.data,
        preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )
    feat_median = train_set.compute_feature_median()

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
        ch_idx = None
        chs_info_subset = chs_info_full
        channel_locations_t = torch.tensor(
            np.array([ch["loc"][:3] for ch in chs_info_full]), dtype=torch.float32,
        ).unsqueeze(0)
        logger.info("Using all %d HBN channels (LUNA topology-agnostic)", len(chs_info_full))
    else:
        ch_idx = _hbn_to_1020_indices(chs_info_full, subset=spec.channel_subset)
        chs_info_subset = []
        for src_idx, ten20_name in zip(ch_idx, spec.channel_subset):
            ch = dict(chs_info_full[src_idx])
            ch["ch_name"] = ten20_name
            chs_info_subset.append(ch)
        channel_locations_t = None
        logger.info("Selected %d channels for %s", len(spec.channel_subset), model)

    fm = _load_pretrained(model, spec, chs_info_subset).to(device)
    if channel_locations_t is not None:
        channel_locations_t = channel_locations_t.to(device)

    # Determine encoder output dim before unfreezing
    fm.eval()
    emb_dim = _infer_emb_dim(fm, spec, spec.n_chans, channel_locations_t, device)
    logger.info("Encoder emb_dim=%d", emb_dim)

    # Build the FT wrapper, unfreeze encoder
    net = FullFTModel(
        fm=fm, spec=spec, src_sfreq=src_sfreq,
        emb_dim=emb_dim, n_outputs=n_features,
        channel_locations=channel_locations_t,
    ).to(device)
    # BIOT/CBraMod store integer index buffers as parameters; only floating
    # tensors can carry gradients, so guard the unfreeze.
    for p in net.fm.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("FT params=%d (encoder + head, all trainable)", n_params)

    # Discriminative LRs: small encoder LR (1e-5) + larger head LR (1e-3)
    encoder_params = list(net.fm.parameters())
    head_params = list(net.head.parameters())
    opt = AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = LambdaLR(opt, lr_lambda)

    val_loader = eval_loaders[splits_list[0]]
    best_val_corr = -math.inf
    best_state = None
    epochs_since_improve = 0
    for ep in range(1, epochs + 1):
        tr = _epoch(net, train_loader, ch_idx, device,
                    opt=opt, sched=sched, train=True)
        with torch.no_grad():
            vl = _epoch(net, val_loader, ch_idx, device, train=False)
        val_metrics = _metrics_from_eval(vl, feat_median, feat_names)
        val_corr = float(np.mean([val_metrics[f"reg_{n}_corr"] for n in feat_names]))
        logger.info(
            "ep %d/%d  train -r=%.4f | val -r=%.4f mean_corr=%.4f",
            ep, epochs, tr["loss"], vl["loss"], val_corr,
        )
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= early_stop_patience:
                logger.info("early stop at ep %d (best val mean corr=%.4f)", ep, best_val_corr)
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    # Test-set + per-clip predictions
    all_metrics = {}
    for split in splits_list:
        with torch.no_grad():
            out = _epoch(net, eval_loaders[split], ch_idx, device, train=False)
        m = _metrics_from_eval(out, feat_median, feat_names)
        for k, v in m.items():
            all_metrics[f"{split}/{k}"] = v

    # ------------------------------------------------------------------
    # Per-recording subject probes — extract penultimate from FT encoder
    # ------------------------------------------------------------------
    logger.info("Computing per-recording embeddings (post-FT)...")
    fm.eval()
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
            eval_sets[split], fm, spec, ch_idx, src_sfreq, device,
            channel_locations=channel_locations_t,
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
    print(f"\n=== Tier 4 full-FT: {model} (seed={seed}, params={n_params}) ===")
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
            "n_params": n_params,
            "target_sfreq": spec.target_sfreq,
            "best_val_mean_corr": best_val_corr,
            "metrics": all_metrics,
        }, indent=2))
        logger.info("wrote metrics to %s", out_path)

    return all_metrics


if __name__ == "__main__":
    fire.Fire(run)
