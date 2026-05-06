"""Extract Tier 2 supervised CNN features at JEPA-canonical scale/density/seeds.

Trains the Tier 2 supervised model in-process (matches existing per-seed
end-to-end runs exactly via `setup_seed` + same training loop), then runs
canonical per-(rec, n_passes=20) feature extraction on the trained backbone
and saves NPZs in the schema unified_probe_eval consumes.

Output schema per split file at {out_dir}/{split}_seed{seed_for_split}.npz:
    embs   [n_rec, n_passes, D_FM]
    labels [n_rec, n_passes, n_features]

Matches JEPA Protocol B exactly:
    - n_passes = 20 (canonical density)
    - train: outer p × inner randperm(rec_idx) — matches JEPA train_order
    - val/test: sequential rec × passes
    - random clip starts via torch.randint on a seeded generator
    - per-clip label = mean over windows of dataset stim features
    - per-window backbone embedding mean-pooled to per-clip D
"""
import sys
from pathlib import Path

import fire
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.eeg_jepa.tier2_supervised import (
    DualHeadModel,
    MODEL_REGISTRY,
    NativePreprocWrapper,
    _build_model,
    _embedding,
    _epoch,
    _metrics_from_eval,
)
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.training_utils import load_config, setup_device, setup_seed
from eb_jepa.logging import get_logger
from experiments.eeg_jepa.main import resolve_preprocessed_dir
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _extract_canonical_per_clip(
    dataset, model, device, n_passes: int, seed: int, train_order: bool,
):
    """Per-(rec, pass) backbone features using random clip_start sampling.

    Mirrors the JEPA `_extract` function's iteration order so train sees
    outer-pass × inner-randperm-rec while val/test see sequential rec × passes.
    """
    rng_order = torch.Generator().manual_seed(seed)
    rng_clip = torch.Generator().manual_seed(seed + 100003)
    n_rec = len(dataset)
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1

    if train_order:
        order = []
        for p in range(n_passes):
            for rec_idx in torch.randperm(n_rec, generator=rng_order).tolist():
                order.append((p, rec_idx))
    else:
        order = [(p, r) for r in range(n_rec) for p in range(n_passes)]

    n_features = len(dataset.feature_names)
    D = None
    embs_buckets = [[None] * n_passes for _ in range(n_rec)]
    labels_buckets = [[None] * n_passes for _ in range(n_rec)]

    for (p, rec_idx) in order:
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        n_clips = n_total - required + 1
        if n_clips <= 0:
            embs_buckets[rec_idx][p] = None
            labels_buckets[rec_idx][p] = np.full(n_features, np.nan, dtype=np.float32)
            continue
        start = int(torch.randint(0, n_clips, (1,), generator=rng_clip).item())
        indices = list(range(start, start + required, dataset.temporal_stride))

        # Read EEG via dataset's standard path (handles per-recording norm + CorrCA flag).
        # Tier 2's run() sets corrca_filters=None so dataset returns raw 129ch.
        from eb_jepa.datasets.hbn import _read_raw_windows
        eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
        eeg = torch.from_numpy(eeg_np)
        if dataset._norm_mode == "per_recording":
            rm = eeg.mean(dim=(0, 2), keepdim=True)
            rs = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            eeg = (eeg - rm) / rs
        elif dataset._norm_mode != "none":
            eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std

        x = eeg.to(device)  # [n_windows, C, T]
        with torch.no_grad():
            emb = _embedding(model, x)  # [n_windows, D]
        emb_clip = emb.mean(dim=0).cpu().numpy().astype(np.float32)
        if D is None:
            D = emb_clip.shape[0]

        feats = dataset.feature_recordings[rec_idx][indices]
        label_clip = feats.mean(dim=0).numpy().astype(np.float32)

        embs_buckets[rec_idx][p] = emb_clip
        labels_buckets[rec_idx][p] = label_clip

    if D is None:
        raise RuntimeError("No clips extracted")
    zero_pad = np.zeros(D, dtype=np.float32)
    embs = np.stack(
        [np.stack([e if e is not None else zero_pad for e in row]) for row in embs_buckets]
    )
    labels = np.stack([np.stack(row) for row in labels_buckets])
    return embs, labels


def run(
    model: str,
    seed: int,
    out_dir: str,
    n_windows: int = 2,
    window_size_seconds: int = 4,
    n_passes: int = 20,
    norm_mode: str = "per_recording",
    epochs: int = 30,
    early_stop_patience: int = 16,
    lr: float = 1e-3,
    batch_size: int = 64,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
):
    """Train Tier 2 model + extract canonical per-clip NPZs."""
    assert model in MODEL_REGISTRY, f"unknown model {model}; choose from {list(MODEL_REGISTRY)}"
    setup_seed(seed)
    device = setup_device("auto")

    overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": 4,
        "data.norm_mode": norm_mode,
        "data.corrca_filters": None,
    }
    cfg = load_config(fname, overrides)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feat_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))
    n_features = len(feat_names)

    train_set = JEPAMovieDataset(
        split="train", n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feat_names, cfg=cfg.data,
        preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )
    val_set = JEPAMovieDataset(
        split="val", n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feat_names, eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data, preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )
    test_set = JEPAMovieDataset(
        split="test", n_windows=n_windows, window_size_seconds=window_size_seconds,
        feature_names=feat_names, eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data, preprocessed=preprocessed, preprocessed_dir=preprocessed_dir,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    n_chans = train_set.n_chans
    n_times = train_set.n_times
    feat_stats = train_set.compute_feature_stats()
    feat_median = train_set.compute_feature_median()
    feat_mean = feat_stats["mean"].to(device)
    feat_std = feat_stats["std"].to(device)
    feat_median_t = feat_median.to(device)

    net = _build_model(model, n_chans, n_times, n_features).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    import math
    best_val_corr = -math.inf
    best_state = None
    epochs_since_improve = 0
    for ep in range(epochs):
        tr = _epoch(net, train_loader, opt, feat_mean, feat_std, feat_median_t, device,
                    n_features=n_features, sched=sched, train=True)
        with torch.no_grad():
            vl = _epoch(net, val_loader, None, feat_mean, feat_std, feat_median_t, device,
                        n_features=n_features, sched=None, train=False)
        val_metrics = _metrics_from_eval(vl, feat_mean, feat_std, feat_median, feat_names)
        val_corr = float(np.mean([val_metrics[f"reg_{n}_corr"] for n in feat_names]))
        logger.info("ep %d/%d  train -r=%.4f | val mean_corr=%.4f", ep, epochs, tr["loss"], val_corr)
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= early_stop_patience:
                logger.info("early stop at ep %d (best val mean_corr=%.4f)", ep, best_val_corr)
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting train canonical NPZ (n_passes=%d, train_order=True, seed=%d)", n_passes, seed)
    Etr, Ltr = _extract_canonical_per_clip(train_set, net, device, n_passes, seed, train_order=True)
    np.savez_compressed(out / f"train_seed{seed}.npz", embs=Etr, labels=Ltr)
    logger.info("train_seed%d.npz: embs=%s labels=%s", seed, Etr.shape, Ltr.shape)

    logger.info("Extracting val canonical NPZ (seed=%d)", seed + 1)
    Ev, Lv = _extract_canonical_per_clip(val_set, net, device, n_passes, seed + 1, train_order=False)
    np.savez_compressed(out / f"val_seed{seed + 1}.npz", embs=Ev, labels=Lv)
    logger.info("val_seed%d.npz: embs=%s labels=%s", seed + 1, Ev.shape, Lv.shape)

    logger.info("Extracting test canonical NPZ (seed=%d)", seed + 2)
    Et, Lt = _extract_canonical_per_clip(test_set, net, device, n_passes, seed + 2, train_order=False)
    np.savez_compressed(out / f"test_seed{seed + 2}.npz", embs=Et, labels=Lt)
    logger.info("test_seed%d.npz: embs=%s labels=%s", seed + 2, Et.shape, Lt.shape)

    logger.info("Tier 2 canonical extraction done: model=%s seed=%d → %s (best val mean_corr=%.4f)",
                model, seed, out, best_val_corr)


if __name__ == "__main__":
    fire.Fire(run)
