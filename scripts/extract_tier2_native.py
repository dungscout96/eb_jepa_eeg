"""Train Tier 2 supervised CNN end-to-end and save native regression head
predictions (no separate Ridge probe) in the tier-native NPZ schema that
`scripts/bootstrap_tier_native.py` consumes.

Output: {out_dir}/test_seed{seed+2}.npz with keys
  movie_reg_preds   (n_rec, n_passes, n_features)  — denormalized predictions
  movie_targets     (n_rec, n_passes, n_features)
  feature_names     (n_features,)
  feature_mean / feature_std / feature_median  (n_features,)
  rec_ids           (n_rec,)

Matches Phase D's (scale, size, density, seeds) for apples-to-apples bootstrap:
  - n_passes = 20 (canonical density)
  - random clip starts via torch.randint on a seeded generator
  - per-clip prediction = mean over n_windows of the model's per-window reg_pred
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
    _epoch,
    _metrics_from_eval,
)
from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.training_utils import load_config, setup_device, setup_seed
from eb_jepa.logging import get_logger
from experiments.eeg_jepa.main import resolve_preprocessed_dir
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _predict_native_per_clip(
    dataset, model, device, feat_mean, feat_std, n_passes: int, seed: int,
):
    """Per-(rec, pass) NATIVE regression head predictions on random clip starts.

    Returns:
      preds   [n_rec, n_passes, n_features]  — denormalized to original space
      targets [n_rec, n_passes, n_features]  — in original space
    """
    rng = torch.Generator().manual_seed(seed + 100003)
    n_rec = len(dataset)
    n_features = len(dataset.feature_names)
    required = (dataset.n_windows - 1) * dataset.temporal_stride + 1

    preds = np.full((n_rec, n_passes, n_features), np.nan, dtype=np.float32)
    targets = np.full((n_rec, n_passes, n_features), np.nan, dtype=np.float32)

    fmean_np = feat_mean.detach().cpu().numpy()
    fstd_np = feat_std.detach().cpu().numpy()

    model.eval()
    for rec_idx in range(n_rec):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        n_clips = n_total - required + 1
        if n_clips <= 0:
            continue
        for p in range(n_passes):
            start = int(torch.randint(0, n_clips, (1,), generator=rng).item())
            indices = list(range(start, start + required, dataset.temporal_stride))
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
                reg_norm, _ = model(x)  # [n_windows, n_features] in z-score space
            reg_clip_norm = reg_norm.mean(dim=0).cpu().numpy().astype(np.float32)
            reg_clip = reg_clip_norm * fstd_np + fmean_np

            feats = dataset.feature_recordings[rec_idx][indices]
            target_clip = feats.mean(dim=0).numpy().astype(np.float32)

            preds[rec_idx, p] = reg_clip
            targets[rec_idx, p] = target_clip
    return preds, targets


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
    sfreq = float(train_set.sfreq)
    chs_info = train_set.get_chs_info()
    feat_stats = train_set.compute_feature_stats()
    feat_median = train_set.compute_feature_median()
    feat_mean = feat_stats["mean"].to(device)
    feat_std = feat_stats["std"].to(device)
    feat_median_t = feat_median.to(device)

    net = DualHeadModel(
        model, n_chans, n_times, n_features, sfreq, chs_info,
        native_preproc=False, source_sfreq=None,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    import math
    best_val_corr = -math.inf
    best_state = None
    epochs_since_improve = 0
    for ep in range(1, epochs + 1):
        tr = _epoch(net, train_loader, opt, feat_mean, feat_std, feat_median_t, device, train=True)
        with torch.no_grad():
            vl = _epoch(net, val_loader, opt, feat_mean, feat_std, feat_median_t, device, train=False)
        val_metrics = _metrics_from_eval(vl, feat_mean, feat_std, feat_median, feat_names)
        val_corr = float(np.mean([val_metrics[f"reg_{n}_corr"] for n in feat_names]))
        logger.info(
            "ep %d/%d  train reg=%.4f cls=%.4f | val mean_corr=%.4f",
            ep, epochs, tr["reg_loss"], tr["cls_loss"], val_corr,
        )
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

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Predicting test native head (n_passes=%d, seed=%d)", n_passes, seed + 2)
    test_preds, test_targets = _predict_native_per_clip(
        test_set, net, device, feat_mean, feat_std,
        n_passes=n_passes, seed=seed + 2,
    )
    rec_ids = np.array([Path(p).stem for p in test_set._fif_paths])
    feature_mean_np = feat_mean.detach().cpu().numpy().astype(np.float32)
    feature_std_np = feat_std.detach().cpu().numpy().astype(np.float32)
    feature_median_np = feat_median.detach().cpu().numpy().astype(np.float32)

    npz_path = out / f"test_seed{seed + 2}.npz"
    np.savez_compressed(
        npz_path,
        movie_reg_preds=test_preds,
        movie_targets=test_targets,
        feature_names=np.array(feat_names),
        feature_mean=feature_mean_np,
        feature_std=feature_std_np,
        feature_median=feature_median_np,
        rec_ids=rec_ids,
    )
    logger.info(
        "Tier 2 native end-to-end done: model=%s seed=%d → %s "
        "(best val mean_corr=%.4f, preds=%s)",
        model, seed, npz_path, best_val_corr, test_preds.shape,
    )


if __name__ == "__main__":
    fire.Fire(run)
