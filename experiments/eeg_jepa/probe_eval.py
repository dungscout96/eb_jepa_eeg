"""Post-hoc linear probe evaluation for trained EEG JEPA checkpoints.

Loads a trained encoder from a checkpoint, freezes it, trains fresh linear
probes on the train set, then evaluates on val (and optionally test).

Two probe types:
  1. Movie-feature probes (regression + classification) — evaluated per
     temporal clip, same as the online probes trained during training.
     Labels: contrast_rms, luminance_mean, position_in_movie, narrative_event_score.

  2. Subject-trait probe — evaluated per recording (one embedding per
     subject, pooled over all clips of that recording). Labels: age >
     median or sex (M/F), matching the SanityCheckHook label.
     The probe is trained and evaluated at recording level to avoid
     subject-label leakage across clips.

Usage
-----
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \\
    --checkpoint=/abs/path/to/latest.pth.tar \\
    --n_windows=4 --window_size_seconds=4 --batch_size=32

See scripts/eval_phase1_probes_delta.py to submit all Phase 1 checkpoints.
"""

import copy
import math
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import EEGEncoderTokens, MaskedPredictor, MovieFeatureHead, Projector
from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.jepa import MaskedJEPA, MaskedJEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import VCLoss
from eb_jepa.masking import MultiBlockMaskCollator
from eb_jepa.training_utils import load_checkpoint, load_config, setup_device, setup_seed
from experiments.eeg_jepa.eval import validation_loop
from experiments.eeg_jepa.main import (
    ClassificationLoss,
    RegressionLoss,
    _PREPROCESSED_DIRS,
    resolve_preprocessed_dir,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Subject-trait probe helpers
# ---------------------------------------------------------------------------

def _embed_all_clips(dataset, jepa, device, batch_size, num_workers,
                     max_clips_per_rec=4):
    """Encode a few clips per recording, return per-recording mean embeddings.

    Sub-samples up to ``max_clips_per_rec`` evenly spaced clips per recording
    to keep embedding cost manageable (~4 clips x 701 recordings = 2800 forward
    passes vs ~22K with 32 clips).

    Returns
    -------
    embeddings : np.ndarray  [N_recordings, D]
    metadata   : list[dict]  per-recording metadata (age, sex, ...)
    """
    jepa.eval()
    all_embs = []
    all_meta = []

    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
        n_clips = n_total - required + 1

        if n_clips <= 0:
            continue

        # Evenly spaced sub-sample of clips
        n_sample = min(max_clips_per_rec, n_clips)
        starts = np.linspace(0, n_clips - 1, n_sample, dtype=int)

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
            if dataset._corrca_W is not None:
                eeg = torch.einsum("wct,ck->wkt", eeg, dataset._corrca_W)
            eeg = eeg.unsqueeze(0).to(device)  # [1, n_windows, C, T]

            with torch.no_grad():
                tokens = jepa.context_encoder.encode_tokens(eeg, mask=None)
                emb = tokens.mean(dim=1)  # [1, D]
            clip_embs.append(emb.squeeze(0).cpu())

        if clip_embs:
            rec_emb = torch.stack(clip_embs).mean(dim=0)  # [D] — mean over clips
            all_embs.append(rec_emb.numpy())
            all_meta.append(dataset._recording_metadata[rec_idx])

    return np.stack(all_embs), all_meta


def _extract_subject_labels(metadata_list, median_age=None):
    """Extract age (float), sex (0/1), and age_binary (0/1) from metadata.

    Args:
        metadata_list: list of per-recording metadata dicts.
        median_age: if provided, use this threshold for age binary classification
                    (ensures eval splits use the same threshold as train).

    Returns dict of label_name → np.ndarray[N] (NaN where unavailable).
    """
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
    # Age regression (raw float)
    valid_ages = ages[~np.isnan(ages)]
    if len(valid_ages) >= 10:
        labels["age_reg"] = ages

    # Age binary classification (> median)
    # Use provided median_age if given (for eval splits to match train threshold)
    if len(valid_ages) >= 10:
        if median_age is None:
            median_age = float(np.median(valid_ages))
        age_bin = np.where(np.isnan(ages), np.nan,
                           (ages > median_age).astype(float))
        labels["age_cls"] = age_bin

    # Sex classification
    valid_sex = sexes[~np.isnan(sexes)]
    if len(valid_sex) >= 10:
        labels["sex"] = sexes

    return labels


def _train_cls_probe(train_embs, train_labels, device, probe_epochs, probe_lr):
    """Train a linear probe for binary classification (age>median or sex)."""
    valid = ~np.isnan(train_labels)
    X = torch.from_numpy(train_embs[valid]).float().to(device)
    y = torch.from_numpy(train_labels[valid]).float().to(device)

    D = X.shape[1]
    probe = nn.Linear(D, 1).to(device)
    opt = Adam(probe.parameters(), lr=probe_lr)

    probe.train()
    for epoch in range(probe_epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            probe(X).squeeze(-1), y
        )
        loss.backward()
        opt.step()

    return probe


def _train_reg_probe(train_embs, train_labels, device, probe_epochs, probe_lr):
    """Train a linear probe for regression (age prediction)."""
    valid = ~np.isnan(train_labels)
    X = torch.from_numpy(train_embs[valid]).float().to(device)
    y = torch.from_numpy(train_labels[valid]).float().to(device)

    # Standardize targets for stable training
    y_mean, y_std = y.mean(), y.std().clamp(min=1e-6)
    y_norm = (y - y_mean) / y_std

    D = X.shape[1]
    probe = nn.Linear(D, 1).to(device)
    opt = Adam(probe.parameters(), lr=probe_lr)

    probe.train()
    for epoch in range(probe_epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(probe(X).squeeze(-1), y_norm)
        loss.backward()
        opt.step()

    return probe, y_mean.item(), y_std.item()


def _eval_cls_probe(probe, embs, labels, device):
    """Evaluate binary classification probe. Returns bal_acc and auc."""
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return {"bal_acc": float("nan"), "auc": float("nan")}

    X = torch.from_numpy(embs[valid]).float().to(device)
    y_true = labels[valid]

    probe.eval()
    with torch.no_grad():
        logits = probe(X).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    bal_acc = balanced_accuracy_score(y_true.astype(int), preds)
    try:
        auc = roc_auc_score(y_true.astype(int), probs)
    except ValueError:
        auc = float("nan")

    return {"bal_acc": bal_acc, "auc": auc}


def _eval_reg_probe(probe, embs, labels, device, y_mean, y_std):
    """Evaluate regression probe. Returns MAE, correlation, and R²."""
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return {"mae": float("nan"), "corr": float("nan"), "r2": float("nan")}

    X = torch.from_numpy(embs[valid]).float().to(device)
    y_true = labels[valid]

    probe.eval()
    with torch.no_grad():
        y_pred_norm = probe(X).squeeze(-1).cpu().numpy()
    y_pred = y_pred_norm * y_std + y_mean

    mae = float(np.mean(np.abs(y_pred - y_true)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mae": mae, "corr": corr, "r2": r2}


# ---------------------------------------------------------------------------
# Movie identity probe helpers
# ---------------------------------------------------------------------------

def _embed_clips_with_position(loader, jepa, device, pos_feature_idx):
    """Encode all clips in a DataLoader, return embeddings and positions.

    Returns
    -------
    embs : np.ndarray [N_clips, D]
    positions : np.ndarray [N_clips]  — position_in_movie value per clip
    """
    jepa.eval()
    all_embs = []
    all_pos = []
    for eeg, features, _ in loader:
        eeg = eeg.to(device)
        with torch.no_grad():
            tokens = jepa.context_encoder.encode_tokens(eeg, mask=None)
            emb = tokens.mean(dim=1)  # [B, D]
        all_embs.append(emb.cpu().numpy())
        # position_in_movie: mean over n_windows dimension
        pos = features[:, :, pos_feature_idx].mean(dim=1).numpy()  # [B]
        all_pos.append(pos)
    return np.concatenate(all_embs), np.concatenate(all_pos)


def _train_movie_id_probe(embs, positions, device, n_bins, probe_epochs, probe_lr):
    """Train a K-way linear probe to predict temporal bin from clip embeddings."""
    # Discretize positions into n_bins equal-width bins
    bin_edges = np.linspace(positions.min(), positions.max() + 1e-8, n_bins + 1)
    bin_labels = np.digitize(positions, bin_edges) - 1
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)

    X = torch.from_numpy(embs).float().to(device)
    y = torch.from_numpy(bin_labels).long().to(device)

    D = X.shape[1]
    probe = nn.Linear(D, n_bins).to(device)
    opt = Adam(probe.parameters(), lr=probe_lr)

    probe.train()
    for epoch in range(probe_epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(probe(X), y)
        loss.backward()
        opt.step()

    return probe, bin_edges


def _eval_movie_id_probe(probe, embs, positions, device, bin_edges):
    """Evaluate movie identity probe. Returns top-1 acc and top-5 acc."""
    n_bins = len(bin_edges) - 1
    bin_labels = np.digitize(positions, bin_edges) - 1
    bin_labels = np.clip(bin_labels, 0, n_bins - 1)

    X = torch.from_numpy(embs).float().to(device)
    y_true = torch.from_numpy(bin_labels).long()

    probe.eval()
    with torch.no_grad():
        logits = probe(X).cpu()

    # Top-1
    preds = logits.argmax(dim=1)
    top1 = float((preds == y_true).float().mean())

    # Top-5
    k = min(5, n_bins)
    _, top_k = logits.topk(k, dim=1)
    top5 = float((top_k == y_true.unsqueeze(1)).any(dim=1).float().mean())

    # Chance level = 1/n_bins
    chance = 1.0 / n_bins

    return {"top1_acc": top1, "top5_acc": top5, "chance": chance, "n_bins": n_bins}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    checkpoint: str,
    # Data config — must match the checkpoint's training config
    n_windows: int = 4,
    window_size_seconds: int = 2,
    batch_size: int = 64,
    num_workers: int = 4,
    # Movie-feature probe training
    probe_epochs: int = 20,
    probe_lr: float = 1e-3,
    # Subject-trait probe training
    subject_probe_epochs: int = 100,
    subject_probe_lr: float = 1e-3,
    # Eval splits: "val", "test", or "val,test"
    splits: str = "val,test",
    # Data preprocessing overrides (must match training config)
    norm_mode: str = "",
    add_envelope: bool = False,
    corrca_filters: str = "",
    # Run modes
    subject_only: bool = False,
    # W&B
    wandb_run_id: str = "",
    wandb_project: str = "eb_jepa",
    wandb_group: str = "probe_eval_phase1",
    # Config
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    seed: int = 2025,
):
    """Train fresh linear probes on a frozen encoder and evaluate on val/test.

    Args:
        checkpoint: Absolute path to a saved latest.pth.tar.
        n_windows: Must match the checkpoint's --data.n_windows.
        window_size_seconds: Must match --data.window_size_seconds.
        batch_size: Batch size for data loading.
        num_workers: DataLoader workers.
        probe_epochs: Epochs to train the movie-feature probe (default 20).
        probe_lr: Movie-feature probe learning rate.
        subject_probe_epochs: Epochs to train the subject-trait probe (default 100).
        subject_probe_lr: Subject-trait probe learning rate.
        splits: Comma-separated eval splits ("val", "test").
        wandb_run_id: If set, resumes this W&B run to append probe_eval/* metrics.
        wandb_project: W&B project name.
        wandb_group: W&B group for standalone eval runs.
        fname: Config file path.
        seed: Random seed.
    """
    setup_seed(seed)
    device = setup_device("auto")

    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    # Infer encoder_depth and predictor_dim from checkpoint state dict
    _ckpt_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("model_state_dict", {})
    _depth = max(
        int(k.split(".")[3]) + 1
        for k in _ckpt_sd
        if k.startswith("context_encoder.transformer.layers.")
    )
    # Infer predictor_embed_dim from checkpoint:
    # - If predictor.input_proj exists → narrow predictor (predictor_dim < embed_dim)
    # - If predictor.output_proj exists but no input_proj → predictor_dim == embed_dim
    # - If neither exists → predictor_dim is None (no projection layers)
    if "predictor.input_proj.weight" in _ckpt_sd:
        _pred_dim = int(_ckpt_sd["predictor.input_proj.weight"].shape[0])
    elif "predictor.output_proj.weight" in _ckpt_sd:
        _pred_dim = int(_ckpt_sd["predictor.output_proj.weight"].shape[0])
    else:
        _pred_dim = None

    _overrides = {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "model.encoder_depth": _depth,
        "model.predictor_embed_dim": _pred_dim,
    }
    if norm_mode:
        _overrides["data.norm_mode"] = norm_mode
    if add_envelope:
        _overrides["data.add_envelope"] = True
    if corrca_filters:
        _overrides["data.corrca_filters"] = corrca_filters
    cfg = load_config(fname, _overrides)

    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)
    feature_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading train set...")
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
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    eval_sets = {}
    eval_loaders = {}
    if isinstance(splits, str):
        splits = [s.strip() for s in splits.split(",")]
    for split in splits:
        split = split.strip()
        split_set = JEPAMovieDataset(
            split=split,
            n_windows=n_windows,
            window_size_seconds=window_size_seconds,
            feature_names=feature_names,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
        )
        eval_sets[split] = split_set
        eval_loaders[split] = DataLoader(
            split_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info("Split '%s': %d recordings", split, len(split_set))

    # ------------------------------------------------------------------
    # Build and load frozen encoder
    # ------------------------------------------------------------------
    n_chans = train_set.n_chans
    n_times = train_set.n_times
    n_features = len(feature_names)
    embed_dim = cfg.model.encoder_embed_dim
    chs_info = train_set.get_chs_info()
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
        predictor_dim=cfg.model.get("predictor_embed_dim", None),
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
    # Infer regularizer type from checkpoint state dict:
    # VCLoss saves regularizer.proj.* keys; SIGRegLoss has no learnable params
    regularizer = None
    if any(k.startswith("regularizer.") for k in _ckpt_sd):
        projector = Projector(f"{embed_dim}-{embed_dim * 4}-{embed_dim * 4}")
        regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)

    jepa = MaskedJEPA(
        encoder, target_encoder, predictor, mask_collator, regularizer,
    ).to(device)

    ckpt_info = load_checkpoint(checkpoint_path, jepa, optimizer=None, device=device, strict=False)
    logger.info("Loaded checkpoint at epoch %d", ckpt_info.get("epoch", "?"))

    for p in jepa.parameters():
        p.requires_grad_(False)
    jepa.eval()

    # ------------------------------------------------------------------
    # Movie-feature probes (per-clip, same as online probes during training)
    # ------------------------------------------------------------------
    regression_probe = classification_probe = None
    if not subject_only:
        reg_loss_fn = RegressionLoss(
            feature_stats["mean"].to(device),
            feature_stats["std"].to(device),
        )
        cls_loss_fn = ClassificationLoss(feature_median.to(device))

        reg_head = MovieFeatureHead(embed_dim, cfg.model.hdec, n_features)
        cls_head = MovieFeatureHead(embed_dim, cfg.model.hdec, n_features)
        regression_probe = MaskedJEPAProbe(jepa, reg_head, reg_loss_fn).to(device)
        classification_probe = MaskedJEPAProbe(jepa, cls_head, cls_loss_fn).to(device)

        movie_probe_opt = Adam(
            list(regression_probe.head.parameters())
            + list(classification_probe.head.parameters()),
            lr=probe_lr,
        )

        logger.info("Training movie-feature probes for %d epochs...", probe_epochs)
        for epoch in range(probe_epochs):
            regression_probe.train()
            classification_probe.train()
            reg_total = cls_total = 0.0
            n = 0
            for eeg, features, _ in tqdm(train_loader, desc=f"Movie probe {epoch+1}/{probe_epochs}", leave=False):
                eeg = eeg.to(device)
                features = features.to(device)
                movie_probe_opt.zero_grad()
                reg_loss = regression_probe(eeg, features)
                cls_loss = classification_probe(eeg, features)
                (reg_loss + cls_loss).backward()
                movie_probe_opt.step()
                reg_total += reg_loss.item()
                cls_total += cls_loss.item()
                n += 1
            logger.info(
                "Movie probe ep %d/%d  reg=%.4f  cls=%.4f",
                epoch + 1, probe_epochs, reg_total / max(n, 1), cls_total / max(n, 1),
            )

    # ------------------------------------------------------------------
    # Subject-trait probes (per-recording embeddings, pooled over clips)
    #   - age_binary: age > median classification
    #   - sex: M/F classification
    #   - age_reg: age regression (MAE, correlation, R²)
    # ------------------------------------------------------------------
    # Check if any subject metadata is available before expensive embedding
    has_metadata = any("age" in m or "sex" in m for m in train_set._recording_metadata)
    subject_probes = {}  # label_name → (probe, extras)
    train_embs = None
    _train_median_age = None

    if has_metadata:
        logger.info("Embedding train recordings for subject-trait probes...")
        train_embs, train_meta = _embed_all_clips(train_set, jepa, device, batch_size, num_workers)
        train_labels_dict = _extract_subject_labels(train_meta)
        # Capture train median age so eval splits use the same threshold
        train_ages = np.array([float(m["age"]) for m in train_meta if "age" in m])
        _train_median_age = float(np.median(train_ages)) if len(train_ages) >= 10 else None
        logger.info(
            "Train: %d recordings, labels available: %s (age median=%.1f)",
            len(train_embs), list(train_labels_dict.keys()),
            _train_median_age if _train_median_age else 0,
        )

        for label_name, labels in train_labels_dict.items():
            n_valid = int((~np.isnan(labels)).sum())
            if label_name == "age_reg":
                logger.info("Training age regression probe (%d valid)...", n_valid)
                probe, y_mean, y_std = _train_reg_probe(
                    train_embs, labels, device, subject_probe_epochs, subject_probe_lr
                )
                subject_probes[label_name] = ("reg", probe, y_mean, y_std)
            else:
                logger.info("Training %s classification probe (%d valid)...", label_name, n_valid)
                probe = _train_cls_probe(
                    train_embs, labels, device, subject_probe_epochs, subject_probe_lr
                )
                subject_probes[label_name] = ("cls", probe)
    else:
        logger.warning("No subject metadata available — skipping subject probes")

    # ------------------------------------------------------------------
    # Movie identity probe (per-clip: predict temporal segment of movie)
    # ------------------------------------------------------------------
    n_bins = 20  # discretize 3:23 movie into 20 segments (~10s each)
    pos_idx = feature_names.index("position_in_movie") if "position_in_movie" in feature_names else None
    movie_id_probe = None
    movie_id_bin_edges = None

    if pos_idx is not None:
        logger.info("Encoding train clips for movie identity probe (%d bins)...", n_bins)
        train_clip_embs, train_positions = _embed_clips_with_position(
            train_loader, jepa, device, pos_idx
        )
        logger.info("  %d clips, position range [%.1f, %.1f]",
                     len(train_clip_embs), train_positions.min(), train_positions.max())
        movie_id_probe, movie_id_bin_edges = _train_movie_id_probe(
            train_clip_embs, train_positions, device, n_bins,
            probe_epochs, probe_lr,
        )
        # Train-set accuracy for reference
        train_movie_metrics = _eval_movie_id_probe(
            movie_id_probe, train_clip_embs, train_positions, device, movie_id_bin_edges
        )
        logger.info("  Train movie-id: top1=%.4f  top5=%.4f  (chance=%.4f)",
                     train_movie_metrics["top1_acc"], train_movie_metrics["top5_acc"],
                     train_movie_metrics["chance"])

    # ------------------------------------------------------------------
    # Evaluate on each split
    # ------------------------------------------------------------------
    all_metrics = {}

    for split, loader in eval_loaders.items():
        # Movie-feature metrics
        if regression_probe is not None:
            logger.info("Evaluating movie-feature probes on %s...", split)
            movie_metrics = validation_loop(
                loader, jepa, regression_probe, classification_probe,
                device, feature_stats, feature_median, feature_names,
            )
            for k, v in movie_metrics.items():
                all_metrics[f"probe_eval/{split}/{k.split('/', 1)[-1]}"] = v

        # Movie identity metrics
        if movie_id_probe is not None:
            logger.info("Evaluating movie identity probe on %s...", split)
            eval_clip_embs, eval_positions = _embed_clips_with_position(
                loader, jepa, device, pos_idx
            )
            mid_metrics = _eval_movie_id_probe(
                movie_id_probe, eval_clip_embs, eval_positions, device, movie_id_bin_edges
            )
            for k, v in mid_metrics.items():
                all_metrics[f"probe_eval/{split}/movie_id/{k}"] = v

        # Subject-trait metrics
        if subject_probes:
            logger.info("Embedding %s recordings for subject-trait probes...", split)
            eval_embs, eval_meta = _embed_all_clips(
                eval_sets[split], jepa, device, batch_size, num_workers
            )
            eval_labels_dict = _extract_subject_labels(eval_meta, median_age=_train_median_age)
            logger.info("  %d recordings", len(eval_embs))

            for label_name, probe_info in subject_probes.items():
                eval_labels = eval_labels_dict.get(label_name)
                if eval_labels is None:
                    continue
                if probe_info[0] == "cls":
                    metrics = _eval_cls_probe(probe_info[1], eval_embs, eval_labels, device)
                    for k, v in metrics.items():
                        all_metrics[f"probe_eval/{split}/subject/{label_name}/{k}"] = v
                else:  # reg
                    _, probe, y_mean, y_std = probe_info
                    metrics = _eval_reg_probe(probe, eval_embs, eval_labels, device, y_mean, y_std)
                    for k, v in metrics.items():
                        all_metrics[f"probe_eval/{split}/subject/{label_name}/{k}"] = v

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n=== Probe Eval: {checkpoint_path.parent.name} ===")
    print(f"Config: nw={n_windows}, ws={window_size_seconds}s")
    print()
    for k, v in sorted(all_metrics.items()):
        if any(x in k for x in ("bal_acc", "auc", "corr", "subject", "mae", "r2", "movie_id", "top1", "top5")):
            vstr = f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v)
            print(f"  {k}: {vstr}")

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    try:
        import wandb
        if wandb_run_id:
            wandb_run = wandb.init(
                project=wandb_project, id=wandb_run_id, resume="must",
            )
        else:
            ckpt_name = checkpoint_path.parent.name
            wandb_run = wandb.init(
                project=wandb_project,
                group=wandb_group,
                name=f"probe_eval_{ckpt_name}",
                config={
                    "checkpoint": str(checkpoint_path),
                    "n_windows": n_windows,
                    "window_size_seconds": window_size_seconds,
                    "probe_epochs": probe_epochs,
                    "subject_probe_epochs": subject_probe_epochs,
                    "splits": splits,
                    "probe_label": train_set.probe_label_name,
                },
            )
        wandb_run.log(all_metrics)
        wandb_run.finish()
    except Exception as e:
        logger.warning("W&B logging failed: %s", e)

    return all_metrics


if __name__ == "__main__":
    fire.Fire(run)
