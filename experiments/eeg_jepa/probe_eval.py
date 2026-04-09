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

def _embed_all_clips(dataset, jepa, device, batch_size, num_workers):
    """Encode every clip of every recording, return per-recording mean embeddings.

    Iterates deterministically over all clips in each recording (no random
    crop), encodes with the frozen encoder, and mean-pools over clips to get
    one [D] vector per recording.

    Returns
    -------
    embeddings : np.ndarray  [N_recordings, D]
    labels     : np.ndarray  [N_recordings]  float, NaN where unavailable
    """
    jepa.eval()
    all_embs = []
    all_labels = []

    for rec_idx in range(len(dataset)):
        crop_inds = dataset._crop_inds[rec_idx]
        n_total = len(crop_inds)
        required = (dataset.n_windows - 1) * dataset.temporal_stride + 1
        n_clips = n_total - required + 1

        if n_clips <= 0:
            continue

        # Build all valid start positions for this recording
        clip_embs = []
        for start in range(0, n_clips, max(1, n_clips // 32)):
            # Sub-sample up to 32 clips per recording to keep cost reasonable
            indices = list(range(start, start + required, dataset.temporal_stride))
            eeg_np = _read_raw_windows(dataset._fif_paths[rec_idx], crop_inds[indices])
            eeg = torch.from_numpy(eeg_np).unsqueeze(0)  # [1, n_windows, C, T]
            eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
            eeg = eeg.to(device)

            with torch.no_grad():
                tokens = jepa.context_encoder.encode_tokens(eeg, mask=None)
                emb = tokens.mean(dim=1)  # [1, D]
            clip_embs.append(emb.squeeze(0).cpu())

        if clip_embs:
            rec_emb = torch.stack(clip_embs).mean(dim=0)  # [D] — mean over clips
            all_embs.append(rec_emb.numpy())
            all_labels.append(dataset._probe_labels[rec_idx])

    return np.stack(all_embs), np.array(all_labels, dtype=float)


def _train_subject_probe(train_embs, train_labels, device, probe_epochs, probe_lr):
    """Train a linear probe for subject-trait classification.

    Args:
        train_embs:   np.ndarray [N, D]
        train_labels: np.ndarray [N]  (0/1 float; NaN entries excluded)

    Returns the trained nn.Linear.
    """
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


def _eval_subject_probe(probe, embs, labels, device):
    """Evaluate subject-trait probe on a set of per-recording embeddings.

    Returns dict with bal_acc and auc (NaN-safe).
    """
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return {"subject_trait/bal_acc": float("nan"), "subject_trait/auc": float("nan")}

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

    return {"subject_trait/bal_acc": bal_acc, "subject_trait/auc": auc}


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

    # Infer encoder_depth from checkpoint state dict to avoid mismatch
    _ckpt_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("model_state_dict", {})
    _depth = max(
        int(k.split(".")[3]) + 1
        for k in _ckpt_sd
        if k.startswith("context_encoder.transformer.layers.")
    )

    cfg = load_config(fname, {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "model.encoder_depth": _depth,
    })

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
    if cfg.loss.std_coeff > 0 or cfg.loss.cov_coeff > 0:
        projector = Projector(f"{embed_dim}-{embed_dim * 4}-{embed_dim * 4}")
        regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)

    jepa = MaskedJEPA(
        encoder, target_encoder, predictor, mask_collator, regularizer,
    ).to(device)

    ckpt_info = load_checkpoint(checkpoint_path, jepa, optimizer=None, device=device)
    logger.info("Loaded checkpoint at epoch %d", ckpt_info.get("epoch", "?"))

    for p in jepa.parameters():
        p.requires_grad_(False)
    jepa.eval()

    # ------------------------------------------------------------------
    # Movie-feature probes (per-clip, same as online probes during training)
    # ------------------------------------------------------------------
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
    # Subject-trait probe (per-recording embeddings, pooled over all clips)
    # ------------------------------------------------------------------
    # Check if subject labels are available before expensive embedding
    n_valid_train_labels = sum(1 for v in train_set._probe_labels if not math.isnan(v))
    subject_probe = None
    if n_valid_train_labels >= 10:
        logger.info("Embedding train recordings for subject-trait probe...")
        train_embs, train_labels = _embed_all_clips(train_set, jepa, device, batch_size, num_workers)
        logger.info(
            "Train: %d recordings, %d with valid subject labels  (label: %s)",
            len(train_embs), int((~np.isnan(train_labels)).sum()), train_set.probe_label_name,
        )
        logger.info("Training subject-trait probe for %d epochs...", subject_probe_epochs)
        subject_probe = _train_subject_probe(
            train_embs, train_labels, device, subject_probe_epochs, subject_probe_lr
        )
    else:
        logger.warning(
            "Too few valid subject labels (%d) — skipping subject probe",
            n_valid_train_labels,
        )

    # ------------------------------------------------------------------
    # Evaluate on each split
    # ------------------------------------------------------------------
    all_metrics = {}

    for split, loader in eval_loaders.items():
        # Movie-feature metrics
        logger.info("Evaluating movie-feature probes on %s...", split)
        movie_metrics = validation_loop(
            loader, jepa, regression_probe, classification_probe,
            device, feature_stats, feature_median, feature_names,
        )
        for k, v in movie_metrics.items():
            all_metrics[f"probe_eval/{split}/{k.split('/', 1)[-1]}"] = v

        # Subject-trait metrics
        if subject_probe is not None:
            logger.info("Embedding %s recordings for subject-trait probe...", split)
            eval_embs, eval_labels = _embed_all_clips(
                eval_sets[split], jepa, device, batch_size, num_workers
            )
            n_valid_eval = int((~np.isnan(eval_labels)).sum())
            logger.info("  %d recordings, %d with valid labels", len(eval_embs), n_valid_eval)
            subject_metrics = _eval_subject_probe(subject_probe, eval_embs, eval_labels, device)
            for k, v in subject_metrics.items():
                all_metrics[f"probe_eval/{split}/{k}"] = v

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n=== Probe Eval: {checkpoint_path.parent.name} ===")
    print(f"Config: nw={n_windows}, ws={window_size_seconds}s  |  label: {train_set.probe_label_name}")
    print()
    for k, v in sorted(all_metrics.items()):
        if any(x in k for x in ("bal_acc", "auc", "corr", "subject")):
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
