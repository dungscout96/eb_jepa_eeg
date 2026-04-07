"""Post-hoc linear probe evaluation for trained EEG JEPA checkpoints.

Loads a trained encoder from a checkpoint, freezes it, trains a fresh linear
probe on the train set, then evaluates on val (and optionally test).

The online probes logged during training are trained jointly with the encoder
and are NOT saved in the checkpoint. This script instead trains a fresh probe
from scratch on the frozen encoder representations — giving a cleaner measure
of representation quality that is independent of the online probe's training
trajectory.

Usage
-----
# Single checkpoint:
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \\
    --checkpoint=/abs/path/to/latest.pth.tar \\
    --n_windows=4 --window_size_seconds=4 --batch_size=32

# With W&B logging:
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \\
    --checkpoint=/abs/path/to/latest.pth.tar \\
    --n_windows=4 --window_size_seconds=4 --batch_size=32 \\
    --wandb_run_id=<original_run_id>

# Evaluate all Phase 1 sweep checkpoints (see scripts/eval_phase1_probes.py).
"""

import copy
from pathlib import Path

import fire
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import EEGEncoderTokens, MovieFeatureHead
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.jepa import MaskedJEPA
from eb_jepa.logging import get_logger
from eb_jepa.masking import MultiBlockMaskCollator
from eb_jepa.losses import VCLoss
from eb_jepa.training_utils import load_config, load_checkpoint, setup_device, setup_seed
from experiments.eeg_jepa.eval import validation_loop
from experiments.eeg_jepa.main import (
    NUMERIC_FEATURES,
    RegressionLoss,
    ClassificationLoss,
    resolve_preprocessed_dir,
    _PREPROCESSED_DIRS,
)

logger = get_logger(__name__)


def run(
    checkpoint: str,
    # Data config — must match the checkpoint's training config
    n_windows: int = 4,
    window_size_seconds: int = 2,
    batch_size: int = 64,
    num_workers: int = 4,
    # Probe training
    probe_epochs: int = 20,
    probe_lr: float = 1e-3,
    # Eval splits: "val", "test", or "val,test"
    splits: str = "val",
    # W&B (optional — resume the original run to append eval metrics)
    wandb_run_id: str = "",
    wandb_project: str = "eb_jepa",
    wandb_group: str = "probe_eval",
    # Config file (uses training defaults)
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    seed: int = 2025,
):
    """Train a fresh linear probe on a frozen encoder and evaluate on val/test.

    Args:
        checkpoint: Absolute path to a saved latest.pth.tar.
        n_windows: Must match the checkpoint's --data.n_windows.
        window_size_seconds: Must match --data.window_size_seconds.
        batch_size: Batch size for data loading.
        num_workers: DataLoader workers.
        probe_epochs: Epochs to train the fresh probe (default 20).
        probe_lr: Probe learning rate.
        splits: Comma-separated list of splits to evaluate ("val", "test").
        wandb_run_id: If set, resumes this W&B run to log probe_eval/* metrics.
        wandb_project: W&B project name.
        wandb_group: W&B group for standalone eval runs.
        fname: Config file path.
        seed: Random seed.
    """
    setup_seed(seed)
    device = setup_device(None)
    cfg = load_config(fname, {
        "data.n_windows": n_windows,
        "data.window_size_seconds": window_size_seconds,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
    })

    checkpoint_path = Path(checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    preprocessed = cfg.data.get("preprocessed", False)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading train set (for norm stats and probe training)...")
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=n_windows,
        window_size_seconds=window_size_seconds,
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

    eval_loaders = {}
    for split in splits.split(","):
        split = split.strip()
        split_set = JEPAMovieDataset(
            split=split,
            n_windows=n_windows,
            window_size_seconds=window_size_seconds,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
        )
        eval_loaders[split] = DataLoader(
            split_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info("Split '%s': %d recordings", split, len(split_set))

    # ------------------------------------------------------------------
    # Encoder (frozen)
    # ------------------------------------------------------------------
    n_chans = train_set.n_chans
    n_times = train_set.n_times
    n_features = len(NUMERIC_FEATURES)
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
    predictor_stub = None  # not needed for eval but MaskedJEPA requires it

    from eb_jepa.architectures import MaskedPredictor
    predictor_stub = MaskedPredictor(
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
    projector = None
    reg_type = cfg.loss.get("regularizer", "vc")
    if reg_type == "vc" and (cfg.loss.std_coeff > 0 or cfg.loss.cov_coeff > 0):
        from eb_jepa.architectures import Projector
        projector = Projector(f"{embed_dim}-{embed_dim * 4}-{embed_dim * 4}")
        regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    else:
        regularizer = None

    jepa = MaskedJEPA(
        encoder, target_encoder, predictor_stub, mask_collator, regularizer,
    ).to(device)

    # Load checkpoint weights
    optimizer_stub = Adam(jepa.parameters(), lr=1e-3)  # needed by load_checkpoint
    ckpt_info = load_checkpoint(checkpoint_path, jepa, optimizer_stub, device=device)
    logger.info("Loaded checkpoint at epoch %d", ckpt_info.get("epoch", "?"))

    # Freeze encoder completely
    for p in jepa.parameters():
        p.requires_grad_(False)
    jepa.eval()

    # ------------------------------------------------------------------
    # Fresh probe heads
    # ------------------------------------------------------------------
    from eb_jepa.jepa import MaskedJEPAProbe

    reg_loss_fn = RegressionLoss(
        feature_stats["mean"].to(device),
        feature_stats["std"].to(device),
    )
    cls_loss_fn = ClassificationLoss(feature_median.to(device))

    reg_head = MovieFeatureHead(embed_dim, cfg.model.hdec, n_features)
    cls_head = MovieFeatureHead(embed_dim, cfg.model.hdec, n_features)
    regression_probe = MaskedJEPAProbe(jepa, reg_head, reg_loss_fn).to(device)
    classification_probe = MaskedJEPAProbe(jepa, cls_head, cls_loss_fn).to(device)

    probe_optimizer = Adam(
        list(regression_probe.head.parameters())
        + list(classification_probe.head.parameters()),
        lr=probe_lr,
    )

    # ------------------------------------------------------------------
    # Train probe on frozen encoder (train set)
    # ------------------------------------------------------------------
    logger.info("Training probe for %d epochs on frozen encoder...", probe_epochs)
    for epoch in range(probe_epochs):
        regression_probe.train()
        classification_probe.train()
        reg_total = cls_total = 0.0
        n = 0
        for eeg, features, _ in tqdm(train_loader, desc=f"Probe ep {epoch+1}/{probe_epochs}", leave=False):
            eeg = eeg.to(device)
            features = features.to(device)
            probe_optimizer.zero_grad()
            reg_loss = regression_probe(eeg, features)
            cls_loss = classification_probe(eeg, features)
            (reg_loss + cls_loss).backward()
            probe_optimizer.step()
            reg_total += reg_loss.item()
            cls_total += cls_loss.item()
            n += 1
        logger.info(
            "Probe ep %d/%d — reg_loss=%.4f  cls_loss=%.4f",
            epoch + 1, probe_epochs,
            reg_total / max(n, 1),
            cls_total / max(n, 1),
        )

    # ------------------------------------------------------------------
    # Evaluate on each split
    # ------------------------------------------------------------------
    all_metrics = {}
    for split, loader in eval_loaders.items():
        logger.info("Evaluating on %s split...", split)
        metrics = validation_loop(
            loader,
            jepa,
            regression_probe,
            classification_probe,
            device,
            feature_stats,
            feature_median,
            NUMERIC_FEATURES,
        )
        # Re-key as probe_eval/{split}/...
        split_metrics = {f"probe_eval/{split}/{k.split('/', 1)[-1]}": v for k, v in metrics.items()}
        all_metrics.update(split_metrics)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n=== Probe Eval Results ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: nw={n_windows}, ws={window_size_seconds}s")
    print()
    for k, v in sorted(all_metrics.items()):
        if "bal_acc" in k or "auc" in k or "corr" in k:
            print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # W&B logging (optional)
    # ------------------------------------------------------------------
    if wandb_run_id:
        try:
            import wandb
            run_ref = f"{wandb_project}/{wandb_run_id}"
            logger.info("Resuming W&B run %s to log probe_eval metrics", run_ref)
            wandb_run = wandb.init(
                project=wandb_project,
                id=wandb_run_id,
                resume="must",
            )
            wandb_run.log(all_metrics)
            wandb_run.finish()
        except Exception as e:
            logger.warning("W&B logging failed: %s", e)
    else:
        # Log as a new run tagged with checkpoint path
        try:
            import wandb
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
                    "splits": splits,
                },
            )
            wandb_run.log(all_metrics)
            wandb_run.finish()
        except Exception as e:
            logger.warning("W&B logging failed: %s", e)

    return all_metrics


if __name__ == "__main__":
    fire.Fire(run)
