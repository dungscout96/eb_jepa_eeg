"""
EEG JEPA Training Script

Train a self-supervised EEG prediction model on HBN movie-watching data using
Joint Embedding Predictive Architecture (JEPA) with VC regularization.

Two evaluation decoder probes are trained alongside JEPA:
  1. Regression probe: predicts continuous movie features (MSELoss)
  2. Classification probe: predicts binary movie feature labels (BCEWithLogitsLoss)
"""

from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import (
    MovieFeatureHead,
    Projector,
    EEGEncoder,
    MLPEEGPredictor,
    StateOnlyPredictor,
)
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VCLoss
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from experiments.eeg_jepa.eval import validation_loop

logger = get_logger(__name__)

NUMERIC_FEATURES = JEPAMovieDataset.DEFAULT_FEATURES


# ---------------------------------------------------------------------------
# Loss functions for movie-feature probes
# ---------------------------------------------------------------------------


class RegressionLoss(nn.Module):
    """MSE loss with target z-normalization using training set statistics."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)  # [n_features]
        self.register_buffer("std", std)  # [n_features]

    def forward(self, pred, target):
        # pred: [B, T, n_features], target: [B, T, n_features]
        target_norm = (target - self.mean) / (self.std + 1e-8)
        return F.mse_loss(pred, target_norm)


class ClassificationLoss(nn.Module):
    """BCE loss with median-based binary discretization of continuous targets."""

    def __init__(self, median):
        super().__init__()
        self.register_buffer("median", median)  # [n_features]

    def forward(self, pred, target):
        # pred: [B, T, n_features] logits, target: [B, T, n_features] continuous
        binary = (target > self.median).float()
        return F.binary_cross_entropy_with_logits(pred, binary)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run(
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train an EEG JEPA model on HBN movie-watching data.

    Args:
        fname: Path to YAML config file
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., model.lr=0.001)
    """
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Setup
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)
    temporal_stride = cfg.data.get("temporal_stride", 1)

    # Create experiment directory
    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir = Path(cfg.meta.model_folder)
            folder_name = exp_dir.name
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            stride_suffix = f"_stride{temporal_stride}" if temporal_stride > 1 else ""
            exp_name = (
                f"eeg_jepa_bs{cfg.data.batch_size}"
                f"_lr{cfg.optim.lr}"
                f"_std{cfg.loss.std_coeff}"
                f"_cov{cfg.loss.cov_coeff}"
                f"{stride_suffix}"
            )
            exp_dir = get_unified_experiment_dir(
                example_name="eeg_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        folder_name = exp_dir.name
        exp_name = folder_name.rsplit("_seed", 1)[0]

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "eeg_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["eeg_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    logger.info("Loading HBN Movie datasets...")
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        temporal_stride=temporal_stride,
        cfg=cfg.data,
    )
    val_set = JEPAMovieDataset(
        split="val",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        temporal_stride=temporal_stride,
        eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data,
    )

    # Compute feature statistics from training set (used by losses)
    feature_stats = train_set.compute_feature_stats()
    feature_median = train_set.compute_feature_median()

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    log_data_info(
        "HBN EEG Movie",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_set),
        val_samples=len(val_set),
    )

    # Auto-detect EEG dimensions from data
    n_chans = train_set.n_chans
    n_features = len(NUMERIC_FEATURES)
    chs_info = train_set.get_chs_info()
    logger.info("EEG channels: %d, Movie features: %d", n_chans, n_features)

    # ------------------------------------------------------------------
    # Initialize JEPA model
    # ------------------------------------------------------------------
    logger.info("Initializing model...")
    n_times = train_set.n_times
    encoder_kwargs = {}
    for key in ("encoder_embed_dim", "encoder_depth", "encoder_heads", "encoder_head_dim"):
        if cfg.model.get(key) is not None:
            encoder_kwargs[key.replace("encoder_", "")] = cfg.model[key]
    encoder = EEGEncoder(
        n_chans, cfg.model.henc, cfg.model.dstc,
        chs_info=chs_info, n_times=n_times, **encoder_kwargs,
    )
    predictor_model = MLPEEGPredictor(
        cfg.model.dstc * 2, cfg.model.hpre, cfg.model.dstc
    )
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(
        f"{cfg.model.dstc}-{cfg.model.dstc * 4}-{cfg.model.dstc * 4}"
    )
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    # ------------------------------------------------------------------
    # Initialize evaluation decoder probes
    # ------------------------------------------------------------------

    # 1. Regression probe: continuous movie feature prediction (MSELoss)
    reg_head = MovieFeatureHead(cfg.model.dstc, cfg.model.hdec, n_features)
    reg_loss_fn = RegressionLoss(
        feature_stats["mean"].to(device),
        feature_stats["std"].to(device),
    )
    regression_probe = JEPAProbe(jepa, reg_head, reg_loss_fn).to(device)

    # 2. Classification probe: binary movie feature prediction (BCEWithLogitsLoss)
    cls_head = MovieFeatureHead(cfg.model.dstc, cfg.model.hdec, n_features)
    cls_loss_fn = ClassificationLoss(feature_median.to(device))
    classification_probe = JEPAProbe(jepa, cls_head, cls_loss_fn).to(device)

    # Log model info
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    reg_head_params = sum(p.numel() for p in reg_head.parameters())
    cls_head_params = sum(p.numel() for p in cls_head.parameters())
    log_model_info(
        jepa,
        {
            "encoder": encoder_params,
            "predictor": predictor_params,
            "reg_head": reg_head_params,
            "cls_head": cls_head_params,
        },
    )

    jepa.train()
    regression_probe.train()
    classification_probe.train()

    # Separate optimizers: JEPA is purely self-supervised, probes are online eval
    optimizer = Adam(jepa.parameters(), lr=cfg.optim.lr)
    probe_optimizer = Adam(
        list(regression_probe.head.parameters())
        + list(classification_probe.head.parameters()),
        lr=cfg.optim.lr,
    )

    # Log configuration
    log_config(cfg)

    # Load checkpoint if requested
    start_epoch = 0
    global_step = 0
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(ckpt_path, jepa, optimizer, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        global_step = ckpt_info.get("step", 0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Starting training for %d epochs...", cfg.optim.epochs)

    for epoch in range(start_epoch, cfg.optim.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=cfg.logging.get("tqdm_silent", False),
        )

        for eeg, features in pbar:
            # eeg: [B, T, C, W] -> [B, 1, T, C, W] for JEPA encoder
            x = eeg.unsqueeze(1).to(device)
            features = features.to(device)  # [B, T, n_features]

            # --- JEPA pretraining (self-supervised, no labels) ---
            optimizer.zero_grad()
            _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                x,
                actions=None,
                nsteps=cfg.model.steps,
                unroll_mode="parallel",
                compute_loss=True,
                return_all_steps=False,
            )
            jepa_loss.backward()
            optimizer.step()

            # --- Probe training (online eval on frozen encoder) ---
            probe_optimizer.zero_grad()
            reg_loss = regression_probe(x, features)
            cls_loss = classification_probe(x, features)
            (reg_loss + cls_loss).backward()
            probe_optimizer.step()

            pbar.set_postfix(
                {
                    "loss": f"{jepa_loss.item():.4f}",
                    "vc": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                    "reg": f"{reg_loss.item():.4f}",
                    "cls": f"{cls_loss.item():.4f}",
                }
            )

            # Per-step wandb logging for training metrics
            if wandb_run:
                import wandb

                step_metrics = {
                    "train_step/jepa_loss": jepa_loss.item(),
                    "train_step/vc_loss": regl.item(),
                    "train_step/pred_loss": pl.item(),
                    "train_step/reg_loss": reg_loss.item(),
                    "train_step/cls_loss": cls_loss.item(),
                }
                for k, v in regldict.items():
                    step_metrics[f"train_step/{k}"] = float(v)
                wandb.log(step_metrics, step=global_step)

            global_step += 1

        # Validation and logging
        if epoch % cfg.logging.log_every == 0:
            val_logs = validation_loop(
                val_loader,
                jepa,
                regression_probe,
                classification_probe,
                cfg.model.steps,
                device,
                feature_stats,
                feature_median,
                NUMERIC_FEATURES,
            )

            train_metrics = {
                "epoch": epoch,
                "train/loss": jepa_loss.item(),
                "train/vc_loss": regl.item(),
                "train/pred_loss": pl.item(),
                "train/reg_loss": reg_loss.item(),
                "train/cls_loss": cls_loss.item(),
            }
            for k, v in regldict.items():
                train_metrics[f"train/{k}"] = float(v)

            all_metrics = {**train_metrics, **val_logs}

            if wandb_run:
                import wandb

                wandb.log(all_metrics, step=global_step)

            log_epoch(
                epoch,
                {
                    "loss": jepa_loss.item(),
                    "vc": regl.item(),
                    "pred": pl.item(),
                    "val_reg": val_logs.get("val/reg_loss", 0),
                    "val_cls": val_logs.get("val/cls_loss", 0),
                },
                total_epochs=cfg.optim.epochs,
            )

        # Save checkpoint
        save_checkpoint(
            exp_dir / "latest.pth.tar",
            model=jepa,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
        )
        if epoch % cfg.logging.save_every == 0 and epoch > 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch}.pth.tar",
                model=jepa,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
            )

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating on test set...")
    test_set = JEPAMovieDataset(
        split="test",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        temporal_stride=temporal_stride,
        eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    test_logs = validation_loop(
        test_loader,
        jepa,
        regression_probe,
        classification_probe,
        cfg.model.steps,
        device,
        feature_stats,
        feature_median,
        NUMERIC_FEATURES,
    )
    # Rename val/ -> test/ for clarity
    test_metrics = {k.replace("val/", "test/"): v for k, v in test_logs.items()}

    if wandb_run:
        import wandb

        wandb.log(test_metrics, step=global_step)
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    fire.Fire(run)
