"""Train an EEG encoder via symmetric CLIP InfoNCE against V-JEPA-2 frame embeddings.

Distinct entry point from `jepa_pretrain.py`. The two scripts share the dataset,
encoder builder, and most helpers, but the CLIP path has no masking / predictor /
anti-collapse. Optionally warm-starts the encoder from a JEPA checkpoint via
`--meta.encoder_init_from=<path>`; in that case the optimizer / LR schedule /
step counter are still fresh.
"""
import math
from pathlib import Path

import fire
import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import MovieCLIPHead
from eb_jepa.clip import CLIPPretrain
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.logging import get_logger
from eb_jepa.paths import resolve_preprocessed_dir
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_unified_experiment_dir,
    load_config,
    load_encoder_weights,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)

logger = get_logger(__name__)

_DEFAULT_CFG_PATH = str(
    Path(__file__).resolve().parents[2] / "config" / "clip_pretrain.yaml"
)


def run(
    fname: str = _DEFAULT_CFG_PATH,
    cfg=None,
    folder=None,
    **overrides,
):
    """Train a CLIP-aligned EEG encoder on HBN movie-watching data."""
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)
    temporal_stride = cfg.data.get("temporal_stride", 1)

    # ------------------------------------------------------------------
    # Experiment directory + W&B
    # ------------------------------------------------------------------
    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir = Path(cfg.meta.model_folder)
            exp_name = exp_dir.name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            stride_suffix = f"_stride{temporal_stride}" if temporal_stride > 1 else ""
            nw_suffix = f"_nw{cfg.data.n_windows}_ws{cfg.data.window_size_seconds}s"
            init_suffix = "_warm" if cfg.meta.get("encoder_init_from") else ""
            exp_name = (
                f"eeg_clip_bs{cfg.data.batch_size}"
                f"_lr{cfg.optim.lr}"
                f"_T{cfg.loss.temperature}"
                f"_P{cfg.loss.proj_dim}"
                f"{nw_suffix}{stride_suffix}{init_suffix}"
            )
            exp_dir = get_unified_experiment_dir(
                example_name="eeg_clip",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_name = exp_dir.name.rsplit("_seed", 1)[0]

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "eeg_clip", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["eeg_clip", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    logger.info("Loading HBN Movie dataset...")
    preprocessed = cfg.data.get("preprocessed", False)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    task_cfg = cfg.data.get("task", "ThePresent")
    task = list(task_cfg) if not isinstance(task_cfg, str) else task_cfg

    train_set = JEPAMovieDataset(
        split="train",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        task=task,
        temporal_stride=temporal_stride,
        feature_names=[],
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
    )
    if train_set.frame_embedding_dim == 0:
        raise RuntimeError(
            "JEPAMovieDataset reports frame_embedding_dim=0 — no V-JEPA-2 .npz "
            "was loaded. Check MOVIE_METADATA paths."
        )

    num_workers = cfg.data.num_workers
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    log_data_info(
        "HBN EEG Movie",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_set),
    )

    n_chans = train_set.n_chans
    chs_info = train_set.get_chs_info()
    n_times = train_set.n_times
    logger.info("EEG channels: %d, V-JEPA-2 dim: %d", n_chans, train_set.frame_embedding_dim)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Initializing CLIP model...")
    encoder = build_encoder(
        cfg, n_chans=n_chans, n_times=n_times, chs_info=chs_info,
        n_windows=cfg.data.n_windows,
    )
    encoder_init_from = cfg.meta.get("encoder_init_from")
    if encoder_init_from:
        load_encoder_weights(encoder, encoder_init_from, device=torch.device("cpu"))
    clip_head = MovieCLIPHead(
        eeg_in_dim=cfg.model.encoder_embed_dim,
        vision_in_dim=train_set.frame_embedding_dim,
        proj_dim=int(cfg.loss.proj_dim),
        temperature=float(cfg.loss.temperature),
    )
    model = CLIPPretrain(encoder, clip_head).to(device)

    log_model_info(
        model,
        {
            "encoder": sum(p.numel() for p in model.encoder.parameters()),
            "clip_head": sum(p.numel() for p in model.clip_head.parameters()),
        },
    )

    model.train()
    optimizer = Adam(model.parameters(), lr=cfg.optim.lr)

    lr_min = cfg.optim.get("lr_min", 0.0)
    warmup_epochs = cfg.optim.get("warmup_epochs", 0)
    total_epochs = cfg.optim.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        if lr_min == 0.0:
            return 1.0
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min / cfg.optim.lr + (1 - lr_min / cfg.optim.lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_config(cfg)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Starting CLIP pretraining for %d epochs...", cfg.optim.epochs)
    global_step = 0
    for epoch in range(cfg.optim.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=cfg.logging.get("tqdm_silent", False),
        )

        for eeg, _features, embeds, _shot_ids, _probe_labels in pbar:
            eeg = eeg.to(device, non_blocking=True)
            embeds = embeds.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, loss_dict = model(eeg, embeds)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "e2v":  f"{loss_dict['clip_top1_e2v']:.3f}",
                "v2e":  f"{loss_dict['clip_top1_v2e']:.3f}",
            })

            if wandb_run:
                import wandb
                wandb.log(
                    {f"train_step/{k}": float(v) for k, v in loss_dict.items()},
                    step=global_step,
                )

            global_step += 1

        # Epoch logging
        if epoch % cfg.logging.log_every == 0:
            epoch_metrics = {
                "epoch": epoch,
                "train/loss": loss.item(),
                "train/clip_top1_e2v": loss_dict["clip_top1_e2v"],
                "train/clip_top1_v2e": loss_dict["clip_top1_v2e"],
                "train/clip_logit_scale": loss_dict["clip_logit_scale"],
                "train/lr": scheduler.get_last_lr()[0],
            }
            if wandb_run:
                import wandb
                wandb.log(epoch_metrics, step=global_step)
            log_epoch(
                epoch,
                {
                    "loss": loss.item(),
                    "e2v":  loss_dict["clip_top1_e2v"],
                    "v2e":  loss_dict["clip_top1_v2e"],
                },
                total_epochs=cfg.optim.epochs,
            )

        save_checkpoint(
            exp_dir / "latest.pth.tar",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
        )
        if epoch % cfg.logging.save_every == 0 and epoch > 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch}.pth.tar",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
            )

        scheduler.step()

    if wandb_run:
        import wandb
        wandb.finish()

    logger.info("CLIP pretraining complete!")


if __name__ == "__main__":
    fire.Fire(run)
