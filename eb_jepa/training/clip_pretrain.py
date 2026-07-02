"""Train an EEG encoder via symmetric CLIP InfoNCE against V-JEPA-2 frame embeddings.

Distinct entry point from `jepa_pretrain.py`. The two scripts share the dataset,
encoder builder, and most helpers, but the CLIP path has no masking / predictor /
anti-collapse. Optionally warm-starts the encoder from a JEPA checkpoint via
`--meta.encoder_init_from=<path>`; in that case the optimizer / LR schedule /
step counter are still fresh.
"""
import math
import random
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from eb_jepa.architectures import MovieCLIPHead
from eb_jepa.clip import CLIPPretrain, SceneCLIPPretrain
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.logging import get_logger
from eb_jepa.paths import resolve_preprocessed_dir
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_unified_experiment_dir,
    load_checkpoint,
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


@torch.no_grad()
def evaluate_recipe(model, val_loader, device, *, max_windows: int = 2000) -> dict:
    """Per-epoch diagnostics for the §9 recipe.

    Accumulates ``z_eeg, z_vis, shot_ids, scene_ids`` across val batches (up to
    ``max_windows``), then computes:

    - ``val/vision_shot_auc`` / ``val/vision_scene_auc``: pairwise cosine on
      ``z_vis`` classified by same-shot / same-scene. Sanity check on the
      preprocessing — should hit the doc's 0.92 / 0.94 numbers if centering and
      shot-mean replacement worked.
    - ``val/clip_shot_auc`` / ``val/clip_scene_auc``: same labels but pairwise
      cosine of ``z_eeg @ z_vis.T``. The actual cross-modal alignment metric.
    - ``val/clip_nn_top1``: top-1 retrieval purity at shot level (exclude self).
    - ``val/scene_collapse_ratio``: ``mean(within_scene_spread) /
      between_scene_spread`` on ``z_eeg``. >> 1 = healthy; → 0 = scene collapse.
    """
    model.eval()
    zs_eeg, zs_vis, all_shot, all_scene = [], [], [], []
    head = getattr(model, "clip_head", None)
    if head is None:
        model.train()
        return {}
    n_accum = 0
    for batch in val_loader:
        eeg, _features, embeds, shot_ids, scene_ids, _t_starts, _probe = batch
        eeg = eeg.to(device, non_blocking=True)
        embeds = embeds.to(device, non_blocking=True)
        tokens = model.encoder.encode_tokens(eeg, mask=None)
        pooled = model.encoder.pool_to_windows(tokens)
        z_eeg = head.project_eeg(pooled)
        z_vis = head.project_vision(embeds.reshape(-1, embeds.shape[-1]))
        zs_eeg.append(z_eeg.float().cpu())
        zs_vis.append(z_vis.float().cpu())
        all_shot.append(shot_ids.reshape(-1).cpu())
        all_scene.append(scene_ids.reshape(-1).cpu())
        n_accum += z_eeg.shape[0]
        if n_accum >= max_windows:
            break
    model.train()
    if not zs_eeg:
        return {}
    Z_eeg = torch.cat(zs_eeg, dim=0)[:max_windows]
    Z_vis = torch.cat(zs_vis, dim=0)[:max_windows]
    shot = torch.cat(all_shot, dim=0)[:max_windows]
    scene = torch.cat(all_scene, dim=0)[:max_windows]
    N = Z_eeg.shape[0]
    if N < 4:
        return {}

    valid = (shot >= 0) & (scene >= 0)
    Z_eeg, Z_vis, shot, scene = Z_eeg[valid], Z_vis[valid], shot[valid], scene[valid]
    if Z_eeg.shape[0] < 4:
        return {}

    S_vv = (Z_vis @ Z_vis.T).numpy()
    S_ev = (Z_eeg @ Z_vis.T).numpy()
    shot_np = shot.numpy()
    scene_np = scene.numpy()
    N = shot_np.shape[0]
    iu = np.triu_indices(N, k=1)
    shot_eq = (shot_np[iu[0]] == shot_np[iu[1]]).astype(np.int32)
    scene_eq = (scene_np[iu[0]] == scene_np[iu[1]]).astype(np.int32)
    sym_ev = 0.5 * (S_ev + S_ev.T)

    metrics: dict[str, float] = {}
    try:
        from sklearn.metrics import roc_auc_score
        if shot_eq.sum() > 0 and shot_eq.sum() < shot_eq.shape[0]:
            metrics["val/vision_shot_auc"] = float(roc_auc_score(shot_eq, S_vv[iu]))
            metrics["val/clip_shot_auc"] = float(roc_auc_score(shot_eq, sym_ev[iu]))
        if scene_eq.sum() > 0 and scene_eq.sum() < scene_eq.shape[0]:
            metrics["val/vision_scene_auc"] = float(roc_auc_score(scene_eq, S_vv[iu]))
            metrics["val/clip_scene_auc"] = float(roc_auc_score(scene_eq, sym_ev[iu]))
    except ImportError:
        logger.warning("sklearn not available — skipping AUC metrics")

    # NN top-1 (shot level), excluding self
    sim_self_excluded = S_ev.copy()
    np.fill_diagonal(sim_self_excluded, -np.inf)
    nn = sim_self_excluded.argmax(axis=1)
    metrics["val/clip_nn_top1"] = float((shot_np[nn] == shot_np).mean())

    # Scene-collapse on z_eeg: within-scene spread / between-scene spread
    Z_eeg_np = Z_eeg.numpy()
    centroids = []
    within = []
    for s in np.unique(scene_np):
        members = Z_eeg_np[scene_np == s]
        if members.shape[0] < 2:
            continue
        c = members.mean(axis=0)
        centroids.append(c)
        within.append(np.linalg.norm(members - c, axis=1).mean())
    if len(centroids) >= 2:
        C = np.stack(centroids)
        between = np.linalg.norm(C - C.mean(axis=0), axis=1).mean()
        within_mean = float(np.mean(within)) if within else 0.0
        metrics["val/scene_collapse_ratio"] = (
            float(within_mean / between) if between > 1e-8 else float("inf")
        )

    metrics["val/n_windows"] = float(N)
    return metrics


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

    loss_mode = str(cfg.loss.get("mode", "clip"))
    if loss_mode not in {"clip", "scene_clip"}:
        raise ValueError(f"Unknown loss.mode={loss_mode!r}; expected 'clip' or 'scene_clip'.")
    recipe_mode = loss_mode == "scene_clip"
    recipe_target_kind = str(cfg.loss.get("target_kind", "shot_mean"))
    recipe_mean_center = bool(cfg.loss.get("mean_center", True))
    temporal_buffer_s = float(cfg.loss.get("temporal_buffer_s", 2.0))

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
        recipe_mode=recipe_mode,
        recipe_target_kind=recipe_target_kind,
        recipe_mean_center=recipe_mean_center,
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

    # Optional val loader for recipe diagnostics (only in scene_clip mode).
    val_loader = None
    val_every = int(cfg.get("eval", {}).get("val_every", 0))
    if recipe_mode and val_every > 0:
        val_max_windows = int(cfg.eval.get("val_max_windows", 2000))
        val_rec_frac = float(cfg.eval.get("val_recording_fraction", 0.1))
        val_set = JEPAMovieDataset(
            split="val",
            n_windows=cfg.data.n_windows,
            window_size_seconds=cfg.data.window_size_seconds,
            task=task,
            temporal_stride=temporal_stride,
            feature_names=[],
            cfg=cfg.data,
            preprocessed=preprocessed,
            preprocessed_dir=preprocessed_dir,
            recipe_mode=True,
            recipe_target_kind=recipe_target_kind,
            recipe_mean_center=recipe_mean_center,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
        )
        # Deterministic recording subset for stable per-epoch diagnostics.
        n_val_rec = max(1, int(len(val_set) * val_rec_frac))
        rng = random.Random(cfg.meta.seed)
        sub_idx = rng.sample(range(len(val_set)), min(n_val_rec, len(val_set)))
        val_loader = DataLoader(
            Subset(val_set, sub_idx),
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        logger.info(
            "Val diagnostics enabled: %d/%d val recordings sampled, cap=%d windows",
            n_val_rec, len(val_set), val_max_windows,
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
    resume_from = cfg.meta.get("resume_from")
    if encoder_init_from and resume_from:
        raise ValueError("Set either meta.encoder_init_from or meta.resume_from, not both.")
    if encoder_init_from:
        load_encoder_weights(encoder, encoder_init_from, device=torch.device("cpu"))
    clip_head = MovieCLIPHead(
        eeg_in_dim=cfg.model.encoder_embed_dim,
        vision_in_dim=train_set.frame_embedding_dim,
        proj_dim=int(cfg.loss.proj_dim),
        temperature=float(cfg.loss.temperature),
        drop_proj=float(cfg.loss.get("drop_proj", 0.5)),
        vision_passthrough=bool(cfg.loss.get("vision_passthrough", True)),
        n_residual_blocks=int(cfg.loss.get("n_residual_blocks", 1)),
    )
    if loss_mode == "scene_clip":
        model = SceneCLIPPretrain(
            encoder, clip_head, temporal_buffer_s=temporal_buffer_s
        ).to(device)
    else:
        model = CLIPPretrain(encoder, clip_head).to(device)

    # Optional: freeze the EEG encoder (train only clip_head). Useful for
    # warm-start ablations where you want to measure projector-only alignment.
    if bool(cfg.meta.get("freeze_encoder", False)):
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()  # disable dropout/etc in frozen encoder
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info("encoder frozen: %d / %d params trainable (%.1f%%)",
                    n_trainable, n_total, 100 * n_trainable / n_total)

    log_model_info(
        model,
        {
            "encoder": sum(p.numel() for p in model.encoder.parameters()),
            "clip_head": sum(p.numel() for p in model.clip_head.parameters()),
        },
    )

    model.train()
    # Optimizer: optim.optimizer in {adam, adamw}; AdamW uses decoupled weight
    # decay, applied only to weight matrices (skip biases + 1-D norm params).
    optim_name = str(cfg.optim.get("optimizer", "adam")).lower()
    weight_decay = float(cfg.optim.get("weight_decay", 0.0))
    if optim_name == "adamw":
        decay_p, no_decay_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay_p.append(p)
            else:
                decay_p.append(p)
        optimizer = AdamW([
            {"params": decay_p, "weight_decay": weight_decay},
            {"params": no_decay_p, "weight_decay": 0.0},
        ], lr=cfg.optim.lr)
        logger.info("AdamW: %d decay params, %d no-decay params, wd=%.3g",
                    len(decay_p), len(no_decay_p), weight_decay)
    elif optim_name == "adam":
        # Filter to requires_grad params so frozen encoders don't get optimizer state.
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(trainable, lr=cfg.optim.lr)
    else:
        raise ValueError(f"Unknown optim.optimizer={optim_name!r}; expected adam or adamw.")

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

    # Optional: resume model + optimizer state from a previous run. The new
    # LR schedule (cfg.optim.epochs, lr, lr_min, warmup_epochs) is applied
    # fresh — the resumed optimizer state is the *adaptive moments*, not the
    # learning rate. Useful for continuation runs from a converged checkpoint.
    start_step = 0
    if resume_from:
        state = load_checkpoint(resume_from, model=model, optimizer=optimizer,
                                device=torch.device("cpu"), strict=True)
        start_step = int(state.get("step", 0))
        logger.info("Resumed from %s (step=%d). New LR schedule applies fresh.",
                    resume_from, start_step)

    log_config(cfg)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Starting CLIP pretraining for %d epochs...", cfg.optim.epochs)
    channel_dropout_p = float(cfg.data.get("channel_dropout_p", 0.0))
    global_step = start_step
    for epoch in range(cfg.optim.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=cfg.logging.get("tqdm_silent", False),
        )

        for batch in pbar:
            if recipe_mode:
                eeg, _features, embeds, _shot_ids, scene_ids, t_starts, _probe_labels = batch
                eeg = eeg.to(device, non_blocking=True)
                embeds = embeds.to(device, non_blocking=True)
                scene_ids = scene_ids.to(device, non_blocking=True)
                t_starts = t_starts.to(device, non_blocking=True)
            else:
                eeg, _features, embeds, _shot_ids, _probe_labels = batch
                eeg = eeg.to(device, non_blocking=True)
                embeds = embeds.to(device, non_blocking=True)

            if channel_dropout_p > 0.0:
                # eeg: [B, T, C, W]. Zero entire (window, channel) tiles.
                mask = (
                    torch.rand(eeg.shape[0], eeg.shape[1], eeg.shape[2], 1,
                               device=eeg.device) > channel_dropout_p
                )
                eeg = eeg * mask

            optimizer.zero_grad()
            if recipe_mode:
                loss, loss_dict = model(eeg, embeds, scene_ids, t_starts)
            else:
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

        # Per-epoch val diagnostics (recipe-mode only).
        if val_loader is not None and (epoch % val_every == 0):
            val_metrics = evaluate_recipe(
                model, val_loader, device,
                max_windows=int(cfg.eval.val_max_windows),
            )
            if val_metrics:
                logger.info(
                    "val/ep%d: %s", epoch,
                    " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in val_metrics.items()),
                )
                if wandb_run:
                    import wandb
                    wandb.log(val_metrics, step=global_step)

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
