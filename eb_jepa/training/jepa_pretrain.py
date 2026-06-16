"""eb_jepa.training.jepa_pretrain -- canonical JEPA pretraining entry.

Self-supervised pretraining of an EEG encoder on HBN movie-watching data
using a masked Joint Embedding Predictive Architecture (V-JEPA style).
Labels are not consumed during pretraining; movie features are loaded
only at eval time.

Pipeline
--------
1. Build a `JEPAMovieDataset` over HBN recordings: each sample is
   `n_windows` consecutive EEG windows. Movie-feature labels are skipped
   (`feature_names=[]`). The dataset's visual_processing_delay_s
   defaults to 0.0, so no windows are dropped during pretraining.
2. Build the JEPA model via `eb_jepa.training.builder.build_jepa`:
   encoder + predictor + an `AntiCollapse` regularizer chosen by
   `cfg.loss.anti_collapse`:
     - `vicreg`  — variance + covariance penalty (default).
     - `dino`    — EMA target encoder; cosine momentum schedule from
                   `cfg.loss.dino.ema_momentum{,_end}`.
     - `sigreg`  — sketched Gaussianity test (LeJEPA).
     - `none`    — ablation; representations collapse.
3. Train with Adam + cosine LR (linear warmup). Each step:
   `pred_loss + ac_loss → backward → step → (DINO only) EMA update`.
4. Checkpoint `latest.pth.tar` every epoch and `epoch_N.pth.tar` every
   `cfg.logging.save_every` epochs. No val loop, no best-checkpoint
   tracking — probe metrics are not computed during pretraining.
5. Post-training auto-eval (gated by `cfg.eval.auto_run`, default true):
   `eb_jepa.evaluation.run_probe_eval` fits closed-form sklearn linear
   probes on `latest.pth.tar` (Ridge for stim/age regression,
   LogisticRegression for binary cls + 20-way movie_id), saving per-clip
   predictions to `saved_predictions/preds_seed{seed}.npz`. Then
   `bootstrap_predictions` resamples those predictions at the recording
   level and writes L1 + L2 metrics to `bootstrap_seed{seed}.json`.

Configuration
-------------
Default config: `config/jepa_pretrain.yaml`. Pass `--fname=...` to
override. Sections:
  meta     — seed, device
  data     — batch size, windows, task, preprocessing, normalization
  model    — encoder/predictor architecture, masking patch geometry
  masking  — short/long mask block scales and counts
  loss     — anti_collapse strategy + per-strategy subblock
             (dino / vicreg / sigreg) + pred_loss_type
  optim    — epochs, lr, lr_min, warmup_epochs
  logging  — wandb, log/save cadence
  eval     — auto_run, feature_names, visual_processing_delay_s,
             n_passes, probe_seed, n_bootstrap (post-training only)

Invoke
------
    PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.jepa_pretrain \\
        --optim.lr=5e-4 --data.n_windows=2 --data.window_size_seconds=4

Legacy
------
Promoted from `experiments/eeg_jepa/train.py`, preserved as a frozen
reproducibility snapshot — do not patch.
"""

import math
from pathlib import Path

import fire
import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.anti_collapse import DINOAntiCollapse
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.logging import get_logger
from eb_jepa.paths import resolve_preprocessed_dir
from eb_jepa.training.builder import build_jepa
from eb_jepa.training_utils import (
    get_default_dev_name,
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

logger = get_logger(__name__)

# Default config lives in the repo-root `config/` directory (resolved
# relative to this module so `python -m eb_jepa.training.jepa_pretrain`
# works from any cwd).
_DEFAULT_CFG_PATH = str(
    Path(__file__).resolve().parents[2] / "config" / "jepa_pretrain.yaml"
)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run(
    fname: str = _DEFAULT_CFG_PATH,
    cfg=None,
    folder=None,
    **overrides,
):
    """Train a masked EEG JEPA model on HBN movie-watching data.

    Args:
        fname: Path to YAML config file.
        cfg: Pre-loaded config object (optional, overrides config file).
        folder: Experiment folder path (optional, auto-generated if not provided).
        **overrides: Config overrides in dot notation (e.g., optim.lr=0.001).
    """
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)
    temporal_stride = cfg.data.get("temporal_stride", 1)

    # Create experiment directory
    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir = Path(cfg.meta.model_folder)
            exp_name = exp_dir.name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            stride_suffix = f"_stride{temporal_stride}" if temporal_stride > 1 else ""
            ac_type = cfg.loss.get("anti_collapse", "vicreg")
            if ac_type == "dino":
                ac_suffix = "_dino"
            elif ac_type == "sigreg":
                ac_suffix = f"_sigreg{cfg.loss.sigreg.get('coeff', 0.1)}"
            elif ac_type == "vicreg":
                vicreg_cfg = cfg.loss.get("vicreg", {})
                _up = vicreg_cfg.get("use_projector", True)
                use_proj = _up if isinstance(_up, bool) else str(_up).lower() not in ("false", "0", "no")
                proj_suffix = "" if use_proj else "_noproj"
                ac_suffix = (
                    f"_vicreg_std{vicreg_cfg.get('std_coeff', 1.0)}"
                    f"_cov{vicreg_cfg.get('cov_coeff', 1.0)}{proj_suffix}"
                )
            else:
                ac_suffix = "_noac"
            nw_suffix = f"_nw{cfg.data.n_windows}_ws{cfg.data.window_size_seconds}s"
            exp_name = (
                f"eeg_jepa_bs{cfg.data.batch_size}"
                f"_lr{cfg.optim.lr}"
                f"{ac_suffix}"
                f"{nw_suffix}"
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
        exp_name = exp_dir.name.rsplit("_seed", 1)[0]

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
    # Datasets
    # ------------------------------------------------------------------
    logger.info("Loading HBN Movie datasets...")
    preprocessed = cfg.data.get("preprocessed", False)
    preprocessed_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    task_cfg = cfg.data.get("task", "ThePresent")
    task = list(task_cfg) if not isinstance(task_cfg, str) else task_cfg

    # Pretraining is self-supervised: pass feature_names=[] to skip per-window
    # movie-feature tensor construction. visual_processing_delay_s defaults to
    # 0.0 — irrelevant since labels are unused; relevant only at eval time.
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
    logger.info("EEG channels: %d", n_chans)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Initializing model...")

    jepa = build_jepa(
        cfg,
        n_chans=n_chans,
        n_times=n_times,
        chs_info=chs_info,
        n_windows=cfg.data.n_windows,
    ).to(device)
    is_dino = isinstance(jepa.anti_collapse, DINOAntiCollapse)

    log_model_info(
        jepa,
        {
            "encoder": sum(p.numel() for p in jepa.encoder.parameters()),
            "predictor": sum(p.numel() for p in jepa.predictor.parameters()),
        },
    )

    jepa.train()

    # Encoder + predictor + any trainable params owned by the anti-collapse
    # strategy (e.g. VICReg's projector). DINO's target encoder has
    # requires_grad=False so it contributes nothing here.
    jepa_params = (
        list(jepa.encoder.parameters())
        + list(jepa.predictor.parameters())
        + [p for p in jepa.anti_collapse.parameters() if p.requires_grad]
    )
    optimizer = Adam(jepa_params, lr=cfg.optim.lr)

    # Cosine LR schedule with linear warmup (disabled if lr_min == 0)
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
    dino_cfg = cfg.loss.get("dino", {}) or {}
    ema_start = dino_cfg.get("ema_momentum", 0.996)
    ema_end = dino_cfg.get("ema_momentum_end", 1.0)
    total_steps = cfg.optim.epochs * len(train_loader)

    for epoch in range(start_epoch, cfg.optim.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=cfg.logging.get("tqdm_silent", False),
        )

        for eeg, _features, _embeds, _shot_ids, _probe_labels in pbar:
            eeg = eeg.to(device)

            # --- Masked JEPA pretraining ---
            optimizer.zero_grad()
            jepa_loss, loss_dict = jepa(eeg, global_step=global_step)
            jepa_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(jepa.encoder.parameters())
                + list(jepa.predictor.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            # EMA update of target encoder (cosine momentum schedule).
            # No-op for non-DINO strategies; gated to skip the schedule
            # computation entirely when unused.
            if is_dino:
                momentum = ema_end - (ema_end - ema_start) * (
                    math.cos(math.pi * global_step / total_steps) + 1
                ) / 2
                jepa.update_target_encoder(momentum)

            acl = loss_dict.get("ac_loss", 0.0)
            pl = loss_dict.get("pred_loss", 0.0)
            ac_subdict = {
                k: v for k, v in loss_dict.items()
                if k not in ("total_loss", "ac_loss", "pred_loss")
            }

            pbar.set_postfix({
                "loss": f"{jepa_loss.item():.4f}",
                "ac":   f"{float(acl):.4f}",
                "pred": f"{float(pl):.4f}",
            })

            if wandb_run:
                import wandb
                step_metrics = {
                    "train_step/jepa_loss": jepa_loss.item(),
                    "train_step/ac_loss":   float(acl),
                    "train_step/pred_loss": float(pl),
                }
                for k, v in ac_subdict.items():
                    step_metrics[f"train_step/{k}"] = float(v)
                wandb.log(step_metrics, step=global_step)

            global_step += 1

        # Epoch logging
        if epoch % cfg.logging.log_every == 0:
            train_metrics = {
                "epoch":           epoch,
                "train/loss":      jepa_loss.item(),
                "train/ac_loss":   float(acl),
                "train/pred_loss": float(pl),
                "train/lr":        scheduler.get_last_lr()[0],
            }
            for k, v in ac_subdict.items():
                train_metrics[f"train/{k}"] = float(v)

            if wandb_run:
                import wandb
                wandb.log(train_metrics, step=global_step)

            log_epoch(
                epoch,
                {
                    "loss": jepa_loss.item(),
                    "ac":   float(acl),
                    "pred": float(pl),
                },
                total_epochs=cfg.optim.epochs,
            )

        # Checkpointing
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

        scheduler.step()

    wandb_run_id = wandb_run.id if wandb_run else ""
    if wandb_run:
        import wandb
        wandb.finish()

    logger.info("Training complete!")

    # ------------------------------------------------------------------
    # Auto-eval: probe_eval -> bootstrap_predictions on the saved checkpoint.
    # Gated by cfg.eval.auto_run (default true). Sweeps that submit eval as
    # a separate SLURM job should set cfg.eval.auto_run=false.
    # ------------------------------------------------------------------
    eval_cfg = cfg.get("eval", None)
    if eval_cfg is None or eval_cfg.get("auto_run", True):
        _run_auto_eval(cfg, exp_dir, fname, wandb_run_id=wandb_run_id)


def _run_auto_eval(cfg, exp_dir, cfg_fname, *, wandb_run_id: str = ""):
    """Run probe_eval + bootstrap on the just-saved checkpoint.

    Errors are logged but do not fail the training run -- the checkpoint
    is already saved and a sweep can re-run eval manually.
    """
    from eb_jepa.evaluation import bootstrap_predictions, run_probe_eval

    eval_cfg = cfg.get("eval", {}) or {}
    seed = cfg.meta.seed
    checkpoint_path = exp_dir / "latest.pth.tar"
    save_predictions_dir = exp_dir / "saved_predictions"

    n_passes = eval_cfg.get("n_passes", 20)
    probe_seed = eval_cfg.get("probe_seed", 42)

    logger.info("Auto-eval: running probe_eval on %s", checkpoint_path)
    try:
        run_probe_eval(
            checkpoint=str(checkpoint_path),
            n_windows=cfg.data.n_windows,
            window_size_seconds=cfg.data.window_size_seconds,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            norm_mode=cfg.data.norm_mode,
            add_envelope=cfg.data.add_envelope,
            corrca_filters=cfg.data.corrca_filters or "",
            n_passes=n_passes,
            probe_seed=probe_seed,
            save_predictions_dir=str(save_predictions_dir),
            wandb_run_id=wandb_run_id,
            fname=cfg_fname,
            seed=seed,
        )
    except Exception as e:
        logger.warning("Auto-eval probe_eval failed: %s", e)
        return

    predictions_npz = save_predictions_dir / f"preds_seed{seed}.npz"
    bootstrap_out_json = exp_dir / f"bootstrap_seed{seed}.json"
    n_bootstrap = eval_cfg.get("n_bootstrap", 2000)
    logger.info("Auto-eval: running bootstrap on %s", predictions_npz)
    try:
        bootstrap_predictions(
            predictions_npz=str(predictions_npz),
            out_json=str(bootstrap_out_json),
            n_bootstrap=n_bootstrap,
            seed=seed,
            wandb_run_id=wandb_run_id,
        )
    except Exception as e:
        logger.warning("Auto-eval bootstrap failed: %s", e)


if __name__ == "__main__":
    fire.Fire(run)
