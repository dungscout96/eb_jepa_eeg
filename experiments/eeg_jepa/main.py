"""
EEG JEPA Training Script

Train a self-supervised EEG prediction model on HBN movie-watching data using
the masked Joint Embedding Predictive Architecture (V-JEPA style) with VC
regularization.

Two evaluation decoder probes are trained alongside JEPA:
  1. Regression probe: predicts continuous movie features (MSELoss)
  2. Classification probe: predicts binary movie feature labels (BCEWithLogitsLoss)
"""

import copy
import math
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
    EEGEncoderTokens,
    MaskedPredictor,
    MovieFeatureHead,
    Projector,
    TemporalMovieFeatureHead,
)
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.jepa import MaskedJEPA, MaskedJEPANoEMA, MaskedJEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import VCLoss, SIGRegLoss
from eb_jepa.masking import MultiBlockMaskCollator
from eb_jepa.sanity_checks import SanityCheckHook
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
from experiments.eeg_jepa.eval import validation_loop

logger = get_logger(__name__)

NUMERIC_FEATURES = None  # resolved from cfg.data.feature_names at runtime

# Known preprocessed data locations (checked in order for auto-detection)
_PREPROCESSED_DIRS = [
    Path("/mnt/v1/dtyoung/data/eb_jepa_eeg/hbn_preprocessed"),
    Path("/expanse/projects/nemar/dtyoung/.cache/eb_jepa_eeg/hbn_preprocessed"),
    Path("/projects/bbnv/kkokate/hbn_preprocessed"),  # Delta
]


def resolve_preprocessed_dir(configured: str | None) -> Path | None:
    """Return preprocessed_dir: use explicit config if set, else auto-detect."""
    if configured:
        return Path(configured)
    for p in _PREPROCESSED_DIRS:
        if p.exists():
            logger.info("Auto-detected preprocessed_dir: %s", p)
            return p
    return None


# ---------------------------------------------------------------------------
# Loss functions for movie-feature probes
# ---------------------------------------------------------------------------


class RegressionLoss(nn.Module):
    """MSE loss with target z-normalization using training set statistics."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)  # [n_features]
        self.register_buffer("std", std)    # [n_features]

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
            reg_type = cfg.loss.get("regularizer", "vc")
            _up = cfg.loss.get("use_projector", True)
            use_proj = _up if isinstance(_up, bool) else str(_up).lower() not in ("false", "0", "no")
            if reg_type == "sigreg":
                reg_suffix = f"_sigreg{cfg.loss.sigreg.get('coeff', 0.1)}"
            elif cfg.loss.std_coeff > 0 or cfg.loss.cov_coeff > 0:
                proj_suffix = "" if use_proj else "_noproj"
                reg_suffix = f"_std{cfg.loss.std_coeff}_cov{cfg.loss.cov_coeff}{proj_suffix}"
            else:
                reg_suffix = "_noreg"
            nw_suffix = f"_nw{cfg.data.n_windows}_ws{cfg.data.window_size_seconds}s"
            exp_name = (
                f"eeg_jepa_bs{cfg.data.batch_size}"
                f"_lr{cfg.optim.lr}"
                f"{reg_suffix}"
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
    feature_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))
    global NUMERIC_FEATURES
    NUMERIC_FEATURES = feature_names

    # Lever 1: when the stim-aligned InfoNCE auxiliary is active, the dataloader
    # must emit (rec_idx, movie_id, position_bucket). Auto-enable so callers
    # only need to set --loss.stim_nce_coeff=...
    stim_nce_active = float(cfg.loss.get("stim_nce_coeff", 0.0)) > 0
    # Cell B: cross-subject masked latent prediction also requires paired meta
    # AND the K=2 structured sampler. Auto-enable both together so callers only
    # need to set --loss.xsub_coeff=...
    xsub_active = float(cfg.loss.get("xsub_coeff", 0.0)) > 0
    if (stim_nce_active or xsub_active) and not cfg.data.get("return_stim_meta", False):
        OmegaConf.update(cfg, "data.return_stim_meta", True, force_add=True)
        logger.info("Auto-enabled data.return_stim_meta=True (stim_nce or xsub active)")
    if xsub_active:
        if not cfg.data.get("stim_aligned_sampler", False):
            OmegaConf.update(cfg, "data.stim_aligned_sampler", True, force_add=True)
            logger.info("Auto-enabled data.stim_aligned_sampler=True (loss.xsub_coeff > 0)")
        # Cell B requires K=2 (one cross-subject pair per bucket).
        OmegaConf.update(cfg, "data.stim_aligned_K", 2, force_add=True)
        logger.info("Forced data.stim_aligned_K=2 (Cell B paired-batch requirement)")
    # Lever 1 v2: structured cross-subject batches.
    use_aligned_sampler = bool(cfg.data.get("stim_aligned_sampler", False))
    if use_aligned_sampler and not (stim_nce_active or xsub_active):
        logger.warning("data.stim_aligned_sampler set but no stim_nce/xsub coeff; sampler will run anyway")
    if use_aligned_sampler and not cfg.data.get("stim_aligned_flat_index", False):
        OmegaConf.update(cfg, "data.stim_aligned_flat_index", True, force_add=True)
        logger.info("Auto-enabled data.stim_aligned_flat_index=True for sampler")

    # Multi-movie support: cfg.data.tasks is a list (or comma-separated string).
    # Defaults to the dataset's default single movie when unspecified.
    _tasks = cfg.data.get("tasks", None)
    if _tasks is None:
        _tasks_arg = JEPAMovieDataset.__init__.__defaults__[2]  # DEFAULT_TASK fallback
        # Use module default by simply not passing task=
        train_kwargs = {}
    else:
        if isinstance(_tasks, str):
            _tasks = [t.strip() for t in _tasks.split(",") if t.strip()]
        train_kwargs = {"task": list(_tasks)}
        logger.info("Multi-movie pretraining tasks: %s", _tasks)
    train_set = JEPAMovieDataset(
        split="train",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        temporal_stride=temporal_stride,
        feature_names=feature_names,
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
        **train_kwargs,
    )
    # Val set should NOT use the flat-clip index — that turns the validation
    # loop into 449 batches/epoch and adds ~4 min of validation per epoch.
    # Make a shallow copy of cfg.data with flat-index disabled for val_set
    # construction. The train-side sampler keeps using the train_set's flat
    # index (already baked into train_set._stim_flat_index).
    _val_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    OmegaConf.update(_val_cfg, "stim_aligned_flat_index", False, force_add=True)
    # Val set always uses the first task only (typically "ThePresent"), since
    # secondary tasks like DespicableMe have annotation issues in the val split
    # that crash braindecode's create_windows_from_events.
    val_kwargs = {}
    if "task" in train_kwargs:
        val_kwargs["task"] = train_kwargs["task"][0]
    val_set = JEPAMovieDataset(
        split="val",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        temporal_stride=temporal_stride,
        feature_names=feature_names,
        eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=_val_cfg,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
        **val_kwargs,
    )

    feature_stats = train_set.compute_feature_stats()
    feature_median = train_set.compute_feature_median()

    num_workers = cfg.data.num_workers
    if use_aligned_sampler:
        from eb_jepa.datasets.hbn import StimAlignedBatchSampler
        K = int(cfg.data.get("stim_aligned_K", 4))
        steps_per_epoch_cfg = cfg.data.get("stim_aligned_steps_per_epoch", None)
        steps_per_epoch = (
            int(steps_per_epoch_cfg) if steps_per_epoch_cfg is not None else None
        )
        train_sampler = StimAlignedBatchSampler(
            train_set, batch_size=cfg.data.batch_size, K=K,
            steps_per_epoch=steps_per_epoch,
            seed=cfg.meta.get("seed", 0),
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    log_data_info(
        "HBN EEG Movie",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_set),
        val_samples=len(val_set),
    )

    n_chans = train_set.n_chans
    n_features = len(NUMERIC_FEATURES)
    chs_info = train_set.get_chs_info()
    n_times = train_set.n_times
    logger.info("EEG channels: %d, Movie features: %d", n_chans, n_features)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Initializing model...")
    embed_dim = cfg.model.encoder_embed_dim
    masking_cfg = cfg.get("masking", {})

    encoder = EEGEncoderTokens(
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=cfg.data.n_windows,
        patch_size=cfg.model.get("patch_size", 200),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=chs_info,
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
    )
    use_ema = cfg.model.get("use_ema", True)
    if use_ema:
        target_encoder = copy.deepcopy(encoder)
    predictor_dim = cfg.model.get("predictor_embed_dim", None)
    predictor = MaskedPredictor(
        embed_dim=embed_dim,
        depth=cfg.model.get("predictor_depth", 2),
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
        predictor_dim=predictor_dim,
    )
    mask_collator = MultiBlockMaskCollator(
        n_channels=n_chans,
        n_windows=cfg.data.n_windows,
        n_patches_per_window=encoder.n_patches_per_window,
        n_pred_masks_short=masking_cfg.get("n_pred_masks_short", 2),
        n_pred_masks_long=masking_cfg.get("n_pred_masks_long", 2),
        short_channel_scale=tuple(masking_cfg.get("short_channel_scale", [0.08, 0.15])),
        short_patch_scale=tuple(masking_cfg.get("short_patch_scale", [0.3, 0.6])),
        long_channel_scale=tuple(masking_cfg.get("long_channel_scale", [0.15, 0.35])),
        long_patch_scale=tuple(masking_cfg.get("long_patch_scale", [0.5, 1.0])),
        min_context_fraction=masking_cfg.get("min_context_fraction", 0.15),
    )
    # Regularizer — VCLoss, SIGReg, or none
    regularizer = None
    reg_type = cfg.loss.get("regularizer", "vc")
    _use_proj_raw = cfg.loss.get("use_projector", True)
    use_proj = _use_proj_raw if isinstance(_use_proj_raw, bool) else str(_use_proj_raw).lower() not in ("false", "0", "no")
    if reg_type == "sigreg":
        sigreg_cfg = cfg.loss.get("sigreg", {})
        regularizer = SIGRegLoss(
            num_slices=sigreg_cfg.get("num_slices", 256),
            coeff=sigreg_cfg.get("coeff", 0.1),
        )
    elif cfg.loss.std_coeff > 0 or cfg.loss.cov_coeff > 0:
        proj_spec = cfg.loss.get("proj_spec", None)
        if use_proj:
            spec = proj_spec if proj_spec else f"{embed_dim}-{embed_dim * 4}-{embed_dim * 4}"
            projector = Projector(spec)
        else:
            projector = None
        detach_enc = cfg.loss.get("detach_encoder", False)
        regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff,
                             proj=projector, detach_encoder=detach_enc)

    # Prediction loss type: "mse" (default) or "smooth_l1" (Huber, used in V-JEPA)
    pred_loss_type = cfg.loss.get("pred_loss_type", "mse")

    # Direction C: envelope-auxiliary regression on per-window pooled features.
    env_coeff = float(cfg.loss.get("env_coeff", 0.0))
    env_aux_head = None
    if env_coeff > 0:
        from eb_jepa.architectures import EnvelopeAuxHead
        env_aux_hidden = int(cfg.model.get("env_aux_hidden", 64))
        env_aux_head = EnvelopeAuxHead(
            embed_dim=embed_dim, n_features=n_features, hidden=env_aux_hidden,
        )

    # Optional iBOT-lite: global mean-pooled DINO objective on context vs target tokens.
    dino_coeff = float(cfg.loss.get("dino_coeff", 0.0))
    dino_head = dino_target_head = dino_loss_fn = None
    if dino_coeff > 0 and use_ema:
        from eb_jepa.dino_head import DINOHead, DINOLoss
        dino_K = int(cfg.model.get("dino_K", 4096))
        dino_hidden = int(cfg.model.get("dino_hidden", 2048))
        dino_bottleneck = int(cfg.model.get("dino_bottleneck", 256))
        dino_head = DINOHead(in_dim=embed_dim, hidden_dim=dino_hidden,
                             bottleneck_dim=dino_bottleneck, K=dino_K)
        dino_target_head = DINOHead(in_dim=embed_dim, hidden_dim=dino_hidden,
                                    bottleneck_dim=dino_bottleneck, K=dino_K)
        dino_target_head.load_state_dict(dino_head.state_dict())
        dino_loss_fn = DINOLoss(K=dino_K,
                                t_s=float(cfg.loss.get("dino_t_s", 0.1)),
                                t_t=float(cfg.loss.get("dino_t_t", 0.04)),
                                m_c=float(cfg.loss.get("dino_m_c", 0.9)))

    # Lever 1: cross-subject stim-aligned InfoNCE auxiliary loss.
    stim_nce_coeff = float(cfg.loss.get("stim_nce_coeff", 0.0))
    stim_nce_loss = None
    if stim_nce_coeff > 0:
        from eb_jepa.losses import CrossSubjectStimNCELoss
        stim_nce_loss = CrossSubjectStimNCELoss(
            tau=float(cfg.loss.get("stim_nce_tau", 0.1))
        )

    xsub_coeff = float(cfg.loss.get("xsub_coeff", 0.0))
    xsub_symmetric = bool(cfg.loss.get("xsub_symmetric", True))
    if use_ema:
        jepa = MaskedJEPA(
            encoder, target_encoder, predictor, mask_collator, regularizer,
            pred_loss_type=pred_loss_type,
            dino_head=dino_head, dino_target_head=dino_target_head,
            dino_loss_fn=dino_loss_fn, dino_coeff=dino_coeff,
            env_aux_head=env_aux_head, env_coeff=env_coeff,
            stim_nce_loss=stim_nce_loss, stim_nce_coeff=stim_nce_coeff,
            xsub_coeff=xsub_coeff, xsub_symmetric=xsub_symmetric,
        ).to(device)
    else:
        jepa = MaskedJEPANoEMA(
            encoder, predictor, mask_collator, regularizer,
            pred_loss_type=pred_loss_type,
        ).to(device)

    # ------------------------------------------------------------------
    # Online evaluation probes (trained on frozen encoder representations)
    # ------------------------------------------------------------------
    HeadClass = TemporalMovieFeatureHead if cfg.model.get("temporal_probe", False) else MovieFeatureHead
    # Exp 13: co-trained CorrCA-residual probe — concat raw_corrca box-pool feats
    # with encoder output along D before the probe head.
    probe_corrca_residual = bool(cfg.model.get("probe_corrca_residual", False))
    probe_corrca_dim = int(cfg.model.get("probe_corrca_residual_dim", 100))
    if probe_corrca_residual:
        residual_d = n_chans * probe_corrca_dim  # n_chans is post-CorrCA (5 typically)
        probe_in_dim = embed_dim + residual_d
        logger.info("Co-trained probe with CorrCA residual: enc=%d + raw=%d → probe_in=%d",
                    embed_dim, residual_d, probe_in_dim)
    else:
        probe_in_dim = embed_dim

    reg_head = HeadClass(probe_in_dim, cfg.model.hdec, n_features)
    reg_loss_fn = RegressionLoss(
        feature_stats["mean"].to(device),
        feature_stats["std"].to(device),
    )
    regression_probe = MaskedJEPAProbe(
        jepa, reg_head, reg_loss_fn,
        corrca_residual=probe_corrca_residual,
        corrca_residual_dim=probe_corrca_dim,
    ).to(device)

    cls_head = HeadClass(probe_in_dim, cfg.model.hdec, n_features)
    cls_loss_fn = ClassificationLoss(feature_median.to(device))
    classification_probe = MaskedJEPAProbe(
        jepa, cls_head, cls_loss_fn,
        corrca_residual=probe_corrca_residual,
        corrca_residual_dim=probe_corrca_dim,
    ).to(device)

    log_model_info(
        jepa,
        {
            "encoder": sum(p.numel() for p in encoder.parameters()),
            "predictor": sum(p.numel() for p in predictor.parameters()),
            "reg_head": sum(p.numel() for p in reg_head.parameters()),
            "cls_head": sum(p.numel() for p in cls_head.parameters()),
        },
    )

    jepa.train()
    regression_probe.train()
    classification_probe.train()

    # Context encoder + predictor + regularizer projector (if any);
    # target encoder is updated via EMA
    jepa_params = list(jepa.context_encoder.parameters()) + list(jepa.predictor.parameters())
    if jepa.regularizer is not None:
        jepa_params += [p for p in jepa.regularizer.parameters() if p.requires_grad]
    if getattr(jepa, "dino_head", None) is not None:
        jepa_params += [p for p in jepa.dino_head.parameters() if p.requires_grad]
    if getattr(jepa, "env_aux_head", None) is not None:
        jepa_params += [p for p in jepa.env_aux_head.parameters() if p.requires_grad]
    optimizer = Adam(jepa_params, lr=cfg.optim.lr)
    probe_optimizer = Adam(
        list(regression_probe.head.parameters())
        + list(classification_probe.head.parameters()),
        lr=cfg.optim.lr,
    )

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
    probe_scheduler = torch.optim.lr_scheduler.LambdaLR(probe_optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Sanity check hook
    # ------------------------------------------------------------------
    sanity_cfg = cfg.get("sanity_checks", {})
    probe_label_name = getattr(train_set, "probe_label_name", "none")
    logger.info("Sanity-check linear probe label: '%s'", probe_label_name)
    if wandb_run:
        import wandb
        wandb.config.update(
            {"sanity_checks/probe_label": probe_label_name}, allow_val_change=True
        )
    sanity_hook = SanityCheckHook(
        embed_dim=embed_dim,
        feature_median=feature_median,   # luminance fallback when probe_labels are NaN
        n_pred_masks_short=masking_cfg.get("n_pred_masks_short", 2),
        log_every_steps=sanity_cfg.get("log_every_steps", 50),
        probe_every_steps=sanity_cfg.get("probe_every_steps", 200),
        probe_train_steps=sanity_cfg.get("probe_train_steps", 30),
        probe_buffer_size=sanity_cfg.get("probe_buffer_size", 512),
        cosim_n_pairs=sanity_cfg.get("cosim_n_pairs", 128),
        horizon_every_steps=sanity_cfg.get("horizon_every_steps", 200),
        enabled=sanity_cfg.get("enabled", True),
    )

    log_config(cfg)

    # Early stopping and best checkpoint tracking
    patience = cfg.optim.get("early_stopping_patience", 0)  # 0 = disabled
    best_val_reg = float("inf")
    best_mean_corr = float("-inf")
    epochs_without_improvement = 0

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
    ema_start = cfg.model.get("ema_momentum", 0.996)
    ema_end = cfg.model.get("ema_momentum_end", 1.0)
    total_steps = cfg.optim.epochs * len(train_loader)

    for epoch in range(start_epoch, cfg.optim.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=cfg.logging.get("tqdm_silent", False),
        )

        for batch in pbar:
            # Backwards-compatible unpacking: dataset returns 3-tuple by default,
            # 4-tuple (with stim_meta) when return_stim_meta=True (Lever 1).
            if len(batch) == 4:
                eeg, features, probe_labels, stim_meta = batch
                stim_meta = stim_meta.to(device)
            else:
                eeg, features, probe_labels = batch
                stim_meta = None
            eeg = eeg.to(device)
            features = features.to(device)  # [B, T, n_features]
            # probe_labels: [B] float (NaN where no subject metadata) — stays on CPU

            # --- Masked JEPA pretraining ---
            optimizer.zero_grad()
            # Pass per-window features as env_targets when envelope-aux is active;
            # MaskedJEPA.forward only consumes them when env_coeff > 0.
            # Pass stim_meta when Lever-1 InfoNCE auxiliary is active.
            jepa_loss, loss_dict = jepa(eeg, env_targets=features, stim_meta=stim_meta)
            jepa_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(jepa.context_encoder.parameters())
                + list(jepa.predictor.parameters()),
                max_norm=1.0,
            )
            sanity_metrics = sanity_hook.step(
                global_step, eeg, features, jepa, loss_dict,
                probe_labels=probe_labels,
            )
            optimizer.step()

            # EMA update of target encoder (cosine momentum schedule)
            momentum = ema_end - (ema_end - ema_start) * (
                math.cos(math.pi * global_step / total_steps) + 1
            ) / 2
            jepa.update_target_encoder(momentum)

            regl = loss_dict.get("reg_loss", 0.0)
            pl = loss_dict.get("pred_loss", 0.0)
            regldict = {
                k: v for k, v in loss_dict.items()
                if k not in ("total_loss", "reg_loss", "pred_loss")
            }

            # --- Online probe training (frozen encoder) ---
            probe_optimizer.zero_grad()
            reg_loss = regression_probe(eeg, features)
            cls_loss = classification_probe(eeg, features)
            (reg_loss + cls_loss).backward()
            probe_optimizer.step()

            pbar.set_postfix({
                "loss": f"{jepa_loss.item():.4f}",
                "vc":   f"{float(regl):.4f}",
                "pred": f"{float(pl):.4f}",
                "reg":  f"{reg_loss.item():.4f}",
                "cls":  f"{cls_loss.item():.4f}",
            })

            if wandb_run:
                import wandb
                step_metrics = {
                    "train_step/jepa_loss": jepa_loss.item(),
                    "train_step/vc_loss":   float(regl),
                    "train_step/pred_loss": float(pl),
                    "train_step/reg_loss":  reg_loss.item(),
                    "train_step/cls_loss":  cls_loss.item(),
                    **sanity_metrics,
                }
                for k, v in regldict.items():
                    step_metrics[f"train_step/{k}"] = float(v)
                wandb.log(step_metrics, step=global_step)

            global_step += 1

        # Validation
        if epoch % cfg.logging.log_every == 0:
            val_logs = validation_loop(
                val_loader,
                jepa,
                regression_probe,
                classification_probe,
                device,
                feature_stats,
                feature_median,
                NUMERIC_FEATURES,
            )

            train_metrics = {
                "epoch":            epoch,
                "train/loss":       jepa_loss.item(),
                "train/vc_loss":    float(regl),
                "train/pred_loss":  float(pl),
                "train/reg_loss":   reg_loss.item(),
                "train/cls_loss":   cls_loss.item(),
            }
            for k, v in regldict.items():
                train_metrics[f"train/{k}"] = float(v)

            train_metrics["train/lr"] = scheduler.get_last_lr()[0]

            if wandb_run:
                import wandb
                wandb.log({**train_metrics, **val_logs}, step=global_step)

            log_epoch(
                epoch,
                {
                    "loss":    jepa_loss.item(),
                    "vc":      float(regl),
                    "pred":    float(pl),
                    "val_reg": val_logs.get("val/reg_loss", 0),
                    "val_cls": val_logs.get("val/cls_loss", 0),
                },
                total_epochs=cfg.optim.epochs,
            )

            # Track best val/reg_loss and save best checkpoint.
            # Only write best.pth.tar when early stopping is enabled (patience>0),
            # since val/reg_loss is the noisy auxiliary online probe loss
            # (issue #8 diagnosis). With patience=0, downstream should consume
            # latest.pth.tar instead.
            current_val_reg = val_logs.get("val/reg_loss", float("inf"))
            if current_val_reg < best_val_reg:
                best_val_reg = current_val_reg
                epochs_without_improvement = 0
                if patience > 0:
                    save_checkpoint(
                        exp_dir / "best.pth.tar",
                        model=jepa,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=global_step,
                    )
                    logger.info("New best val/reg_loss=%.4f at epoch %d", best_val_reg, epoch)
            else:
                epochs_without_improvement += 1

            # Settled policy (memo §2): always track mean of val/reg_*_corr
            # over the configured stim probes and save best_by_online_probe.pth.tar.
            # Decouples downstream eval from the "probes peak then decline"
            # phenomenon. Independent of early-stopping (patience).
            corr_vals = [
                v for k, v in val_logs.items()
                if k.startswith("val/reg_") and k.endswith("_corr")
            ]
            if corr_vals:
                current_mean_corr = float(sum(corr_vals) / len(corr_vals))
                if current_mean_corr > best_mean_corr:
                    best_mean_corr = current_mean_corr
                    save_checkpoint(
                        exp_dir / "best_by_online_probe.pth.tar",
                        model=jepa,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=global_step,
                    )
                    logger.info(
                        "New best mean(val/reg_*_corr)=%.4f at epoch %d (%d corrs)",
                        best_mean_corr, epoch, len(corr_vals),
                    )

            # Early stopping
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs, "
                    "best val/reg_loss=%.4f)",
                    epoch, patience, best_val_reg,
                )
                break

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
        probe_scheduler.step()

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating on test set...")
    test_set = JEPAMovieDataset(
        split="test",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        temporal_stride=temporal_stride,
        feature_names=feature_names,
        eeg_norm_stats=train_set.get_eeg_norm_stats(),
        cfg=cfg.data,
        preprocessed=preprocessed,
        preprocessed_dir=preprocessed_dir,
        **val_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_logs = validation_loop(
        test_loader,
        jepa,
        regression_probe,
        classification_probe,
        device,
        feature_stats,
        feature_median,
        NUMERIC_FEATURES,
    )
    test_metrics = {k.replace("val/", "test/"): v for k, v in test_logs.items()}

    if wandb_run:
        import wandb
        wandb.log(test_metrics, step=global_step)
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    fire.Fire(run)
