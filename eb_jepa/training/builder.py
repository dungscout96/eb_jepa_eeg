"""Shared builder for MaskedJEPA used by pretrain and probe-eval entry points.

The training script and the probe evaluator both need to reconstruct the
same model from a config. Centralizing it here keeps the anti-collapse
selection in one place and prevents the two call sites from drifting.
"""

from __future__ import annotations

import copy
from typing import Iterable

from eb_jepa.anti_collapse import (
    AntiCollapse,
    DINOAntiCollapse,
    SIGRegAntiCollapse,
    VICRegAntiCollapse,
)
from eb_jepa.architectures import EEGEncoderTokens, MaskedPredictor, Projector
from eb_jepa.jepa import MaskedJEPA
from eb_jepa.losses import SIGRegLoss, VCLoss
from eb_jepa.masking import MultiBlockMaskCollator


def build_anti_collapse(cfg, encoder) -> AntiCollapse:
    """Construct the AntiCollapse strategy named by ``cfg.loss.anti_collapse``.

    Recognized values: ``"dino"``, ``"vicreg"``, ``"sigreg"``, ``"none"``.
    For DINO, an EMA copy of ``encoder`` is created here so the caller keeps
    a single source of truth for encoder construction.
    """
    ac_type = cfg.loss.get("anti_collapse", "vicreg")
    embed_dim = cfg.model.encoder_embed_dim

    if ac_type == "dino":
        target_encoder = copy.deepcopy(encoder)
        return DINOAntiCollapse(target_encoder)

    if ac_type == "sigreg":
        sigreg_cfg = cfg.loss.get("sigreg", {})
        sigreg = SIGRegLoss(
            num_slices=sigreg_cfg.get("num_slices", 1024),
            coeff=sigreg_cfg.get("coeff", 0.05),
            ep_t_range=sigreg_cfg.get("ep_t_range", 5.0),
            ep_n_points=sigreg_cfg.get("ep_n_points", 17),
        )
        return SIGRegAntiCollapse(sigreg)

    if ac_type == "vicreg":
        vicreg_cfg = cfg.loss.get("vicreg", {})
        _use_proj_raw = vicreg_cfg.get("use_projector", True)
        use_proj = (
            _use_proj_raw if isinstance(_use_proj_raw, bool)
            else str(_use_proj_raw).lower() not in ("false", "0", "no")
        )
        projector = (
            Projector(f"{embed_dim}-{embed_dim * 4}-{embed_dim * 4}")
            if use_proj else None
        )
        vc = VCLoss(
            vicreg_cfg.get("std_coeff", 1.0),
            vicreg_cfg.get("cov_coeff", 1.0),
            proj=projector,
        )
        return VICRegAntiCollapse(vc)

    if ac_type == "none":
        return AntiCollapse()

    raise ValueError(
        f"Unknown loss.anti_collapse={ac_type!r}. "
        "Expected one of: dino, vicreg, sigreg, none."
    )


def build_jepa(cfg, *, n_chans: int, n_times: int, chs_info,
               n_windows: int) -> MaskedJEPA:
    """Build a MaskedJEPA from a config.

    Args:
        cfg: OmegaConf config with ``model``, ``loss``, ``masking`` sections.
        n_chans / n_times / chs_info: dataset-derived inputs.
        n_windows: number of windows per sample.

    Returns:
        A fully assembled ``MaskedJEPA`` on CPU. The caller moves it to a device.
    """
    embed_dim = cfg.model.encoder_embed_dim
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
    anti_collapse = build_anti_collapse(cfg, encoder)
    pred_loss_type = cfg.loss.get("pred_loss_type", "mse")

    return MaskedJEPA(
        encoder, predictor, mask_collator, anti_collapse,
        pred_loss_type=pred_loss_type,
    )


def check_old_checkpoint_format(state_dict: Iterable[str]) -> None:
    """Raise a clear error if the checkpoint was saved under the pre-refactor
    module layout (``context_encoder.*`` / top-level ``target_encoder.*`` /
    top-level ``regularizer.*``)."""
    keys = list(state_dict)
    old_markers = [
        ("context_encoder.", "context_encoder.* (renamed to encoder.*)"),
        ("target_encoder.", "top-level target_encoder.* (now anti_collapse.target_encoder.*)"),
        ("regularizer.", "top-level regularizer.* (now anti_collapse.vc_loss.* or anti_collapse.sigreg_loss.*)"),
    ]
    for prefix, description in old_markers:
        if any(k.startswith(prefix) for k in keys):
            raise RuntimeError(
                f"Checkpoint uses the pre-refactor key layout ({description}). "
                "The MaskedJEPA module tree was reorganized; old checkpoints "
                "cannot be loaded silently as it would leave new submodules "
                "freshly initialized. Either retrain, or pin the previous "
                "commit (before this refactor) for evaluation."
            )
