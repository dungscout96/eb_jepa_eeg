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
from eb_jepa.jepa import CrossSubjectJEPA, MaskedJEPA
from eb_jepa.mjepa import MJEPA
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
        combine_mode = sigreg_cfg.get("combine_mode", "convex")
        return SIGRegAntiCollapse(sigreg, combine_mode=combine_mode)

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


def build_encoder(cfg, *, n_chans: int, n_times: int, chs_info,
                  n_windows: int) -> EEGEncoderTokens:
    """Build the EEG token encoder shared by JEPA and CLIP pretraining paths."""
    return EEGEncoderTokens(
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=cfg.model.encoder_embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=n_windows,
        patch_size=cfg.model.get("patch_size", 200),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=chs_info,
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
        init_depth_scaled=cfg.model.get("init_depth_scaled", False),
    )


def _build_components(cfg, *, n_chans: int, n_times: int, chs_info,
                      n_windows: int):
    """Construct the four pieces every JEPA variant shares.

    Returns:
        (encoder, predictor, mask_collator, anti_collapse)
    """
    embed_dim = cfg.model.encoder_embed_dim
    masking_cfg = cfg.get("masking", {})

    encoder = build_encoder(
        cfg, n_chans=n_chans, n_times=n_times, chs_info=chs_info, n_windows=n_windows,
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
    return encoder, predictor, mask_collator, anti_collapse


def build_jepa(cfg, *, n_chans: int, n_times: int, chs_info,
               n_windows: int, vision_dim: int = 0) -> MaskedJEPA:
    """Build a JEPA model from a config.

    ``cfg.loss.objective`` selects the variant:

    - ``"masked"`` (default): ``MaskedJEPA`` — within-subject masked prediction.
    - ``"cross_subject"``: ``CrossSubjectJEPA`` — predict a different subject's
      tokens at the same movie time. Requires ``PairedSubjectJEPADataset``,
      which yields ``[B, 2, T, C, W]`` batches.
    - ``"mjepa"``: ``MJEPA`` — masked prediction plus cross-modal L1 regression
      to frozen V-JEPA-2 embeddings. Requires ``recipe_mode`` on the dataset and
      a non-zero ``vision_dim``; batches are ``([B,T,C,W], [B,T,vision_dim])``.

    All three share the same submodule layout, so a checkpoint from any of them
    loads into the encoder-only evaluators unchanged.

    Args:
        cfg: OmegaConf config with ``model``, ``loss``, ``masking`` sections.
        n_chans / n_times / chs_info: dataset-derived inputs.
        n_windows: number of windows per sample.
        vision_dim: frozen movie-embedding dim (``train_set.frame_embedding_dim``,
            1408 for V-JEPA-2). Required by ``mjepa``, ignored otherwise.

    Returns:
        A fully assembled model on CPU. The caller moves it to a device.
    """
    encoder, predictor, mask_collator, anti_collapse = _build_components(
        cfg, n_chans=n_chans, n_times=n_times, chs_info=chs_info,
        n_windows=n_windows,
    )
    pred_loss_type = cfg.loss.get("pred_loss_type", "mse")
    objective = str(cfg.loss.get("objective", "masked"))

    if objective == "masked":
        return MaskedJEPA(
            encoder, predictor, mask_collator, anti_collapse,
            pred_loss_type=pred_loss_type,
        )
    if objective == "cross_subject":
        cs_cfg = cfg.loss.get("cross_subject", {}) or {}
        return CrossSubjectJEPA(
            encoder, predictor, mask_collator, anti_collapse,
            pred_loss_type=pred_loss_type,
            symmetric=bool(cs_cfg.get("symmetric", True)),
            context_mask_mode=str(cs_cfg.get("context_mask_mode", "masked")),
            pred_target_mode=str(cs_cfg.get("pred_target_mode", "masked")),
            within_subject_weight=float(cs_cfg.get("within_subject_weight", 0.0)),
            diagnostic_every=int(cs_cfg.get("diagnostic_every", 50)),
        )
    if objective == "mjepa":
        mj_cfg = cfg.loss.get("mjepa", {}) or {}
        lambda_intra = float(mj_cfg.get("lambda_intra", 1.0))
        ac_name = str(cfg.loss.get("anti_collapse", "none")).lower()
        # With no masked term there is nothing for the existing anti-collapse
        # strategies to act on: all four consume `target_representations` from
        # the masked branch, which lambda=0 skips entirely. Requiring "none"
        # keeps lambda=0 a clean "pure cross-modal regression" arm rather than a
        # silent no-op.
        #
        # NB this is a *structural* constraint, not a claim that lambda=0 cannot
        # collapse. An earlier version of this comment argued there was "no
        # collapse attractor" because L_e2v regresses to a fixed external target
        # the model cannot influence. That reasoning rules out *total* collapse
        # (a constant encoder is underfitting, not an optimum) but says nothing
        # about *dimensional* collapse, and measurement contradicts it: the
        # lambda=0 from-scratch arm collapsed to participation ratio 1.02 with
        # ev_gap ~= 2e-4, probing at the noise floor (experiments/mjepa/
        # RESULTS.md). The lambda=0 REVE warm-start arm dips to pr=1.24 by epoch
        # 48 before recovering to 6.31. If a future arm needs regularizing at
        # lambda=0, the fix is to apply SIGReg to `pooled` directly -- not to
        # relax this check.
        if lambda_intra == 0.0 and ac_name != "none":
            raise ValueError(
                f"loss.mjepa.lambda_intra=0 requires loss.anti_collapse='none', "
                f"got {ac_name!r}. At lambda=0 the masked term is skipped entirely, "
                "so the anti-collapse loss would be applied to nothing it can act on."
            )
        return MJEPA(
            encoder, predictor, mask_collator, anti_collapse,
            pred_loss_type=pred_loss_type,
            vision_dim=vision_dim,
            lambda_intra=lambda_intra,
            cross_loss_type=str(mj_cfg.get("cross_loss_type", "l1")),
            predictor_hidden=int(mj_cfg.get("predictor_hidden", 2048)),
            predictor_depth=int(mj_cfg.get("predictor_depth", 3)),
            cross_source=str(mj_cfg.get("cross_source", "auto")),
            ve_enabled=bool(mj_cfg.get("ve_enabled", True)),
            ve_stopgrad=bool(mj_cfg.get("ve_stopgrad", True)),
            standardize_targets=bool(mj_cfg.get("standardize_targets", True)),
            standardizer_momentum=float(mj_cfg.get("standardizer_momentum", 0.01)),
            diagnostic_every=int(mj_cfg.get("diagnostic_every", 1)),
        )
    raise ValueError(
        f"Unknown loss.objective={objective!r}. "
        "Expected 'masked', 'cross_subject' or 'mjepa'."
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
