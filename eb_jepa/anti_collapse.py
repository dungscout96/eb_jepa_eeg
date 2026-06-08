"""Anti-collapse strategies for JEPA-style training.

Each strategy answers two questions:

1. How is the prediction target computed? (DINO uses an EMA copy of the
   encoder; VICReg uses the online encoder with a stop-gradient; SIGReg
   uses the online encoder with gradients flowing.)
2. Is there an auxiliary loss term, and how does it combine with the
   prediction loss? (DINO has none; VICReg adds variance/covariance
   terms; SIGReg adds an Epps-Pulley Gaussianity statistic combined
   convexly with the prediction loss.)

The three concrete strategies wrap the existing loss primitives in
``eb_jepa.losses`` rather than re-implementing them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eb_jepa.losses import SIGRegLoss, VCLoss


class AntiCollapse(nn.Module):
    """Base anti-collapse strategy. Default: no-op (representations will collapse)."""

    combine_mode: str = "additive"

    def target_representations(self, encoder, eeg):
        """Compute the target representations the predictor regresses against.

        Default: detached online encoder output. Subclasses override to use
        an EMA copy (DINO) or to let gradients flow (SIGReg).
        """
        with torch.no_grad():
            return encoder.encode_tokens(eeg, mask=None)

    def auxiliary_loss(self, *, context_tokens, target_tokens, pooled, global_step):
        return torch.zeros((), device=context_tokens.device), {}

    def step(self, encoder, momentum):
        """Post-optimizer-step hook (e.g. EMA update). Default: no-op."""
        return None

    @property
    def coeff(self) -> float:
        """λ for convex combine_mode; unused for additive."""
        return 0.0


class DINOAntiCollapse(AntiCollapse):
    """DINO-style anti-collapse: an EMA copy of the encoder produces targets.

    No explicit loss term — collapse is prevented by the asymmetry between
    the trainable encoder (which sees masked context) and the slow-moving
    momentum encoder (which produces frozen targets).
    """

    combine_mode = "additive"

    def __init__(self, target_encoder: nn.Module):
        super().__init__()
        self.target_encoder = target_encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def target_representations(self, encoder, eeg):
        with torch.no_grad():
            return self.target_encoder.encode_tokens(eeg, mask=None)

    def step(self, encoder, momentum):
        for p_enc, p_tgt in zip(encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.lerp_(p_enc.data, 1.0 - momentum)


class VICRegAntiCollapse(AntiCollapse):
    """VICReg: variance/covariance penalties on the online encoder's context tokens.

    Targets are produced by the online encoder with a stop-gradient (matching
    the existing single-encoder code path). The auxiliary loss is added to
    the prediction loss.
    """

    combine_mode = "additive"

    def __init__(self, vc_loss: VCLoss):
        super().__init__()
        self.vc_loss = vc_loss

    def auxiliary_loss(self, *, context_tokens, target_tokens, pooled, global_step):
        # VCLoss expects [B, C, T, H, W] where C is feature dim. Context tokens
        # are [B, n_ctx, D]; reshape to [B, D, n_ctx, 1, 1].
        ctx = context_tokens.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        loss, _, loss_dict = self.vc_loss(ctx)
        return loss, loss_dict


class SIGRegAntiCollapse(AntiCollapse):
    """SIGReg (LeJEPA): sketched Gaussianity test on per-window pooled embeddings.

    Targets are produced by the online encoder with gradients flowing — this
    matches LeJEPA's recipe of constraining the very representations probes
    will read. The auxiliary loss is combined convexly with the prediction
    loss (λ from the inner ``SIGRegLoss.coeff``).
    """

    combine_mode = "convex"

    def __init__(self, sigreg_loss: SIGRegLoss):
        super().__init__()
        self.sigreg_loss = sigreg_loss

    def target_representations(self, encoder, eeg):
        # NO stop-gradient: SIGReg is meant to shape these representations.
        return encoder.encode_tokens(eeg, mask=None)

    def auxiliary_loss(self, *, context_tokens, target_tokens, pooled, global_step):
        loss, _, loss_dict = self.sigreg_loss(pooled, global_step)
        return loss, loss_dict

    @property
    def coeff(self) -> float:
        return float(self.sigreg_loss.coeff)
