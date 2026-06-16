"""CLIP-style pretraining model for EEG ↔ V-JEPA-2 alignment.

Distinct from `eb_jepa.jepa.MaskedJEPA`: this path does not mask, does not run
a predictor, and has no anti-collapse term. The encoder produces per-window
embeddings that are pulled toward the matching V-JEPA-2 mean-pooled vision
vector via symmetric InfoNCE (standard CLIP loss).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPPretrain(nn.Module):
    """Symmetric CLIP InfoNCE between EEG window embeddings and V-JEPA-2 vectors.

    Forward consumes ``(eeg, frame_embedding_target)`` from JEPAMovieDataset's
    dataloader output (the dataset already mean-pools V-JEPA-2 clips within each
    EEG window). One unmasked encoder pass per batch.

    Caveat: per-window vision vectors can duplicate when two recordings in the
    same batch are sampled at the same window position. With B=64, T=8 this is
    rare in practice and treated as a hard-negative collision (no dedup).
    """

    def __init__(self, encoder, clip_head):
        super().__init__()
        self.encoder = encoder
        self.clip_head = clip_head

    def forward(
        self,
        eeg: torch.Tensor,
        frame_embedding_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        pooled = self.encoder.pool_to_windows(tokens)            # [B, D, T, 1, 1]
        z_eeg = self.clip_head.project_eeg(pooled)               # [B*T, P]
        tgt = frame_embedding_target.to(z_eeg.dtype)
        z_vis = self.clip_head.project_vision(
            tgt.reshape(-1, tgt.shape[-1])
        )                                                        # [B*T, P]
        scale = self.clip_head.logit_scale.exp().clamp(max=100.0)
        logits = scale * (z_eeg @ z_vis.T)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_e2v = F.cross_entropy(logits, labels)
        loss_v2e = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_e2v + loss_v2e)
        with torch.no_grad():
            loss_dict = {
                "clip_loss": loss.item(),
                "clip_loss_e2v": loss_e2v.item(),
                "clip_loss_v2e": loss_v2e.item(),
                "clip_top1_e2v": (logits.argmax(-1) == labels).float().mean().item(),
                "clip_top1_v2e": (logits.argmax(0) == labels).float().mean().item(),
                "clip_logit_scale": scale.item(),
                "total_loss": loss.item(),
            }
        return loss, loss_dict

    @torch.no_grad()
    def encode(self, eeg: torch.Tensor, keep_channels: bool = False) -> torch.Tensor:
        """Probe-side encoder; mirrors ``MaskedJEPA.encode`` for downstream eval."""
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        return self.encoder.pool_to_windows(tokens, keep_channels=keep_channels)
