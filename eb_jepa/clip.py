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


class SceneCLIPPretrain(nn.Module):
    """Supervised-contrastive CLIP per the §9 bottom-line recipe.

    Differences from ``CLIPPretrain``:
    - Positive mask: ``scene_id[i] == scene_id[j]`` (multi-positive, not diagonal).
    - Negative-exclusion mask: cross-scene pairs with ``|t_start[i] - t_start[j]| < buffer_s``
      are dropped from the denominator. Self always positive.
    - Dataset is expected to supply shot-mean, mean-centered V-JEPA-2 targets (so
      same-shot rows of ``z_vis`` collide — the positive mask handles this correctly).

    Forward consumes ``(eeg, frame_embedding_target, scene_ids, t_starts)`` from
    ``JEPAMovieDataset`` in recipe_mode.
    """

    def __init__(self, encoder, clip_head, *, temporal_buffer_s: float = 2.0):
        super().__init__()
        self.encoder = encoder
        self.clip_head = clip_head
        self.temporal_buffer_s = float(temporal_buffer_s)

    def forward(
        self,
        eeg: torch.Tensor,
        frame_embedding_target: torch.Tensor,
        scene_ids: torch.Tensor,
        t_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        pooled = self.encoder.pool_to_windows(tokens)            # [B, D, T, 1, 1]
        z_eeg = self.clip_head.project_eeg(pooled)               # [B*T, P]
        tgt = frame_embedding_target.to(z_eeg.dtype)
        z_vis = self.clip_head.project_vision(
            tgt.reshape(-1, tgt.shape[-1])
        )                                                        # [B*T, P]
        scale = self.clip_head.logit_scale.exp().clamp(max=100.0)
        logits = scale * (z_eeg @ z_vis.T)                       # [N, N]
        n = logits.shape[0]
        device = logits.device

        sid = scene_ids.reshape(-1).to(device)
        ts = t_starts.reshape(-1).to(device)

        # Positive mask (always includes self). Windows with scene_id == -1 are
        # treated as singleton positives (self only) so they cannot pollute another
        # anchor's positive set.
        valid = sid >= 0
        pos_mask = (sid.unsqueeze(0) == sid.unsqueeze(1)) & valid.unsqueeze(0) & valid.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=device)
        pos_mask = pos_mask | eye

        # Exclude cross-scene pairs within the temporal buffer from the denominator.
        # Positives are never excluded (self-pair already handled by `eye`).
        dt = (ts.unsqueeze(0) - ts.unsqueeze(1)).abs()
        excl_mask = (~pos_mask) & (dt < self.temporal_buffer_s)

        # Symmetric supervised-contrastive InfoNCE.
        neg_inf = torch.finfo(logits.dtype).min
        denom_mask = ~excl_mask
        loss_e2v = self._supcon(logits, pos_mask, denom_mask, neg_inf)
        loss_v2e = self._supcon(logits.T, pos_mask.T, denom_mask.T, neg_inf)
        loss = 0.5 * (loss_e2v + loss_v2e)

        with torch.no_grad():
            n_pos = pos_mask.float().sum(dim=-1).mean().item()
            n_excl = excl_mask.float().sum(dim=-1).mean().item()
            loss_dict = {
                "clip_loss": loss.item(),
                "clip_loss_e2v": loss_e2v.item(),
                "clip_loss_v2e": loss_v2e.item(),
                "clip_top1_e2v": (logits.argmax(-1) == torch.arange(n, device=device)).float().mean().item(),
                "clip_top1_v2e": (logits.argmax(0) == torch.arange(n, device=device)).float().mean().item(),
                "clip_logit_scale": scale.item(),
                "clip_n_positives_mean": n_pos,
                "clip_n_excluded_mean": n_excl,
                "total_loss": loss.item(),
            }
        return loss, loss_dict

    @staticmethod
    def _supcon(
        logits: torch.Tensor,
        pos_mask: torch.Tensor,
        denom_mask: torch.Tensor,
        neg_inf: float,
    ) -> torch.Tensor:
        """SupCon: -mean_i logsumexp(logits[i] | pos) + logsumexp(logits[i] | denom)."""
        pos_logits = logits.masked_fill(~pos_mask, neg_inf)
        den_logits = logits.masked_fill(~denom_mask, neg_inf)
        pos_lse = torch.logsumexp(pos_logits, dim=-1)
        den_lse = torch.logsumexp(den_logits, dim=-1)
        return (den_lse - pos_lse).mean()

    @torch.no_grad()
    def encode(self, eeg: torch.Tensor, keep_channels: bool = False) -> torch.Tensor:
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        return self.encoder.pool_to_windows(tokens, keep_channels=keep_channels)
