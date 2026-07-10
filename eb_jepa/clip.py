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


class SoftTargetCLIPPretrain(nn.Module):
    """Soft-target CLIP via Andonian-2022-style distillation from V-JEPA-2 similarity.

    Blends a hard diagonal target with a teacher distribution built from the
    pairwise cosine similarity of the (mean-centered, shot-mean) V-JEPA-2
    targets themselves. The teacher is *fixed* — V-JEPA-2 is frozen — so the
    Andonian progressive schedule collapses to a fixed alpha.

    Objective for the e→v direction:
        p_ij  = softmax_j(logits_ij)                   # logits = scale · z_eeg z_vis^T
        S_ij  = <v_i, v_j> / (‖v_i‖ ‖v_j‖)             # centered-cosine on the target space
        q_ij  = softmax_j(S_ij / tau_teacher)          # teacher soft label
        y_ij  = 1{i = j}
        t     = (1 - alpha) · y + alpha · q            # blended target
        L_e2v = -mean_i Σ_j t_ij log p_ij
    Symmetric v→e uses logits.T and t.T (q is symmetric by construction, so t is too).

    A temporal buffer masks cross-pairs with |Δt| < buffer_s out of both the
    student softmax denominator *and* the teacher softmax denominator, so those
    shot-cut-boundary pairs contribute to neither the target nor the prediction.
    Self-pairs are never masked.

    Forward consumes ``(eeg, frame_embedding_target, scene_ids, t_starts)`` —
    scene_ids is accepted for interface parity with ``SceneCLIPPretrain`` but is
    unused (all label structure comes from the teacher).
    """

    def __init__(
        self,
        encoder,
        clip_head,
        *,
        alpha: float = 0.5,
        tau_teacher: float = 0.1,
        temporal_buffer_s: float = 2.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.clip_head = clip_head
        self.alpha = float(alpha)
        self.tau_teacher = float(tau_teacher)
        self.temporal_buffer_s = float(temporal_buffer_s)

    def forward(
        self,
        eeg: torch.Tensor,
        frame_embedding_target: torch.Tensor,
        scene_ids: torch.Tensor,
        t_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        del scene_ids  # interface parity with SceneCLIPPretrain; teacher carries the label
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        pooled = self.encoder.pool_to_windows(tokens)            # [B, D, T, 1, 1]
        z_eeg = self.clip_head.project_eeg(pooled)               # [B*T, P]
        tgt = frame_embedding_target.to(z_eeg.dtype)
        tgt_flat = tgt.reshape(-1, tgt.shape[-1])                # [N, D_vjepa]
        z_vis = self.clip_head.project_vision(tgt_flat)          # [N, P]
        scale = self.clip_head.logit_scale.exp().clamp(max=100.0)
        logits = scale * (z_eeg @ z_vis.T)                       # [N, N]
        n = logits.shape[0]
        device = logits.device

        # Temporal-buffer mask on cross-pairs (self always kept).
        ts = t_starts.reshape(-1).to(device)
        dt = (ts.unsqueeze(0) - ts.unsqueeze(1)).abs()
        eye = torch.eye(n, dtype=torch.bool, device=device)
        buffer_mask = (dt < self.temporal_buffer_s) & ~eye        # [N, N]

        # Teacher soft targets from centered-cosine on the target space.
        v = F.normalize(tgt_flat.float(), dim=-1)
        S = v @ v.T                                              # [N, N] in [-1, 1]
        neg_inf = torch.finfo(S.dtype).min
        S_masked = S.masked_fill(buffer_mask, neg_inf)
        q = F.softmax(S_masked / self.tau_teacher, dim=-1)       # [N, N]

        # Blended target: (1 - α) · diag + α · q. q already zero on buffer cells.
        y = torch.eye(n, dtype=q.dtype, device=device)
        t = (1.0 - self.alpha) * y + self.alpha * q

        # Student log-probs with the same buffer mask.
        logits_masked = logits.masked_fill(buffer_mask, torch.finfo(logits.dtype).min)
        log_p_e2v = F.log_softmax(logits_masked, dim=-1)
        log_p_v2e = F.log_softmax(logits_masked.T, dim=-1)

        loss_e2v = -(t.to(log_p_e2v.dtype) * log_p_e2v).sum(dim=-1).mean()
        loss_v2e = -(t.T.to(log_p_v2e.dtype) * log_p_v2e).sum(dim=-1).mean()
        loss = 0.5 * (loss_e2v + loss_v2e)

        with torch.no_grad():
            labels = torch.arange(n, device=device)
            q_entropy = -(q.clamp_min(1e-12).log() * q).sum(dim=-1).mean()
            eff_pos = q_entropy.exp()
            n_excl = buffer_mask.float().sum(dim=-1).mean().item()
            loss_dict = {
                "clip_loss": loss.item(),
                "clip_loss_e2v": loss_e2v.item(),
                "clip_loss_v2e": loss_v2e.item(),
                "clip_top1_e2v": (logits.argmax(-1) == labels).float().mean().item(),
                "clip_top1_v2e": (logits.argmax(0) == labels).float().mean().item(),
                "clip_logit_scale": scale.item(),
                "clip_teacher_entropy": q_entropy.item(),
                "clip_teacher_eff_positives": eff_pos.item(),
                "clip_n_excluded_mean": n_excl,
                "clip_alpha": self.alpha,
                "clip_tau_teacher": self.tau_teacher,
                "total_loss": loss.item(),
            }
        return loss, loss_dict

    @torch.no_grad()
    def encode(self, eeg: torch.Tensor, keep_channels: bool = False) -> torch.Tensor:
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        return self.encoder.pool_to_windows(tokens, keep_channels=keep_channels)


class CSAlignerCLIPPretrain(nn.Module):
    """CS-Aligner: InfoNCE + Cauchy–Schwarz divergence on the batch marginals.

    Combines pairwise InfoNCE (per CLIPPretrain) with a distributional term —
    the Cauchy–Schwarz divergence between the empirical distributions of
    projected EEG and V-JEPA-2 embeddings — following Yin et al., 2025
    ("Distributional Vision-Language Alignment by Cauchy–Schwarz Divergence",
    arXiv:2502.17028). InfoNCE closes the pair-level gap; CS closes the
    global modality gap that pairwise objectives cannot reach.

    Objective:
        L = L_InfoNCE + cs_weight · D_CS(P_eeg, P_vis)

    Empirical CS estimator (paper Eq. 8), Gaussian kernel κ_σ(z,z') =
    exp(-‖z-z'‖²/(2σ²)). On L2-normalized z the kernel simplifies to
    exp((z·z' − 1)/σ²) via ‖z-z'‖² = 2(1 − z·z'):

        D_CS = log_mean_K(K_xx) + log_mean_K(K_yy) − 2·log_mean_K(K_xy)

    Numerically implemented as ``logsumexp((z·z' − 1)/σ²) − log(N·M)`` per
    Gram matrix. Self-similarity diagonals of K_xx / K_yy are the constant
    exp(0)=1 and included per the paper.

    Bandwidth σ: adaptive median heuristic — σ² = median(‖z_eeg − z_vis‖²)/2
    over the cross Gram matrix, detached from the graph so it does not
    backprop through the median. Pass a float via ``kernel_bandwidth`` to
    override with a fixed value.

    Temporal buffer: the 2 s |Δt| exclusion is applied to the InfoNCE
    denominator only (matches scene_clip / soft_target_clip so cs_aligner
    is directly comparable). The CS term never gets the buffer — masking
    cross-pairs would corrupt the kernel-density estimator on the marginals.

    ``logit_scale`` participates only in the InfoNCE term; CS uses raw
    cosine similarity to keep the temperature knob and CS decoupled.

    Forward consumes ``(eeg, frame_embedding_target, scene_ids, t_starts)`` —
    scene_ids is accepted for interface parity with ``SceneCLIPPretrain`` but
    is unused.
    """

    def __init__(
        self,
        encoder,
        clip_head,
        *,
        cs_weight: float = 1.0,
        kernel_bandwidth: float | None = None,
        temporal_buffer_s: float = 2.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.clip_head = clip_head
        self.cs_weight = float(cs_weight)
        self.kernel_bandwidth = (
            float(kernel_bandwidth) if kernel_bandwidth is not None else None
        )
        self.temporal_buffer_s = float(temporal_buffer_s)

    @staticmethod
    def _log_mean_kernel(A: torch.Tensor) -> torch.Tensor:
        """log(mean(exp(A))) via logsumexp for numerical stability."""
        flat = A.reshape(-1)
        return torch.logsumexp(flat, dim=0) - torch.log(
            torch.tensor(flat.numel(), dtype=flat.dtype, device=flat.device)
        )

    def _cs_divergence(
        self, z_eeg: torch.Tensor, z_vis: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """CS divergence on L2-normalized embeddings via cosine similarity.

        Returns (D_CS, diagnostics_dict). D_CS ≥ 0 with equality only when
        the two batch distributions match (up to kernel resolution)."""
        # Compute in fp32 to keep logsumexp numerically clean under amp.
        x = z_eeg.float()
        y = z_vis.float()

        # Cross squared distance (used for both median heuristic and K_xy).
        # ||x - y||² = 2 - 2·x·y for unit-norm vectors.
        cos_xy = x @ y.T                             # [M, N]
        d2_xy = (2.0 - 2.0 * cos_xy).clamp_min(0.0)

        if self.kernel_bandwidth is not None:
            sigma2 = torch.tensor(
                self.kernel_bandwidth ** 2, dtype=x.dtype, device=x.device
            )
        else:
            # Median heuristic: σ² = median(d²)/2. Detached so it doesn't
            # backprop through the median statistic.
            sigma2 = (d2_xy.detach().reshape(-1).median() / 2.0).clamp_min(1e-8)

        # Same-modality Gram matrices — diagonals are exactly 0 in log-space
        # (K_xx[i,i] = exp(0) = 1) so fill_diagonal_ eliminates float drift.
        cos_xx = x @ x.T
        cos_yy = y @ y.T
        A_xx = (cos_xx - 1.0) / sigma2
        A_yy = (cos_yy - 1.0) / sigma2
        A_xy = (cos_xy - 1.0) / sigma2
        A_xx.fill_diagonal_(0.0)
        A_yy.fill_diagonal_(0.0)

        log_kxx = self._log_mean_kernel(A_xx)
        log_kyy = self._log_mean_kernel(A_yy)
        log_kxy = self._log_mean_kernel(A_xy)
        d_cs = log_kxx + log_kyy - 2.0 * log_kxy

        diagnostics = {
            "cs_sigma": float(sigma2.sqrt().item()),
            "cs_log_kxx": float(log_kxx.item()),
            "cs_log_kyy": float(log_kyy.item()),
            "cs_log_kxy": float(log_kxy.item()),
        }
        return d_cs, diagnostics

    def forward(
        self,
        eeg: torch.Tensor,
        frame_embedding_target: torch.Tensor,
        scene_ids: torch.Tensor,
        t_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        del scene_ids  # interface parity; cs_aligner uses only paired InfoNCE + CS
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        pooled = self.encoder.pool_to_windows(tokens)            # [B, D, T, 1, 1]
        z_eeg = self.clip_head.project_eeg(pooled)               # [N, P]
        tgt = frame_embedding_target.to(z_eeg.dtype)
        z_vis = self.clip_head.project_vision(
            tgt.reshape(-1, tgt.shape[-1])
        )                                                        # [N, P]
        scale = self.clip_head.logit_scale.exp().clamp(max=100.0)
        logits = scale * (z_eeg @ z_vis.T)                       # [N, N]
        n = logits.shape[0]
        device = logits.device
        labels = torch.arange(n, device=device)

        # Temporal-buffer mask on cross-pairs for InfoNCE only. Self-pair kept.
        ts = t_starts.reshape(-1).to(device)
        dt = (ts.unsqueeze(0) - ts.unsqueeze(1)).abs()
        eye = torch.eye(n, dtype=torch.bool, device=device)
        buffer_mask = (dt < self.temporal_buffer_s) & ~eye
        neg_inf = torch.finfo(logits.dtype).min
        logits_masked = logits.masked_fill(buffer_mask, neg_inf)

        loss_e2v = F.cross_entropy(logits_masked, labels)
        loss_v2e = F.cross_entropy(logits_masked.T, labels)
        loss_infonce = 0.5 * (loss_e2v + loss_v2e)

        d_cs, cs_diag = self._cs_divergence(z_eeg, z_vis)
        if not torch.isfinite(d_cs):
            raise RuntimeError(
                f"cs_aligner: D_CS non-finite ({d_cs.item()}); "
                f"diagnostics={cs_diag}"
            )
        loss = loss_infonce + self.cs_weight * d_cs

        with torch.no_grad():
            n_excl = buffer_mask.float().sum(dim=-1).mean().item()
            loss_dict = {
                "clip_loss": loss.item(),
                "clip_loss_e2v": loss_e2v.item(),
                "clip_loss_v2e": loss_v2e.item(),
                "clip_top1_e2v": (logits.argmax(-1) == labels).float().mean().item(),
                "clip_top1_v2e": (logits.argmax(0) == labels).float().mean().item(),
                "clip_logit_scale": scale.item(),
                "clip_n_excluded_mean": n_excl,
                "cs_infonce": loss_infonce.item(),
                "cs_div": d_cs.item(),
                "cs_weight": self.cs_weight,
                **cs_diag,
                "total_loss": loss.item(),
            }
        return loss, loss_dict

    @torch.no_grad()
    def encode(self, eeg: torch.Tensor, keep_channels: bool = False) -> torch.Tensor:
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        return self.encoder.pool_to_windows(tokens, keep_channels=keep_channels)
