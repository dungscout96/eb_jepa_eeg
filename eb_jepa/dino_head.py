"""DINO self-distillation head (Caron et al. 2021, Sec. 3.2).

Used for an iBOT-lite auxiliary objective on the pooled global representation:
student = mean-pool(context_encoder visible tokens), teacher = mean-pool(target_encoder all tokens).
Cross-entropy with sharpened-and-centered teacher → forces encoder to retain
inter-subject-shared global structure (narrative envelope, scene-level semantics)
without forcing the encoder to satisfy a hard variance/decorrelation constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """3-layer MLP + L2-norm + weight-norm linear to K prototypes."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, K: int = 4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, K, bias=False))
        # freeze the magnitude scale so only direction is updated
        self.last.weight_g.data.fill_(1.0)
        self.last.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        return self.last(x)


class DINOLoss(nn.Module):
    """Self-distillation cross-entropy with running teacher centering."""

    def __init__(self, K: int = 4096, t_s: float = 0.1, t_t: float = 0.04,
                 m_c: float = 0.9):
        super().__init__()
        self.t_s = t_s
        self.t_t = t_t
        self.m_c = m_c
        self.register_buffer("center", torch.zeros(1, K))

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor) -> torch.Tensor:
        # teacher comes from EMA branch; detach defensively
        teacher_logits = teacher_logits.detach()
        t = F.softmax((teacher_logits - self.center) / self.t_t, dim=-1)
        s = F.log_softmax(student_logits / self.t_s, dim=-1)
        loss = -(t * s).sum(dim=-1).mean()
        # update center toward batch teacher mean
        with torch.no_grad():
            batch_center = teacher_logits.mean(dim=0, keepdim=True)
            self.center.mul_(self.m_c).add_(batch_center, alpha=1.0 - self.m_c)
        return loss
