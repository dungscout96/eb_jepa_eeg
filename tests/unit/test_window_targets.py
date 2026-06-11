"""Unit tests for window-anchored frame embedding + shot ID helpers."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from eb_jepa.architectures import MovieCLIPHead
from eb_jepa.datasets.hbn import (
    get_window_frame_embedding,
    get_window_shot_id,
)


def test_frame_embedding_mean_pool_full_window():
    # 2 Hz clips, 5 clips covering 0..2s (timestamps 0.0, 0.5, 1.0, 1.5, 2.0)
    embeddings = np.arange(20, dtype=np.float32).reshape(5, 4)
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    # Window [0, 2): half-open → includes clips at 0.0, 0.5, 1.0, 1.5 (NOT 2.0)
    out = get_window_frame_embedding(0, 24.0, 2.0, embeddings, timestamps)
    np.testing.assert_allclose(out, embeddings[:4].mean(axis=0))


def test_frame_embedding_partial_window():
    embeddings = np.arange(20, dtype=np.float32).reshape(5, 4)
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    # Window starts at frame 12 (0.5s), 1s long → covers [0.5, 1.5)
    out = get_window_frame_embedding(12, 24.0, 1.0, embeddings, timestamps)
    np.testing.assert_allclose(out, embeddings[1:3].mean(axis=0))


def test_frame_embedding_out_of_bounds():
    embeddings = np.zeros((5, 4), dtype=np.float32)
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)

    # Window starts at 100s, far past the last timestamp
    out = get_window_frame_embedding(2400, 24.0, 2.0, embeddings, timestamps)
    assert out is None


def test_shot_id_inside_single_shot():
    # Shots: [0, 100) [100, 250) [250, +inf)
    boundaries = np.array([100, 250], dtype=np.int64)

    # Window fully inside shot 0
    sid, crossed = get_window_shot_id(10, 48, boundaries)
    assert sid == 0 and crossed is False


def test_shot_id_dominant_on_boundary_window():
    # Boundary at 100; window [90, 138) → shot 0 overlap=10, shot 1 overlap=38
    boundaries = np.array([100, 250], dtype=np.int64)
    sid, crossed = get_window_shot_id(90, 48, boundaries)
    assert sid == 1 and crossed is True


def test_shot_id_dominant_when_window_mostly_in_first_shot():
    # Boundary at 100; window [60, 108) → shot 0 overlap=40, shot 1 overlap=8
    boundaries = np.array([100, 250], dtype=np.int64)
    sid, crossed = get_window_shot_id(60, 48, boundaries)
    assert sid == 0 and crossed is True


def test_shot_id_final_shot():
    boundaries = np.array([100, 250], dtype=np.int64)
    # Window deep in shot 2 (last shot, unbounded)
    sid, crossed = get_window_shot_id(500, 48, boundaries)
    assert sid == 2 and crossed is False


# ---------------------------------------------------------------------------
# MovieCLIPHead — symmetric InfoNCE sanity
# ---------------------------------------------------------------------------


def _clip_loss(head: MovieCLIPHead, eeg_pooled: torch.Tensor, vis: torch.Tensor):
    """Replicate the symmetric InfoNCE used in MaskedJEPA.forward."""
    z_eeg = head.project_eeg(eeg_pooled)
    z_vis = head.project_vision(vis)
    scale = head.logit_scale.exp().clamp(max=100.0)
    logits = scale * (z_eeg @ z_vis.T)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_e2v = F.cross_entropy(logits, labels)
    loss_v2e = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_e2v + loss_v2e), logits, labels


def test_movie_clip_head_random_init_near_log_bt():
    """At random init the symmetric CLIP loss should be close to log(B*T)."""
    torch.manual_seed(0)
    B, T, D, V = 4, 8, 32, 1408
    head = MovieCLIPHead(eeg_in_dim=D, vision_in_dim=V, proj_dim=64, temperature=1.0)
    head.eval()
    eeg = torch.randn(B, D, T, 1, 1)
    vis = torch.randn(B * T, V)
    loss, logits, _ = _clip_loss(head, eeg, vis)
    expected = math.log(B * T)
    # With temperature=1 (scale=e) and random projections, logits are small and
    # the loss sits close to log(N). Allow a 25% band for finite-batch noise.
    assert abs(loss.item() - expected) / expected < 0.25, (
        f"loss={loss.item():.3f} vs log(B*T)={expected:.3f}"
    )


def test_movie_clip_head_temperature_is_learnable():
    """logit_scale must receive gradient when its loss is non-trivial."""
    torch.manual_seed(0)
    B, T, D, V = 2, 4, 16, 64
    head = MovieCLIPHead(eeg_in_dim=D, vision_in_dim=V, proj_dim=32)
    eeg = torch.randn(B, D, T, 1, 1, requires_grad=True)
    vis = torch.randn(B * T, V)
    loss, _, _ = _clip_loss(head, eeg, vis)
    loss.backward()
    assert head.logit_scale.grad is not None
    assert head.logit_scale.grad.abs().item() > 0
    assert head.eeg_proj.weight.grad.abs().sum().item() > 0
    assert head.vision_proj.weight.grad.abs().sum().item() > 0


def test_movie_clip_head_perfect_alignment_low_loss():
    """If z_eeg == z_vis (identical paired vectors), loss drops far below log(N)."""
    torch.manual_seed(0)
    B, T, D, V = 2, 4, 8, 8
    head = MovieCLIPHead(eeg_in_dim=D, vision_in_dim=V, proj_dim=8, temperature=0.07)
    # Force both projections to identity by reusing one tensor as both inputs.
    head.eeg_proj.weight.data.copy_(torch.eye(8))
    head.eeg_proj.bias.data.zero_()
    head.vision_proj.weight.data.copy_(torch.eye(8))
    head.vision_proj.bias.data.zero_()
    x = torch.randn(B * T, 8)
    # Fake the [B, D, T, 1, 1] shape from a per-window vector.
    eeg_pooled = x.T.view(D, B, T).permute(1, 0, 2).unsqueeze(-1).unsqueeze(-1)
    loss, _, _ = _clip_loss(head, eeg_pooled, x)
    assert loss.item() < math.log(B * T) * 0.5
