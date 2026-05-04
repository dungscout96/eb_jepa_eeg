"""Cell L (multi-horizon latent rollout) sanity tests."""
import copy

import pytest
import torch
import torch.nn as nn

from eb_jepa.architectures import EEGEncoderTokens, MaskedPredictor
from eb_jepa.jepa import MaskedJEPA
from eb_jepa.masking import ContiguousTimeMaskCollator


def _build_jepa(horizon_embed=True, decompose=True, mf=0.5):
    C, T, P, embed_dim, patch_size = 5, 4, 12, 64, 50
    n_times = patch_size + (P - 1) * (patch_size - 20)
    chs_info = [{"ch_name": f"E{i+1}"} for i in range(C)]
    enc = EEGEncoderTokens(
        n_chans=C, n_times=n_times,
        embed_dim=embed_dim, depth=2, heads=4, head_dim=16,
        n_windows=T, patch_size=patch_size, patch_overlap=20,
        freqs=4, chs_info=chs_info, mlp_dim_ratio=2.0,
    )
    target_enc = copy.deepcopy(enc)
    pred = MaskedPredictor(embed_dim=64, depth=1, heads=4, head_dim=16,
                           mlp_dim_ratio=2.0, predictor_dim=24)
    coll = ContiguousTimeMaskCollator(n_channels=C, n_windows=T,
                                      n_patches_per_window=P,
                                      mask_window_fraction=mf,
                                      mask_position="tail")
    h_emb = nn.Embedding(T, embed_dim) if horizon_embed else None
    if h_emb is not None:
        nn.init.normal_(h_emb.weight, std=0.02)
    jepa = MaskedJEPA(enc, target_enc, pred, coll, regularizer=None,
                      pred_loss_type="smooth_l1",
                      horizon_embed=h_emb,
                      horizon_loss_decompose=decompose)
    return jepa, n_times


def test_horizons_returned_by_collator():
    coll = ContiguousTimeMaskCollator(n_channels=5, n_windows=4,
                                      n_patches_per_window=12,
                                      mask_window_fraction=0.5,
                                      mask_position="tail")
    res = coll()
    assert res.horizons is not None
    assert res.horizons.shape == (5 * 4 * 12,)
    # Tail mask 0.5 → mask windows {2, 3}; last_ctx_window = 1
    # Horizon for window 2 = |2-1| = 1; window 3 = |3-1| = 2
    P, T = 12, 4
    h_w2 = res.horizons[0 * T * P + 2 * P + 0].item()  # c=0, t=2, p=0
    h_w3 = res.horizons[0 * T * P + 3 * P + 0].item()  # c=0, t=3, p=0
    assert h_w2 == 1
    assert h_w3 == 2


def test_cell_l_forward_with_horizon_embedding():
    jepa, n_times = _build_jepa(horizon_embed=True, decompose=True, mf=0.5)
    eeg = torch.randn(4, 4, 5, n_times)
    total, ld = jepa(eeg)
    assert torch.isfinite(total)
    assert "pred_loss" in ld
    # Per-horizon loss decomposition logged
    assert "pred_loss_h1" in ld
    assert "pred_loss_h2" in ld


def test_cell_l_no_decompose_no_per_horizon_keys():
    jepa, n_times = _build_jepa(horizon_embed=True, decompose=False, mf=0.5)
    eeg = torch.randn(2, 4, 5, n_times)
    _, ld = jepa(eeg)
    assert "pred_loss" in ld
    # Decomposition disabled → no h_k keys
    h_keys = [k for k in ld if k.startswith("pred_loss_h")]
    assert h_keys == []


def test_cell_l_grad_flows_into_horizon_embed():
    jepa, n_times = _build_jepa(horizon_embed=True, decompose=False, mf=0.5)
    eeg = torch.randn(2, 4, 5, n_times)
    total, _ = jepa(eeg)
    total.backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0
             for p in jepa.horizon_embed.parameters()]
    assert any(grads)


def test_cell_l_falls_back_when_no_horizon_embed():
    """horizon_embed=None should leave forward unchanged (acts like Cell J)."""
    jepa, n_times = _build_jepa(horizon_embed=False, decompose=False, mf=0.5)
    eeg = torch.randn(2, 4, 5, n_times)
    total, ld = jepa(eeg)
    assert torch.isfinite(total)
    assert "pred_loss" in ld
