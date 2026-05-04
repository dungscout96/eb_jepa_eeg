"""Cell J (Cross-Time mask) + Cell K (PARS Δt regression) sanity tests."""
import copy

import pytest
import torch
import torch.nn as nn

from eb_jepa.architectures import EEGEncoderTokens, MaskedPredictor
from eb_jepa.jepa import MaskedJEPA
from eb_jepa.masking import ContiguousTimeMaskCollator, MultiBlockMaskCollator


def _build_encoder(C=5, T=4, P=12, embed_dim=64, depth=2, patch_size=50):
    n_times = patch_size + (P - 1) * (patch_size - 20)
    chs_info = [{"ch_name": f"E{i+1}"} for i in range(C)]
    enc = EEGEncoderTokens(
        n_chans=C, n_times=n_times,
        embed_dim=embed_dim, depth=depth, heads=4, head_dim=16,
        n_windows=T, patch_size=patch_size, patch_overlap=20,
        freqs=4, chs_info=chs_info, mlp_dim_ratio=2.0,
    )
    return enc, n_times


# ============== Cell J: Cross-Time mask ==============

def test_cross_time_mask_shape():
    coll = ContiguousTimeMaskCollator(n_channels=5, n_windows=4, n_patches_per_window=12,
                                      mask_window_fraction=0.5, mask_position="tail")
    res = coll()
    assert res.context_mask.shape == (5 * 4 * 12,)
    # Half windows masked: 2 of 4 windows × 5 channels × 12 patches = 120 masked
    assert (~res.context_mask).sum().item() == 5 * 2 * 12
    assert res.context_mask.sum().item() == 5 * 2 * 12  # 2 visible windows
    assert len(res.pred_masks) == 1


def test_cross_time_tail_position():
    """Tail mask must hit the LAST mask_window_fraction of windows."""
    coll = ContiguousTimeMaskCollator(n_channels=5, n_windows=4, n_patches_per_window=12,
                                      mask_window_fraction=0.25, mask_position="tail")
    res = coll()
    masked = res.pred_masks[0].tolist()
    T, P = 4, 12
    # token_idx = c*T*P + t*P + p; masked windows should all have t=T-1=3
    masked_t = sorted({(idx % (T * P)) // P for idx in masked})
    assert masked_t == [3]


def test_cross_time_head_position():
    coll = ContiguousTimeMaskCollator(n_channels=5, n_windows=4, n_patches_per_window=12,
                                      mask_window_fraction=0.25, mask_position="head")
    res = coll()
    masked = res.pred_masks[0].tolist()
    T, P = 4, 12
    masked_t = sorted({(idx % (T * P)) // P for idx in masked})
    assert masked_t == [0]


def test_cross_time_jepa_forward():
    """End-to-end JEPA forward with Cross-Time mask collator."""
    enc, n_times = _build_encoder()
    target_enc = copy.deepcopy(enc)
    pred = MaskedPredictor(embed_dim=64, depth=1, heads=4, head_dim=16,
                           mlp_dim_ratio=2.0, predictor_dim=24)
    coll = ContiguousTimeMaskCollator(n_channels=5, n_windows=4, n_patches_per_window=12)
    jepa = MaskedJEPA(enc, target_enc, pred, coll, regularizer=None,
                      pred_loss_type="smooth_l1")
    eeg = torch.randn(4, 4, 5, n_times)
    total_loss, loss_dict = jepa(eeg)
    assert torch.isfinite(total_loss)
    assert "pred_loss" in loss_dict
    assert loss_dict["pred_loss"] > 0


def test_cross_time_invalid_fraction():
    with pytest.raises(ValueError):
        ContiguousTimeMaskCollator(n_channels=5, n_windows=4, n_patches_per_window=12,
                                   mask_window_fraction=1.0, mask_position="tail")


# ============== Cell K: PARS Δt regression ==============

def _build_pars_jepa(pars_coeff=0.5, max_delta=1000.0):
    enc, n_times = _build_encoder()
    target_enc = copy.deepcopy(enc)
    pred = MaskedPredictor(embed_dim=64, depth=1, heads=4, head_dim=16,
                           mlp_dim_ratio=2.0, predictor_dim=24)
    coll = MultiBlockMaskCollator(n_channels=5, n_windows=4, n_patches_per_window=12,
                                  n_pred_masks_short=1, n_pred_masks_long=1)
    pars_head = nn.Sequential(
        nn.Linear(2 * 64, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh()
    )
    jepa = MaskedJEPA(enc, target_enc, pred, coll, regularizer=None,
                      pred_loss_type="smooth_l1",
                      pars_head=pars_head, pars_coeff=pars_coeff,
                      pars_max_delta=max_delta)
    return jepa, n_times


def test_pars_forward_runs():
    jepa, n_times = _build_pars_jepa(pars_coeff=0.5, max_delta=1000.0)
    eeg = torch.randn(4, 4, 5, n_times)
    pair = torch.randn(4, 4, 5, n_times)
    delta = torch.tensor([100, -200, 500, -700], dtype=torch.long)
    total, ld = jepa(eeg, pars_pair_eeg=pair, pars_delta=delta)
    assert torch.isfinite(total)
    assert "pars_loss" in ld
    assert ld["pars_loss"] > 0


def test_pars_off_when_coeff_zero():
    """pars_coeff=0 → no PARS branch."""
    jepa, n_times = _build_pars_jepa(pars_coeff=0.0)
    eeg = torch.randn(2, 4, 5, n_times)
    total, ld = jepa(eeg)
    assert "pars_loss" not in ld


def test_pars_delta_normalization():
    """Predicted Δ is in [-1, 1] (Tanh output); target is Δ/max_delta clamped."""
    jepa, n_times = _build_pars_jepa(pars_coeff=1.0, max_delta=1000.0)
    eeg = torch.randn(2, 4, 5, n_times)
    pair = torch.randn(2, 4, 5, n_times)
    # Δ outside max_delta should be clamped to [-1, 1] in target
    delta = torch.tensor([5000, -5000], dtype=torch.long)
    _, ld = jepa(eeg, pars_pair_eeg=pair, pars_delta=delta)
    assert "pars_target_mean" in ld
    # target_mean of ±1 average → 0
    assert abs(ld["pars_target_mean"]) < 0.01


def test_pars_grad_flows():
    """Gradient from PARS loss reaches the encoder."""
    jepa, n_times = _build_pars_jepa(pars_coeff=1.0)
    eeg = torch.randn(2, 4, 5, n_times)
    pair = torch.randn(2, 4, 5, n_times)
    delta = torch.tensor([100, -300], dtype=torch.long)
    total, _ = jepa(eeg, pars_pair_eeg=pair, pars_delta=delta)
    total.backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0
             for p in jepa.context_encoder.parameters()]
    assert any(grads)
    head_grads = [p.grad is not None and p.grad.abs().sum() > 0
                  for p in jepa.pars_head.parameters()]
    assert any(head_grads)
