"""Cell B (cross-subject masked latent prediction) forward-pass sanity test."""
import copy

import pytest
import torch

from eb_jepa.architectures import EEGEncoderTokens, MaskedPredictor
from eb_jepa.jepa import MaskedJEPA
from eb_jepa.masking import MultiBlockMaskCollator


def _build_jepa(xsub_coeff=0.5, xsub_symmetric=True):
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
    pred = MaskedPredictor(
        embed_dim=embed_dim, depth=1, heads=4, head_dim=16,
        mlp_dim_ratio=2.0, predictor_dim=24,
    )
    mask_coll = MultiBlockMaskCollator(
        n_channels=C, n_windows=T, n_patches_per_window=P,
        n_pred_masks_short=1, n_pred_masks_long=1,
    )
    jepa = MaskedJEPA(
        enc, target_enc, pred, mask_coll, regularizer=None,
        pred_loss_type="smooth_l1",
        xsub_coeff=xsub_coeff, xsub_symmetric=xsub_symmetric,
    )
    return jepa, n_times, C, T


def _paired_stim_meta(B):
    """Build stim_meta where eeg[0::2][i] and eeg[1::2][i] share (movie, pos)."""
    assert B % 2 == 0
    half = B // 2
    rows = []
    for i in range(half):
        movie_id, pos_bucket = 7, 100 + i
        rows.append([2 * i,     movie_id, pos_bucket])
        rows.append([2 * i + 1, movie_id, pos_bucket])
    return torch.tensor(rows, dtype=torch.long)


def test_cell_b_forward_paired():
    jepa, n_times, C, T = _build_jepa(xsub_coeff=0.5, xsub_symmetric=True)
    B = 8
    eeg = torch.randn(B, T, C, n_times)
    meta = _paired_stim_meta(B)

    total_loss, loss_dict = jepa(eeg, stim_meta=meta)
    assert torch.isfinite(total_loss)
    assert "pred_loss" in loss_dict
    assert "xsub_pred_loss" in loss_dict
    assert loss_dict["xsub_pred_loss"] > 0  # nonzero for randomly-init weights
    # total_loss should include both pred_loss and xsub_coeff * xsub_pred_loss
    expected_total = loss_dict["pred_loss"] + 0.5 * loss_dict["xsub_pred_loss"]
    assert abs(loss_dict["total_loss"] - expected_total) < 1e-4


def test_cell_b_asymmetric():
    """xsub_symmetric=False: only A-from-B prediction, half the compute."""
    jepa, n_times, C, T = _build_jepa(xsub_coeff=0.5, xsub_symmetric=False)
    B = 8
    eeg = torch.randn(B, T, C, n_times)
    meta = _paired_stim_meta(B)
    total_loss, loss_dict = jepa(eeg, stim_meta=meta)
    assert torch.isfinite(total_loss)
    assert "xsub_pred_loss" in loss_dict


def test_cell_b_pair_mismatch_raises():
    jepa, n_times, C, T = _build_jepa(xsub_coeff=0.5)
    B = 4
    eeg = torch.randn(B, T, C, n_times)
    # Bad meta: pairs do NOT share (movie, position_bucket)
    meta = torch.tensor([
        [0, 7, 100],
        [1, 7, 200],  # mismatched pos with row 0
        [2, 7, 300],
        [3, 7, 400],  # mismatched pos with row 2
    ], dtype=torch.long)
    with pytest.raises(ValueError, match="xsub pair mismatch"):
        jepa(eeg, stim_meta=meta)


def test_cell_b_odd_batch_raises():
    jepa, n_times, C, T = _build_jepa(xsub_coeff=0.5)
    B = 5
    eeg = torch.randn(B, T, C, n_times)
    meta = torch.zeros(B, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="even batch size"):
        jepa(eeg, stim_meta=meta)


def test_cell_b_off_when_coeff_zero():
    """xsub_coeff=0 → no xsub branch; behaves like standard JEPA."""
    jepa, n_times, C, T = _build_jepa(xsub_coeff=0.0)
    B = 4
    eeg = torch.randn(B, T, C, n_times)
    total_loss, loss_dict = jepa(eeg)
    assert torch.isfinite(total_loss)
    assert "xsub_pred_loss" not in loss_dict


def test_cell_b_grad_flows():
    """Gradient flows through xsub branch into the encoder."""
    jepa, n_times, C, T = _build_jepa(xsub_coeff=1.0, xsub_symmetric=True)
    B = 4
    eeg = torch.randn(B, T, C, n_times)
    meta = _paired_stim_meta(B)
    total_loss, _ = jepa(eeg, stim_meta=meta)
    total_loss.backward()
    grads_seen = [
        p.grad is not None and p.grad.abs().sum() > 0
        for p in jepa.context_encoder.parameters()
    ]
    assert any(grads_seen), "no gradient flowed into context encoder"
