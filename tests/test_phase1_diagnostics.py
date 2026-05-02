"""Smoke test the new Phase-1 diagnostic flags on EEGEncoderTokens / MaskedJEPA."""
import copy
import torch

from eb_jepa.architectures import (
    EEGEncoderTokens,
    MaskedPredictor,
    MovieFeatureHead,
)
from eb_jepa.jepa import MaskedJEPA, MaskedJEPAProbe
from eb_jepa.masking import MultiBlockMaskCollator


def build_encoder(C=5, T=4, P=12, embed_dim=64, depth=2, patch_size=50):
    n_times = patch_size + (P - 1) * (patch_size - 20)  # match P with overlap=20
    chs_info = [{"ch_name": f"E{i+1}"} for i in range(C)]
    enc = EEGEncoderTokens(
        n_chans=C, n_times=n_times,
        embed_dim=embed_dim, depth=depth, heads=4, head_dim=16,
        n_windows=T, patch_size=patch_size, patch_overlap=20,
        freqs=4, chs_info=chs_info, mlp_dim_ratio=2.0,
    )
    return enc, n_times


def test_encode_at_layer_shape():
    enc, n_times = build_encoder()
    eeg = torch.randn(2, 4, 5, n_times)
    final = enc.encode_at_layer(eeg, layer="final")
    pe = enc.encode_at_layer(eeg, layer="patch_embed")
    b0 = enc.encode_at_layer(eeg, layer="block0")
    b1 = enc.encode_at_layer(eeg, layer="block1")
    expected = (2, 5 * 4 * 12, 64)
    for name, x in [("final", final), ("patch_embed", pe), ("block0", b0), ("block1", b1)]:
        assert x.shape == expected, f"{name}: {x.shape}"
    # b1 should equal final (depth=2 → block1 is last block)
    assert torch.allclose(b1, final)
    # patch_embed != final
    assert not torch.allclose(pe, final)
    print("encode_at_layer shapes OK")


def test_encode_at_layer_invalid():
    enc, n_times = build_encoder(depth=2)
    try:
        enc.encode_at_layer(torch.randn(1, 4, 5, n_times), layer="block5")
    except ValueError as e:
        assert "out of range" in str(e)
        print("invalid layer rejected:", e)
    else:
        raise AssertionError("should have raised")


def test_pool_to_windows_modes():
    enc, _ = build_encoder()
    # synth tokens
    B, total = 2, 5 * 4 * 12
    tokens = torch.randn(B, total, 64)
    # default
    x_default = enc.pool_to_windows(tokens)
    assert x_default.shape == (2, 64, 4, 1, 1)
    # keep_channels
    x_kc = enc.pool_to_windows(tokens, keep_channels=True)
    assert x_kc.shape == (2, 5 * 64, 4, 1, 1), f"{x_kc.shape}"
    # select_channel
    x_sc = enc.pool_to_windows(tokens, select_channel=2)
    assert x_sc.shape == (2, 64, 4, 1, 1)
    # mutual exclusivity not enforced at pool level — caller validates
    print("pool_to_windows shapes OK")


def test_jepa_encode_diagnostic_flags():
    enc, n_times = build_encoder()
    target_enc = copy.deepcopy(enc)
    pred = MaskedPredictor(
        embed_dim=64, depth=1, heads=4, head_dim=16,
        mlp_dim_ratio=2.0, predictor_dim=24,
    )
    mask_coll = MultiBlockMaskCollator(
        n_channels=5, n_windows=4, n_patches_per_window=12,
        n_pred_masks_short=1, n_pred_masks_long=1,
    )
    jepa = MaskedJEPA(enc, target_enc, pred, mask_coll, regularizer=None)
    eeg = torch.randn(2, 4, 5, n_times)

    # Default
    out = jepa.encode(eeg)
    assert out.shape == (2, 64, 4, 1, 1), out.shape
    # Layer tap
    out_b0 = jepa.encode(eeg, probe_layer="block0")
    assert out_b0.shape == (2, 64, 4, 1, 1)
    # Teacher
    out_t = jepa.encode(eeg, use_teacher=True)
    assert out_t.shape == (2, 64, 4, 1, 1)
    # keep_channels
    out_kc = jepa.encode(eeg, keep_channels=True)
    assert out_kc.shape == (2, 5 * 64, 4, 1, 1)
    # select_channel
    out_sc = jepa.encode(eeg, select_channel=0)
    assert out_sc.shape == (2, 64, 4, 1, 1)
    # prepred (24-d bottleneck)
    out_pp = jepa.encode(eeg, prepred=True)
    assert out_pp.shape == (2, 24, 4, 1, 1), out_pp.shape
    print("jepa.encode diagnostic flags OK")


def test_probe_construction_dims():
    """MaskedJEPAProbe head sizing must match the post-flag feature dim."""
    enc, n_times = build_encoder()
    target_enc = copy.deepcopy(enc)
    pred = MaskedPredictor(64, 1, 4, 16, 2.0, predictor_dim=24)
    mask_coll = MultiBlockMaskCollator(5, 4, 12, n_pred_masks_short=1, n_pred_masks_long=1)
    jepa = MaskedJEPA(enc, target_enc, pred, mask_coll, regularizer=None)

    # default: head expects D=64
    head = MovieFeatureHead(64, 32, 4)
    probe = MaskedJEPAProbe(jepa, head, hcost=lambda y, t: y.sum())
    out = probe._features(torch.randn(2, 4, 5, n_times))
    assert out.shape == (2, 64, 4, 1, 1)

    # keep_channels: head expects 5*64=320
    head_kc = MovieFeatureHead(320, 32, 4)
    probe_kc = MaskedJEPAProbe(jepa, head_kc, hcost=lambda y, t: y.sum(), keep_channels=True)
    out_kc = probe_kc._features(torch.randn(2, 4, 5, n_times))
    assert out_kc.shape == (2, 320, 4, 1, 1)

    # prepred: head expects 24
    head_pp = MovieFeatureHead(24, 32, 4)
    probe_pp = MaskedJEPAProbe(jepa, head_pp, hcost=lambda y, t: y.sum(), prepred=True)
    out_pp = probe_pp._features(torch.randn(2, 4, 5, n_times))
    assert out_pp.shape == (2, 24, 4, 1, 1)
    print("probe head dim sizing OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_encode_at_layer_shape()
    test_encode_at_layer_invalid()
    test_pool_to_windows_modes()
    test_jepa_encode_diagnostic_flags()
    test_probe_construction_dims()
    print("\nALL PHASE-1 DIAGNOSTIC TESTS PASSED")
