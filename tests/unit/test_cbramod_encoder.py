"""Tests for the CBraMod port behind the EEGEncoderTokens interface.

What can go wrong silently: a token order that differs from (c, t, p) makes
``pool_to_windows`` average the wrong things and still return the right shape;
a state_dict rename that misses a key trains a half-random "warm start" that
looks fine in every curve. Both are shape-preserving, so both are tested by
value here rather than by shape alone.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from eb_jepa.architectures import EEGEncoderTokens
from eb_jepa.cbramod import CBraModEncoderTokens
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_encoder_weights

_SPEC = importlib.util.spec_from_file_location(
    "prepare_cbramod_checkpoint",
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "clip_pretraining"
    / "scene_clip_from_checkpoint"
    / "prepare_cbramod_checkpoint.py",
)
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


def _enc(n_chans=5, n_times=400, n_windows=1, depth=2, **kw):
    torch.manual_seed(0)
    return CBraModEncoderTokens(
        n_chans=n_chans, n_times=n_times, n_windows=n_windows, depth=depth, **kw
    ).eval()


def test_token_and_pool_shapes():
    enc = _enc(n_chans=5, n_times=400, n_windows=3)
    eeg = torch.randn(2, 3, 5, 400)
    tokens = enc.encode_tokens(eeg)
    assert tokens.shape == (2, 5 * 3 * 2, 200)
    assert enc.n_patches_per_window == 2
    pooled = enc.pool_to_windows(tokens)
    assert pooled.shape == (2, 200, 3, 1, 1)
    kept = enc.pool_to_windows(tokens, keep_channels=True)
    assert kept.shape == (2, 5 * 200, 3, 1, 1)


def test_token_order_is_channel_window_patch():
    """Token index must be c*(T*P) + t*P + p, the order the REVE encoder uses.

    Tokens cannot be isolated by zeroing the rest of the input -- the patch
    encoder's GroupNorm pools statistics over every token in the sample -- so
    the order is checked at the two places it is decided: the [B, C, N, patch]
    grid handed to the patch encoder, and the flattening of its output.
    """
    enc = _enc(n_chans=3, n_times=400, n_windows=2)
    eeg = torch.randn(1, 2, 3, 400)
    T, P = 2, 2
    grid = enc._to_grid(eeg)
    assert grid.shape == (1, 3, T * P, 200)
    for c in range(3):
        for t in range(T):
            for p in range(P):
                assert torch.equal(
                    grid[0, c, t * P + p], eeg[0, t, c, p * 200 : (p + 1) * 200]
                )

    tokens, _ = enc.tokenize(eeg)
    grid_tokens, _ = enc.patch_embedding(grid)
    for c in range(3):
        for n in range(T * P):
            assert torch.equal(tokens[0, c * (T * P) + n], grid_tokens[0, c, n])


def test_mask_is_refused():
    enc = _enc()
    with pytest.raises(NotImplementedError):
        enc.encode_tokens(
            torch.randn(1, 1, 5, 400), mask=torch.ones(10, dtype=torch.bool)
        )


def test_input_scale_changes_the_spectral_branch_only_in_scale():
    """input_scale=s must equal feeding s*x -- a pure pre-multiplication."""
    a = _enc(input_scale=0.2)
    b = _enc(input_scale=1.0)
    b.load_state_dict(a.state_dict())
    eeg = torch.randn(1, 1, 5, 400)
    assert torch.allclose(a.encode_tokens(eeg), b.encode_tokens(0.2 * eeg), atol=1e-5)


def test_rejects_window_not_multiple_of_patch():
    with pytest.raises(ValueError):
        CBraModEncoderTokens(n_chans=4, n_times=410)


def test_build_encoder_dispatch():
    base = dict(
        encoder_embed_dim=200,
        encoder_depth=2,
        encoder_heads=8,
        encoder_head_dim=25,
        patch_size=200,
        patch_overlap=0,
    )
    cb = OmegaConf.create(
        {"model": {**base, "encoder_arch": "cbramod", "input_scale": 0.2}}
    )
    enc = build_encoder(cb, n_chans=4, n_times=400, chs_info=None, n_windows=1)
    assert isinstance(enc, CBraModEncoderTokens)
    assert enc.input_scale == 0.2

    reve = OmegaConf.create(
        {
            "model": {
                "encoder_embed_dim": 16,
                "encoder_depth": 1,
                "encoder_heads": 2,
                "encoder_head_dim": 8,
                "patch_size": 200,
                "patch_overlap": 0,
                "freqs": 2,
            }
        }
    )
    assert isinstance(
        build_encoder(reve, n_chans=4, n_times=400, chs_info=None, n_windows=1),
        EEGEncoderTokens,
    )
    with pytest.raises(ValueError):
        build_encoder(
            OmegaConf.create({"model": {**base, "encoder_arch": "luna"}}),
            n_chans=4,
            n_times=400,
            chs_info=None,
            n_windows=1,
        )
    with pytest.raises(ValueError):
        build_encoder(
            OmegaConf.create(
                {
                    "model": {
                        **base,
                        "encoder_arch": "cbramod",
                        "init_depth_scaled": True,
                    }
                }
            ),
            n_chans=4,
            n_times=400,
            chs_info=None,
            n_windows=1,
        )


def _upstream_style_state_dict(enc):
    """Rewrite our keys into the released checkpoint's naming, plus proj_out."""
    sd = {}
    for k, v in enc.state_dict().items():
        sd[k.replace("transformer.layers.", "encoder.layers.", 1)] = v.clone()
    sd["proj_out.0.weight"] = torch.zeros(200, 200)
    sd["proj_out.0.bias"] = torch.zeros(200)
    return sd


def test_remap_round_trips_through_strict_loader(tmp_path):
    src = _enc(depth=2)
    upstream = _upstream_style_state_dict(src)
    remapped = prep.remap_cbramod_to_eet(upstream)
    assert all(k.startswith("encoder.") for k in remapped)
    assert not any("proj_out" in k for k in remapped)
    assert len(remapped) == len(upstream) - 2
    path = tmp_path / "cbramod_init.pth.tar"
    torch.save({"model_state_dict": remapped}, path)

    # A different channel count on purpose: no CBraMod parameter depends on it.
    dst = _enc(n_chans=129, depth=2)
    info = load_encoder_weights(dst, path)
    assert info["missing"] == [] and info["unexpected"] == []
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_depth_mismatch_still_raises(tmp_path):
    path = tmp_path / "deep.pth.tar"
    torch.save(
        {
            "model_state_dict": prep.remap_cbramod_to_eet(
                _upstream_style_state_dict(_enc(depth=3))
            )
        },
        path,
    )
    with pytest.raises(RuntimeError):
        load_encoder_weights(_enc(depth=2), path)
