"""Depth-scaled residual init: touches the residual writers, nothing else.

Guards the flag that makes depth > 22 trainable. A pre-norm stack adds every
block's output onto the residual stream, so stream variance grows with depth;
without rescaling the residual OUTPUT projections, deep encoders train poorly
and produce a null that looks like "capacity does not help" (CaiT reports
12 -> 36 layers costing ~10 points top-1 at fixed lr/wd).

The flag must stay off by default, or every result recorded before it existed
stops reproducing.
"""
import math

import pytest
import torch

from eb_jepa.architectures import EEGEncoderTokens
from eb_jepa.nn_utils import apply_depth_scaled_residual_init


def _encoder(depth, **kw):
    return EEGEncoderTokens(
        n_chans=8, n_times=400, embed_dim=64, depth=depth, heads=2,
        head_dim=32, n_windows=1, patch_size=200, patch_overlap=20,
        freqs=3, mlp_dim_ratio=2.66, **kw,
    )


@pytest.mark.parametrize("depth", [4, 22, 44])
def test_scales_both_residual_writers_per_block(depth):
    enc = _encoder(depth)
    n = apply_depth_scaled_residual_init(enc.transformer, depth)
    assert n == 2 * depth, "one attention out-proj and one FFN out-proj per block"

    target = 0.02 / math.sqrt(2.0 * depth)
    for attn, ff in enc.transformer.layers:
        ffn_out = [m for m in ff.net if isinstance(m, torch.nn.Linear)][-1]
        for w in (attn.to_out.weight, ffn_out.weight):
            # Sample std, so allow generous tolerance on small tensors.
            assert w.std().item() == pytest.approx(target, rel=0.35)


def test_leaves_qkv_and_ffn_input_untouched():
    depth = 12
    enc = _encoder(depth)
    before = {
        "qkv": enc.transformer.layers[0][0].to_qkv.weight.clone(),
        "ffn_in": [m for m in enc.transformer.layers[0][1].net
                   if isinstance(m, torch.nn.Linear)][0].weight.clone(),
        "patch": enc.to_patch_embedding.weight.clone(),
    }
    apply_depth_scaled_residual_init(enc.transformer, depth)
    assert torch.equal(enc.transformer.layers[0][0].to_qkv.weight, before["qkv"])
    ffn_in = [m for m in enc.transformer.layers[0][1].net
              if isinstance(m, torch.nn.Linear)][0]
    assert torch.equal(ffn_in.weight, before["ffn_in"])
    assert torch.equal(enc.to_patch_embedding.weight, before["patch"])


def test_flag_defaults_off_and_is_opt_in():
    torch.manual_seed(0)
    default = _encoder(8)
    torch.manual_seed(0)
    explicit_off = _encoder(8, init_depth_scaled=False)
    torch.manual_seed(0)
    on = _encoder(8, init_depth_scaled=True)

    a = default.transformer.layers[0][0].to_out.weight
    b = explicit_off.transformer.layers[0][0].to_out.weight
    c = on.transformer.layers[0][0].to_out.weight
    assert torch.equal(a, b), "default must be a no-op"
    assert not torch.equal(a, c), "the flag must actually change the init"
    assert c.std().item() < a.std().item(), "scaled init must shrink the writer"


def test_deeper_gets_smaller_scale():
    stds = []
    for depth in (8, 32):
        enc = _encoder(depth, init_depth_scaled=True)
        stds.append(enc.transformer.layers[0][0].to_out.weight.std().item())
    assert stds[1] < stds[0], "1/sqrt(2*depth) must shrink as depth grows"


def test_forward_still_runs_with_flag_on():
    enc = _encoder(22, init_depth_scaled=True)
    out = enc(torch.randn(2, 1, 8, 400))
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()
