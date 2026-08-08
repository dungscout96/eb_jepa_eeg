"""Tests for the strict-load guard in ``load_encoder_weights``.

The failure this guards is silent. ``load_state_dict(..., strict=False)`` reports
a depth-22 REVE checkpoint loaded into a depth-12 encoder as ``missing=0,
unexpected=61`` and returns normally: the run then trains on a truncated
backbone, converges, and yields plausible metrics that no downstream check can
distinguish from a correct warm-start. Only the load-time key comparison can
catch it, so it must raise rather than log.
"""

import pytest
import torch

from eb_jepa.architectures import EEGEncoderTokens
from eb_jepa.training_utils import load_encoder_weights


def _encoder(depth=2, embed_dim=8):
    return EEGEncoderTokens(
        n_chans=2,
        n_times=32,
        embed_dim=embed_dim,
        depth=depth,
        heads=2,
        head_dim=4,
        n_windows=1,
        patch_size=16,
        patch_overlap=0,
        freqs=2,
        mlp_dim_ratio=2.0,
    )


def _write_ckpt(tmp_path, encoder, extra=None, name="init.pth.tar"):
    """Save `encoder` the way the trainers do: keys prefixed with `encoder.`."""
    sd = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    if extra:
        sd.update(extra)
    path = tmp_path / name
    torch.save({"model_state_dict": sd, "epoch": -1, "step": 0}, path)
    return path


def test_matching_shapes_load_cleanly(tmp_path):
    src = _encoder()
    ckpt = _write_ckpt(tmp_path, src)

    dst = _encoder()
    info = load_encoder_weights(dst, ckpt)

    assert info["missing"] == [] and info["unexpected"] == []
    assert info["n_loaded"] == len(src.state_dict())
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_depth_mismatch_raises(tmp_path):
    """The E0.4 accident: a deeper checkpoint into a shallower encoder."""
    ckpt = _write_ckpt(tmp_path, _encoder(depth=4))

    with pytest.raises(RuntimeError, match="does not match this encoder"):
        load_encoder_weights(_encoder(depth=2), ckpt)


def test_depth_mismatch_allowed_when_partial_is_opted_in(tmp_path):
    ckpt = _write_ckpt(tmp_path, _encoder(depth=4))

    dst = _encoder(depth=2)
    info = load_encoder_weights(dst, ckpt, allow_partial=True)

    assert info["missing"] == []
    assert any(k.startswith("transformer.layers.2.") for k in info["unexpected"])


def test_reve_position_bank_is_not_a_mismatch(tmp_path):
    """REVE ships a fixed channel-position buffer; EEGEncoderTokens recomputes
    it, so it is the one leftover key that must not abort a warm-start."""
    src = _encoder()
    ckpt = _write_ckpt(
        tmp_path,
        src,
        extra={"encoder._position_bank.embedding": torch.zeros(543, 3)},
    )

    info = load_encoder_weights(_encoder(), ckpt)

    assert info["unexpected"] == ["_position_bank.embedding"]
    assert info["missing"] == []


def test_position_bank_does_not_excuse_a_missing_key(tmp_path):
    src = _encoder()
    sd = {f"encoder.{k}": v for k, v in src.state_dict().items()}
    sd.pop("encoder.to_patch_embedding.weight")
    sd["encoder._position_bank.embedding"] = torch.zeros(543, 3)
    path = tmp_path / "holed.pth.tar"
    torch.save({"model_state_dict": sd}, path)

    with pytest.raises(RuntimeError, match="to_patch_embedding.weight"):
        load_encoder_weights(_encoder(), path)


def test_non_encoder_keys_are_dropped_not_counted_as_mismatch(tmp_path):
    src = _encoder()
    ckpt = _write_ckpt(
        tmp_path,
        src,
        extra={"predictor.blocks.0.weight": torch.zeros(4, 4)},
    )

    info = load_encoder_weights(_encoder(), ckpt)

    assert info["dropped"] == ["predictor.blocks.0.weight"]
    assert info["missing"] == [] and info["unexpected"] == []
