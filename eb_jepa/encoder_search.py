"""Mutable post-tokenization encoder body for autoresearch architecture search.

This is the ONLY file the autoresearch loop edits.

Contract
--------
Export `build_encoder_body(embed_dim: int) -> nn.Module` such that the returned
module has forward signature:

    Input:  [B, N, embed_dim]   (N can be < total tokens when masking is active)
    Output: [B, N, embed_dim]   (must preserve N; dim must stay embed_dim)

The body is plugged in at `eb_jepa.architectures.EEGEncoderTokens.transformer`,
slotted between tokenization (Linear patch embed + 4D Fourier pos) and pooling
(`pool_to_windows`). Tokenization, masking, predictor, and pool layers are all
LOCKED for the autoresearch loop — do not change them here.

Baseline (this commit) reproduces the current production best:
REVETransformerBackbone, depth=2, heads=4, head_dim=16, mlp_dim_ratio=2.66.
"""

from torch import nn

from eb_jepa.architectures import REVETransformerBackbone


def build_encoder_body(embed_dim: int) -> nn.Module:
    """Build the post-tokenization encoder body.

    Returns a module mapping [B, N, embed_dim] -> [B, N, embed_dim].
    """
    return REVETransformerBackbone(
        dim=embed_dim,
        depth=2,
        heads=4,
        head_dim=16,
        mlp_dim=int(embed_dim * 2.66),
        geglu=True,
    )
