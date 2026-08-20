"""Shared utilities for neural network initialization and common patterns."""

import math

import torch.nn as nn
from einops import rearrange


def init_module_weights(m, std: float = 0.02):
    """
    Initialize weights for common layer types using truncated normal distribution.

    This is a unified weight initialization function used across the codebase.
    Apply it via module.apply(init_module_weights) or as a method wrapper.

    Args:
        m: PyTorch module to initialize
        std: Standard deviation for truncated normal initialization (default: 0.02)
    """
    if isinstance(
        m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    ):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def apply_depth_scaled_residual_init(backbone, depth: int, std: float = 0.02):
    """Rescale each residual branch's OUTPUT projection by 1/sqrt(2 * depth).

    A pre-norm transformer adds every block's output straight onto the residual
    stream, so the stream's variance grows roughly linearly with depth. Left
    alone this caps trainable depth: CaiT reports 12 -> 36 layers costing ~10
    points top-1 at fixed lr/wd, and REVE's own pretraining uses
    ``init_method: full_megatron`` over encoder_depth + decoder_depth for exactly
    this reason.

    The two projections that write into the residual stream are
    ``REVEAttention.to_out`` and the second Linear of ``REVEFeedForward.net``.
    Only those are touched; QKV and the FFN's expanding Linear keep their
    default init.

    Off by default. Turning it on changes the initial weights of every depth,
    including 22 -- so an arm that enables it is only comparable to another arm
    that also enables it.
    """
    scale = std / math.sqrt(2.0 * max(depth, 1))
    n = 0
    for attn, ff in backbone.layers:
        nn.init.normal_(attn.to_out.weight, mean=0.0, std=scale)
        n += 1
        for m in reversed(list(ff.net)):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=scale)
                n += 1
                break
    return n


class TemporalBatchMixin:
    """
    Mixin class that handles automatic temporal batching for 4D/5D tensors.

    This mixin provides a unified forward() method that:
    - For 5D tensors [B, C, T, H, W]: flattens temporal dim, applies _forward(), restores shape
    - For 4D tensors [B, C, H, W]: directly applies _forward()

    Subclasses must implement _forward(self, x) for 4D tensors.
    """

    def _forward(self, x):
        """
        Process 4D tensor [B, C, H, W]. Must be implemented by subclasses.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C_out, H_out, W_out]
        """
        raise NotImplementedError("Subclasses must implement _forward()")

    def forward(self, x):
        """
        Forward pass supporting both 4D and 5D tensors.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, C, T, H, W]

        Returns:
            Output tensor with same batch and temporal dimensions as input
        """
        assert x.ndim in [
            4,
            5,
        ], "Supports only 4D [B, C, H, W] or 5D [B, C, T, H, W] tensors"
        if x.ndim == 5:
            b = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            out = self._forward(x)
            out = rearrange(out, "(b t) c h w -> b c t h w", b=b)
            return out
        else:
            return self._forward(x)