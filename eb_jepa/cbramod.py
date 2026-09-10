"""CBraMod (Wang et al., ICLR 2025) as a drop-in token encoder for the CLIP path.

Why this exists. REVE-base -- the only warm start the subject-scaling sweeps
have used -- was pretrained on HBN R1-R9, which includes the R6 test release,
so its warm-started arm cannot be described as evaluated on unseen subjects.
CBraMod was pretrained on TUEG alone (19-channel 10-20 clinical EEG, 200 Hz,
1 s patches), so a CBraMod warm start carries no HBN exposure at all. See
experiments/snr_scaling/PLAN_cbramod.md for why it was chosen over LUNA.

What this is. A faithful re-implementation of ``models/cbramod.py`` and
``models/criss_cross_transformer.py`` from github.com/wjq-learning/CBraMod
(MIT), reorganised to expose the same token interface as
:class:`eb_jepa.architectures.EEGEncoderTokens` -- ``tokenize`` /
``encode_tokens`` / ``pool_to_windows`` plus the shape attributes the CLIP
models and evaluators read -- so nothing downstream has to know which backbone
it is looking at. Submodule and parameter names inside ``patch_embedding`` and
each transformer layer are kept IDENTICAL to the upstream module so the
released ``pretrained_weights.pth`` maps onto this class by a prefix rename
alone (see ``prepare_cbramod_checkpoint.py``).

Two things about the upstream model that shape everything here:

* **The token grid is 2-D and must stay 2-D.** Criss-cross attention runs
  spatial attention across channels at each time patch and temporal attention
  across patches within each channel, on half the feature dim each. It needs
  the full [C, P] grid, so the JEPA-style "drop masked tokens" path cannot be
  supported; ``encode_tokens`` refuses a mask rather than silently attending
  over a broken grid.
* **Positional information is a convolution over the grid, not coordinates.**
  The asymmetric conditional positional encoding is a depthwise (19, 7) conv
  over (channel, patch). Channel identity therefore comes only from index
  order -- the GSN-HydroCel-129 numbering is spatially coherent enough for a
  19-wide kernel to see neighbours, but no electrode coordinate ever enters
  the model, which is the main respect in which it is a weaker spatial prior
  than REVE's 4-D Fourier positions.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Upstream constants. The patch encoder's first conv (kernel 49, stride 25,
# pad 24) turns a 200-sample patch into 8 steps x 25 filters = 200 features,
# which is what makes d_model=200 and patch_size=200 a matched pair rather than
# two free knobs. Changing either without the other breaks the reshape.
CBRAMOD_PATCH_SIZE = 200
CBRAMOD_D_MODEL = 200
_TIME_CONV_FILTERS = 25
_TIME_CONV_KERNEL = 49
_TIME_CONV_STRIDE = 25
_TIME_CONV_PAD = 24
_SPECTRAL_BINS = CBRAMOD_PATCH_SIZE // 2 + 1  # rfft of a 200-sample patch
_ACPE_KERNEL = (19, 7)
_ACPE_PAD = (9, 3)


class CBraModPatchEmbedding(nn.Module):
    """Time-conv + spectral patch encoder with asymmetric conditional PE.

    Input  ``[B, C, N, patch]`` (N = patches per channel, in time order).
    Output ``(tokens, pos)`` each ``[B, C, N, d_model]`` where ``tokens``
    already includes ``pos``; ``pos`` is returned separately only so the
    outer encoder can honour the ``tokenize`` contract.
    """

    def __init__(
        self,
        d_model: int = CBRAMOD_D_MODEL,
        patch_size: int = CBRAMOD_PATCH_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_steps = (
            patch_size + 2 * _TIME_CONV_PAD - _TIME_CONV_KERNEL
        ) // _TIME_CONV_STRIDE + 1
        if n_steps * _TIME_CONV_FILTERS != d_model:
            raise ValueError(
                f"CBraMod patch encoder maps patch_size={patch_size} to "
                f"{n_steps * _TIME_CONV_FILTERS} features, not d_model={d_model}. "
                "The released weights fix both at 200."
            )
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_steps = n_steps

        # Names below are the upstream state_dict names. Do not rename.
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=_ACPE_KERNEL,
                stride=(1, 1),
                padding=_ACPE_PAD,
                groups=d_model,
            ),
        )
        # Pretraining mask token; unused here but kept so the released
        # checkpoint loads with zero missing / unexpected keys under the strict
        # guard in load_encoder_weights.
        self.register_buffer("mask_encoding", torch.zeros(patch_size))
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                1,
                _TIME_CONV_FILTERS,
                kernel_size=(1, _TIME_CONV_KERNEL),
                stride=(1, _TIME_CONV_STRIDE),
                padding=(0, _TIME_CONV_PAD),
            ),
            nn.GroupNorm(5, _TIME_CONV_FILTERS),
            nn.GELU(),
            nn.Conv2d(
                _TIME_CONV_FILTERS,
                _TIME_CONV_FILTERS,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, _TIME_CONV_FILTERS),
            nn.GELU(),
            nn.Conv2d(
                _TIME_CONV_FILTERS,
                _TIME_CONV_FILTERS,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, _TIME_CONV_FILTERS),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(_SPECTRAL_BINS, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, N, P = x.shape
        flat = x.contiguous().view(B, 1, C * N, P)
        emb = self.proj_in(flat)  # [B, 25, C*N, 8]
        emb = emb.permute(0, 2, 1, 3).contiguous().view(B, C, N, self.d_model)

        spectral = torch.fft.rfft(flat.view(B * C * N, P), dim=-1, norm="forward")
        spectral = torch.abs(spectral).view(B, C, N, _SPECTRAL_BINS)
        emb = emb + self.spectral_proj(spectral)

        pos = self.positional_encoding(emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return emb + pos, pos


class CrissCrossEncoderLayer(nn.Module):
    """Pre-norm transformer layer whose attention is split in two halves:
    the first ``d_model // 2`` features attend across channels at a fixed
    patch, the second half across patches within a fixed channel."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if d_model % 2 or nhead % 2:
            raise ValueError("criss-cross attention needs even d_model and nhead")
        half = d_model // 2
        self.self_attn_s = nn.MultiheadAttention(
            half, nhead // 2, dropout=dropout, batch_first=True
        )
        self.self_attn_t = nn.MultiheadAttention(
            half, nhead // 2, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._sa_block(self.norm1(x))
        return x + self._ff_block(self.norm2(x))

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, D = x.shape
        half = D // 2
        xs = x[..., :half].transpose(1, 2).contiguous().view(B * N, C, half)
        xt = x[..., half:].contiguous().view(B * C, N, half)
        xs = self.self_attn_s(xs, xs, xs, need_weights=False)[0]
        xs = xs.contiguous().view(B, N, C, half).transpose(1, 2)
        xt = self.self_attn_t(xt, xt, xt, need_weights=False)[0]
        xt = xt.contiguous().view(B, C, N, half)
        return self.dropout1(torch.cat((xs, xt), dim=3))

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(x)))))


class CrissCrossEncoder(nn.Module):
    """Stack of :class:`CrissCrossEncoderLayer`; ``layers`` mirrors upstream."""

    def __init__(self, layer: CrissCrossEncoderLayer, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _cbramod_weights_init(m: nn.Module) -> None:
    """Upstream ``_weights_init``: kaiming-normal on every nn.Linear (which
    includes attention out_proj). Convs and LayerNorms keep torch defaults."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class CBraModEncoderTokens(nn.Module):
    """CBraMod backbone behind the ``EEGEncoderTokens`` token interface.

    Input:  ``[B, T, C, W]`` raw (z-scored) EEG, ``W = n_times``.
    Tokens: ``[B, C*T*P, d_model]`` in (c, t, p) order, ``P = n_times // 200``,
            i.e. exactly the order ``pool_to_windows`` and the masking code
            already assume for the REVE encoder.

    ``input_scale`` multiplies the raw input before the patch encoder. CBraMod
    was pretrained on microvolts divided by 100; this repo feeds per-channel
    z-scores, whose unit variance is several times that scale. The spectral
    branch is linear in amplitude while the time branch is GroupNorm-ed, so a
    scale mismatch shifts the balance between the two. Default 1.0 keeps the
    module a pure port; the frozen sweep config sets it explicitly.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_windows: int = 1,
        d_model: int = CBRAMOD_D_MODEL,
        depth: int = 12,
        heads: int = 8,
        dim_feedforward: int = 800,
        patch_size: int = CBRAMOD_PATCH_SIZE,
        dropout: float = 0.1,
        input_scale: float = 1.0,
    ):
        super().__init__()
        if n_times % patch_size:
            raise ValueError(
                f"n_times ({n_times}) must be a multiple of patch_size ({patch_size}); "
                "CBraMod patches tile the window with no overlap."
            )
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_windows = n_windows
        self.embed_dim = d_model
        self.patch_size = patch_size
        self.patch_overlap = 0
        self.input_scale = float(input_scale)
        self.n_patches_per_window = n_times // patch_size
        self.n_tokens_per_window = n_chans * self.n_patches_per_window
        self.total_patches_per_channel = n_windows * self.n_patches_per_window

        self.patch_embedding = CBraModPatchEmbedding(d_model, patch_size, dropout)
        layer = CrissCrossEncoderLayer(d_model, heads, dim_feedforward, dropout)
        self.transformer = CrissCrossEncoder(layer, depth)
        self.apply(_cbramod_weights_init)

    # -- helpers ----------------------------------------------------------
    def _to_grid(self, eeg: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, W]`` -> ``[B, C, T*P, patch]`` with (t, p) time order."""
        B, T, C, W = eeg.shape
        if W != self.n_times or C != self.n_chans:
            raise ValueError(
                f"expected [B, T, {self.n_chans}, {self.n_times}], got {tuple(eeg.shape)}"
            )
        x = eeg * self.input_scale if self.input_scale != 1.0 else eeg
        return x.permute(0, 2, 1, 3).reshape(
            B, C, T * self.n_patches_per_window, self.patch_size
        )

    @staticmethod
    def _flatten(grid: torch.Tensor) -> torch.Tensor:
        B, C, N, D = grid.shape
        return grid.reshape(B, C * N, D)

    # -- token interface --------------------------------------------------
    def tokenize(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, pos = self.patch_embedding(self._to_grid(eeg))
        return self._flatten(tokens), self._flatten(pos)

    def encode_tokens(
        self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            raise NotImplementedError(
                "CBraModEncoderTokens cannot drop tokens: criss-cross attention "
                "needs the full [channel, patch] grid. It serves the CLIP path only."
            )
        tokens, _ = self.patch_embedding(self._to_grid(eeg))
        return self._flatten(self.transformer(tokens))

    def pool_to_windows(
        self, tokens: torch.Tensor, keep_channels: bool = False
    ) -> torch.Tensor:
        """Same contract as ``EEGEncoderTokens.pool_to_windows``."""
        B = tokens.shape[0]
        C, T, P, D = (
            self.n_chans,
            self.n_windows,
            self.n_patches_per_window,
            self.embed_dim,
        )
        x = tokens.view(B, C, T, P, D)
        if keep_channels:
            x = x.mean(dim=3)
            x = x.permute(0, 2, 1, 3).reshape(B, T, C * D)
        else:
            x = x.mean(dim=(1, 3))
        return x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
