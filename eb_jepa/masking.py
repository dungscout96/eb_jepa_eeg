"""V-JEPA multi-block masking for EEG tokens.

Generates contiguous 2D block masks on the [C, P] grid (channels × patches-per-window),
replicated identically across all T windows.

Key invariant: if (channel c, patch p) is masked, it is masked for ALL T windows.
This prevents temporal information leakage from EEG's high autocorrelation.

Token grid: [C, T, P] — 3D. Flattened using (c, t, p) ordering:
    token_idx = c * (T * P) + t * P + p
"""

import random
from dataclasses import dataclass

import torch


@dataclass
class MaskResult:
    """Result of mask generation.

    Attributes:
        context_mask: [C*T*P] bool — True for context (visible) tokens, False for masked
        pred_masks: list of [n_pred_i] int64 tensors — flat token indices for each prediction block
        n_total_tokens: total number of tokens in the grid
        horizons: optional [n_total_tokens] int64 — per-token "horizon" (window-distance from
            last context window). Used by Cell L (multi-horizon predictor); 0 for non-cross-
            time mask collators that don't have a meaningful horizon axis.
    """
    context_mask: torch.Tensor
    pred_masks: list[torch.Tensor]
    n_total_tokens: int
    horizons: torch.Tensor = None


class MultiBlockMaskCollator:
    """Generates multi-block masks for V-JEPA training on EEG tokens.

    Each mask block is a contiguous rectangle on the [C, P] grid (channels × patches-per-window),
    replicated across all T windows. Two types of blocks:
    - Short-range: smaller channel/patch extent
    - Long-range: larger channel/patch extent

    The same mask is used for all samples in a batch.

    Args:
        n_channels: Number of EEG channels (e.g., 129)
        n_windows: Number of temporal windows (e.g., 16)
        n_patches_per_window: Patches per window per channel (e.g., 1 or 2)
        n_pred_masks_short: Number of short-range prediction mask blocks
        n_pred_masks_long: Number of long-range prediction mask blocks
        short_channel_scale: (min, max) fraction of channels per short mask
        short_patch_scale: (min, max) fraction of patches-per-window per short mask
        long_channel_scale: (min, max) fraction of channels per long mask
        long_patch_scale: (min, max) fraction of patches-per-window per long mask
        min_context_fraction: Minimum fraction of (C, P) cells that must remain as context
    """

    def __init__(
        self,
        n_channels: int = 129,
        n_windows: int = 16,
        n_patches_per_window: int = 1,
        n_pred_masks_short: int = 2,
        n_pred_masks_long: int = 2,
        short_channel_scale: tuple[float, float] = (0.08, 0.15),
        short_patch_scale: tuple[float, float] = (0.3, 0.6),
        long_channel_scale: tuple[float, float] = (0.15, 0.35),
        long_patch_scale: tuple[float, float] = (0.5, 1.0),
        min_context_fraction: float = 0.15,
    ):
        self.n_channels = n_channels
        self.n_windows = n_windows
        self.n_patches_per_window = n_patches_per_window
        self.n_pred_masks_short = n_pred_masks_short
        self.n_pred_masks_long = n_pred_masks_long
        self.short_channel_scale = short_channel_scale
        self.short_patch_scale = short_patch_scale
        self.long_channel_scale = long_channel_scale
        self.long_patch_scale = long_patch_scale
        self.min_context_fraction = min_context_fraction

        self.n_total_tokens = n_channels * n_windows * n_patches_per_window
        self.n_cp_cells = n_channels * n_patches_per_window  # cells on [C, P] grid

    def _sample_block_size(
        self, channel_scale: tuple[float, float], patch_scale: tuple[float, float]
    ) -> tuple[int, int]:
        """Sample a block size on the [C, P] grid.

        Returns:
            (ch_size, p_size) — number of channels and patches in the block
        """
        ch_min = max(1, int(self.n_channels * channel_scale[0]))
        ch_max = max(ch_min, int(self.n_channels * channel_scale[1]))
        ch_size = random.randint(ch_min, ch_max)

        p_min = max(1, int(self.n_patches_per_window * patch_scale[0]))
        p_max = max(p_min, min(self.n_patches_per_window, int(self.n_patches_per_window * patch_scale[1])))
        p_size = random.randint(p_min, p_max)

        return ch_size, p_size

    def _block_to_flat_indices(
        self, ch_start: int, ch_size: int, p_start: int, p_size: int
    ) -> torch.Tensor:
        """Convert a 2D block on [C, P] to flat token indices, replicated across T.

        Flat index = c * (T * P) + t * P + p
        """
        T = self.n_windows
        P = self.n_patches_per_window
        indices = []
        for c in range(ch_start, ch_start + ch_size):
            for t in range(T):
                for p in range(p_start, p_start + p_size):
                    indices.append(c * (T * P) + t * P + p)
        return torch.tensor(indices, dtype=torch.long)

    def __call__(self) -> MaskResult:
        """Generate masks for one batch.

        Returns:
            MaskResult with context_mask and pred_masks
        """
        C = self.n_channels
        P = self.n_patches_per_window
        max_masked_cells = int(self.n_cp_cells * (1 - self.min_context_fraction))

        # Track which (c, p) cells are masked on the 2D grid
        masked_cp = set()
        pred_blocks = []  # list of (ch_start, ch_size, p_start, p_size)

        # Sample short-range blocks
        for _ in range(self.n_pred_masks_short):
            ch_size, p_size = self._sample_block_size(self.short_channel_scale, self.short_patch_scale)
            ch_start = random.randint(0, C - ch_size)
            p_start = random.randint(0, P - p_size) if P > p_size else 0

            # Check if adding this block would exceed max masked
            new_cells = {(c, p) for c in range(ch_start, ch_start + ch_size)
                         for p in range(p_start, p_start + p_size)}
            if len(masked_cp | new_cells) <= max_masked_cells:
                masked_cp |= new_cells
                pred_blocks.append((ch_start, ch_size, p_start, p_size))

        # Sample long-range blocks
        for _ in range(self.n_pred_masks_long):
            ch_size, p_size = self._sample_block_size(self.long_channel_scale, self.long_patch_scale)
            ch_start = random.randint(0, C - ch_size)
            p_start = random.randint(0, P - p_size) if P > p_size else 0

            new_cells = {(c, p) for c in range(ch_start, ch_start + ch_size)
                         for p in range(p_start, p_start + p_size)}
            if len(masked_cp | new_cells) <= max_masked_cells:
                masked_cp |= new_cells
                pred_blocks.append((ch_start, ch_size, p_start, p_size))

        # Convert blocks to flat token indices
        pred_masks = []
        for ch_start, ch_size, p_start, p_size in pred_blocks:
            indices = self._block_to_flat_indices(ch_start, ch_size, p_start, p_size)
            pred_masks.append(indices)

        # Build context mask: True for visible, False for masked
        context_mask = torch.ones(self.n_total_tokens, dtype=torch.bool)
        if pred_masks:
            all_masked_indices = torch.cat(pred_masks).unique()
            context_mask[all_masked_indices] = False

        return MaskResult(
            context_mask=context_mask,
            pred_masks=pred_masks,
            n_total_tokens=self.n_total_tokens,
        )


class ContiguousTimeMaskCollator:
    """Cross-Time JEPA mask (Cell J / Brain-JEPA "Cross-Time" geometry).

    Masks ALL channels × ALL patches at a contiguous block of windows;
    keeps the remaining windows fully visible as context. Forces the encoder
    to capture temporal evolution within a clip — the dimension CorrCA does
    NOT trivialize, since CorrCA operates per-time-point.

    Token grid: [C, T, P] flattened by (c, t, p) order:
        token_idx = c * (T * P) + t * P + p

    Args:
        n_channels: number of EEG channels (post-CorrCA, e.g., 5)
        n_windows: number of temporal windows in a clip (e.g., 4)
        n_patches_per_window: patches per window (e.g., 12)
        mask_window_fraction: fraction of windows to mask (e.g., 0.5 → mask half)
        mask_position: where to put the masked block: "tail" (predict future from past),
            "head" (predict past from future), or "random" (sample location).
    """

    def __init__(
        self,
        n_channels: int = 5,
        n_windows: int = 4,
        n_patches_per_window: int = 12,
        mask_window_fraction: float = 0.5,
        mask_position: str = "tail",
    ):
        if mask_position not in ("tail", "head", "random"):
            raise ValueError(f"mask_position must be tail|head|random, got {mask_position}")
        self.n_channels = n_channels
        self.n_windows = n_windows
        self.n_patches_per_window = n_patches_per_window
        self.mask_window_fraction = float(mask_window_fraction)
        self.mask_position = mask_position
        self.n_total_tokens = n_channels * n_windows * n_patches_per_window
        n_mask = max(1, int(round(n_windows * self.mask_window_fraction)))
        if n_mask >= n_windows:
            raise ValueError(
                f"mask_window_fraction={mask_window_fraction} → {n_mask}/{n_windows} masked; "
                "must leave at least one context window"
            )
        self.n_mask_windows = n_mask

    def __call__(self) -> MaskResult:
        T = self.n_windows
        n_mask = self.n_mask_windows
        if self.mask_position == "tail":
            t_start = T - n_mask
            last_ctx_window = t_start - 1
        elif self.mask_position == "head":
            t_start = 0
            last_ctx_window = t_start + n_mask
        else:
            t_start = random.randint(0, T - n_mask)
            # for random, "last_ctx_window" reference is the closest context window
            # before the masked block (or after, if mask is at head).
            last_ctx_window = t_start - 1 if t_start > 0 else t_start + n_mask
        masked_windows = list(range(t_start, t_start + n_mask))

        C = self.n_channels
        P = self.n_patches_per_window
        indices = []
        for c in range(C):
            for t in masked_windows:
                for p in range(P):
                    indices.append(c * (T * P) + t * P + p)
        pred_masks = [torch.tensor(indices, dtype=torch.long)]

        context_mask = torch.ones(self.n_total_tokens, dtype=torch.bool)
        context_mask[pred_masks[0]] = False

        # Compute per-token horizons (window-distance from last_ctx_window).
        # For Cell L's horizon-conditioned predictor; harmless for Cell J use.
        horizons = torch.zeros(self.n_total_tokens, dtype=torch.long)
        for c in range(C):
            for t in range(T):
                k = abs(t - last_ctx_window)
                for p in range(P):
                    horizons[c * (T * P) + t * P + p] = k

        return MaskResult(
            context_mask=context_mask,
            pred_masks=pred_masks,
            n_total_tokens=self.n_total_tokens,
            horizons=horizons,
        )
