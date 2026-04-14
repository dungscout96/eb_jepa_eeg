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
    """
    context_mask: torch.Tensor
    pred_masks: list[torch.Tensor]
    n_total_tokens: int


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


class TemporalDominantMaskCollator:
    """Temporal-dominant masking that defeats the spatial interpolation shortcut.

    Primary axis: mask entire time windows across ALL channels (no spatial
    neighbors at masked timepoints — forces temporal prediction).
    Secondary axis: mask some channels in visible windows (mild spatial task).

    Token grid: [C, T, P] flattened as token_idx = c * (T*P) + t*P + p.

    References:
        - EEG2Rep (KDD 2024): 50% temporal masking optimal for EEG SSL
        - Laya (arXiv 2026): temporal-masked LeJEPA for EEG
        - REVE (NeurIPS 2025): block masking needed against spatial redundancy
    """

    def __init__(
        self,
        n_channels: int,
        n_windows: int,
        n_patches_per_window: int,
        temporal_mask_ratio: float = 0.5,
        spatial_mask_ratio: float = 0.3,
        min_visible_windows: int = 1,
    ):
        self.C = n_channels
        self.T = n_windows
        self.P = n_patches_per_window
        self.n_total_tokens = n_channels * n_windows * n_patches_per_window
        self.temporal_mask_ratio = temporal_mask_ratio
        self.spatial_mask_ratio = spatial_mask_ratio
        self.min_visible_windows = min_visible_windows

    def __call__(self) -> MaskResult:
        C, T, P = self.C, self.T, self.P

        # Phase 1: Temporal masking — mask k entire time windows across ALL channels
        n_temporal_mask = max(1, int(T * self.temporal_mask_ratio))
        n_temporal_mask = min(n_temporal_mask, T - self.min_visible_windows)
        masked_windows = sorted(random.sample(range(T), k=n_temporal_mask))
        visible_windows = [t for t in range(T) if t not in masked_windows]

        # All tokens at masked windows → prediction target
        temporal_pred_indices = []
        for t in masked_windows:
            for c in range(C):
                for p in range(P):
                    temporal_pred_indices.append(c * (T * P) + t * P + p)

        # Phase 2: Spatial masking in visible windows — mask some channels
        n_chan_mask = max(1, int(C * self.spatial_mask_ratio))
        masked_channels = sorted(random.sample(range(C), k=n_chan_mask))

        spatial_pred_indices = []
        for t in visible_windows:
            for c in masked_channels:
                for p in range(P):
                    spatial_pred_indices.append(c * (T * P) + t * P + p)

        # Build prediction masks (two blocks: temporal + spatial)
        pred_masks = []
        if temporal_pred_indices:
            pred_masks.append(torch.tensor(temporal_pred_indices, dtype=torch.long))
        if spatial_pred_indices:
            pred_masks.append(torch.tensor(spatial_pred_indices, dtype=torch.long))

        # Context mask: everything not in either prediction set
        context_mask = torch.ones(self.n_total_tokens, dtype=torch.bool)
        if pred_masks:
            all_masked = torch.cat(pred_masks).unique()
            context_mask[all_masked] = False

        return MaskResult(
            context_mask=context_mask,
            pred_masks=pred_masks,
            n_total_tokens=self.n_total_tokens,
        )
