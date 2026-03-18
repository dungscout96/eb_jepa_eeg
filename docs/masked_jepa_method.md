# EEG-JEPA: Masked Joint Embedding Predictive Architecture for EEG

## Overview

We adapt the V-JEPA framework (Bardes et al., 2024) to self-supervised EEG representation learning, combining the REVE tokenization scheme (El Ouahidi et al., 2025) with a spatio-temporal block masking strategy designed for the unique properties of multi-channel EEG signals.

## Tokenization

We adopt REVE's patch-based tokenization for converting raw EEG into a sequence of tokens. Given a multi-channel EEG input of shape `[B, T, C, W]` where `B` is batch size, `T` is the number of temporal windows, `C` is the number of channels, and `W` is the number of timepoints per window:

1. **Patch extraction**: Each window's signal is segmented into overlapping temporal patches using `torch.unfold` with configurable `patch_size` and `patch_overlap`. For channel `c`, window `t`, this produces `P` patches per channel per window, where `P = floor((W - patch_size) / (patch_size - patch_overlap)) + 1`.

2. **Linear projection**: Each patch of raw values is projected to an embedding space via a learned linear layer: `patch ∈ R^{patch_size} → token ∈ R^{embed_dim}`.

3. **4D positional encoding**: Each token is assigned a 4D coordinate `(x, y, z, t)` where `(x, y, z)` are the 3D spatial coordinates of its EEG channel (from a standardized electrode position bank) and `t = window_index * P + patch_index` is the temporal index. The positional encoding combines:
   - **Fourier embedding**: Sinusoidal encoding at multiple frequencies for smooth interpolation to unseen positions
   - **MLP embedding**: A learned linear projection from `R^4 → R^{embed_dim}` with GELU activation and LayerNorm
   - Both are summed and normalized via LayerNorm

4. **Token sequence**: All tokens across channels, windows, and patches are flattened into a single sequence of length `C × T × P` using `(c, t, p)` ordering (token index = `c * T * P + t * P + p`). This jointly encodes spatial and temporal structure.

### Token Grid

The token grid is conceptually 3D: `[C, T, P]`. With the default configuration (`patch_size=200`, `window_size=200 samples`), `P=1`, yielding `C × T` tokens (e.g., 129 × 16 = 2,064 tokens). Smaller patch sizes increase `P` proportionally.

## Transformer Encoder

Tokens are processed by a transformer encoder following REVE's architecture:

- **RMSNorm** pre-normalization (more stable than LayerNorm)
- **Multi-head self-attention** via scaled dot-product attention (flash attention when available)
- **GEGLU feedforward** networks (gated GELU activation)
- **Residual connections** around both attention and feedforward sublayers
- Configurable depth, number of heads, head dimension, and FFN ratio

## Block Masking Strategy

We introduce a masking strategy tailored to EEG's spatial and temporal redundancy. Masks are defined as contiguous 2D rectangular blocks on the `[C, P]` grid (channels × patches-per-window), replicated identically across all `T` windows.

### Temporal Replication Invariant

**Key constraint**: If token `(channel c, patch p)` is masked, it is masked for **all** `T` windows. This prevents information leakage: without this constraint, the model could trivially predict a masked channel's activity at one timepoint by observing the same channel at adjacent timepoints, due to EEG's high temporal autocorrelation.

### Short-Range and Long-Range Masks

Following V-JEPA, we sample multiple prediction mask blocks of two types:

- **Short-range masks**: Smaller blocks covering fewer channels and/or patches (e.g., 8-15% of channels, 30-60% of patches). These encourage learning fine-grained spatial relationships between nearby electrodes.

- **Long-range masks**: Larger blocks covering more channels and/or patches (e.g., 15-35% of channels, 50-100% of patches). These force the model to predict broad spatial patterns from distant context, learning global brain dynamics.

### Mask Generation Algorithm

1. Sample `M_short` short-range + `M_long` long-range rectangular blocks on the `[C, P]` grid
2. For each block: uniformly sample `(ch_start, ch_size)` and `(p_start, p_size)` from configured scale ranges
3. Compute the union of all prediction blocks as the masked set
4. Enforce a minimum context fraction (default 15%) — if exceeded, discard the largest block
5. **Context** = all tokens NOT in any prediction block (replicated across all T windows)
6. The same mask is used for all samples in the batch

## Architecture

### Context Encoder

The context encoder (the online network) receives only **unmasked (context) tokens** and encodes them through the transformer. Masked tokens are excluded from the input entirely, reducing computation proportionally to the masking ratio.

### Target Encoder

The target encoder is an exponential moving average (EMA) copy of the context encoder. It receives **all tokens** (no masking) and produces target representations. Its parameters are updated as:

```
θ_target ← m · θ_target + (1 - m) · θ_context
```

where momentum `m` follows a cosine schedule from 0.996 to 1.0 over training.

### Predictor

A lightweight transformer predictor maps context representations to predicted representations at masked positions:

1. Context tokens receive their positional embeddings
2. A learned **mask token** (single shared vector) is expanded to all target positions and receives the target positional embeddings
3. Context and mask tokens are concatenated and processed by a shallow transformer (default depth=2, vs depth=4 for the encoder)
4. The last `n_pred` output tokens (corresponding to mask token positions) are projected as predictions

### Loss

The training objective combines:

- **Prediction loss**: Mean squared error between predictor output and target encoder representations at masked positions: `L_pred = MSE(predictor(context), target_encoder(all)_at_masked)`

- **Variance-Covariance regularization** (VICReg-style): Applied to context encoder output to prevent representation collapse:
  - Hinge variance loss: ensures each feature dimension maintains minimum standard deviation
  - Covariance loss: decorrelates feature dimensions

## Evaluation Probes

Online linear probes are trained alongside the self-supervised objective (with frozen encoder gradients) to monitor representation quality:

1. **Regression probe**: Predicts continuous movie features (contrast, luminance, entropy, scene naturalness) via MSE loss
2. **Classification probe**: Predicts binary movie feature labels via BCE loss

For probes, the full (unmasked) encoder output is mean-pooled over channels and patches per window to produce per-window representations: `[B, C*T*P, D] → [B, T, D]`.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 64 | Token embedding dimension |
| `encoder_depth` | 4 | Transformer encoder layers |
| `heads` | 4 | Attention heads |
| `head_dim` | 16 | Dimension per head |
| `patch_size` | 200 | Temporal patch size (samples) |
| `patch_overlap` | 20 | Patch overlap (samples) |
| `predictor_depth` | 2 | Predictor transformer layers |
| `n_pred_masks_short` | 2 | Short-range mask blocks |
| `n_pred_masks_long` | 2 | Long-range mask blocks |
| `short_channel_scale` | [0.08, 0.15] | Channel fraction for short masks |
| `long_channel_scale` | [0.15, 0.35] | Channel fraction for long masks |
| `min_context_fraction` | 0.15 | Minimum visible token fraction |
| `ema_momentum` | 0.996 → 1.0 | Target encoder EMA (cosine schedule) |

## References

- Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., & Balestriero, R. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video. *arXiv:2404.08471*.
- El Ouahidi, Y., Lys, J., Tholke, P., Farrugia, N., Pasdeloup, B., Gripon, V., Jerbi, K., & Lioi, G. (2025). REVE: A Foundation Model for EEG. *NeurIPS 2025*.
