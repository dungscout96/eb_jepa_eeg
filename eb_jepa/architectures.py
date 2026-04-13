import importlib
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import average_precision_score
from torch.nn.attention import SDPBackend, sdpa_kernel

from eb_jepa.nn_utils import TemporalBatchMixin, init_module_weights


# ===========================================================================
# REVE Transformer Components
# Adapted from braindecode (BSD-3-Clause License)
# Original: El Ouahidi et al. (2025), "REVE: A Foundation Model for EEG"
# https://github.com/braindecode/braindecode/blob/master/braindecode/models/reve.py
# ===========================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more stable than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GEGLU(nn.Module):
    """Gated GELU activation: splits input, uses GELU-gated half to modulate other."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class REVEFeedForward(nn.Module):
    """Transformer feedforward sublayer: RMSNorm → Linear → GEGLU → Linear."""

    def __init__(self, dim: int, hidden_dim: int, geglu: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if geglu else hidden_dim, bias=False),
            GEGLU() if geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REVEClassicalAttention(nn.Module):
    """Multi-head attention using PyTorch SDPA (flash attention when available)."""

    def __init__(self, heads: int, use_sdpa: bool = True):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.heads = heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (
            rearrange(t, "batch seq (heads dim) -> batch heads seq dim", heads=self.heads)
            for t in (q, k, v)
        )
        if self.use_sdpa:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = q.shape[-1] ** -0.5
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = torch.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)
        out = rearrange(out, "batch heads seq dim -> batch seq (heads dim)")
        return out


class REVEAttention(nn.Module):
    """Self-attention sublayer: RMSNorm → QKV projection → SDPA → output projection."""

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attend = REVEClassicalAttention(self.heads, use_sdpa=True)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


class REVETransformerBackbone(nn.Module):
    """Transformer backbone: stacks of [Attention + FeedForward] with residual connections."""

    def __init__(self, dim, depth, heads, head_dim, mlp_dim, geglu=True):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    REVEAttention(self.dim, heads=heads, head_dim=head_dim),
                    REVEFeedForward(self.dim, mlp_dim, geglu),
                ])
            )

    def forward(self, x) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class FourierEmb4D(nn.Module):
    """4D Fourier positional encoding for (x, y, z, t) coordinates.

    Uses sinusoidal features at multiple frequencies for smooth interpolation
    to unseen positions. The time dimension is scaled by increment_time.
    """

    def __init__(self, dimension: int, freqs: int, increment_time=0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_: torch.Tensor) -> torch.Tensor:
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        input_shape = positions.shape
        batch_dims = list(input_shape[:-1])

        freqs_w = torch.arange(self.freqs).to(positions)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width
        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x
            + positions[..., 1] * p_y
            + positions[..., 2] * p_z
            + positions[..., 3] * p_w
        )
        batch_dims.append(-1)
        loc = loc.view(batch_dims)

        half_dim = self.dimension // 2
        current_dim = loc.shape[-1]
        if current_dim != half_dim:
            if current_dim > half_dim:
                loc = loc[..., :half_dim]
            else:
                raise ValueError(
                    f"Input dimension ({current_dim}) is too small for target "
                    f"embedding dimension ({self.dimension}). Expected at least {half_dim}."
                )

        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Expand position tensor [B, C, 3] by adding time dimension → [B, C*num_patches, 4].

        Each channel position is repeated for each time patch index.
        """
        batch, nchans, _ = pos.shape
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)
        time_values = torch.arange(0, num_patches, 1, device=pos.device).float()
        time_values = time_values.view(1, 1, num_patches, 1).expand(batch, nchans, num_patches, 1)
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)
        pos_with_time = pos_with_time.view(batch, nchans * num_patches, 4)
        return pos_with_time


# ===========================================================================
# Original architectures (unchanged)
# ===========================================================================


class conv3d2(nn.Sequential):
    """Simple 3D convnet with 2 layers."""

    def __init__(self, in_d, h_d, out_d, tk, ts, sk, ss, pad):
        super(conv3d2, self).__init__(
            nn.Conv3d(
                in_d, h_d, kernel_size=(tk, sk, sk), stride=(1, 1, 1), padding=pad
            ),
            nn.ReLU(),
            nn.Conv3d(
                h_d, out_d, kernel_size=(tk, sk, sk), stride=(ts, ss, ss), padding=pad
            ),
        )
        self.apply(init_module_weights)
        self.input_dim = in_d
        self.hidden_dim = h_d
        self.output_dim = out_d
        # t_shift is the index (in the time dimension) of the first output
        # cannot see its coresponding input
        if pad == "valid":
            self.t_shift = 2 * tk - 1
        elif pad == "same":
            self.t_shift = 2 * (tk - 1)
        else:
            raise NameError("invalid padding for con3d2. Must be 'valid' or 'same'")


class ResidualBlock(nn.Module):
    """Standard residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet5(TemporalBatchMixin, nn.Module):
    """
    A lightweight ResNet with 5 layers (2 blocks).
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else torch.nn.Identity()

    def _forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        if self.avg_pool:
            out = out.flatten(1)
        return out


class SimplePredictor(nn.Module):
    """Wrapper that concatenates states and actions channel-wise before prediction."""

    def __init__(self, predictor, context_length):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a):
        return self.predictor(torch.cat([x, a], dim=1))


class StateOnlyPredictor(SimplePredictor):
    """Wrapper for a simple predictor which concatenates states and actions channel wise."""

    def forward(self, x, a):
        # action not used on purpose
        prev_state = x[:, :, :-1]  # [B, C, T-1, H, W]
        next_state = x[:, :, 1:]  # [B, C, T-1, H, W]
        combined_xa = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined_xa)


class ResUNet(TemporalBatchMixin, nn.Module):
    """
    A small UNet with residual encoder blocks and transposed-conv upsampling.
    Channels scale like h, 2h, 4h, 8h. Output keeps the input HxW.
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, is_rnn=False):
        super().__init__()
        self.is_rnn = is_rnn
        # Stem
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.enc1 = ResidualBlock(h_d, h_d, stride=1)  # H, W
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)  # H/2, W/2
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)  # H/4, W/4
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)  # H/8, W/8

        # Decoder upsamples, then fuses skip with a residual block that reduces channels
        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride=1)

        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)

        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)

        # Head
        self.head = nn.Conv2d(h_d, out_d, kernel_size=1)

    @staticmethod
    def _match_size(x, ref):
        # Guards against odd input sizes by resizing the upsample to the skip spatial dims
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return x

    def _forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))

        # Encoder with skips
        s1 = self.enc1(x0)  # h
        s2 = self.enc2(s1)  # 2h
        s3 = self.enc3(s2)  # 4h
        b = self.bott(s3)  # 8h

        # Decoder stage 3
        d3 = self.up3(b)
        d3 = self._match_size(d3, s3)
        d3 = torch.cat([d3, s3], dim=1)  # 4h + 4h = 8h
        d3 = self.dec3(d3)  # → 4h

        # Decoder stage 2
        d2 = self.up2(d3)
        d2 = self._match_size(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)  # 2h + 2h = 4h
        d2 = self.dec2(d2)  # → 2h

        # Decoder stage 1
        d1 = self.up1(d2)
        d1 = self._match_size(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)  # h + h = 2h
        d1 = self.dec1(d1)  # → h

        out = self.head(d1)  # → out_d channels
        return out


class Projector(nn.Module):
    """MLP projector built from a spec string like '256-512-128'."""

    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]  # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)


class DetHead(nn.Module):
    """Detection head that pools features and predicts binary maps."""

    def __init__(self, in_d, h_d, out_d):
        super().__init__()
        self.head = nn.Sequential(conv3d2(in_d, h_d, out_d, 1, 1, 3, 1, "same"))
        self.apply(init_module_weights)

    def forward(self, x):
        """Forward pass on predictor output of shape (B, C, T, H, W)."""
        # (Batch, Feature, Time, Height, Width)
        # [8, 8, T, 8, 8]
        x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
        x = torch.stack(x, 2)
        # [8, T, 8, 8]
        x = self.head(x).squeeze(1)

        return torch.sigmoid(x)

    @torch.no_grad()
    def score(self, preds, targets):

        scores = []
        for T in range(len(preds) - 1):
            x = preds[T]
            x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
            x = torch.stack(x, 2)
            x = self.head(x).squeeze(1)

            y = targets[:, T:]
            x = x[:, T:]

            ap = average_precision_score(
                y.flatten().detach().long().cpu().numpy(),
                x.flatten().detach().cpu().numpy(),
                average="weighted",
            )
            scores.append(ap)

        return scores


class ResnetBlock(nn.Module):
    """ResNet Block."""

    def __init__(self, num_features):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = nn.Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2, 65, 65),
    ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        input_channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=input_channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        # Compute MLP input dimension dynamically
        with torch.no_grad():
            # Create a dummy input (assuming typical input size for this encoder)
            dummy_input = torch.zeros(1, *self.input_shape)  # (1, C, H, W)
            conv_out = dummy_input
            for stack_block in self.stack_blocks:
                conv_out = stack_block(conv_out)  # b c w h
            flattened_dim = conv_out.view(conv_out.size(0), -1).shape[1]  # c * w * h

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)

        if final_ln:
            self.final_ln = nn.LayerNorm(self.mlp_output_dim)
        else:
            self.final_ln = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            out: [B, D, T, 1, 1]
        """

        # [B, C, T, H, W] --> [T, B, C, H, W]
        (
            _,
            _,
            t,
            _,
            _,
        ) = x.shape
        x = x.permute(2, 0, 1, 3, 4)

        features = []

        for i in range(t):

            conv_out = x[i]

            for i, stack_block in enumerate(self.stack_blocks):
                conv_out = stack_block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)

            conv_out = F.relu(conv_out)
            if self.layer_norm:
                conv_out = nn.LayerNorm(conv_out.size()[1:])(conv_out)  # b c w h
            # flatten
            out = conv_out.view(conv_out.size(0), -1)
            out = self.mlp(out)
            out = self.final_ln(out)

            features.append(out)

        features = torch.stack(features, dim=1)

        features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        return features


class RNNPredictor(nn.Module):
    """GRU-based predictor for single-step state propagation."""

    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: Optional[int] = 2,
        num_layers: int = 1,
        final_ln: Optional[torch.nn.Module] = None,
    ):
        super(RNNPredictor, self).__init__()

        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.final_ln = final_ln
        self.is_rnn = True
        self.context_length = 0

    def forward(self, state, action):
        """
        Propagate one step forward.

        Args:
            state: [B, D, 1, 1, 1]
            action: [B, A, 1]
        Returns:
            next_state: [B, D, 1, 1, 1]
        """
        # This only does one step
        rnn_state = state.flatten(1, 4).unsqueeze(0).contiguous()  # [1, B, D]
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous()  # [1, B, A]

        next_state, _ = self.rnn(rnn_input, rnn_state)

        next_state = self.final_ln(next_state)

        return next_state[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused a transition from state_t to state_t_plus_1.
    Used as auxiliary task for representation learning.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1):
        """
        Args:
            state_t: State at time t, shape [B, D]
            state_t_plus_1: State at time t+1, shape [B, D]
        Returns:
            predicted_action: Action predicted to transform state_t to state_t_plus_1, shape [B, A]
        """
        combined_states = torch.cat([state_t, state_t_plus_1], dim=1)
        return self.model(combined_states)

class EEGEncoder(TemporalBatchMixin, nn.Module):
    """
    EEG encoder that wraps a Braindecode model specified by name.
    Supports both 4D [B, 1, C, W] and 5D [B, 1, T, C, W] inputs via TemporalBatchMixin.
    """
    def __init__(self, in_d, h_d, out_d, name: str="REVE", chs_info=None, attention_pooling=False, n_times=None, **encoder_kwargs):
        super().__init__()
        import importlib
        module = importlib.import_module("braindecode.models")
        self.encoder = getattr(module, name)(n_chans=in_d, n_outputs=out_d, n_times=n_times, chs_info=chs_info, attention_pooling=attention_pooling, **encoder_kwargs)

    def _forward(self, x):
        """
        Forward pass for the encoder that handles EEG data input.

        Removes the singleton dimension from EEG input, processes through the encoder,
        and restores the singleton dimensions for compatibility with the computer vision framework.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, C, W] where B is batch size,
                              1 is the singleton input dimension, C is channels, and W is width.

        Returns:
            torch.Tensor: Output tensor of shape [B, C, 1, 1] with restored singleton dimensions
                          for compatibility with CV framework operations.

        Raises:
            ValueError: If input tensor shape[1] is not equal to 1.
        """
        if x.shape[1] != 1:
            raise ValueError(f"Expected input with shape [B, 1, C, W], got {x.shape}")
        out = self.encoder(x.squeeze(1))  # Remove singleton input dim for EEG data vs image
        if out.ndim == 2:
            out = out.unsqueeze(2).unsqueeze(3)  # Add singleton dims back for compatibility with CV framework
        return out

class EEGEncoderTokens(nn.Module):
    """REVE-based EEG encoder exposing token-level representations.

    Processes all T windows jointly as a single token sequence.
    Token grid is 3D: [C, T, P] where P = patches per window per channel.
    Flattened token order: (c, t, p) → index = c * (T*P) + t * P + p.

    Input:  [B, T, C, W] raw EEG
    Output: [B, C*T*P, embed_dim] token representations
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        embed_dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        head_dim: int = 16,
        n_windows: int = 16,
        patch_size: int = 200,
        patch_overlap: int = 20,
        freqs: int = 4,
        chs_info=None,
        mlp_dim_ratio: float = 2.66,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.embed_dim = embed_dim
        self.n_windows = n_windows
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.freqs = freqs

        # Compute patches per window (matches torch.unfold output count)
        step = patch_size - patch_overlap
        assert n_times >= patch_size, f"n_times ({n_times}) must be >= patch_size ({patch_size})"
        self.n_patches_per_window = (n_times - patch_size) // step + 1

        self.n_tokens_per_window = n_chans * self.n_patches_per_window
        self.total_patches_per_channel = n_windows * self.n_patches_per_window

        # Patch embedding
        self.to_patch_embedding = nn.Linear(patch_size, embed_dim)

        # 4D positional encoding
        self.fourier4d = FourierEmb4D(embed_dim, freqs=freqs)
        self.mlp4d = nn.Sequential(
            nn.Linear(4, embed_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.ln = nn.LayerNorm(embed_dim)

        # Transformer backbone
        mlp_dim = int(embed_dim * mlp_dim_ratio)
        self.transformer = REVETransformerBackbone(
            dim=embed_dim, depth=depth, heads=heads,
            head_dim=head_dim, mlp_dim=mlp_dim, geglu=True,
        )

        # Channel positions from REVE position bank
        self.default_pos = None
        if chs_info is not None:
            from braindecode.models.reve import RevePositionBank
            position_bank = RevePositionBank()
            self.default_pos = position_bank.forward(
                [ch["ch_name"] for ch in chs_info]
            )

    def _compute_pos_embed(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute 4D positional embeddings for the full [C, T, P] token grid.

        Returns: [B, C*T*P, embed_dim]
        """
        if self.default_pos is None:
            raise ValueError("No channel positions available. Provide chs_info at init.")

        pos = self.default_pos.to(device)  # [C, 3]
        C = pos.shape[0]
        T = self.n_windows
        P = self.n_patches_per_window

        # Build 4D positions: (x, y, z, t) for each token in (c, t, p) order
        # token_idx = c * (T*P) + t * P + p
        positions_4d = []
        for c in range(C):
            for t in range(T):
                for p in range(P):
                    xyz = pos[c]  # [3]
                    time_idx = t * P + p
                    pos_4d = torch.cat([xyz, torch.tensor([time_idx], device=device, dtype=xyz.dtype)])
                    positions_4d.append(pos_4d)

        positions_4d = torch.stack(positions_4d, dim=0)  # [C*T*P, 4]
        positions_4d = positions_4d.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C*T*P, 4]

        pos_embed = self.ln(self.fourier4d(positions_4d) + self.mlp4d(positions_4d))
        return pos_embed

    def tokenize(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract patch tokens and compute positional embeddings.

        Args:
            eeg: [B, T, C, W] raw EEG

        Returns:
            tokens: [B, C*T*P, embed_dim] patch embeddings + positional encoding
            pos_embed: [B, C*T*P, embed_dim] positional embeddings (for predictor)
        """
        B, T, C, W = eeg.shape
        step = self.patch_size - self.patch_overlap
        P = self.n_patches_per_window

        # Extract patches from each window: [B, T, C, P, patch_size]
        patches = eeg.unfold(dimension=3, size=self.patch_size, step=step)
        # patches shape: [B, T, C, P, patch_size]

        # Linear projection: [B, T, C, P, embed_dim]
        patch_embeds = self.to_patch_embedding(patches)

        # Reorder to (c, t, p) flattening: [B, C*T*P, embed_dim]
        # Current: [B, T, C, P, embed_dim] → need [B, C, T, P, embed_dim] → [B, C*T*P, embed_dim]
        patch_embeds = patch_embeds.permute(0, 2, 1, 3, 4)  # [B, C, T, P, embed_dim]
        patch_embeds = patch_embeds.reshape(B, C * T * P, self.embed_dim)

        # Compute positional embeddings
        pos_embed = self._compute_pos_embed(B, eeg.device)

        # Add positional encoding
        tokens = patch_embeds + pos_embed

        return tokens, pos_embed

    def encode_tokens(
        self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Tokenize, optionally mask, and encode through transformer.

        Args:
            eeg: [B, T, C, W] raw EEG
            mask: [C*T*P] bool tensor — True = keep (context), False = mask out.
                  If None, all tokens are kept.

        Returns:
            [B, n_visible, embed_dim] encoded token representations
        """
        tokens, _ = self.tokenize(eeg)

        if mask is not None:
            tokens = tokens[:, mask]  # [B, n_visible, embed_dim]

        tokens = self.transformer(tokens)
        return tokens

    def pool_to_windows(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool full (unmasked) token sequence to per-window representations.

        Reshapes [B, C*T*P, D] → [B, C, T, P, D], mean-pools over C and P
        per window → [B, T, D], then formats as [B, D, T, 1, 1] for probes.

        Args:
            tokens: [B, C*T*P, embed_dim] — must be full (unmasked) token sequence

        Returns:
            [B, embed_dim, T, 1, 1]
        """
        B = tokens.shape[0]
        C = self.n_chans
        T = self.n_windows
        P = self.n_patches_per_window
        D = self.embed_dim

        # Reshape: [B, C, T, P, D]
        x = tokens.view(B, C, T, P, D)
        # Mean pool over channels and patches: [B, T, D]
        x = x.mean(dim=(1, 3))
        # Format for MovieFeatureHead: [B, D, T, 1, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return x


class MaskedPredictor(nn.Module):
    """V-JEPA transformer predictor with optional narrow bottleneck.

    Takes context encoder output + positional info for masked positions,
    predicts representations at masked positions.

    When ``predictor_dim`` < ``embed_dim``, the predictor operates in a
    narrower latent space (as recommended by I-JEPA / V-JEPA) to act as
    an information bottleneck that forces the encoder to learn richer
    representations.

    Input:  context_tokens [B, n_ctx, D], context_pos [B, n_ctx, D], target_pos [B, n_pred, D]
    Output: predictions [B, n_pred, D]
    """

    def __init__(
        self,
        embed_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        head_dim: int = 16,
        mlp_dim_ratio: float = 2.66,
        predictor_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim or embed_dim

        # Project encoder dim -> predictor dim (identity when equal)
        if self.predictor_dim != embed_dim:
            self.input_proj = nn.Linear(embed_dim, self.predictor_dim)
            self.pos_proj = nn.Linear(embed_dim, self.predictor_dim)
        else:
            self.input_proj = nn.Identity()
            self.pos_proj = nn.Identity()

        self.mask_token = nn.Parameter(torch.randn(1, 1, self.predictor_dim) * 0.02)

        mlp_dim = int(self.predictor_dim * mlp_dim_ratio)
        self.transformer = REVETransformerBackbone(
            dim=self.predictor_dim, depth=depth, heads=heads,
            head_dim=head_dim, mlp_dim=mlp_dim, geglu=True,
        )
        self.output_proj = nn.Linear(self.predictor_dim, embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_tokens: [B, n_ctx, D] encoded context representations
            context_pos: [B, n_ctx, D] positional embeddings for context tokens
            target_pos: [B, n_pred, D] positional embeddings for prediction targets

        Returns:
            predictions: [B, n_pred, D]
        """
        B, n_pred, D = target_pos.shape
        n_ctx = context_tokens.shape[1]

        # Project to predictor dimension
        ctx_proj = self.input_proj(context_tokens)  # [B, n_ctx, predictor_dim]
        ctx_pos_proj = self.pos_proj(context_pos)    # [B, n_ctx, predictor_dim]
        tgt_pos_proj = self.pos_proj(target_pos)     # [B, n_pred, predictor_dim]

        # Create mask tokens with target positions
        mask_tokens = self.mask_token.expand(B, n_pred, -1) + tgt_pos_proj

        # Add position to context tokens
        context_with_pos = ctx_proj + ctx_pos_proj

        # Concatenate: [context; mask_tokens]
        x = torch.cat([context_with_pos, mask_tokens], dim=1)  # [B, n_ctx + n_pred, predictor_dim]

        # Transformer
        x = self.transformer(x)

        # Extract prediction tokens (last n_pred) and project back to embed_dim
        pred_tokens = x[:, n_ctx:]  # [B, n_pred, predictor_dim]

        return self.output_proj(pred_tokens)


class MovieFeatureHead(nn.Module):
    """MLP head for predicting per-timestep movie features from JEPA representations.

    Takes [B, D, T, 1, 1] encoder output and predicts [B, T, n_features].
    Used as a probe in JEPAProbe for evaluating representation quality.
    """

    def __init__(self, in_dim, hidden_dim, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        self.apply(init_module_weights)

    def forward(self, x):
        # x: [B, D, T, 1, 1] from JEPA encoder
        B, D, T = x.shape[:3]
        x = x.view(B, D, T).permute(0, 2, 1).reshape(B * T, D)
        out = self.net(x)  # [B*T, n_features]
        return out.view(B, T, -1)  # [B, T, n_features]


class TemporalMovieFeatureHead(nn.Module):
    """MLP head with temporal context for predicting per-timestep movie features.

    Like MovieFeatureHead but concatenates a global temporal mean to each
    window's representation before the MLP, giving the probe access to
    cross-window context.  Input/output shapes are identical to MovieFeatureHead.
    """

    def __init__(self, in_dim, hidden_dim, n_features):
        super().__init__()
        # Input is 2*in_dim: local window repr + global mean across windows
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        self.apply(init_module_weights)

    def forward(self, x):
        # x: [B, D, T, 1, 1] from JEPA encoder
        B, D, T = x.shape[:3]
        x = x.view(B, D, T).permute(0, 2, 1)  # [B, T, D]
        x_global = x.mean(dim=1, keepdim=True).expand_as(x)  # [B, T, D]
        x = torch.cat([x, x_global], dim=-1)  # [B, T, 2D]
        x = x.reshape(B * T, D * 2)
        out = self.net(x)  # [B*T, n_features]
        return out.view(B, T, -1)  # [B, T, n_features]


class MLPEEGPredictor(TemporalBatchMixin, nn.Module):
    """MLP predictor for flat EEG embeddings.

    Pairs consecutive timesteps and predicts the next embedding.
    Input x: [B, 1, T, D], Output: [B, 1, T-1, D]

    Args:
        in_d: Embedding dimension D
        h_d: Hidden dimension
        out_d: Output embedding dimension (typically same as in_d)
    TODO: This is a simple predictor for testing. For better performance, consider adding temporal convolutions or an RNN predictor that can capture longer-range dependencies.
    """

    def __init__(self, in_d, h_d, out_d):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_d, h_d),
            nn.ReLU(),
            nn.Linear(h_d, out_d),
        )
        self.is_rnn = False

    def _forward(self, x):
        if x.shape[2] != 1 or x.shape[3] != 1:
            raise ValueError(f"Expected input with shape [B, C, 1, 1], got {x.shape}")
        output = self.predictor(x.squeeze(2, 3))
        output = output.unsqueeze(2).unsqueeze(3)  # [B, D, 1, 1]
        return output