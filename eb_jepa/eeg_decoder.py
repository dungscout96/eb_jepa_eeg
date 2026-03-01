import torch.nn as nn

from eb_jepa.nn_utils import TemporalBatchMixin, init_module_weights

class EEGDecoder(TemporalBatchMixin, nn.Module):
    """
    EEG encoder that wraps a Braindecode model specified by name.
    Supports both 4D [B, 1, C, W] and 5D [B, 1, T, C, W] inputs via TemporalBatchMixin.
    """
    def __init__(self, in_d, h_d, out_d, name: str="REVE", chs_info=None, attention_pooling=False):
        super().__init__()
        import importlib
        module = importlib.import_module("braindecode.models")
        self.encoder = getattr(module, name)(n_chans=in_d, n_outputs=out_d, n_times=1000, chs_info=chs_info, attention_pooling=attention_pooling)

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

class ImageDecoder(TemporalBatchMixin, nn.Module):
    """
    Simple 2D convolutional decoder for reconstructing images from representations.
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(
        self,
        in_dim,
        out_dim=1,
        hidden_dim=16,
        tk=1,  # unused in 2D; kept for API compatibility
        ts=1,  # unused in 2D; kept for API compatibility
        sk=4,  # spatial kernel for ConvTranspose2d
        ss=2,  # spatial stride (controls the upsample factor)
        pad_mode="same",
        scale_factor=1.0,
        shift_factor=0.0,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1),
        )

        self.apply(init_module_weights)

    def _forward(self, x):
        # x: (B,C,H,W)
        y = self.net(x)
        return y
