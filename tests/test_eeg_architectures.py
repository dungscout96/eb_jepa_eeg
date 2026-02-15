from eb_jepa.architectures import EEGEncoder
import torch

# 64 standard EEG channel names recognized by REVE's position bank
MOCK_CHS_INFO = [{"ch_name": ch} for ch in [
    "A1", "A2", "C3", "C4", "CZ", "F3", "F4", "F7", "F8", "FP1",
    "FP2", "FZ", "O1", "O2", "P3", "P4", "PZ", "T3", "T4", "T5",
    "T6", "OZ", "Cz", "Fpz", "Fz", "P7", "P8", "Pz", "T7", "T8",
    "C1", "C2", "C5", "C6", "CP1", "CP2", "CP3", "CP4", "CPz", "FC1",
    "FC2", "FC3", "FC4", "FCz", "P1", "P2", "POz", "AFz", "P5", "P6",
    "PO3", "PO4", "AF3", "AF4", "AF7", "AF8", "CP5", "CP6", "F1", "F2",
    "F5", "F6", "FC5", "FC6",
]]

def test_eeg_encoder():
    """Test EEGEncoder with a sample input."""
    in_d = 64  # Number of EEG channels
    h_d = 128  # Hidden dimension (not used in this encoder but required by interface)
    out_d = 256  # Output dimension of the encoder
    name = "REVE"  # Name of the Braindecode model to use

    encoder = EEGEncoder(in_d, h_d, out_d, name, chs_info=MOCK_CHS_INFO)

    # Create a sample input tensor with shape [B, 1, C, W]
    batch_size = 8
    time_steps = 1000
    x = torch.randn(batch_size, 1, in_d, time_steps)

    # Forward pass through the encoder
    output = encoder(x)

    # Check output shape
    assert output.shape == (batch_size, out_d), f"Expected output shape {(batch_size, out_d)}, got {output.shape}"

def test_eeg_predictor():
    """Test EEGPredictor with a sample input."""
    in_d = 256  # Input dimension from the encoder
    h_d = 128  # Hidden dimension (not used in this predictor but required by interface)
    out_d = 256  # Output dimension of the predictor

    from eb_jepa.architectures import MLPEEGPredictor
    predictor = MLPEEGPredictor(in_d, h_d, out_d)

    # Create a sample input tensor with shape [B, D]
    batch_size = 8
    n_windows = 16 # predictor must be able to take in window dimension as well.
    x = torch.randn(batch_size, 1, n_windows, in_d)

    # Forward pass through the predictor
    output = predictor(x, None)

    # Check output shape
    assert output.shape == (batch_size, 1, n_windows-1, out_d), f"Expected output shape {(batch_size, 1, n_windows-1, out_d)}, got {output.shape}"