from eb_jepa.architectures import EEGEncoder, StateOnlyPredictor
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

def create_eeg_encoder(in_d=64, h_d=128, out_d=256, name="REVE", device="cpu"):
    """Helper function to create an EEGEncoder with mock channel info."""
    encoder = EEGEncoder(in_d, h_d, out_d, name, chs_info=MOCK_CHS_INFO)
    return encoder.to(device)

def test_eeg_encoder_4d():
    """Test EEGEncoder with a sample input."""
    in_d = 64  # Number of EEG channels
    h_d = 128  # Hidden dimension (not used in this encoder but required by interface)
    out_d = 256  # Output dimension of the encoder
    name = "REVE"  # Name of the Braindecode model to use

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = create_eeg_encoder(in_d, h_d, out_d, name, device=device)

    # Create a sample input tensor with shape [B, 1, C, W]
    batch_size = 8
    time_steps = 1000
    x = torch.randn(batch_size, 1, in_d, time_steps)
    x = x.to(device)

    # Forward pass through the encoder
    output = encoder(x)

    # Check output shape
    assert output.shape == (batch_size, out_d, 1, 1), f"Expected output shape {(batch_size, out_d, 1, 1)}, got {output.shape}"

def test_eeg_encoder_5d():
    """Test EEGEncoder with a sample input."""
    in_d = 64  # Number of EEG channels
    h_d = 128  # Hidden dimension (not used in this encoder but required by interface)
    out_d = 256  # Output dimension of the encoder
    name = "REVE"  # Name of the Braindecode model to use

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = create_eeg_encoder(in_d, h_d, out_d, name, device=device)

    # Create a sample input tensor with shape [B, 1, C, W]
    batch_size = 8
    time_steps = 1000
    n_windows = 16
    x = torch.randn(batch_size, 1, n_windows, in_d, time_steps)
    x = x.to(device)

    # Forward pass through the encoder
    output = encoder(x)

    # Check output shape
    assert output.shape == (batch_size, out_d, n_windows, 1, 1), f"Expected output shape {(batch_size, out_d, n_windows, 1, 1)}, got {output.shape}"

def test_eeg_mlp_predictor():
    """Test EEGPredictor with a sample input."""
    in_d = 256  # Input dimension from the encoder
    h_d = 128  # Hidden dimension (not used in this predictor but required by interface)
    out_d = 256  # Output dimension of the predictor

    from eb_jepa.architectures import MLPEEGPredictor
    predictor = MLPEEGPredictor(in_d, h_d, out_d)

    # Create a sample input tensor with shape [B, D]
    batch_size = 8
    n_windows = 16 # predictor must be able to take in window dimension as well.
    x = torch.randn(batch_size, in_d, n_windows, 1, 1)  # [B, D, T, 1, 1] to match encoder output shape

    # Forward pass through the predictor
    output = predictor(x)

    # Check output shape
    assert output.shape == (batch_size, out_d, n_windows, 1, 1), f"Expected output shape {(batch_size, out_d, n_windows-1, 1, 1)}, got {output.shape}"

def test_eeg_state_only_predictor():
    """Test EEGPredictor with a sample input."""
    in_d = 256  # Input dimension from the encoder
    h_d = 128  # Hidden dimension (not used in this predictor but required by interface)
    out_d = 256  # Output dimension of the predictor

    from eb_jepa.architectures import MLPEEGPredictor
    predictor_model = MLPEEGPredictor(in_d*2, h_d, out_d)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)

    # Create a sample input tensor with shape [B, D]
    batch_size = 8
    n_windows = 16 # predictor must be able to take in window dimension as well.
    x = torch.randn(batch_size, in_d, n_windows, 1, 1)  # [B, D, T, 1, 1] to match encoder output shape

    # Forward pass through the predictor
    output = predictor(x, None)

    # Check output shape
    assert output.shape == (batch_size, out_d, n_windows-1, 1, 1), f"Expected output shape {(batch_size, out_d, n_windows-1, 1, 1)}, got {output.shape}"