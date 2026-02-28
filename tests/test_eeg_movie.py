import torch
from braindecode.models import ShallowFBCSPNet
from eb_jepa.datasets.hbn import HBNMovieDataset
from torch.nn import functional as F


def test_hbn_movie_dataset(cfg):
    window_size_seconds = 2
    window_size_samples = int(window_size_seconds * 100)  # example for 100Hz sampling rate
    dataset = HBNMovieDataset(split="train", window_size_seconds=window_size_seconds, cfg=cfg.data)

    assert len(dataset) > 0, "Dataset should contain at least one recording"

    selected = "luminance_mean"
    model = ShallowFBCSPNet(
        n_chans=129,
        n_outputs=1,
        n_times=window_size_samples,
    )

    # Each item is one recording's worth of windows (like a movie clip)
    X, features = dataset[0]

    n_windows = X.shape[0]
    assert X.ndim == 3, f"Expected (W, C, T) but got ndim={X.ndim}"
    assert X.shape[1:] == (129, window_size_samples), (
        f"Expected (W, 129, {window_size_samples}) but got {X.shape}"
    )
    assert len(features) == n_windows, (
        f"Number of feature rows ({len(features)}) should match number of windows ({n_windows})"
    )

    # Forward pass: feed all windows from the clip as a batch
    outputs = model(X)
    assert outputs.shape == (n_windows, 1)

    movie_features = features.apply(lambda row: row[selected])
    targets = torch.tensor(movie_features.values, dtype=torch.float32)
    loss = F.mse_loss(outputs.squeeze(), targets)
    print(f"Recording 0: {n_windows} windows, loss={loss.item():.4f}")
