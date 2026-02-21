from braindecode.models import ShallowFBCSPNet
from eb_jepa.datasets.hbn import HBNMovieProbeDataset
from torch.utils.data import DataLoader
from torch.functional import F

def test_hbn_movie_probe_dataset():
    window_size_seconds = 2
    window_size_samples = int(window_size_seconds * 100)  # example for 100Hz sampling rate
    dataset = HBNMovieProbeDataset(split="train", window_size_seconds=window_size_seconds)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    selected = "luminance_mean"  # example feature name from the movie metadata
    model = ShallowFBCSPNet(
        n_chans=129,  # example number of EEG channels
        n_outputs=1,   # regression task
        n_times=window_size_samples,  # example number of time points in each window
    )
    for batch in dataloader:
        # batch is a tuple of (X, features), where:
        # - X is the EEG data tensor of shape (B, C, T)
        # - features is a dictionary of movie features for each sample in the batch
        X, features = batch
        windows = X  # (B, C, T)
        movie_features = features[selected] # (B, F)

        assert len(windows) == len(movie_features)
        assert windows.shape[1:] == (129, window_size_samples)  # example shape

        # Forward pass through the model (example)
        outputs = model(windows)
        assert outputs.ndim == 2 and outputs.shape[1] == 1  # example output shape for regression
        loss = F.mse_loss(outputs.squeeze(), movie_features)
        print(f"Loss: {loss.item()}")

        

if __name__ == "__main__":
    test_hbn_movie_probe_dataset()