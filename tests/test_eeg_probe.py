from braindecode.models import ShallowFBCSPNet
from eb_jepa.eeg_probe_decoder import HBNMovieProbeDataset
from torch.utils.data import DataLoader
from torch.functional import F

def test_hbn_movie_probe_dataset():
    dataset = HBNMovieProbeDataset(split="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    selected = "luminance_mean"  # example feature name from the movie metadata
    model = ShallowFBCSPNet(
        n_chans=129,  # example number of EEG channels
        n_outputs=1,   # regression task
        n_times=1000,  # example number of time points in each window
    )
    for batch in dataloader:
        # batch is a tuple of (X, features), where:
        # - X is the EEG data tensor of shape (B, C, T)
        # - features is a dictionary of movie features for each sample in the batch
        X, features = batch
        windows = X  # (B, C, T)
        movie_features = features[selected] # (B, F)

        assert windows.shape[0] == movie_features.shape[0] == 4
        assert windows.shape[1:] == (129, 1000)  # example shape

        # Forward pass through the model (example)
        outputs = model(windows)
        assert outputs.shape == (4, 1)  # example output shape for regression
        loss = F.mse_loss(outputs.squeeze(), movie_features)
        print(f"Loss: {loss.item()}")
        break  # just test one batch

        

if __name__ == "__main__":
    test_hbn_movie_probe_dataset()