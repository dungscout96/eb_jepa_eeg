from eb_jepa.datasets.hbn import HBNDataset
from torch.utils.data import DataLoader

def test_hbn_dataset():
    bs = 4
    n_windows = 16
    train_set = HBNDataset(split="train", n_windows=n_windows)
    assert len(train_set) > 0, "Train dataset should not be empty"
    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
    )
    data = next(iter(train_loader))
    assert data.ndim == 4, "Data should be 4-dimensional (batch, windows, channels, time)"
    assert data.shape[0] == bs, f"Batch size should be {bs}"
    assert data.shape[1] == n_windows, f"Window count should be {n_windows}"
