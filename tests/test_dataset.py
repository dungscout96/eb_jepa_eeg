from eb_jepa.datasets.hbn import HBNDataset
from torch.utils.data import DataLoader

def test_hbn_dataset():
    bs = 4
    train_set = HBNDataset(split="train")
    assert len(train_set) > 0, "Train dataset should not be empty"
    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
    )
    batch = next(iter(train_loader))
    assert len(batch) == 3, "Batch should contain data, labels, and window index"
    assert batch[0].ndim == 3, "Data should be 3-dimensional (batch, channels, time)"
    assert batch[0].shape[0] == bs, f"Batch size should be {bs}"
    assert batch[1].ndim == 1, "Labels should be 1-dimensional (batch,)"
