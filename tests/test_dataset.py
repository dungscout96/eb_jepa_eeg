from eb_jepa.datasets.hbn import HBNDataset, collate_fn
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
        collate_fn=collate_fn,
    )
    data, mask = next(iter(train_loader))
    assert data.ndim == 4, "Data should be 4-dimensional (batch, windows, channels, time)"
    assert data.shape[0] == bs, f"Batch size should be {bs}"
    assert mask.ndim == 2, "Mask should be 2-dimensional (batch, windows)"
    assert mask.shape[0] == bs, f"Mask batch size should be {bs}"
