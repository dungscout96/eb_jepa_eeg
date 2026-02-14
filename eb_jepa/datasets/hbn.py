from eegdash.dataset import EEGDashDataset, EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset  # noqa: F401

from pathlib import Path
from torch.utils.data import Dataset
import torch
import random
import os

DATA_DIR = Path("/u/dtyoung/.cache/eb_jepa/datasets/eegdash_cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TASK="RestingState"
TRAIN_RELEASES = {
    "R1": "ds005505"
} # ["R1", "R2", "R3", "R4", "R5"]
VAL_RELEASES = {
    "R6": "ds005505"
}
TEST_RELEASES = {
    "R7": "ds005505"
}

def load_or_download(release, dataset_name):
    dataset_id = f"EEG2025r{release[1:]}mini"
    data_dir = DATA_DIR / dataset_id
    needs_download = not data_dir.exists() or not list(data_dir.glob("**/*.bdf"))

    dataset = EEGChallengeDataset(
        cache_dir=DATA_DIR,
        release=release,
        download=needs_download,
        task=TASK,
        mini=True
    )
    if needs_download:
        dataset.download_all(n_jobs=-1)
    return dataset

def preprocess_dataset(dataset):
    from braindecode.preprocessing import create_fixed_length_windows

    windowed = create_fixed_length_windows(
        dataset,
        window_size_samples=1000,  # e.g., 1000 for ~2s at 500Hz
        window_stride_samples=500,
        drop_last_window=True,
    )
    return windowed

class HBNDataset(Dataset):
    """Dataset where each item is a random crop of n_windows contiguous windows (WxCxT).
    TODO: revisit fixed window number
    """

    def __init__(self, split="train", n_windows=16):
        if split == "train":
            releases = TRAIN_RELEASES
        elif split == "val":
            releases = VAL_RELEASES
        elif split == "test":
            releases = TEST_RELEASES
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        self.n_windows = n_windows
        self.recordings = []
        for release, dataset_name in releases.items():
            dataset = load_or_download(release, dataset_name)
            windowed_dataset = preprocess_dataset(dataset)
            for recording_ds in windowed_dataset.datasets:
                if len(recording_ds) >= n_windows:
                    self.recordings.append(recording_ds)

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, idx):
        recording = self.recordings[idx]
        start = random.randint(0, len(recording) - self.n_windows)
        windows = torch.stack([
            torch.from_numpy(recording[start + i][0])
            for i in range(self.n_windows)
        ])
        return windows
