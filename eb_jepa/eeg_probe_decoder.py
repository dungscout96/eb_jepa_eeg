from eegdash.dataset import EEGDashDataset, EEGChallengeDataset
from braindecode.datasets import BaseConcatDataset  # noqa: F401

from pathlib import Path
import mne
from torch.utils.data import Dataset
import torch
import random
import pandas as pd


DATA_DIR = Path.home() / ".cache" / "eb_jepa" / "datasets" / "eegdash_cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TASK="ThePresent" # "RestingState"
TRAIN_RELEASES = {
    "R1": "ds005505"
} # ["R1", "R2", "R3", "R4", "R5"]
VAL_RELEASES = {
    "R6": "ds005505"
}
TEST_RELEASES = {
    "R7": "ds005505"
}
MOVIE_METADATA = { # computed by get_movie_metadata()
    "ThePresent": {
        "duration": 203.29166666666669,  # 3 minutes 23 seconds, 291 milliseconds
        "fps": 24,
        "frame_count": 4878,
        "feature_csv": "/home/dung/Documents/eb_jepa_eeg/data/output/The_Present/features.csv"
    }
}

def get_movie_metadata(task):
    movie_filepath = {
        "ThePresent": "/home/dung/Documents/eb_jepa_eeg/data/movies/The_Present.mp4",
        "RestingState": "/home/dung/Documents/eb_jepa_eeg/data/movies/Resting_State.mp4"
    }[task]
    # load the movie and compute its duration in seconds
    import cv2
    cap = cv2.VideoCapture(movie_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    movie_duration_seconds = frame_count / fps
    return movie_duration_seconds, fps, frame_count

def get_window_movie_metadata(
        window_onset: int, 
        sfreq: int, 
        movie: str, 
        movie_features: pd.DataFrame
    ):
    # compute the movie timestamp corresponding to the window onset
    movie_timestamp = window_onset / sfreq
    # compute the corresponding frame index in the movie
    frame_index = int(movie_timestamp * MOVIE_METADATA[movie]["fps"])
    features = movie_features.iloc[frame_index].to_dict()
    return features

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

def get_movie_recording_duration(raw: mne.io.BaseRaw):
    events, event_id = mne.events_from_annotations(raw)
    # compute the duration from start to stop
    events_filtered = mne.pick_events(
        events, 
        include=[event_id["video_start"], event_id["video_stop"]]
    )
    duration_sample = events_filtered[1, 0] - events_filtered[0, 0]
    duration_seconds = duration_sample / raw.info["sfreq"]
    return duration_seconds

def preprocess_dataset(dataset):
    from braindecode.preprocessing import create_fixed_length_windows

    windowed = create_fixed_length_windows(
        dataset,
        window_size_samples=1000,  # e.g., 1000 for ~2s at 500Hz
        window_stride_samples=500,
        drop_last_window=True,
    )
    return windowed

class HBNMovieProbeDataset(Dataset):
    """Dataset where each window is associated with the movie features at the corresponding timestamp.
    """

    def __init__(self, split="train"):
        if split == "train":
            releases = TRAIN_RELEASES
        elif split == "val":
            releases = VAL_RELEASES
        elif split == "test":
            releases = TEST_RELEASES
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        self.data = []
        for release, dataset_name in releases.items():
            dataset = load_or_download(release, dataset_name)
            windowed_dataset = preprocess_dataset(dataset)
            self.data.append(windowed_dataset)
        
        self.data = BaseConcatDataset(self.data)
        self.sfreq = self.data.datasets[0].raw.info["sfreq"]
        self._preload_movie_features()

    def _preload_movie_features(self):
        self.movie_features = {
            "ThePresent": pd.read_csv(MOVIE_METADATA["ThePresent"]["feature_csv"]),
        }
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y, crop_inds = self.data[idx]
        window_onset = crop_inds[1]  # i_start_in_trial (samples)
        features = get_window_movie_metadata(
            window_onset=window_onset,
            sfreq=self.sfreq,
            movie=TASK,
            movie_features=self.movie_features[TASK],
        )
        return torch.from_numpy(X).float(), features

