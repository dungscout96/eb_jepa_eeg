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
    '''
    Window onset is with respect to the start of movie recording, not the start of the trial. 
    We assume that the movie starts at the same time as the recording, and that there are no dropped frames in the movie.
    '''
    # compute the movie timestamp corresponding to the window onset
    movie_timestamp = window_onset / sfreq
    # compute the corresponding frame index in the movie
    frame_index = int(movie_timestamp * MOVIE_METADATA[movie]["fps"])
    # assume visual processing of EEG still happening for 2 seconds after the movie ends, 
    # so we can still use the features of the last frame for windows that are within 2 seconds after the movie ends. 
    # This is important for handling cases where the recording is slightly longer than the movie due to annotation imprecision.
    if (frame_index - MOVIE_METADATA[movie]["frame_count"]) < (MOVIE_METADATA[movie]["fps"] * 2):
        frame_index = len(movie_features) - 1
    # frame_index = min(frame_index, len(movie_features) - 1)
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

def get_movie_recording_duration(raw: mne.io.BaseRaw, movie:str):
    events, event_id = mne.events_from_annotations(raw)
    # compute the duration from start to stop
    events_filtered = mne.pick_events(
        events, 
        include=[event_id["video_start"], event_id["video_stop"]]
    )
    duration_sample = events_filtered[1, 0] - events_filtered[0, 0]
    duration_seconds = duration_sample / raw.info["sfreq"]
    if duration_seconds > MOVIE_METADATA[movie]["duration"] + 60:
        raise ValueError(f"Recording has duration {duration_seconds}s (+60s) for movie '{movie}', which is unexpected. Please check the recording.")
    return duration_seconds

def reject_recording(raw: mne.io.BaseRaw, movie:str):
    # reject recording that don't have viedo_start and video_stop annotations
    events, event_id = mne.events_from_annotations(raw)
    if "video_start" not in event_id or "video_stop" not in event_id:
        return True

    # reject recording that have duration less than movie duration
    duration_seconds = get_movie_recording_duration(raw, movie)
    if duration_seconds < MOVIE_METADATA[movie]["duration"] - 0.5: # allow 0.5s tolerance for annotation imprecision
        return True

class HBNDataset(Dataset):
    """Dataset where each item is a random crop of n_windows contiguous windows (WxCxT).
    TODO: revisit fixed window number
    """

    def __init__(self, split="train", n_windows=16, window_size_seconds=2):
        if split == "train":
            releases = TRAIN_RELEASES
        elif split == "val":
            releases = VAL_RELEASES
        elif split == "test":
            releases = TEST_RELEASES
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        self.n_windows = n_windows
        self.window_size_seconds = window_size_seconds
        self.recordings = []
        for release, dataset_name in releases.items():
            dataset = load_or_download(release, dataset_name)
            from braindecode.preprocessing import create_fixed_length_windows
            windowed_dataset = create_fixed_length_windows(
                dataset,
                window_size_samples=int(window_size_seconds*dataset.datasets[0].raw.info["sfreq"]),
                window_stride_samples=int(window_size_seconds*dataset.datasets[0].raw.info["sfreq"]), # non-overlapping windows
                drop_last_window=True,
            )
            for recording_ds in windowed_dataset.datasets:
                if len(recording_ds) >= n_windows:
                    self.recordings.append(recording_ds)

        self._preload_movie_features()

    def _preload_movie_features(self):
        self.movie_features = {
            "ThePresent": pd.read_csv(MOVIE_METADATA["ThePresent"]["feature_csv"]),
        }
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

class HBNMovieProbeDataset(Dataset):
    """Dataset where each window is associated with the movie features at the corresponding timestamp.
    """

    def __init__(self, split="train", window_size_seconds=2):
        self.window_size_seconds = window_size_seconds
        if split == "train":
            releases = TRAIN_RELEASES
        elif split == "val":
            releases = VAL_RELEASES
        elif split == "test":
            releases = TEST_RELEASES
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        data = []
        for release, dataset_name in releases.items():
            dataset = load_or_download(release, dataset_name)
            data.append(dataset)
        data = BaseConcatDataset(data)

        selected_recordings = []
        rejected = 0
        for recording_ds in data.datasets: 
            raw = recording_ds.raw

            for idx, ann in enumerate(raw.annotations):
                # set the duration of the "video_start" annotation to the actual movie recording duration, which can be computed from the "video_start" and "video_stop" annotations. This is important for correctly creating windows that align with the movie timestamps.
                if ann["description"] == "video_start":
                    movie_recording_duration = get_movie_recording_duration(raw, movie=TASK)
                    # clipping duration to movie duration, while avoiding extremely long recordings that may be due to annotation errors
                    movie_recording_duration = min(movie_recording_duration, MOVIE_METADATA[TASK]["duration"])
                    print(f"Setting duration of 'video_start' annotation to {movie_recording_duration:.2f}s for recording '{recording_ds}'")
                    raw.annotations.duration[idx] = movie_recording_duration

            if reject_recording(raw, movie=TASK):
                rejected += 1
                continue
            selected_recordings.append(recording_ds)
        print(f"Rejected {rejected}/{len(data.datasets)} recordings")
        data = BaseConcatDataset(selected_recordings)
        from braindecode.preprocessing import create_windows_from_events
        sfreq = data.datasets[0].raw.info["sfreq"]
        windowed = create_windows_from_events(
            data,
            mapping={"video_start": 0},
            trial_stop_offset_samples=-int(0.1*sfreq), # just one big trial for the whole movie
            window_size_samples=int(window_size_seconds*sfreq),
            window_stride_samples=int(window_size_seconds*sfreq), # non-overlapping windows
            drop_last_window=True,
        )
        self.data = windowed
        
        self.sfreq = data.datasets[0].raw.info["sfreq"]
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

