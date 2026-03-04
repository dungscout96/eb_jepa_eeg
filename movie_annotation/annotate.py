"""Main entry point: annotate movie frames with EEG-relevant visual features."""

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from features.lowlevel import extract_lowlevel
from features.motion import extract_motion
from features.depth import DepthEstimator
from features.objects import ObjectDetector
from features.faces import FaceDetector
from features.clip_features import CLIPFeatureExtractor

CHUNK_SIZE = 64  # frames to read at a time from video
GPU_BATCH_SIZE = 16  # frames per GPU inference batch
MOVIES_DIR = "movies"
OUTPUT_DIR = "output"


def get_movie_paths(movies_dir: str) -> list[tuple[str, str]]:
    """Return list of (movie_name, movie_path) for all mp4 files."""
    paths = []
    for fname in sorted(os.listdir(movies_dir)):
        if fname.endswith(".mp4"):
            name = os.path.splitext(fname)[0]
            paths.append((name, os.path.join(movies_dir, fname)))
    return paths


def read_video_chunk(cap: cv2.VideoCapture, chunk_size: int) -> list[np.ndarray]:
    """Read up to chunk_size frames from an open VideoCapture."""
    frames = []
    for _ in range(chunk_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def batched(lst: list, batch_size: int):
    """Yield successive batches from a list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def process_movie(
    movie_name: str,
    movie_path: str,
    output_dir: str,
    depth_estimator: DepthEstimator,
    object_detector: ObjectDetector,
    face_detector: FaceDetector,
    clip_extractor: CLIPFeatureExtractor,
):
    """Process a single movie and save features to disk."""
    cap = cv2.VideoCapture(movie_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing: {movie_name} ({total_frames} frames, {fps:.2f} fps)")

    out_dir = os.path.join(output_dir, movie_name)
    os.makedirs(out_dir, exist_ok=True)

    all_records = []
    all_clip_embeddings = []
    all_clip_indices = []
    frame_idx = 0
    prev_frame = None

    pbar = tqdm(total=total_frames, desc=movie_name, unit="frame")

    while True:
        chunk = read_video_chunk(cap, CHUNK_SIZE)
        if not chunk:
            break

        # --- CPU features: low-level + motion ---
        chunk_cpu_features = []
        for frame in chunk:
            ll = extract_lowlevel(frame)
            mot = extract_motion(prev_frame, frame)
            ll.update(mot)
            ll["frame_idx"] = frame_idx
            ll["timestamp_s"] = frame_idx / fps if fps > 0 else 0.0
            chunk_cpu_features.append(ll)
            prev_frame = frame
            frame_idx += 1

        # --- Face detection (CPU, per-frame) ---
        for i, frame in enumerate(chunk):
            face_feats = face_detector.extract(frame)
            chunk_cpu_features[i].update(face_feats)

        # --- GPU features: batched depth, objects, CLIP ---
        chunk_depth = []
        chunk_objects = []
        chunk_clip_embeds = []
        chunk_clip_feats = []

        for batch in batched(chunk, GPU_BATCH_SIZE):
            chunk_depth.extend(depth_estimator.extract_batch(batch))
            chunk_objects.extend(object_detector.extract_batch(batch))
            embeds, clip_feats = clip_extractor.extract_batch(batch)
            chunk_clip_embeds.append(embeds)
            chunk_clip_feats.extend(clip_feats)

        # Merge all features for this chunk
        for i in range(len(chunk)):
            record = chunk_cpu_features[i]
            record.update(chunk_depth[i])
            record.update(chunk_objects[i])
            record.update(chunk_clip_feats[i])
            all_records.append(record)

        # Collect CLIP embeddings
        chunk_embeds_arr = np.concatenate(chunk_clip_embeds, axis=0)
        all_clip_embeddings.append(chunk_embeds_arr)
        start_idx = frame_idx - len(chunk)
        all_clip_indices.extend(range(start_idx, start_idx + len(chunk)))

        pbar.update(len(chunk))

    pbar.close()
    cap.release()

    # Save Parquet
    df = pd.DataFrame(all_records)
    parquet_path = os.path.join(out_dir, "features.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved {parquet_path} ({len(df)} rows, {len(df.columns)} columns)")

    # Save CLIP embeddings as .npz
    if all_clip_embeddings:
        embeddings = np.concatenate(all_clip_embeddings, axis=0)
        indices = np.array(all_clip_indices)
        npz_path = os.path.join(out_dir, "clip_embeddings.npz")
        np.savez(npz_path, embeddings=embeddings, frame_indices=indices)
        print(f"  Saved {npz_path} (shape: {embeddings.shape})")


def main():
    parser = argparse.ArgumentParser(description="Annotate movie frames with EEG-relevant visual features")
    parser.add_argument("--movies-dir", default=MOVIES_DIR, help="Directory containing .mp4 files")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--movie", default=None, help="Process a single movie (filename without extension)")
    args = parser.parse_args()

    output_dir = args.output_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize models
    print("Loading models...")
    t0 = time.time()
    depth_estimator = DepthEstimator(device=device)
    object_detector = ObjectDetector(device=device)
    face_detector = FaceDetector(cache_dir=os.path.join(output_dir, ".cache"))
    clip_extractor = CLIPFeatureExtractor(device=device)
    print(f"Models loaded in {time.time() - t0:.1f}s")

    # Get movies
    movies = get_movie_paths(args.movies_dir)
    if args.movie:
        movies = [(n, p) for n, p in movies if n == args.movie]
        if not movies:
            print(f"Movie '{args.movie}' not found in {args.movies_dir}")
            return

    print(f"Movies to process: {[n for n, _ in movies]}")

    total_start = time.time()
    for movie_name, movie_path in movies:
        t0 = time.time()
        process_movie(
            movie_name, movie_path, output_dir,
            depth_estimator, object_detector, face_detector, clip_extractor,
        )
        print(f"  Finished {movie_name} in {time.time() - t0:.1f}s")

    print(f"\nAll done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
