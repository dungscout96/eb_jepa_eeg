"""
Shot boundary detection for The_Present using three methods:
  1. PySceneDetect (content-aware heuristic)
  2. TransNetV2 (deep learning)
  3. FFmpeg (scene change filter)

Computes pairwise and three-way agreement scores, and saves
boundary frames for visual review.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────
MOVIES_DIR = Path("movies")
OUTPUT_DIR = Path("output")


# ── Helper: get video metadata ─────────────────────────────────────────────
def get_video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


# ── Method 1: PySceneDetect ───────────────────────────────────────────────
def detect_pyscenedetect(video_path: Path, threshold: float = 27.0) -> list[int]:
    """Return list of frame indices where new shots begin."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list()
    # Each scene is (start_timecode, end_timecode). The shot boundary
    # is the start frame of each scene (except the very first scene at 0).
    boundaries = []
    for start, _end in scene_list:
        frame = start.get_frames()
        if frame > 0:
            boundaries.append(frame)
    return sorted(boundaries)


# ── Method 2: TransNetV2 ──────────────────────────────────────────────────
def detect_transnetv2(video_path: Path, threshold: float = 0.5) -> list[int]:
    """Return list of frame indices where shots begin according to TransNetV2."""
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="auto")
    scenes = model.detect_scenes(str(video_path), threshold=threshold)

    # Each scene dict has start_frame, end_frame, etc.
    # Shot boundary = start_frame of each scene after the first.
    boundaries = []
    for scene in scenes:
        sf = scene["start_frame"]
        if sf > 0:
            boundaries.append(int(sf))
    return sorted(boundaries)


# ── Method 3: FFmpeg scene detection ──────────────────────────────────────
def detect_ffmpeg(video_path: Path, threshold: float = 0.3) -> list[int]:
    """
    Use ffmpeg's select filter to find scene changes.
    Returns list of frame indices where shots begin.
    """
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse showinfo output for pts_time and frame number
    # showinfo lines look like:
    # [Parsed_showinfo_1 @ ...] n:   0 pts:    0 ... pts_time:0
    boundaries = []
    fps = get_video_info(video_path)["fps"]
    for line in result.stderr.split("\n"):
        if "pts_time:" in line and "showinfo" in line:
            # Extract pts_time
            for part in line.split():
                if part.startswith("pts_time:"):
                    pts_time = float(part.split(":")[1])
                    frame_idx = round(pts_time * fps)
                    boundaries.append(frame_idx)
                    break
    return sorted(boundaries)


# ── Save boundary frames ──────────────────────────────────────────────────
def save_boundary_frames(
    video_path: Path, boundaries: list[int], method_name: str, out_dir: Path
) -> None:
    """Save frames at each boundary (and 1 frame before) as images for review."""
    method_dir = out_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    for i, frame_idx in enumerate(boundaries):
        # Save the frame just before the cut and the cut frame
        for offset, label in [(-1, "before"), (0, "cut")]:
            idx = max(0, frame_idx + offset)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                ts = idx / fps
                fname = f"shot{i+1:03d}_f{idx:05d}_{ts:.2f}s_{label}.jpg"
                cv2.imwrite(str(method_dir / fname), frame)

    cap.release()
    print(f"  Saved {len(boundaries)} boundary frame pairs → {method_dir}/")


# ── Agreement metrics ─────────────────────────────────────────────────────
def compute_agreement(
    methods: dict[str, list[int]], n_frames: int, tolerance: int = 12
) -> dict:
    """
    Compute pairwise and 3-way agreement between shot boundary detectors.

    tolerance: frames within this window are considered a match (default=12
               frames = 0.5s at 24fps).

    Returns dict with:
      - pairwise precision/recall/F1 for each pair
      - per-method counts
      - 3-way agreement count
    """
    names = list(methods.keys())
    results = {"per_method": {}, "pairwise": {}, "three_way": {}}

    for name, bounds in methods.items():
        results["per_method"][name] = {"n_boundaries": len(bounds)}

    # Pairwise
    def match_count(ref: list[int], pred: list[int], tol: int) -> int:
        """Count how many pred boundaries match a ref boundary within tol."""
        ref_arr = np.array(ref)
        matched = 0
        for p in pred:
            if len(ref_arr) > 0 and np.min(np.abs(ref_arr - p)) <= tol:
                matched += 1
        return matched

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            bounds_a, bounds_b = methods[a], methods[b]

            if len(bounds_b) > 0 and len(bounds_a) > 0:
                hits_a_in_b = match_count(bounds_a, bounds_b, tolerance)
                hits_b_in_a = match_count(bounds_b, bounds_a, tolerance)
                precision = hits_a_in_b / len(bounds_b) if bounds_b else 0
                recall = hits_b_in_a / len(bounds_a) if bounds_a else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
            else:
                precision = recall = f1 = 0.0

            results["pairwise"][f"{a}_vs_{b}"] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                f"n_{a}_matched_in_{b}": hits_b_in_a if len(bounds_a) > 0 else 0,
                f"n_{b}_matched_in_{a}": hits_a_in_b if len(bounds_b) > 0 else 0,
            }

    # Three-way agreement: boundaries found by ALL three methods
    if len(names) == 3:
        three_way = []
        # Use the method with fewest boundaries as anchor
        anchor_name = min(names, key=lambda n: len(methods[n]))
        others = [n for n in names if n != anchor_name]
        for b in methods[anchor_name]:
            matched_all = True
            for other in others:
                other_arr = np.array(methods[other])
                if len(other_arr) == 0 or np.min(np.abs(other_arr - b)) > tolerance:
                    matched_all = False
                    break
            if matched_all:
                three_way.append(b)
        results["three_way"] = {
            "n_agreed": len(three_way),
            "frames": three_way,
            "anchor_method": anchor_name,
        }

    return results


# ── Main ───────────────────────────────────────────────────────────────────
def process_movie(video_path: Path):
    """Run all 3 shot detection methods on a single movie."""
    movie_name = video_path.stem
    outdir = OUTPUT_DIR / movie_name / "shot_detection"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'━' * 70}")
    print(f"  Processing: {movie_name}")
    print(f"{'━' * 70}")
    info = get_video_info(video_path)
    print(f"  FPS: {info['fps']}, Frames: {info['n_frames']}, "
          f"Size: {info['width']}x{info['height']}")
    print()

    methods = {}

    # 1. PySceneDetect
    print("═══ Method 1: PySceneDetect (ContentDetector, threshold=27) ═══")
    boundaries_psd = detect_pyscenedetect(video_path)
    methods["pyscenedetect"] = boundaries_psd
    print(f"  Found {len(boundaries_psd)} shot boundaries")
    print(f"  Frames: {boundaries_psd}")
    save_boundary_frames(video_path, boundaries_psd, "pyscenedetect", outdir)
    print()

    # 2. TransNetV2
    print("═══ Method 2: TransNetV2 (threshold=0.5) ═══")
    boundaries_tv2 = detect_transnetv2(video_path)
    methods["transnetv2"] = boundaries_tv2
    print(f"  Found {len(boundaries_tv2)} shot boundaries")
    print(f"  Frames: {boundaries_tv2}")
    save_boundary_frames(video_path, boundaries_tv2, "transnetv2", outdir)
    print()

    # 3. FFmpeg
    print("═══ Method 3: FFmpeg (scene filter, threshold=0.3) ═══")
    boundaries_ffm = detect_ffmpeg(video_path)
    methods["ffmpeg"] = boundaries_ffm
    print(f"  Found {len(boundaries_ffm)} shot boundaries")
    print(f"  Frames: {boundaries_ffm}")
    save_boundary_frames(video_path, boundaries_ffm, "ffmpeg", outdir)
    print()

    # ── Agreement ─────────────────────────────────────────────────────────
    print("═══ Agreement Analysis (tolerance = 12 frames / 0.5s) ═══")
    agreement = compute_agreement(methods, info["n_frames"], tolerance=12)

    for name, stats in agreement["per_method"].items():
        print(f"  {name}: {stats['n_boundaries']} boundaries")
    print()

    for pair, stats in agreement["pairwise"].items():
        print(f"  {pair}:")
        print(f"    Precision={stats['precision']:.3f}  "
              f"Recall={stats['recall']:.3f}  F1={stats['f1']:.3f}")
    print()

    tw = agreement["three_way"]
    print(f"  Three-way agreement: {tw['n_agreed']} boundaries "
          f"(anchor: {tw['anchor_method']})")
    if tw["frames"]:
        print(f"  Agreed frames: {tw['frames']}")
    print()

    # Save three-way agreed boundary frames too
    if tw["frames"]:
        save_boundary_frames(video_path, tw["frames"], "three_way_agreed", outdir)

    # ── Save results ──────────────────────────────────────────────────────
    summary = {
        "video": str(video_path),
        "video_info": info,
        "tolerance_frames": 12,
        "methods": {name: {"boundaries": bounds} for name, bounds in methods.items()},
        "agreement": agreement,
    }
    summary_path = outdir / "shot_detection_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {summary_path}")

    # Per-method CSVs for easy review
    fps = info["fps"]
    for name, bounds in methods.items():
        df = pd.DataFrame({
            "shot_number": range(1, len(bounds) + 1),
            "boundary_frame": bounds,
            "timestamp_s": [b / fps for b in bounds],
        })
        csv_path = outdir / f"{name}_boundaries.csv"
        df.to_csv(csv_path, index=False)
        print(f"  {name} CSV → {csv_path}")

    # Combined comparison CSV
    max_len = max(len(b) for b in methods.values())
    combined = {}
    for name, bounds in methods.items():
        padded = bounds + [None] * (max_len - len(bounds))
        combined[f"{name}_frame"] = padded
        combined[f"{name}_time_s"] = [
            round(b / fps, 3) if b is not None else None for b in padded
        ]
    df_combined = pd.DataFrame(combined)
    combined_path = outdir / "all_methods_comparison.csv"
    df_combined.to_csv(combined_path, index=False)
    print(f"  Combined comparison → {combined_path}")


def main():
    # Process specific movies from CLI args, or all movies in the directory
    if len(sys.argv) > 1:
        videos = [Path(v) for v in sys.argv[1:]]
    else:
        videos = sorted(MOVIES_DIR.glob("*.mp4"))

    for video_path in videos:
        process_movie(video_path)


if __name__ == "__main__":
    main()
