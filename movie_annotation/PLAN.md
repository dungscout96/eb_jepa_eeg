# Plan: EEG-Relevant Visual Feature Annotation of Movie Frames

## Context
Annotate every frame of 4 movies (~21K total frames at 640x360) with visual features known to be detectable via EEG, based on the literature scan in `skill.md`. The output will be per-movie Parquet files with frame-level features and `.npz` files with high-dimensional CLIP embeddings. Code runs as a Slurm GPU job on NCSA Delta.

## Project Structure
```
movie_annotation/
├── pyproject.toml              # uv project with dependencies
├── annotate.py                 # Main entry point (CLI)
├── features/
│   ├── __init__.py
│   ├── lowlevel.py             # CPU-based low-level visual features
│   ├── motion.py               # Optical flow + scene cuts
│   ├── depth.py                # Monocular depth estimation (DPT)
│   ├── objects.py              # Object detection (DETR)
│   ├── faces.py                # Face detection (OpenCV DNN)
│   └── clip_features.py        # CLIP embeddings + zero-shot scene classification
├── submit_job.sh               # Slurm sbatch script
├── movies/                     # (existing) 4 mp4 files
├── output/                     # (created at runtime)
│   ├── The_Present/
│   │   ├── features.parquet
│   │   └── clip_embeddings.npz
│   ├── despicable_me/  ...
│   ├── diary_of_a_whimpy_kid/  ...
│   └── fun_with_fractals/  ...
└── skill.md                    # (existing) task description
```

## Features Per Frame (Parquet Columns)

### 1. Physical & Spatial (Low-Level) — literature sections 1 & 3
| Column | Description | Method |
|--------|-------------|--------|
| `frame_idx` | Frame number (0-indexed) | — |
| `timestamp_s` | Time in seconds | frame_idx / fps |
| `luminance_mean` | Mean brightness | Mean of grayscale |
| `contrast_rms` | RMS contrast | Std of grayscale |
| `color_r_mean/g/b` | Mean RGB channels | Per-channel mean |
| `saturation_mean` | Mean color saturation | HSV saturation channel |
| `edge_density` | Fraction of edge pixels | Canny edge detection |
| `spatial_freq_energy` | High-freq energy ratio | 2D FFT power spectrum |
| `entropy` | Visual complexity | Shannon entropy of grayscale histogram |

### 2. Depth — literature section 1 (perceived depth, retinal vs real-world size)
| Column | Description | Method |
|--------|-------------|--------|
| `depth_mean` | Mean predicted depth | Intel/dpt-large (MiDaS) |
| `depth_std` | Depth variation | std of depth map |
| `depth_range` | Depth range (max-min) | range of depth map |

### 3. Dynamic & Temporal — literature section 3 (neural tracking, motion)
| Column | Description | Method |
|--------|-------------|--------|
| `motion_energy` | Mean optical flow magnitude | Farneback dense optical flow |
| `scene_cut` | Scene transition flag (bool) | Histogram correlation threshold |

### 4. Semantic & Categorical — literature section 2
| Column | Description | Method |
|--------|-------------|--------|
| `n_objects` | Number of detected objects | facebook/detr-resnet-50 |
| `object_categories` | JSON: {category: count} | DETR labels |
| `n_faces` | Number of detected faces | OpenCV DNN face detector |
| `face_area_frac` | Total face area / frame area | Face bounding boxes |
| `scene_category` | Top-1 scene class | CLIP zero-shot (15 scene types) |
| `scene_category_score` | Confidence of top-1 | CLIP similarity |
| `scene_natural_score` | Nature vs urban score | CLIP: "natural scene" |
| `scene_open_score` | Open vs enclosed score | CLIP: "open outdoor scene" |

### 5. CLIP Embeddings (separate .npz)
| Key | Shape | Description |
|-----|-------|-------------|
| `embeddings` | (n_frames, 512) | CLIP ViT-B/32 visual features |
| `frame_indices` | (n_frames,) | Corresponding frame indices |

## HuggingFace Models
1. **`Intel/dpt-large`** — monocular depth estimation (captures perceived depth)
2. **`facebook/detr-resnet-50`** — object detection + categorization
3. **`openai/clip-vit-base-patch32`** — visual embeddings + zero-shot scene classification

## Processing Strategy
- **Chunked processing**: Read 64 frames at a time from video
- For each chunk: compute all CPU features, then batch through each GPU model
- GPU batch size: 16 (safe for A40 48GB VRAM at 640x360)
- Process all 4 movies sequentially within one job

## Dependencies (pyproject.toml via uv)
- `opencv-python-headless`, `numpy`, `pandas`, `scipy`
- `torch`, `torchvision` (CUDA 12.4 wheels)
- `transformers`, `Pillow`, `pyarrow`, `tqdm`

## Slurm Job Config (submit_job.sh)
- Partition: `gpuA40x4` (0.5x charge factor)
- GPUs: 1, CPUs: 8, Memory: 64GB
- Account: `bbnv-delta-gpu`
- Wall time: 4 hours
- Loads `cuda/12.8` module
- Runs `uv run python annotate.py`

## Implementation Steps
1. Create `pyproject.toml` with uv config and all dependencies
2. Create `features/__init__.py`
3. Create `features/lowlevel.py` — luminance, contrast, color, edges, spatial freq, entropy
4. Create `features/motion.py` — optical flow, scene cuts
5. Create `features/depth.py` — DPT depth estimation with batched inference
6. Create `features/objects.py` — DETR object detection with batched inference
7. Create `features/faces.py` — OpenCV DNN face detection
8. Create `features/clip_features.py` — CLIP embeddings + zero-shot scene labels
9. Create `annotate.py` — CLI entry point, orchestrates chunked processing, saves output
10. Create `submit_job.sh` — Slurm batch script
11. Run `uv sync` on login node to install dependencies
12. Submit job with `sbatch submit_job.sh`

## Verification
1. `uv sync` completes without errors on login node
2. `sbatch submit_job.sh` submits successfully
3. After job completes: check `output/<movie>/features.parquet` has expected columns and row counts matching frame counts
4. Check `output/<movie>/clip_embeddings.npz` has correct shapes
5. Spot-check: `scene_cut` flags should be sparse, `motion_energy` should be 0 for first frame, `n_faces` should be non-negative integers
