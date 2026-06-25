"""Precompute V-JEPA-2 recipe artifacts for a movie.

Writes ``movie_annotation/output/<task>/vjepa2_recipe.npz`` containing:
- ``global_mean[D]``: mean V-JEPA-2 vector across all clips
- ``shot_means[n_shots, D]``: per-shot mean V-JEPA-2 vector
- ``scene_id_per_shot[n_shots]``: scene ID for each shot (from scene_map.csv)
- ``n_shots``: scalar

Reads paths from ``eb_jepa.datasets.hbn.MOVIE_METADATA[task]``. Required inputs:
``vjepa2_embeddings``, ``shot_boundaries``, ``scene_map``. Fails if any is missing.

Usage:
    python -m experiments.embedding_feature_correlation.precompute_vjepa2_recipe \\
        --task ThePresent
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eb_jepa.datasets.hbn import MOVIE_METADATA


def build_recipe(task: str) -> dict[str, np.ndarray]:
    meta = MOVIE_METADATA[task]
    fps = float(meta["fps"])

    vjepa_path = meta.get("vjepa2_embeddings")
    shot_path = meta.get("shot_boundaries")
    scene_path = meta.get("scene_map")
    for name, p in [("vjepa2_embeddings", vjepa_path),
                    ("shot_boundaries", shot_path),
                    ("scene_map", scene_path)]:
        if p is None or not Path(p).exists():
            raise FileNotFoundError(
                f"Required artifact '{name}' missing for task '{task}': {p}"
            )

    bundle = np.load(vjepa_path)
    embeddings = bundle["embeddings"].astype(np.float32)        # [N, D]
    timestamps = bundle["timestamps"].astype(np.float64)        # [N]
    n_clips, dim = embeddings.shape

    boundaries = np.sort(
        pd.read_csv(shot_path)["boundary_frame"].to_numpy().astype(np.int64)
    )                                                            # [n_shots-1]
    n_shots = len(boundaries) + 1
    boundary_seconds = boundaries / fps                          # [n_shots-1]

    # Clip-to-shot via clip start time
    shot_of_clip = np.searchsorted(boundary_seconds, timestamps, side="right")
    shot_of_clip = shot_of_clip.astype(np.int64)
    assert shot_of_clip.min() >= 0 and shot_of_clip.max() < n_shots

    shot_means = np.zeros((n_shots, dim), dtype=np.float32)
    shot_counts = np.zeros(n_shots, dtype=np.int64)
    for s in range(n_shots):
        in_shot = shot_of_clip == s
        shot_counts[s] = int(in_shot.sum())
        if shot_counts[s] == 0:
            raise RuntimeError(
                f"Shot {s} has no V-JEPA-2 clips. Movie '{task}' may have a "
                f"shot shorter than the V-JEPA-2 clip spacing (~0.5s)."
            )
        shot_means[s] = embeddings[in_shot].mean(axis=0)

    global_mean = embeddings.mean(axis=0).astype(np.float32)

    scene_df = pd.read_csv(scene_path)
    if not {"shot_id", "scene_id"}.issubset(scene_df.columns):
        raise ValueError(f"scene_map {scene_path} missing required columns shot_id,scene_id")
    scene_map = dict(zip(scene_df["shot_id"].astype(int),
                         scene_df["scene_id"].astype(int)))
    if set(scene_map.keys()) != set(range(n_shots)):
        missing = set(range(n_shots)) - set(scene_map.keys())
        extra = set(scene_map.keys()) - set(range(n_shots))
        raise ValueError(
            f"scene_map shot ids do not match shot count {n_shots}. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    scene_id_per_shot = np.array(
        [scene_map[s] for s in range(n_shots)], dtype=np.int64
    )

    print(
        f"[{task}] n_clips={n_clips} dim={dim} n_shots={n_shots} "
        f"n_scenes={len(set(scene_id_per_shot))} "
        f"clips_per_shot=min{shot_counts.min()}/median{int(np.median(shot_counts))}/max{shot_counts.max()}"
    )

    return {
        "global_mean": global_mean,
        "shot_means": shot_means,
        "scene_id_per_shot": scene_id_per_shot,
        "n_shots": np.int64(n_shots),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ThePresent", help="Movie key in MOVIE_METADATA")
    parser.add_argument("--out", default=None,
                        help="Override output path (defaults to MOVIE_METADATA[task]['vjepa2_recipe'])")
    args = parser.parse_args()

    out_path = args.out or MOVIE_METADATA[args.task]["vjepa2_recipe"]
    if out_path is None:
        raise SystemExit(
            f"Task '{args.task}' has no vjepa2_recipe path configured in MOVIE_METADATA."
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = build_recipe(args.task)
    np.savez(out_path, **artifacts)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
