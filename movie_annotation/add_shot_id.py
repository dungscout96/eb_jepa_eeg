"""Add a `shot_id` integer column to `features_enriched.parquet`.

Maps each frame to its shot using TransNet V2 boundary timestamps. Boundaries
are exclusive starts of the next shot (frame == boundary_frame belongs to the
NEW shot). Shot IDs are 0-indexed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PARQUET = Path("movie_annotation/output/The_Present/features_enriched.parquet")
DEFAULT_BOUNDARIES = Path(
    "movie_annotation/output/The_Present/shot_detection/transnetv2_boundaries.csv"
)


def add_shot_id(parquet_path: Path, boundaries_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    boundaries = pd.read_csv(boundaries_path)["boundary_frame"].to_numpy()
    df["shot_id"] = np.searchsorted(boundaries, df["frame_idx"].to_numpy(), side="right")
    df["shot_id"] = df["shot_id"].astype("int32")
    df.to_parquet(parquet_path, index=False)
    n_shots = int(df["shot_id"].max()) + 1
    print(f"Wrote shot_id to {parquet_path}: {n_shots} shots, "
          f"range [{df['shot_id'].min()}, {df['shot_id'].max()}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    args = ap.parse_args()
    add_shot_id(args.parquet, args.boundaries)


if __name__ == "__main__":
    main()
