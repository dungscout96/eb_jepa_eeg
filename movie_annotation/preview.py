"""Load and preview feature dataframes for all movies."""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("output")

movies = sorted(p.parent.name for p in OUTPUT_DIR.glob("*/features.parquet"))

for movie in movies:
    df = pd.read_parquet(OUTPUT_DIR / movie / "features.parquet")
    emb = np.load(OUTPUT_DIR / movie / "clip_embeddings.npz")

    print(f"=== {movie} ({len(df)} frames) ===")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
    print(f"\nClip embeddings: {emb['embeddings'].shape}")
    print(df.describe())
    print()

    df.to_csv(OUTPUT_DIR / movie / "features.csv", index=False)
