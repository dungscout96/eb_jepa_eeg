# V-JEPA-2 embedding ↔ scalar movie feature correlation

Sanity check: V-JEPA-2 frame embeddings should explain a large fraction of
variance in the existing scalar movie features (luminance, contrast, motion,
faces, depth, etc.) because both are derived from the same frames. If the
embeddings *do not* recover these basic signals, something is wrong with the
extraction or alignment.

## Method

- Align: each V-JEPA-2 clip (2 Hz, 0.5 s span) is paired with the mean of the
  per-frame parquet features falling in `[t, t + 0.5 s)`.
- Regression: per scalar target, fit `RidgeCV` (alphas log-spaced 1e-2…1e4) over
  L2-normalized embeddings; report 5-fold CV R².
- Classification (`shot_id`): multinomial logistic regression; report
  5-fold CV accuracy vs. majority-class chance.

## Run

```bash
PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run.py
```

Outputs `results.csv` and `results.png` alongside this README.

## Expected outcome

- Low-level descriptors (luminance, contrast, edge_density, motion) → R² > 0.5
- Shot_id → accuracy ≫ chance (53 boundaries, ~54 classes, chance ≈ 5%)
- `position_in_movie` → near-perfect R² (V-JEPA-2 is told the clip's content
  but not its time; any signal must come from gradual scene-level drift)
