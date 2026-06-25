"""Sanity check: how much variance in scalar movie features does the V-JEPA-2
frame embedding explain?

For each clip-anchored V-JEPA-2 embedding (~2 Hz, 1408-d), average the parquet
frame features (24 fps) over the clip's 0.5 s span. Then fit a ridge regression
(scalar features) or multinomial logistic regression (shot_id) per target, using
5-fold cross-validation. Report cross-validated R^2 / accuracy.

If V-JEPA-2 truly captures the visual content, low-level descriptors like
luminance and motion should land at high R^2 (>0.5); shot_id should be near
perfect because shot identity is a deterministic function of frame content.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run.py

Outputs:
    experiments/embedding_feature_correlation/results.csv
    experiments/embedding_feature_correlation/results.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "movie_annotation/output/The_Present/vjepa2_embeddings.npz"
DEFAULT_FEATURES = PROJECT_ROOT / "movie_annotation/output/The_Present/features_enriched.parquet"
DEFAULT_OUTDIR = Path(__file__).resolve().parent

SCALAR_TARGETS = [
    "luminance_mean",
    "contrast_rms",
    "saturation_mean",
    "edge_density",
    "spatial_freq_energy",
    "entropy",
    "motion_energy",
    "n_objects",
    "n_faces",
    "face_area_frac",
    "depth_mean",
    "depth_std",
    "scene_natural_score",
    "scene_open_score",
    "position_in_movie",
    "narrative_event_score",
]


def align_features_to_clips(features: pd.DataFrame, timestamps: np.ndarray,
                            freq: float, fps: float) -> pd.DataFrame:
    """For each clip at timestamp t, average frame rows whose timestamp_s falls in [t, t+1/freq)."""
    clip_span_s = 1.0 / freq
    rows = []
    frame_ts = features["timestamp_s"].to_numpy()
    for t in timestamps:
        mask = (frame_ts >= t) & (frame_ts < t + clip_span_s)
        if not mask.any():
            rows.append({c: np.nan for c in features.columns})
            continue
        sub = features.loc[mask]
        row = {}
        for c in sub.columns:
            if pd.api.types.is_numeric_dtype(sub[c]):
                row[c] = sub[c].mean()
            else:
                # categorical / object — take the modal value
                row[c] = sub[c].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_regression(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """5-fold CV ridge regression. Returns mean and std of R^2."""
    valid = ~np.isnan(y)
    Xv, yv = X[valid], y[valid]
    if len(yv) < n_splits or np.std(yv) < 1e-9:
        return {"r2_mean": float("nan"), "r2_std": float("nan"), "n": int(valid.sum()),
                "pearson": float("nan")}
    pipe_X = StandardScaler().fit_transform(Xv)
    model = RidgeCV(alphas=np.logspace(-2, 4, 13))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(model, pipe_X, yv, cv=cv, scoring="r2")
    # Also report Pearson between predicted (full-fit) and true for context.
    model.fit(pipe_X, yv)
    pearson = float(np.corrcoef(model.predict(pipe_X), yv)[0, 1])
    return {
        "r2_mean": float(scores.mean()),
        "r2_std": float(scores.std()),
        "n": int(valid.sum()),
        "pearson_train": pearson,
    }


def evaluate_classification(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """5-fold CV multinomial logistic regression. Returns mean and std accuracy."""
    valid = ~pd.isna(y)
    Xv, yv = X[valid], y[valid].astype(int)
    n_classes = len(np.unique(yv))
    if len(yv) < n_splits or n_classes < 2:
        return {"acc_mean": float("nan"), "acc_std": float("nan"),
                "n": int(valid.sum()), "n_classes": int(n_classes),
                "chance": float("nan")}
    Xs = StandardScaler().fit_transform(Xv)
    model = LogisticRegression(max_iter=2000, C=1.0)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(model, Xs, yv, cv=cv, scoring="accuracy")
    # Chance = majority-class frequency
    _, counts = np.unique(yv, return_counts=True)
    chance = counts.max() / counts.sum()
    return {
        "acc_mean": float(scores.mean()),
        "acc_std": float(scores.std()),
        "n": int(valid.sum()),
        "n_classes": int(n_classes),
        "chance": float(chance),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings)
    X = data["embeddings"].astype(np.float32)
    timestamps = data["timestamps"].astype(np.float64)
    freq = float(data["frequency"]) if "frequency" in data.files else 2.0
    print(f"  X shape={X.shape}, freq={freq} Hz, span=[{timestamps[0]:.2f}, {timestamps[-1]:.2f}] s")

    print(f"Loading per-frame features from {args.features}")
    feats = pd.read_parquet(args.features)
    print(f"  frames={len(feats)}, columns={len(feats.columns)}")

    print("Aligning frame features to clip timestamps...")
    aligned = align_features_to_clips(feats, timestamps, freq=freq, fps=args.fps)
    print(f"  aligned rows={len(aligned)}")

    # Regression targets
    print("\nFitting ridge regression for scalar targets (5-fold CV R^2):")
    print(f"{'feature':<26}  {'R^2 mean':>10}  {'R^2 std':>10}  {'pearson':>10}  {'n':>5}")
    print("-" * 72)
    rows = []
    for col in SCALAR_TARGETS:
        if col not in aligned.columns:
            print(f"{col:<26}  [missing column]")
            continue
        y = aligned[col].to_numpy().astype(np.float32)
        res = evaluate_regression(X, y)
        print(f"{col:<26}  {res['r2_mean']:>10.4f}  {res['r2_std']:>10.4f}  "
              f"{res['pearson_train']:>10.4f}  {res['n']:>5}")
        rows.append({"target": col, "kind": "regression", **res})

    # Categorical target: shot_id
    if "shot_id" in aligned.columns:
        print("\nFitting logistic regression for shot_id (5-fold CV accuracy):")
        # Use modal shot per clip (categorical mean is not defined)
        # align_features_to_clips already took iloc[0] for non-numeric; shot_id is int
        # but pandas may treat it as numeric so we already have the mean — round it.
        shot_aligned = aligned["shot_id"]
        if pd.api.types.is_numeric_dtype(shot_aligned):
            shot_y = shot_aligned.round().to_numpy()
        else:
            shot_y = shot_aligned.to_numpy()
        res = evaluate_classification(X, shot_y)
        print(f"{'shot_id':<26}  acc={res['acc_mean']:.4f} ± {res['acc_std']:.4f}  "
              f"(chance={res['chance']:.4f}, classes={res['n_classes']}, n={res['n']})")
        rows.append({"target": "shot_id", "kind": "classification",
                     "r2_mean": res["acc_mean"], "r2_std": res["acc_std"],
                     "n": res["n"], "pearson_train": float("nan")})

    df = pd.DataFrame(rows)
    out_csv = args.outdir / "results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Plot
    reg = df[df["kind"] == "regression"].sort_values("r2_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(reg))))
    ax.barh(reg["target"], reg["r2_mean"], xerr=reg["r2_std"], color="steelblue")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("5-fold CV R^2")
    ax.set_title("V-JEPA-2 embedding → scalar movie feature")
    ax.set_xlim(left=min(-0.05, reg["r2_mean"].min() - 0.05),
                right=max(1.0, reg["r2_mean"].max() + 0.1))
    fig.tight_layout()
    out_png = args.outdir / "results.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
