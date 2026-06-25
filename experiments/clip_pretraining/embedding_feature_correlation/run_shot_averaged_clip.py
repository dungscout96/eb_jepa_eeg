"""Does averaging OpenAI CLIP frame embeddings *by shot* preserve the signal?

Per-frame CLIP variant of `run_shot_averaged_vjepa2.py`. OpenAI CLIP emits one
512-d embedding per video frame (~24 fps, 4877 frames). Each frame already has a
`frame_idx` → `shot_id` mapping in the parquet, so shot assignment is a direct
lookup (no temporal-modal step needed, unlike the V-JEPA-2 case where each clip
spans 0.5 s of frames).

Three R^2 measurements per scalar target (5-fold CV ridge):
  - `r2_frame`     (n=4877): per-frame embedding -> per-frame feature.
  - `r2_broadcast` (n=4877): each frame's embedding REPLACED by its shot-mean,
                             still predicting the per-frame target. The drop vs
                             per-frame is the pure cost of averaging.
  - `r2_shot`/`r2_shot_pca` (n=54): one mean embedding + mean feature per shot.
                                    Raw 512-d and PCA-20.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_shot_averaged_clip.py

Outputs:
    experiments/embedding_feature_correlation/clip_results_shot_averaged.csv
    experiments/embedding_feature_correlation/clip_results_shot_averaged.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import SCALAR_TARGETS, evaluate_regression  # noqa: E402
from run_shot_averaged_vjepa2 import evaluate_regression_pca  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "movie_annotation/output/The_Present/clip_embeddings.npz"
DEFAULT_FEATURES = PROJECT_ROOT / "movie_annotation/output/The_Present/features_enriched.parquet"
DEFAULT_OUTDIR = Path(__file__).resolve().parent


def load_clip_frames(emb_path: Path, feats: pd.DataFrame):
    """Load CLIP frame embeddings and join per-frame shot_id from the parquet."""
    d = np.load(emb_path)
    X = d["embeddings"].astype(np.float32)
    frame_idx = d["frame_indices"].astype(np.int64)
    if "shot_id" not in feats.columns or "frame_idx" not in feats.columns:
        raise SystemExit("features parquet needs frame_idx and shot_id columns")
    aligned = feats.set_index("frame_idx").reindex(frame_idx)
    valid = aligned["shot_id"].notna().to_numpy()
    return X[valid], aligned.loc[valid].reset_index(drop=True)


def aggregate_by_shot(X: np.ndarray, aligned: pd.DataFrame, shot: np.ndarray):
    """Collapse frames to one row per shot: mean embedding + mean feature."""
    shots = np.unique(shot)
    X_shot = np.zeros((len(shots), X.shape[1]), dtype=X.dtype)
    rows, n_per = [], []
    for j, s in enumerate(shots):
        m = shot == s
        X_shot[j] = X[m].mean(axis=0)
        rows.append(aligned.loc[m].mean(numeric_only=True))
        n_per.append(int(m.sum()))
    return X_shot, pd.DataFrame(rows).reset_index(drop=True), shots, np.array(n_per)


def broadcast_shot_embeddings(X: np.ndarray, shot: np.ndarray) -> np.ndarray:
    out = X.copy()
    for s in np.unique(shot):
        m = shot == s
        out[m] = X[m].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP embeddings from {args.embeddings}")
    feats = pd.read_parquet(args.features)
    X, aligned = load_clip_frames(args.embeddings, feats)
    shot = aligned["shot_id"].to_numpy().astype(int)

    X_shot, aligned_shot, shots, n_per = aggregate_by_shot(X, aligned, shot)
    X_bcast = broadcast_shot_embeddings(X, shot)
    print(f"  frames={X.shape[0]}, shots={len(shots)}, "
          f"frames/shot min/median/max={n_per.min()}/{int(np.median(n_per))}/{n_per.max()}")

    print("\nR^2 comparison (5-fold CV ridge). 'shot' = raw 512-d fit (n=54); "
          "'shotPCA'\nis PCA-reduced and is the trustworthy shot-level read.")
    print(f"{'feature':<24} {'frame':>8} {'broadcast':>10} {'shot':>8} "
          f"{'shotPCA':>9}")
    print("-" * 64)
    rows = []
    for col in SCALAR_TARGETS:
        if col not in aligned_shot.columns:
            print(f"{col:<24}  [missing column]")
            continue
        y_frame = aligned[col].to_numpy().astype(np.float32)
        y_shot = aligned_shot[col].to_numpy().astype(np.float32)
        r_frame = evaluate_regression(X, y_frame)
        r_bcast = evaluate_regression(X_bcast, y_frame)
        r_shot = evaluate_regression(X_shot, y_shot)
        r_shot_pca = evaluate_regression_pca(X_shot, y_shot, n_components=20)
        print(f"{col:<24} {r_frame['r2_mean']:>8.3f} {r_bcast['r2_mean']:>10.3f} "
              f"{r_shot['r2_mean']:>8.3f} {r_shot_pca['r2_mean']:>9.3f}")
        rows.append({
            "target": col,
            "r2_frame": r_frame["r2_mean"], "r2_frame_std": r_frame["r2_std"],
            "r2_broadcast": r_bcast["r2_mean"], "r2_broadcast_std": r_bcast["r2_std"],
            "r2_shot": r_shot["r2_mean"], "r2_shot_std": r_shot["r2_std"],
            "r2_shot_pca": r_shot_pca["r2_mean"], "r2_shot_pca_std": r_shot_pca["r2_std"],
            "delta_bcast_minus_frame": r_bcast["r2_mean"] - r_frame["r2_mean"],
            "n_frame": r_frame["n"], "n_shot": r_shot["n"],
        })

    df = pd.DataFrame(rows)
    out_csv = args.outdir / "clip_results_shot_averaged.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    mean_frame = df["r2_frame"].mean()
    mean_bcast = df["r2_broadcast"].mean()
    mean_shot = df["r2_shot"].mean()
    mean_shot_pca = df["r2_shot_pca"].mean()
    print(f"\nMean R^2 over {len(df)} scalar targets:")
    print(f"  per-frame embedding             : {mean_frame:.3f}")
    print(f"  broadcast shot-mean (n=frames)  : {mean_bcast:.3f}  "
          f"(pure embedding-averaging loss: {mean_bcast - mean_frame:+.3f})")
    print(f"  shot-level raw 512-d (n={len(shots)})    : {mean_shot:.3f}  "
          f"(dominated by small-n CV noise)")
    print(f"  shot-level PCA-20    (n={len(shots)})    : {mean_shot_pca:.3f}  "
          f"(trustworthy shot-level read)")
    print("\nTakeaway: the broadcast control isolates the cost of averaging the "
          "embedding\nitself — if it stays close to per-frame, shot-mean "
          "embeddings preserve the signal.")

    # Plot
    d = df.sort_values("r2_frame", ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(d))))
    ax.barh(y - 0.26, d["r2_frame"], height=0.26, color="steelblue", label="per-frame")
    ax.barh(y, d["r2_broadcast"], height=0.26, color="seagreen",
            label="broadcast shot-mean (n=frames)")
    ax.barh(y + 0.26, d["r2_shot_pca"], height=0.26, color="darkorange",
            label=f"shot-level PCA-20 (n={len(shots)})")
    ax.set_yticks(y)
    ax.set_yticklabels(d["target"])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(left=-0.3)
    ax.set_xlabel("5-fold CV R^2")
    ax.set_title(f"OpenAI CLIP: does shot-averaging preserve feature signal? "
                 f"({len(shots)} shots)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_png = args.outdir / "clip_results_shot_averaged.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
