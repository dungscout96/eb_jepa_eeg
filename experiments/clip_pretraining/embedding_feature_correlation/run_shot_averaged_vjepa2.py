"""Does averaging V-JEPA-2 embeddings *by shot* preserve the feature signal?

The companion `run.py` checks how well a per-clip embedding (2 Hz, 1408-d)
predicts the scalar movie features. Here we ask the downstream question that
matters for shot-level EEG↔video alignment: if we collapse every clip inside a
shot to a single mean embedding, does that shot-averaged vector still carry the
visual content?

Procedure (mirrors run.py so the two are apples-to-apples):
  1. Align each clip to the mean of the 24 fps frame features over its 0.5 s span.
  2. Assign each clip to the shot it spends most of its span in (modal shot_id).
  3. Aggregate to shot level: mean embedding and mean feature per shot.
  4. Fit RidgeCV per scalar target with 5-fold CV, at the *shot* level, and
     compare cross-validated R^2 to the per-clip baseline.

Two effects are folded into the shot-level R^2 and we report enough to separate
them:
  - information loss from averaging the embedding (what we care about), and
  - the change in sample size / target variance (54 shots vs 406 clips).
To isolate the first, we also run a "broadcast" control: keep all 406 clip rows
but replace each clip's embedding with its shot-mean embedding, predicting the
per-clip target. The gap between per-clip and broadcast R^2 is the pure cost of
discarding within-shot embedding detail; the gap between broadcast and shot is
the effect of also aggregating the target.

See `run_shot_averaged_clip.py` for the same analysis on OpenAI CLIP frame
embeddings.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_shot_averaged_vjepa2.py

Outputs:
    experiments/embedding_feature_correlation/vjepa2_results_shot_averaged.csv
    experiments/embedding_feature_correlation/vjepa2_results_shot_averaged.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (  # noqa: E402  reuse the exact same alignment / eval logic
    DEFAULT_EMBEDDINGS,
    DEFAULT_FEATURES,
    SCALAR_TARGETS,
    align_features_to_clips,
    evaluate_regression,
)


def evaluate_regression_pca(X: np.ndarray, y: np.ndarray, n_components: int,
                            n_splits: int = 5) -> dict:
    """Ridge on PCA-reduced embeddings — robust when samples << dims (shot level).

    With only ~54 shots, a 1408-d ridge gives wildly noisy (often negative) CV
    R^2 that reflects sample size, not signal. Reducing to a handful of PCs
    first gives a fairer read on how much shot-level signal survives.
    """
    valid = ~np.isnan(y)
    Xv, yv = X[valid], y[valid]
    n = len(yv)
    if n < n_splits + 1 or np.std(yv) < 1e-9:
        return {"r2_mean": float("nan"), "r2_std": float("nan"), "n": int(n)}
    k = min(n_components, n - n_splits, Xv.shape[1])
    pipe = make_pipeline(StandardScaler(), PCA(n_components=k),
                         RidgeCV(alphas=np.logspace(-2, 4, 13)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, Xv, yv, cv=cv, scoring="r2")
    return {"r2_mean": float(scores.mean()), "r2_std": float(scores.std()),
            "n": int(n), "n_pca": int(k)}

DEFAULT_OUTDIR = Path(__file__).resolve().parent


def assign_clip_shots(features: pd.DataFrame, timestamps: np.ndarray,
                      freq: float) -> np.ndarray:
    """Return the modal shot_id for each clip's [t, t + 1/freq) frame span.

    NaN where no frame falls in the clip span (so the clip can be dropped).
    """
    span = 1.0 / freq
    frame_ts = features["timestamp_s"].to_numpy()
    shot = features["shot_id"].to_numpy()
    out = np.full(len(timestamps), np.nan)
    for i, t in enumerate(timestamps):
        mask = (frame_ts >= t) & (frame_ts < t + span)
        if mask.any():
            vals, counts = np.unique(shot[mask], return_counts=True)
            out[i] = vals[counts.argmax()]
    return out


def aggregate_by_shot(X: np.ndarray, aligned: pd.DataFrame,
                      clip_shot: np.ndarray):
    """Collapse clips to one row per shot: mean embedding + mean feature.

    Returns (X_shot, aligned_shot, shot_ids, clips_per_shot).
    """
    valid = ~np.isnan(clip_shot)
    shots = np.unique(clip_shot[valid]).astype(int)
    X_shot = np.zeros((len(shots), X.shape[1]), dtype=X.dtype)
    rows = []
    clips_per_shot = []
    for j, s in enumerate(shots):
        m = clip_shot == s
        X_shot[j] = X[m].mean(axis=0)
        clips_per_shot.append(int(m.sum()))
        rows.append(aligned.loc[m].mean(numeric_only=True))
    return X_shot, pd.DataFrame(rows).reset_index(drop=True), shots, np.array(clips_per_shot)


def broadcast_shot_embeddings(X: np.ndarray, clip_shot: np.ndarray) -> np.ndarray:
    """Replace each clip's embedding with its shot-mean, keeping all clip rows."""
    Xb = X.copy()
    for s in np.unique(clip_shot[~np.isnan(clip_shot)]):
        m = clip_shot == s
        Xb[m] = X[m].mean(axis=0)
    return Xb


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
    print(f"  X shape={X.shape}, freq={freq} Hz")

    feats = pd.read_parquet(args.features)
    if "shot_id" not in feats.columns:
        raise SystemExit("features parquet has no shot_id column")

    aligned = align_features_to_clips(feats, timestamps, freq=freq, fps=args.fps)
    clip_shot = assign_clip_shots(feats, timestamps, freq=freq)
    n_valid = int((~np.isnan(clip_shot)).sum())

    X_shot, aligned_shot, shots, cps = aggregate_by_shot(X, aligned, clip_shot)
    X_bcast = broadcast_shot_embeddings(X, clip_shot)
    print(f"  clips={X.shape[0]} (valid={n_valid}), shots={len(shots)}, "
          f"clips/shot min/median/max={cps.min()}/{int(np.median(cps))}/{cps.max()}")

    # Per-clip baseline uses only clips with an assigned shot, for a fair n.
    keep = ~np.isnan(clip_shot)
    X_clip = X[keep]
    aligned_clip = aligned.loc[keep].reset_index(drop=True)

    print("\nR^2 comparison (5-fold CV ridge). 'shot' is the raw 1408-d fit; "
          "'shotPCA'\nis PCA-reduced and is the trustworthy shot-level read.")
    print(f"{'feature':<24} {'clip':>8} {'broadcast':>10} {'shot':>8} "
          f"{'shotPCA':>9}")
    print("-" * 64)
    rows = []
    for col in SCALAR_TARGETS:
        if col not in aligned_shot.columns:
            print(f"{col:<24}  [missing column]")
            continue
        y_clip = aligned_clip[col].to_numpy().astype(np.float32)
        y_shot = aligned_shot[col].to_numpy().astype(np.float32)
        r_clip = evaluate_regression(X_clip, y_clip)
        r_bcast = evaluate_regression(X_bcast[keep], y_clip)
        r_shot = evaluate_regression(X_shot, y_shot)
        r_shot_pca = evaluate_regression_pca(X_shot, y_shot, n_components=20)
        print(f"{col:<24} {r_clip['r2_mean']:>8.3f} {r_bcast['r2_mean']:>10.3f} "
              f"{r_shot['r2_mean']:>8.3f} {r_shot_pca['r2_mean']:>9.3f}")
        rows.append({
            "target": col,
            "r2_clip": r_clip["r2_mean"], "r2_clip_std": r_clip["r2_std"],
            "r2_broadcast": r_bcast["r2_mean"], "r2_broadcast_std": r_bcast["r2_std"],
            "r2_shot": r_shot["r2_mean"], "r2_shot_std": r_shot["r2_std"],
            "r2_shot_pca": r_shot_pca["r2_mean"], "r2_shot_pca_std": r_shot_pca["r2_std"],
            "delta_bcast_minus_clip": r_bcast["r2_mean"] - r_clip["r2_mean"],
            "n_clip": r_clip["n"], "n_shot": r_shot["n"],
        })

    df = pd.DataFrame(rows)
    out_csv = args.outdir / "vjepa2_results_shot_averaged.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    mean_clip = df["r2_clip"].mean()
    mean_bcast = df["r2_broadcast"].mean()
    mean_shot = df["r2_shot"].mean()
    mean_shot_pca = df["r2_shot_pca"].mean()
    print(f"\nMean R^2 over {len(df)} scalar targets:")
    print(f"  per-clip embedding              : {mean_clip:.3f}")
    print(f"  broadcast shot-mean (n=clips)   : {mean_bcast:.3f}  "
          f"(pure embedding-averaging loss: {mean_bcast - mean_clip:+.3f})")
    print(f"  shot-level raw 1408-d (n={len(shots)})    : {mean_shot:.3f}  "
          f"(dominated by small-n CV noise)")
    print(f"  shot-level PCA-20     (n={len(shots)})    : {mean_shot_pca:.3f}  "
          f"(trustworthy shot-level read)")
    print("\nTakeaway: the broadcast control isolates the cost of averaging the "
          "embedding\nitself — if it stays close to per-clip, shot-mean "
          "embeddings preserve the signal.")

    # Plot: per-clip vs broadcast vs shot-PCA R^2 per target, sorted by clip R^2.
    d = df.sort_values("r2_clip", ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(d))))
    ax.barh(y - 0.26, d["r2_clip"], height=0.26, color="steelblue", label="per-clip")
    ax.barh(y, d["r2_broadcast"], height=0.26, color="seagreen",
            label="broadcast shot-mean (n=clips)")
    ax.barh(y + 0.26, d["r2_shot_pca"], height=0.26, color="darkorange",
            label=f"shot-level PCA-20 (n={len(shots)})")
    ax.set_yticks(y)
    ax.set_yticklabels(d["target"])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(left=-0.3)
    ax.set_xlabel("5-fold CV R^2")
    ax.set_title(f"Does shot-averaging V-JEPA-2 preserve feature signal? "
                 f"({len(shots)} shots)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_png = args.outdir / "vjepa2_results_shot_averaged.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
