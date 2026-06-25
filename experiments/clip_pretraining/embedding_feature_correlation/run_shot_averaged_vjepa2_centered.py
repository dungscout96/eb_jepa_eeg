"""Does mean-centering V-JEPA-2 embeddings change the shot-averaging story?

The variance analysis showed that *mean-centering* (subtracting the global mean
vector) is the meaningful preprocessing for V-JEPA-2's cosine geometry — it took
ROC-AUC from 0.88 → 0.92. Full PCA-whitening goes further (also decorrelates
and rescales each PC to unit variance), but for the shot-averaging regression
question, mean-centering is the relevant operation to check.

This script replays the 3-way comparison from `run_shot_averaged_vjepa2.py`
with explicit mean-centering inside CV folds, contrasted against the existing
per-dim-standardized "raw" baseline.

Important caveat: ridge regression with `fit_intercept=True` (the sklearn
default used by RidgeCV) is INVARIANT to translation of the input — it learns
its own intercept. So mean-centering alone should not change R^2 vs the raw
baseline at all. This script empirically verifies that, and serves as the
honest answer to "does shot-averaging still preserve signal after centering?":
yes, exactly as well as raw, because centering is regression-invariant.

Per-dim std-scaling (what `StandardScaler` does in the raw baseline) IS a
different operation; this script does NOT replicate that. The "centered"
column here is pure mean-centering only.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_shot_averaged_vjepa2_centered.py

Outputs:
    experiments/embedding_feature_correlation/vjepa2_centered_shot_averaged.csv
    experiments/embedding_feature_correlation/vjepa2_centered_shot_averaged.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (  # noqa: E402
    DEFAULT_EMBEDDINGS,
    DEFAULT_FEATURES,
    SCALAR_TARGETS,
    align_features_to_clips,
    evaluate_regression,
)
from run_shot_averaged_vjepa2 import (  # noqa: E402
    aggregate_by_shot,
    assign_clip_shots,
    broadcast_shot_embeddings,
)

DEFAULT_OUTDIR = Path(__file__).resolve().parent
ALPHAS = np.logspace(-2, 4, 13)


def evaluate_regression_centered(X: np.ndarray, y: np.ndarray,
                                 n_splits: int = 5) -> dict:
    """Ridge after MEAN-CENTERING only — no per-dim scaling.

    StandardScaler(with_std=False) subtracts the column means fit on the train
    fold and applies them to the test fold. Equivalent to the centering used
    in the variance analysis.
    """
    valid = ~np.isnan(y)
    Xv, yv = X[valid], y[valid]
    if len(yv) < n_splits or np.std(yv) < 1e-9:
        return {"r2_mean": float("nan"), "r2_std": float("nan"),
                "n": int(valid.sum())}
    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=False),
                         RidgeCV(alphas=ALPHAS))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, Xv, yv, cv=cv, scoring="r2")
    return {"r2_mean": float(scores.mean()), "r2_std": float(scores.std()),
            "n": int(valid.sum())}


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
    feats = pd.read_parquet(args.features)

    aligned = align_features_to_clips(feats, timestamps, freq=freq, fps=args.fps)
    clip_shot = assign_clip_shots(feats, timestamps, freq=freq)
    keep = ~np.isnan(clip_shot)
    X = X[keep]; clip_shot = clip_shot[keep]
    aligned_clip = aligned.loc[keep].reset_index(drop=True)

    X_shot, aligned_shot, shots, n_per = aggregate_by_shot(X, aligned, clip_shot)
    X_bcast = broadcast_shot_embeddings(X, clip_shot)
    n_clip, n_shot = X.shape[0], X_shot.shape[0]
    print(f"  clips={n_clip}, shots={n_shot}, "
          f"clips/shot min/median/max={n_per.min()}/{int(np.median(n_per))}/{n_per.max()}\n")

    print("R^2 comparison (5-fold CV ridge). 'raw' = StandardScaler "
          "(center + per-dim std) + Ridge;\n'centered' = mean-center only + Ridge. "
          "Ridge with fit_intercept is\ntranslation-invariant, so 'centered' "
          "should match 'raw' up to numerical noise\nWHERE the raw column is "
          "ALSO centered-only — i.e. they differ only because\nraw also applies "
          "per-dim std-scaling.")
    print(f"{'feature':<22} | {'clip':>16} | {'broadcast':>16} | {'shot (n=' + str(n_shot) + ')':>17}")
    print(f"{'':<22} | {'raw':>7}{'cent.':>9} | {'raw':>7}{'cent.':>9} | {'raw':>8}{'cent.':>9}")
    print("-" * 88)

    rows = []
    for col in SCALAR_TARGETS:
        if col not in aligned_shot.columns:
            print(f"{col:<22}  [missing column]")
            continue
        y_clip = aligned_clip[col].to_numpy().astype(np.float32)
        y_shot = aligned_shot[col].to_numpy().astype(np.float32)

        r_clip_raw = evaluate_regression(X, y_clip)
        r_clip_c = evaluate_regression_centered(X, y_clip)
        r_bcast_raw = evaluate_regression(X_bcast, y_clip)
        r_bcast_c = evaluate_regression_centered(X_bcast, y_clip)
        r_shot_raw = evaluate_regression(X_shot, y_shot)
        r_shot_c = evaluate_regression_centered(X_shot, y_shot)

        print(f"{col:<22} | "
              f"{r_clip_raw['r2_mean']:>7.3f}{r_clip_c['r2_mean']:>9.3f} | "
              f"{r_bcast_raw['r2_mean']:>7.3f}{r_bcast_c['r2_mean']:>9.3f} | "
              f"{r_shot_raw['r2_mean']:>8.3f}{r_shot_c['r2_mean']:>9.3f}")
        rows.append({
            "target": col,
            "r2_clip_raw": r_clip_raw["r2_mean"], "r2_clip_centered": r_clip_c["r2_mean"],
            "r2_broadcast_raw": r_bcast_raw["r2_mean"], "r2_broadcast_centered": r_bcast_c["r2_mean"],
            "r2_shot_raw": r_shot_raw["r2_mean"], "r2_shot_centered": r_shot_c["r2_mean"],
        })

    df = pd.DataFrame(rows)
    out_csv = args.outdir / "vjepa2_centered_shot_averaged.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    print(f"\nMean R^2 over {len(df)} scalar targets:")
    print(f"  {'level':<18}{'raw':>10}{'centered':>12}{'delta':>10}")
    for level, raw_col, c_col in [
        ("per-clip", "r2_clip_raw", "r2_clip_centered"),
        ("broadcast (mean)", "r2_broadcast_raw", "r2_broadcast_centered"),
        (f"shot (n={n_shot})", "r2_shot_raw", "r2_shot_centered"),
    ]:
        r = df[raw_col].mean(); c = df[c_col].mean()
        print(f"  {level:<18}{r:>10.3f}{c:>12.3f}{c - r:>+10.3f}")

    bcast_loss_raw = df["r2_broadcast_raw"].mean() - df["r2_clip_raw"].mean()
    bcast_loss_c = df["r2_broadcast_centered"].mean() - df["r2_clip_centered"].mean()
    print(f"\nPure embedding-averaging cost (broadcast - per-clip):")
    print(f"  raw     : {bcast_loss_raw:+.3f}")
    print(f"  centered: {bcast_loss_c:+.3f}")
    print("(If centered ≈ raw, mean-centering is regression-invariant; the\n"
          "raw baseline IS the centered baseline for downstream prediction.)")

    # ---- plot ----
    d = df.sort_values("r2_clip_raw", ascending=True)
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, 0.45 * len(d))),
                             sharey=True)
    for ax, suf, title in [(axes[0], "raw", "Raw (StandardScaler + Ridge)"),
                           (axes[1], "centered", "Mean-centered only + Ridge")]:
        ax.barh(y - 0.26, d[f"r2_clip_{suf}"], height=0.26, color="steelblue",
                label="per-clip")
        ax.barh(y, d[f"r2_broadcast_{suf}"], height=0.26, color="seagreen",
                label="broadcast shot-mean")
        ax.barh(y + 0.26, d[f"r2_shot_{suf}"], height=0.26, color="darkorange",
                label=f"shot-level (n={n_shot})")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlim(left=-0.3, right=1.05)
        ax.set_xlabel("5-fold CV R^2")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_yticks(y); axes[0].set_yticklabels(d["target"])
    fig.suptitle("V-JEPA-2 shot-averaging: raw vs mean-centered embeddings",
                 fontsize=13)
    fig.tight_layout()
    out_png = args.outdir / "vjepa2_centered_shot_averaged.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
