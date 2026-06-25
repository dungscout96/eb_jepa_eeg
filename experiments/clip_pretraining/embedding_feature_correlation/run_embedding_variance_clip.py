"""How dissimilar are OpenAI CLIP frame embeddings within vs across shots?

Per-frame variant of `run_embedding_variance_vjepa2.py`. OpenAI CLIP emits one
512-d embedding per video frame (~24 fps, ~4877 frames for "The_Present"). The
question is the same: does the embedding space's cosine geometry support a
shot-level contrastive objective, and does it need centering / whitening?

Inputs:
    movie_annotation/output/The_Present/clip_embeddings.npz
        - embeddings:    (n_frames, 512) float32
        - frame_indices: (n_frames,) int64
    movie_annotation/output/The_Present/features_enriched.parquet
        - rows keyed by frame_idx, with shot_id and timestamp_s

The 5 sub-analyses mirror the V-JEPA-2 script (variance decomposition, raw vs
centered pairwise cosine, matched-gap shot vs not-shot, per-shot geometry, NN
purity). Time-gap bins are finer (sub-second) since CLIP frames are 24 Hz and
within-shot pairs span tens of milliseconds.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_embedding_variance_clip.py

Outputs (alongside this file):
    clip_embedding_variance_summary.txt
    clip_embedding_variance_per_shot.csv
    clip_embedding_variance.png
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
# Reuse the geometry helpers from the V-JEPA-2 script verbatim.
from run_embedding_variance_vjepa2 import (  # noqa: E402
    matched_gap_table,
    nn_purity,
    pairwise_cosine_stats,
    per_shot_geometry,
    variance_decomposition,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "movie_annotation/output/The_Present/clip_embeddings.npz"
DEFAULT_FEATURES = PROJECT_ROOT / "movie_annotation/output/The_Present/features_enriched.parquet"
DEFAULT_OUTDIR = Path(__file__).resolve().parent


def load_clip_frames(emb_path: Path, feats: pd.DataFrame):
    """Load CLIP frame embeddings and join to per-frame shot_id / timestamp."""
    d = np.load(emb_path)
    X = d["embeddings"].astype(np.float64)
    frame_idx = d["frame_indices"].astype(np.int64)
    needed = {"frame_idx", "shot_id", "timestamp_s"}
    missing = needed - set(feats.columns)
    if missing:
        raise SystemExit(f"features parquet is missing columns: {missing}")
    lookup = feats.set_index("frame_idx")[["shot_id", "timestamp_s"]]
    aligned = lookup.reindex(frame_idx)
    valid = aligned.notna().all(axis=1).to_numpy()
    return (X[valid],
            aligned.loc[valid, "shot_id"].to_numpy().astype(int),
            aligned.loc[valid, "timestamp_s"].to_numpy().astype(float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--subsample", type=int, default=0,
                    help="If >0, randomly subsample this many frames to keep "
                         "the n^2 pairwise computation light.")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(args.features)
    X, shot, t = load_clip_frames(args.embeddings, feats)
    if args.subsample and args.subsample < len(X):
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(X), args.subsample, replace=False))
        X, shot, t = X[sel], shot[sel], t[sel]
        print(f"  (subsampled to {len(X)} frames for n^2 analysis)")

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    Xcn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    print(f"frames={len(Xn)}, shots={len(np.unique(shot))}, dim={Xn.shape[1]}, "
          f"time span={t.min():.1f}–{t.max():.1f} s")

    vd = variance_decomposition(Xn, shot)
    pc_raw = pairwise_cosine_stats(Xn, shot, t)
    pc = pairwise_cosine_stats(Xcn, shot, t)
    # Finer bins than V-JEPA-2: CLIP frames are 24 Hz so within-shot pairs sit
    # at tens of ms; we still want coarse bins out to the movie length.
    edges = np.array([0, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    gap_tab = matched_gap_table(pc["cos"], pc["same"], pc["gap"], edges)
    geom = per_shot_geometry(Xcn, shot)
    purity = nn_purity(Xcn, shot)
    purity_raw = nn_purity(Xn, shot)

    # ---- summary text ----
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 70)
    emit("OpenAI CLIP frame embedding geometry: within-shot vs between-shot")
    emit("=" * 70)
    emit(f"\nframes={vd['n_clips']}  shots={vd['k_shots']}  cosine space (L2-normalized)")
    emit("\n[1] Variance decomposition (fraction of embedding variance):")
    emit(f"    between-shot (eta^2) = {vd['eta2']:.3f}   "
         f"within-shot = {1 - vd['eta2']:.3f}")
    emit(f"    F-statistic          = {vd['f_stat']:.1f}  "
         f"(>>1 means shot identity strongly structures the space)")
    emit("\n[2] Pairwise cosine similarity (RAW embeddings):")
    emit(f"    same-shot (positives) : {pc_raw['pos_mean']:.3f} +/- {pc_raw['pos_std']:.3f}  "
         f"(n={pc_raw['n_pos']:,})")
    emit(f"    diff-shot (negatives) : {pc_raw['neg_mean']:.3f} +/- {pc_raw['neg_std']:.3f}  "
         f"(n={pc_raw['n_neg']:,})")
    emit(f"    separation: d-prime = {pc_raw['dprime']:.2f}   ROC-AUC = {pc_raw['auc']:.3f}")
    emit("\n[2b] Pairwise cosine after MEAN-CENTERING:")
    emit(f"    same-shot (positives) : {pc['pos_mean']:.3f} +/- {pc['pos_std']:.3f}")
    emit(f"    diff-shot (negatives) : {pc['neg_mean']:.3f} +/- {pc['neg_std']:.3f}")
    emit(f"    separation: d-prime = {pc['dprime']:.2f}   ROC-AUC = {pc['auc']:.3f}")
    centering_helps = pc["auc"] - pc_raw["auc"]
    emit(f"    -> centering shifts AUC by {centering_helps:+.3f} "
         f"({'meaningful' if abs(centering_helps) > 0.02 else 'negligible'} for CLIP)")
    emit("\n[3] Same- vs different-shot cosine at matched |time gap| (s):")
    emit(f"    {'gap [s)':<14}{'same':>8}{'(n)':>10}{'diff':>8}{'(n)':>10}")
    for _, r in gap_tab.iterrows():
        same_s = f"{r['same_mean']:.3f}" if not np.isnan(r["same_mean"]) else "  -  "
        diff_s = f"{r['diff_mean']:.3f}" if not np.isnan(r["diff_mean"]) else "  -  "
        emit(f"    [{r['gap_lo']:>5.2f},{r['gap_hi']:>5.1f}){'':<2}{same_s:>8}"
             f"{int(r['same_n']):>10}{diff_s:>8}{int(r['diff_n']):>10}")
    emit("    -> compare same vs diff WITHIN a row: shot effect above adjacency.")
    emit("\n[4] Per-shot geometry (centered, within-shot vs nearest other shot):")
    emit(f"    within-shot cos to own centroid : "
         f"{geom['within_cos_mean'].mean():.3f} (mean over shots)")
    emit(f"    cos to nearest OTHER centroid   : "
         f"{geom['nearest_other_cos'].mean():.3f}")
    emit(f"    margin (own - nearest other)    : {geom['margin'].mean():.3f}  "
         f"[min {geom['margin'].min():.3f}, "
         f"{int((geom['margin'] < 0).sum())} shots overlap a neighbor]")
    emit("\n[5] Nearest-neighbor shot purity (in-batch negative hardness):")
    emit(f"    top-1 neighbor same shot : {purity['top1_purity']:.3f}  "
         f"(raw: {purity_raw['top1_purity']:.3f})")
    emit(f"    top-5 neighbor same shot : {purity['top5_purity']:.3f}  "
         f"(raw: {purity_raw['top5_purity']:.3f})")

    emit("\n" + "-" * 70)
    emit("CLIP design implications (CLIP-encoder space):")
    if abs(centering_helps) > 0.02:
        emit(f"  * Centering helps (AUC {pc_raw['auc']:.2f} -> {pc['auc']:.2f}). CLIP")
        emit("    embeddings carry a non-trivial shared component, similar to V-JEPA-2.")
    else:
        emit(f"  * Centering barely changes AUC ({pc_raw['auc']:.2f} -> {pc['auc']:.2f}).")
        emit("    CLIP's representation is already comparatively isotropic — raw cosine")
        emit("    is usable as a CLIP-objective metric without explicit whitening.")
    if pc["auc"] > 0.85 and vd["eta2"] > 0.4:
        emit(f"  * Strong shot structure (eta^2={vd['eta2']:.2f}, centered AUC="
             f"{pc['auc']:.2f}): same-shot")
        emit("    frames are reliably closer than cross-shot ones; shot-level positives")
        emit("    are well-posed.")
    else:
        emit(f"  * Modest shot structure (eta^2={vd['eta2']:.2f}, AUC={pc['auc']:.2f}): "
             f"some")
        emit("    same/diff overlap — expect noisier negatives than V-JEPA-2.")
    emit("  * At 24 Hz, ADJACENT FRAMES are nearly identical even across shot cuts;")
    emit("    exclude tight-time-gap pairs from the negative pool, or sub-sample to")
    emit("    ~1-2 Hz before contrasting to avoid trivial positives.")
    emit(f"  * top-1 NN purity {purity['top1_purity']:.2f}: ~"
         f"{int(round((1-purity['top1_purity'])*100))}% of frames' nearest neighbor "
         f"lies in another shot.")

    summ = args.outdir / "clip_embedding_variance_summary.txt"
    summ.write_text("\n".join(lines) + "\n")
    geom_csv = args.outdir / "clip_embedding_variance_per_shot.csv"
    geom.to_csv(geom_csv, index=False)
    print(f"\nSaved {summ}")
    print(f"Saved {geom_csv}")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for p, tag, alpha in [(pc_raw, "raw", 0.4), (pc, "centered", 0.6)]:
        lo = min(p["neg"].min(), p["pos"].min())
        bins = np.linspace(lo, 1.0, 60)
        ax.hist(p["neg"], bins=bins, density=True, alpha=alpha, color="crimson",
                label=f"diff-shot {tag} (AUC {p['auc']:.2f})")
        ax.hist(p["pos"], bins=bins, density=True, alpha=alpha, color="steelblue",
                label=f"same-shot {tag}")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("density")
    ax.set_title("Pairwise cosine: raw vs centered")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    g = gap_tab.dropna(subset=["same_mean"])
    mid = (g["gap_lo"] + g["gap_hi"]) / 2
    ax.plot(mid, g["same_mean"], "o-", color="steelblue", label="same-shot")
    gd = gap_tab.dropna(subset=["diff_mean"])
    midd = (gd["gap_lo"] + gd["gap_hi"]) / 2
    ax.plot(midd, gd["diff_mean"], "s-", color="crimson", label="different-shot")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("|time gap| between frames (s)"); ax.set_ylabel("mean cosine")
    ax.set_title("Cosine vs time gap (temporal confound)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(geom["within_cos_mean"], geom["nearest_other_cos"],
               s=10 + 0.4 * geom["n_clips"], c=geom["margin"], cmap="coolwarm_r",
               edgecolor="k", linewidth=0.2)
    lim = [min(geom["within_cos_mean"].min(), geom["nearest_other_cos"].min()) - 0.02,
           1.0]
    ax.plot(lim, lim, "k--", lw=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("within-shot cos to own centroid (centered)")
    ax.set_ylabel("cos to nearest OTHER centroid")
    ax.set_title("Per-shot compactness vs separation\n(below diagonal = separable)")

    ax = axes[1, 1]
    ax.hist(geom["margin"], bins=20, color="seagreen", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(geom["margin"].mean(), color="darkgreen", ls="--",
               label=f"mean {geom['margin'].mean():.3f}")
    ax.set_xlabel("per-shot margin (own - nearest other centroid cos)")
    ax.set_ylabel("# shots")
    ax.set_title("Contrastive margin per shot")
    ax.legend(fontsize=8)

    fig.suptitle(f"OpenAI CLIP frame embedding geometry for contrastive design  "
                 f"(eta^2={vd['eta2']:.2f}, F={vd['f_stat']:.0f})", fontsize=13)
    fig.tight_layout()
    out_png = args.outdir / "clip_embedding_variance.png"
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
