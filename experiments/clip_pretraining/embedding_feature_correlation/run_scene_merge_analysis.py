"""Merge near-duplicate V-JEPA-2 shots into 'scenes' and re-analyze.

Motivation: the per-shot geometry analysis showed 12 of 54 shots overlap a
neighbor shot's spread in centered V-JEPA-2 space (alternating camera angles
within a scene, returning locations). These near-duplicate pairs pollute the
negative pool for a contrastive objective. Merging them into 'scenes' should
give a cleaner membership definition for hard-negative sampling.

What this script does:
  1. Cluster shot centroids (mean-centered V-JEPA-2) with agglomerative cosine
     average-linkage at a configurable threshold -> shot -> scene mapping.
  2. Recompute the variance / geometry analysis (eta^2, AUC, d-prime, per-shot
     margin, NN purity) with scene labels in place of shot labels.
  3. Recompute the shot-averaging feature-regression sweep (per-clip baseline,
     broadcast-shot, broadcast-scene, scene-level) and report the gap.
  4. Produce two figures:
        vjepa2_scenes_overview.png   timeline + centroid heatmap + metric bars
        vjepa2_scenes_frame_grid.png one example frame per scene labeled with
                                     constituent shot IDs

Usage:
    PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_scene_merge_analysis.py [--merge-threshold 0.75]
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import (  # noqa: E402
    DEFAULT_EMBEDDINGS,
    DEFAULT_FEATURES,
    SCALAR_TARGETS,
    align_features_to_clips,
    evaluate_regression,
)
from run_embedding_variance_vjepa2 import (  # noqa: E402
    nn_purity,
    pairwise_cosine_stats,
    per_shot_geometry,
    variance_decomposition,
)
from run_shot_averaged_vjepa2 import (  # noqa: E402
    aggregate_by_shot,
    assign_clip_shots,
    broadcast_shot_embeddings,
    evaluate_regression_pca,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = Path(__file__).resolve().parent
DEFAULT_FRAMES_DIR = (
    PROJECT_ROOT
    / "movie_annotation/output/The_Present/shot_detection/three_way_agreed"
)


def _average_linkage_clusters(cosmat: np.ndarray, threshold: float) -> list:
    """Hand-rolled agglomerative average-linkage clustering on a similarity
    matrix. Merge the pair of clusters with the highest mean cosine until
    no pair has mean cosine > threshold. Returns a list[set[int]] of indices.

    O(n^3) — fine for n ≈ 50 centroids.
    """
    n = cosmat.shape[0]
    clusters = [{i} for i in range(n)]
    sims = cosmat.copy().astype(float)
    np.fill_diagonal(sims, -np.inf)
    while True:
        i, j = np.unravel_index(np.argmax(sims), sims.shape)
        if sims[i, j] <= threshold or len(clusters) <= 1:
            break
        # merge j into i (and remove j)
        new_size = len(clusters[i]) + len(clusters[j])
        for k in range(len(clusters)):
            if k == i or k == j:
                continue
            si, sj = sims[i, k], sims[j, k]
            # weighted mean = average linkage on similarities
            sims[i, k] = sims[k, i] = (
                si * len(clusters[i]) + sj * len(clusters[j])
            ) / new_size
        clusters[i] |= clusters[j]
        clusters.pop(j)
        # shrink the sims matrix
        sims = np.delete(np.delete(sims, j, 0), j, 1)
    return clusters


def cluster_shots_into_scenes(Xcn: np.ndarray, shot: np.ndarray,
                              threshold: float) -> dict:
    """Average-linkage clustering of shot centroids in centered cosine space.
    Merges any pair whose cluster-mean cosine exceeds `threshold`. Scene IDs
    are numbered in order of earliest constituent shot so the timeline reads
    left-to-right.
    """
    shots = np.unique(shot)
    if len(shots) <= 1:
        return {int(shots[0]): 0}
    centroids = np.stack([Xcn[shot == s].mean(axis=0) for s in shots])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    cosmat = centroids @ centroids.T
    clusters = _average_linkage_clusters(cosmat, threshold)
    # idx-of-centroids -> shot_id, then label by earliest shot
    cluster_shot_sets = [{int(shots[i]) for i in c} for c in clusters]
    cluster_shot_sets.sort(key=lambda s: min(s))
    return {s: scene_id for scene_id, members in enumerate(cluster_shot_sets)
            for s in members}


def build_shot_frame_map(frames_dir: Path, feats: pd.DataFrame) -> dict:
    """Map shot_id (parquet) -> path to its first-frame jpg.

    Frames named like `shot{NNN}_f{FFFFF}_{TT.TT}s_cut.jpg`, where `_cut` is
    the first frame of that shot. We match by timestamp to be robust against
    1-vs-0 indexing differences.
    """
    if not frames_dir.exists():
        return {}
    pat = re.compile(r"shot(\d+)_f\d+_([\d.]+)s_cut\.jpg")
    file_ts = {}
    for f in frames_dir.glob("*_cut.jpg"):
        m = pat.match(f.name)
        if m:
            file_ts[float(m.group(2))] = f
    if not file_ts:
        return {}
    file_times = np.array(sorted(file_ts.keys()))
    starts = feats.groupby("shot_id")["timestamp_s"].min().to_dict()
    mapping = {}
    for sid, st in starts.items():
        idx = int(np.argmin(np.abs(file_times - st)))
        if abs(file_times[idx] - st) < 1.0:
            mapping[int(sid)] = file_ts[file_times[idx]]
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--merge-threshold", type=float, default=0.90,
                    help="Centered cosine threshold for merging shot centroids "
                         "into a scene. Higher = stricter, fewer merges. "
                         "Default 0.90 catches the 12 originally-identified "
                         "negative-margin shots without cascading. Try 0.75 "
                         "for aggressive scene-level grouping, 0.95 for "
                         "surgical near-duplicate-only merges.")
    ap.add_argument("--fps", type=float, default=24.0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    data = np.load(args.embeddings)
    X = data["embeddings"].astype(np.float64)
    t = data["timestamps"].astype(np.float64)
    freq = float(data["frequency"]) if "frequency" in data.files else 2.0
    feats = pd.read_parquet(args.features)

    aligned = align_features_to_clips(feats, t, freq=freq, fps=args.fps)
    clip_shot_raw = assign_clip_shots(feats, t, freq=freq)
    keep = ~np.isnan(clip_shot_raw)
    X = X[keep]; t = t[keep]
    clip_shot = clip_shot_raw[keep].astype(int)
    aligned = aligned.loc[keep].reset_index(drop=True)

    Xc = X - X.mean(0)
    Xcn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)

    # ---- cluster shots -> scenes ----
    scene_map = cluster_shots_into_scenes(Xcn, clip_shot, args.merge_threshold)
    clip_scene = np.array([scene_map[int(s)] for s in clip_shot])
    n_shots = len(np.unique(clip_shot))
    n_scenes = len(np.unique(clip_scene))

    scene_to_shots: dict = {}
    for s, sc in scene_map.items():
        scene_to_shots.setdefault(sc, []).append(s)
    multi_scenes = {k: sorted(v) for k, v in scene_to_shots.items() if len(v) > 1}

    print(f"Merged {n_shots} shots -> {n_scenes} scenes "
          f"(centered-cosine threshold {args.merge_threshold})")
    print(f"  {len(multi_scenes)} multi-shot scenes "
          f"({n_shots - n_scenes} shot merges)")
    for sc_id, shots in sorted(multi_scenes.items()):
        print(f"    scene {sc_id:>2}: shots {shots}")

    pd.DataFrame([{"shot_id": s, "scene_id": sc}
                  for s, sc in sorted(scene_map.items())]).to_csv(
        args.outdir / "vjepa2_scenes_map.csv", index=False)

    # ---- geometry: shot vs scene ----
    vd_shot = variance_decomposition(Xcn, clip_shot)
    vd_scene = variance_decomposition(Xcn, clip_scene)
    pc_shot = pairwise_cosine_stats(Xcn, clip_shot, t)
    pc_scene = pairwise_cosine_stats(Xcn, clip_scene, t)
    geom_shot = per_shot_geometry(Xcn, clip_shot)
    geom_scene = per_shot_geometry(Xcn, clip_scene)
    purity_shot = nn_purity(Xcn, clip_shot)
    purity_scene = nn_purity(Xcn, clip_scene)

    print("\n=== Variance decomposition (centered) ===")
    print(f"  {'':<14}{'shot':>10}{'scene':>10}{'delta':>10}")
    print(f"  {'eta^2':<14}{vd_shot['eta2']:>10.3f}{vd_scene['eta2']:>10.3f}"
          f"{vd_scene['eta2'] - vd_shot['eta2']:>+10.3f}")
    print(f"  {'F-stat':<14}{vd_shot['f_stat']:>10.1f}{vd_scene['f_stat']:>10.1f}"
          f"{vd_scene['f_stat'] - vd_shot['f_stat']:>+10.1f}")
    print(f"  {'K groups':<14}{vd_shot['k_shots']:>10d}{vd_scene['k_shots']:>10d}")

    print("\n=== Pairwise cosine (positives = same-{shot,scene}) ===")
    print(f"  {'':<14}{'shot':>10}{'scene':>10}")
    print(f"  {'pos mean':<14}{pc_shot['pos_mean']:>10.3f}{pc_scene['pos_mean']:>10.3f}")
    print(f"  {'neg mean':<14}{pc_shot['neg_mean']:>10.3f}{pc_scene['neg_mean']:>10.3f}")
    print(f"  {'d-prime':<14}{pc_shot['dprime']:>10.2f}{pc_scene['dprime']:>10.2f}")
    print(f"  {'ROC-AUC':<14}{pc_shot['auc']:>10.3f}{pc_scene['auc']:>10.3f}")

    print("\n=== Per-group geometry ===")
    n_neg_shot = int((geom_shot["margin"] < 0).sum())
    n_neg_scene = int((geom_scene["margin"] < 0).sum())
    print(f"  margin (own - nearest other), shot : mean "
          f"{geom_shot['margin'].mean():+.3f}, {n_neg_shot}/{len(geom_shot)} negative")
    print(f"  margin (own - nearest other), scene: mean "
          f"{geom_scene['margin'].mean():+.3f}, {n_neg_scene}/{len(geom_scene)} negative")
    print(f"  NN top-1 purity: shot {purity_shot['top1_purity']:.3f}  "
          f"scene {purity_scene['top1_purity']:.3f}")
    print(f"  NN top-5 purity: shot {purity_shot['top5_purity']:.3f}  "
          f"scene {purity_scene['top5_purity']:.3f}")

    # ---- feature regression ----
    X_scene_mean, aligned_scene, _, _ = aggregate_by_shot(X, aligned, clip_scene)
    X_shot_mean, aligned_shotmean, _, _ = aggregate_by_shot(X, aligned, clip_shot)
    X_bcast_shot = broadcast_shot_embeddings(X, clip_shot)
    X_bcast_scene = broadcast_shot_embeddings(X, clip_scene)

    print("\n=== Feature R^2 (5-fold CV ridge) ===")
    print(f"{'feature':<22} | {'clip':>7} {'b-shot':>8} {'b-scene':>9} | "
          f"{'shot(PCA)':>10} {'scene(PCA)':>11}")
    print("-" * 80)
    rows = []
    for col in SCALAR_TARGETS:
        if col not in aligned_scene.columns:
            continue
        y_clip = aligned[col].to_numpy().astype(np.float32)
        y_shot = aligned_shotmean[col].to_numpy().astype(np.float32)
        y_scene = aligned_scene[col].to_numpy().astype(np.float32)
        r_clip = evaluate_regression(X, y_clip)
        r_bs = evaluate_regression(X_bcast_shot, y_clip)
        r_bsc = evaluate_regression(X_bcast_scene, y_clip)
        r_shot = evaluate_regression_pca(X_shot_mean, y_shot, 20)
        r_scene = evaluate_regression_pca(X_scene_mean, y_scene, 20)
        print(f"{col:<22} | {r_clip['r2_mean']:>7.3f} {r_bs['r2_mean']:>8.3f} "
              f"{r_bsc['r2_mean']:>9.3f} | {r_shot['r2_mean']:>10.3f} "
              f"{r_scene['r2_mean']:>11.3f}")
        rows.append({"target": col,
                     "r2_clip": r_clip["r2_mean"],
                     "r2_broadcast_shot": r_bs["r2_mean"],
                     "r2_broadcast_scene": r_bsc["r2_mean"],
                     "r2_shot_pca20": r_shot["r2_mean"],
                     "r2_scene_pca20": r_scene["r2_mean"]})
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "vjepa2_scenes_shot_averaged.csv", index=False)

    means = df.mean(numeric_only=True)
    print(f"\n  mean R^2: clip={means['r2_clip']:.3f}, "
          f"bcast-shot={means['r2_broadcast_shot']:.3f}, "
          f"bcast-scene={means['r2_broadcast_scene']:.3f}")
    print(f"  broadcast loss vs per-clip: "
          f"shot {means['r2_broadcast_shot'] - means['r2_clip']:+.3f}, "
          f"scene {means['r2_broadcast_scene'] - means['r2_clip']:+.3f}")

    # ---- summary text ----
    lines = ["=" * 70,
             "V-JEPA-2 shot -> scene merge analysis",
             "=" * 70,
             f"\nMerge threshold (centered cosine): {args.merge_threshold}",
             f"Shots: {n_shots}  ->  Scenes: {n_scenes}  "
             f"({n_shots - n_scenes} shot merges)",
             "\nMulti-shot scenes:"]
    for sc_id, shots in sorted(multi_scenes.items()):
        lines.append(f"  scene {sc_id:>2}: shots {shots}")
    lines += ["",
              "[Variance decomposition, centered]",
              f"  shot : eta^2 {vd_shot['eta2']:.3f}  F {vd_shot['f_stat']:.1f}  "
              f"K {vd_shot['k_shots']}",
              f"  scene: eta^2 {vd_scene['eta2']:.3f}  F {vd_scene['f_stat']:.1f}  "
              f"K {vd_scene['k_shots']}",
              "",
              "[Pairwise cosine, centered]",
              f"  shot : pos {pc_shot['pos_mean']:.3f}  neg {pc_shot['neg_mean']:.3f}  "
              f"d' {pc_shot['dprime']:.2f}  AUC {pc_shot['auc']:.3f}",
              f"  scene: pos {pc_scene['pos_mean']:.3f}  neg {pc_scene['neg_mean']:.3f}  "
              f"d' {pc_scene['dprime']:.2f}  AUC {pc_scene['auc']:.3f}",
              "",
              "[Per-group margin and NN purity]",
              f"  shot : margin {geom_shot['margin'].mean():+.3f}  "
              f"negative {n_neg_shot}/{len(geom_shot)}  "
              f"top-1 {purity_shot['top1_purity']:.3f}",
              f"  scene: margin {geom_scene['margin'].mean():+.3f}  "
              f"negative {n_neg_scene}/{len(geom_scene)}  "
              f"top-1 {purity_scene['top1_purity']:.3f}",
              "",
              f"[Feature R^2 mean over {len(df)} targets]",
              f"  per-clip            : {means['r2_clip']:.3f}",
              f"  broadcast shot-mean : {means['r2_broadcast_shot']:.3f}  "
              f"(loss {means['r2_broadcast_shot'] - means['r2_clip']:+.3f})",
              f"  broadcast scene-mean: {means['r2_broadcast_scene']:.3f}  "
              f"(loss {means['r2_broadcast_scene'] - means['r2_clip']:+.3f})"]
    (args.outdir / "vjepa2_scenes_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {args.outdir / 'vjepa2_scenes_summary.txt'}")

    # ---- visualization 1: overview ----
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    cmap = plt.cm.tab20
    scene_colors = {sc: cmap(sc % cmap.N) for sc in np.unique(clip_scene)}

    # (0,0) Scene timeline (Gantt)
    ax_tl = fig.add_subplot(gs[0, 0])
    shots_sorted_t = sorted(np.unique(clip_shot),
                            key=lambda s: t[clip_shot == s].mean())
    for s in shots_sorted_t:
        m = clip_shot == s
        t_start, t_end = float(t[m].min()), float(t[m].max()) + 0.5
        sc = scene_map[int(s)]
        ax_tl.barh(0, t_end - t_start, left=t_start, height=0.8,
                   color=scene_colors[sc], edgecolor="black", linewidth=0.3)
        # tiny shot-id label
        ax_tl.text((t_start + t_end) / 2, 0, str(int(s)), ha="center",
                   va="center", fontsize=5)
    # Multi-shot scene callouts above the bar
    for sc_id, shots in sorted(multi_scenes.items()):
        times = [t[clip_shot == s].mean() for s in shots]
        ax_tl.scatter(times, [0.6] * len(times), s=40,
                      color=scene_colors[sc_id], edgecolor="black", linewidth=0.5)
        ax_tl.text(float(np.mean(times)), 0.85, f"sc{sc_id}", ha="center",
                   fontsize=7, color=scene_colors[sc_id])
    ax_tl.set_xlabel("time (s)"); ax_tl.set_yticks([])
    ax_tl.set_ylim(-0.5, 1.1)
    ax_tl.set_title(f"Scene timeline: {n_shots} shots -> {n_scenes} scenes "
                    f"(centered cos > {args.merge_threshold})")
    ax_tl.set_xlim(-1, float(t.max()) + 1)

    # (0,1) Centroid cosine heatmap sorted by scene
    ax_hm = fig.add_subplot(gs[0, 1])
    shots_by_scene = sorted(np.unique(clip_shot),
                            key=lambda s: (scene_map[int(s)],
                                           float(t[clip_shot == s].mean())))
    cents = np.stack([Xcn[clip_shot == s].mean(axis=0) for s in shots_by_scene])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    cosmat = cents @ cents.T
    im = ax_hm.imshow(cosmat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    fig.colorbar(im, ax=ax_hm, fraction=0.04)
    scenes_in_order = [scene_map[int(s)] for s in shots_by_scene]
    for i in range(1, len(scenes_in_order)):
        if scenes_in_order[i] != scenes_in_order[i - 1]:
            ax_hm.axhline(i - 0.5, color="black", lw=0.4)
            ax_hm.axvline(i - 0.5, color="black", lw=0.4)
    ax_hm.set_title("Shot centroid cosine (sorted by scene)\n"
                    "block-diagonal = clean clusters", fontsize=10)
    ax_hm.set_xlabel("shot (sorted by scene)"); ax_hm.set_ylabel("shot")

    # (1,0) Metric comparison
    ax_bar = fig.add_subplot(gs[1, 0])
    metrics = ["eta^2", "AUC", "NN top-1", "NN top-5", "non-overlap fraction"]
    shot_vals = [vd_shot["eta2"], pc_shot["auc"],
                 purity_shot["top1_purity"], purity_shot["top5_purity"],
                 float((geom_shot["margin"] >= 0).mean())]
    scene_vals = [vd_scene["eta2"], pc_scene["auc"],
                  purity_scene["top1_purity"], purity_scene["top5_purity"],
                  float((geom_scene["margin"] >= 0).mean())]
    x = np.arange(len(metrics))
    ax_bar.bar(x - 0.2, shot_vals, 0.4, label=f"shot (K={n_shots})",
               color="steelblue")
    ax_bar.bar(x + 0.2, scene_vals, 0.4, label=f"scene (K={n_scenes})",
               color="darkorange")
    for xi, sv, scv in zip(x, shot_vals, scene_vals):
        ax_bar.text(xi - 0.2, sv + 0.01, f"{sv:.2f}", ha="center", fontsize=7)
        ax_bar.text(xi + 0.2, scv + 0.01, f"{scv:.2f}", ha="center", fontsize=7)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metrics, rotation=15, ha="right")
    ax_bar.set_ylim(0, 1.1); ax_bar.legend()
    ax_bar.set_title("Geometry metrics: shot vs scene (higher = better)")

    # (1,1) Feature R^2 grouped bars
    ax_r2 = fig.add_subplot(gs[1, 1])
    d = df.sort_values("r2_clip")
    yy = np.arange(len(d))
    ax_r2.barh(yy - 0.27, d["r2_clip"], height=0.27, color="steelblue",
               label="per-clip")
    ax_r2.barh(yy, d["r2_broadcast_shot"], height=0.27, color="seagreen",
               label="broadcast-shot")
    ax_r2.barh(yy + 0.27, d["r2_broadcast_scene"], height=0.27, color="darkorange",
               label="broadcast-scene")
    ax_r2.set_yticks(yy); ax_r2.set_yticklabels(d["target"], fontsize=8)
    ax_r2.axvline(0, color="k", lw=0.5)
    ax_r2.set_xlim(left=-0.05, right=1.05)
    ax_r2.set_xlabel("5-fold CV R^2")
    ax_r2.legend(fontsize=8, loc="lower right")
    ax_r2.set_title("Feature R^2: per-clip vs broadcast (shot vs scene)")

    fig.suptitle(f"V-JEPA-2 shot -> scene merge: "
                 f"{n_shots} shots -> {n_scenes} scenes "
                 f"(cos > {args.merge_threshold})",
                 fontsize=13)
    fig.tight_layout()
    overview_png = args.outdir / "vjepa2_scenes_overview.png"
    fig.savefig(overview_png, dpi=130)
    print(f"Saved: {overview_png}")

    # ---- visualization 2: frame grid (one example frame per scene) ----
    shot_frames = build_shot_frame_map(args.frames_dir, feats)
    if not shot_frames:
        print(f"WARNING: no frames found in {args.frames_dir}")
        return
    scene_list = sorted(scene_to_shots.items(), key=lambda kv: min(kv[1]))
    n_cols = 6
    n_rows = int(np.ceil(len(scene_list) / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.2))
    axes2 = np.atleast_2d(axes2).flatten()
    for ax, (sc_id, shots) in zip(axes2, scene_list):
        sizes = {s: int((clip_shot == s).sum()) for s in shots}
        rep_shot = max(sizes, key=sizes.get)
        frame_path = shot_frames.get(rep_shot)
        if frame_path and frame_path.exists():
            try:
                ax.imshow(imread(str(frame_path)))
            except Exception:
                ax.text(0.5, 0.5, "image err", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "no frame", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        if len(shots) > 1:
            title = f"Sc{sc_id}: shots {shots}"
        else:
            title = f"Sc{sc_id}: shot {shots[0]}"
        ax.set_title(title, fontsize=7,
                     color=scene_colors[sc_id], fontweight="bold")
        # colored border to match timeline
        for spine in ax.spines.values():
            spine.set_edgecolor(scene_colors[sc_id])
            spine.set_linewidth(2.5 if len(shots) > 1 else 1)
    for ax in axes2[len(scene_list):]:
        ax.axis("off")
    fig2.suptitle(
        f"Example frame per scene ({len(scene_list)} scenes; "
        f"thick borders = multi-shot scenes)\n"
        f"frame is the first '_cut' frame of the largest shot in the scene",
        fontsize=10)
    fig2.tight_layout()
    frame_png = args.outdir / "vjepa2_scenes_frame_grid.png"
    fig2.savefig(frame_png, dpi=110)
    print(f"Saved: {frame_png}")


if __name__ == "__main__":
    main()
