"""Split-modality t-SNE plots: run t-SNE *per modality* on the shared-space
embeddings saved by ``plot_modality_gap.py --save-npz``, and lay out EEG on
top row and V-JEPA-2 on bottom row.

Each column recolors the same per-modality points by a different label:

    * Movie (task)
    * Shot id
    * Continuous movie features (columns ``feat_*`` in the npz), one per column

Motivation: the joint t-SNE in ``plot_embedding_structure.py`` shows global
alignment, but flattens each modality's *internal* structure. Splitting the
projection makes it visible whether the encoder alone (or V-JEPA-2 alone)
organizes windows by shot / movie / features. The two rows therefore use
INDEPENDENT t-SNE layouts, not shared coords.

Usage:
    PYTHONPATH=. uv run --group eeg python \\
        experiments/clip_pretraining/cs_aligner/plot_split_modality.py \\
        --npz experiments/clip_pretraining/cs_aligner/panel_b.npz \\
        --output experiments/clip_pretraining/cs_aligner/split_modality.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--output", default="split_modality.png")
    ap.add_argument("--title", default="Per-modality t-SNE")
    ap.add_argument("--tsne-perplexity", type=float, default=30.0)
    ap.add_argument("--tsne-seed", type=int, default=0)
    return ap.parse_args()


def _decode(arr):
    if arr.dtype.kind == "S":
        return np.array([x.decode() for x in arr])
    return arr


def _tsne(X, perplexity, seed):
    return TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=seed,
    ).fit_transform(X.astype(np.float32))


def _style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#888")
        spine.set_linewidth(0.6)


def _plot_categorical(ax, xy, labels, title, palette, legend_max=6):
    uniq = np.array([u for u in np.unique(labels) if u not in ("", b"")])
    color = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
    for u in uniq:
        m = labels == u
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.65,
                   c=[color[u]], edgecolors="none",
                   label=str(u) if len(uniq) <= legend_max else None)
    if len(uniq) <= legend_max:
        ax.legend(loc="lower right", frameon=False, fontsize=7, markerscale=1.5)
    ax.set_title(title, fontsize=10)


def _plot_continuous(ax, xy, values, title, cmap="viridis", invalid_below=None):
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if invalid_below is not None:
        valid &= values >= invalid_below
    if (~valid).any():
        ax.scatter(xy[~valid, 0], xy[~valid, 1], s=6, alpha=0.3,
                   c="#cccccc", edgecolors="none")
    if valid.any():
        sc = ax.scatter(xy[valid, 0], xy[valid, 1], s=6, alpha=0.65,
                        c=values[valid], cmap=cmap, edgecolors="none")
        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)


def main():
    args = parse_args()
    d = np.load(args.npz, allow_pickle=False)
    z = d["z_shared"]
    is_vision = d["is_vision"].astype(bool)
    task = _decode(d["task"])
    shot_id = d["shot_id"]

    feature_keys = sorted(k for k in d.files if k.startswith("feat_"))
    print(f"Loaded {len(z)} points  "
          f"({(~is_vision).sum()} EEG, {is_vision.sum()} V-JEPA-2)  "
          f"features: {[k[len('feat_'):] for k in feature_keys]}")

    print("Running per-modality t-SNE ...")
    xy_eeg = _tsne(z[~is_vision], args.tsne_perplexity, args.tsne_seed)
    xy_vis = _tsne(z[is_vision], args.tsne_perplexity, args.tsne_seed)

    # Two rows (EEG, V-JEPA-2), columns = movie, shot_id, and each feature.
    col_specs = [("movie", "cat"), ("shot_id", "cont")]
    for k in feature_keys:
        col_specs.append((k, "cont"))
    n_cols = len(col_specs)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.4 * n_cols, 7.0),
                             squeeze=False)

    movie_palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    row_specs = [
        ("EEG", ~is_vision, xy_eeg),
        ("V-JEPA-2", is_vision, xy_vis),
    ]

    for r, (row_name, mask, xy) in enumerate(row_specs):
        row_task = task[mask]
        row_shot = shot_id[mask]
        for c, (col_key, kind) in enumerate(col_specs):
            ax = axes[r, c]
            if col_key == "movie":
                _plot_categorical(ax, xy, row_task,
                                  f"{row_name}: Movie", movie_palette)
            elif col_key == "shot_id":
                _plot_continuous(ax, xy, row_shot,
                                 f"{row_name}: Shot id",
                                 cmap="turbo", invalid_below=0)
            else:
                vals = d[col_key][mask]
                pretty = col_key[len("feat_"):]
                _plot_continuous(ax, xy, vals,
                                 f"{row_name}: {pretty}", cmap="viridis")
            _style(ax)

    fig.suptitle(args.title, fontsize=13, y=0.995)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
