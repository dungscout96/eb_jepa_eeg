"""Read a saved embedding .npz (from ``plot_modality_gap.py --save-npz ...``)
and produce a 2x2 t-SNE panel colored by different structural labels:

    (a) Modality (EEG vs V-JEPA-2)      — baseline reference
    (b) Movie (ThePresent vs DespicableMe)
    (c) Scene id                        — categorical (viridis over scene index)
    (d) Shot id                         — categorical (viridis over shot index)

Uses the t-SNE coords already stored in the .npz so this figure and the
modality-gap figure show the same points in the same 2D layout.

Usage:
    PYTHONPATH=. uv run --group eeg python \\
        experiments/clip_pretraining/cs_aligner/plot_embedding_structure.py \\
        --npz experiments/clip_pretraining/cs_aligner/panel_b.npz \\
        --output experiments/clip_pretraining/cs_aligner/embedding_structure.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="Path to .npz saved by plot_modality_gap.py --save-npz")
    ap.add_argument("--output", default="embedding_structure.png")
    ap.add_argument("--title", default="Embedding structure after training")
    return ap.parse_args()


def plot_modality(ax, xy, is_vision):
    ax.scatter(xy[~is_vision, 0], xy[~is_vision, 1], s=6, alpha=0.5,
               c="#1f77b4", edgecolors="none", label="EEG")
    ax.scatter(xy[is_vision, 0], xy[is_vision, 1], s=6, alpha=0.5,
               c="#d62728", edgecolors="none", label="V-JEPA-2")
    ax.legend(loc="lower right", frameon=False, fontsize=8, markerscale=1.5)
    ax.set_title("(a) Modality", fontsize=12)


def plot_categorical(ax, xy, labels, palette, title, legend_max=6):
    """Scatter with a distinct color per unique label. Invalid labels
    (``label < 0`` for ints, empty for strings) are drawn light gray."""
    uniq = np.array([u for u in np.unique(labels) if not _is_invalid(u)])
    color_for = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
    invalid_mask = np.array([_is_invalid(x) for x in labels])
    if invalid_mask.any():
        ax.scatter(xy[invalid_mask, 0], xy[invalid_mask, 1], s=6, alpha=0.35,
                   c="#cccccc", edgecolors="none")
    for u in uniq:
        m = labels == u
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.65,
                   c=[color_for[u]], edgecolors="none",
                   label=str(u) if len(uniq) <= legend_max else None)
    if len(uniq) <= legend_max:
        ax.legend(loc="lower right", frameon=False, fontsize=8, markerscale=1.5)
    ax.set_title(title, fontsize=12)


def plot_continuous(ax, xy, values, title, cmap="viridis"):
    """Scatter colored by a continuous scalar (e.g., scene id, shot id).
    Invalid values (``< 0``) drawn light gray."""
    values = np.asarray(values)
    valid = values >= 0
    if (~valid).any():
        ax.scatter(xy[~valid, 0], xy[~valid, 1], s=6, alpha=0.35,
                   c="#cccccc", edgecolors="none")
    if valid.any():
        sc = ax.scatter(xy[valid, 0], xy[valid, 1], s=6, alpha=0.6,
                        c=values[valid], cmap=cmap, edgecolors="none")
        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=12)


def _is_invalid(x):
    if isinstance(x, (bytes, np.bytes_, str, np.str_)):
        return x in ("", b"")
    try:
        return int(x) < 0
    except (TypeError, ValueError):
        return False


def _style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#888")
        spine.set_linewidth(0.6)


def main():
    args = parse_args()
    data = np.load(args.npz, allow_pickle=False)
    xy = data["xy"]
    is_vision = data["is_vision"].astype(bool)
    task = data["task"]
    if task.dtype.kind == "S":
        task = np.array([t.decode() for t in task])
    shot_id = data["shot_id"]
    scene_id = data["scene_id"]

    print(f"Loaded {len(xy)} points, {is_vision.sum()} V-JEPA-2, "
          f"{(~is_vision).sum()} EEG, {len(np.unique(task))} movies, "
          f"{(shot_id >= 0).sum()} with valid shot id, "
          f"{(scene_id >= 0).sum()} with valid scene id.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    plot_modality(axes[0, 0], xy, is_vision)

    movie_palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    plot_categorical(axes[0, 1], xy, task, movie_palette, "(b) Movie", legend_max=6)

    plot_continuous(axes[1, 0], xy, scene_id, "(c) Scene id", cmap="viridis")
    plot_continuous(axes[1, 1], xy, shot_id, "(d) Shot id", cmap="turbo")

    for ax in axes.ravel():
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
