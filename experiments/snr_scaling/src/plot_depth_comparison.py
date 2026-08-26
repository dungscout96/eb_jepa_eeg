"""Figures for the depth comparison at matched pool, budget, and optimum.

The claim rests on PAIRING -- draw 11 of each arm trains on the identical
subjects -- and on the sign flipping between within-task and cross-task. Two
figures, because those are two different jobs:

  depth_arms    the four readouts as small multiples: what each arm scores.
  depth_paired  the per-pair differences, which is where the evidence actually
                lives. A line chart of two means cannot show that 9 of 9 pairs
                point the same way, and 9/9 is the reason the effect is
                believable at +0.008.

Colour follows the ARM, consistently with plot_optimum_comparison.py: the
depth-22 from-scratch arm keeps the hue it has there, and depth-12 takes a third
that was searched for rather than guessed -- worst all-pairs CVD deltaE 18.5
against both existing hues, against a floor of 8. Purple and teal were tried
first and failed outright; a green passed overall but sat at tritan 5.4 against
the orange, below the floor.

Usage:
    uv run --group eeg python experiments/snr_scaling/src/plot_depth_comparison.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "raw_results"
FIGS = ROOT / "figures"

ORANGE = "#eb6834"   # depth-22 from scratch -- same arm, same hue as elsewhere
PINK = "#a03060"     # depth-12 from scratch
INK, INK_2, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"

D12_SEL = RAW / "d12av_selection_probe.json"
D22_SEL = RAW / "fromscratch_optimum_selection.json"
S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]
HI = [1000, 1400, 1863]
READOUTS = [
    ("Within-task probe", "e04_tt_test", "probe", "mean(12) Pearson r"),
    ("Within-task retrieval", "e04_retr_test", "retr", "time-pool e→v@1"),
    ("Cross-task probe", "xtask_tt_DMtest", "probe", "mean(12) Pearson r"),
    ("Cross-task retrieval", "xtask_retr_DMtest", "retr", "time-pool e→v@1"),
]
S_of = lambda c: int(c.split("_")[1][1:])
draw = lambda c: int(c.rsplit("_d", 1)[1])


def read(p, kind):
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if kind == "probe":
        f = d["features"]
        return sum(v["pearson_r"] for v in f.values()) / len(f)
    return d["levels"]["time"]["e2v_top_k"]["1"]


def pairs(prefix, kind, s_filter=None):
    """(S, draw, d12, d22) for every cell present in both arms."""
    a = json.loads(D12_SEL.read_text())
    b = json.loads(D22_SEL.read_text())
    out = []
    for c in a:
        S, dr = S_of(c), draw(c)
        if s_filter and S not in s_filter:
            continue
        m = [x for x in b if S_of(x) == S and draw(x) == dr]
        if not m:
            continue
        va = read(RAW / f"{prefix}_{c}_epprb.json", kind)
        vb = read(RAW / f"{prefix}_{m[0]}_epprb.json", kind)
        if va is None or vb is None:
            continue
        out.append((S, dr, va, vb))
    return out


def style(ax, xlab, ylab, title, logx=True):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=RULE, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_color(INK_2)
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(S_AXIS)
        ax.set_xticklabels([str(s) for s in S_AXIS], rotation=45, ha="right")
        ax.minorticks_off()
    ax.set_xlabel(xlab, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


def fig_arms():
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), facecolor=SURFACE)
    for ax, (title, prefix, kind, ylab) in zip(axes.ravel(), READOUTS):
        ps = pairs(prefix, kind)
        by = {}
        for S, _, a, b in ps:
            by.setdefault(S, ([], []))
            by[S][0].append(a); by[S][1].append(b)
        xs = sorted(by)
        for vals, colour, mark, lbl in ((0, PINK, "D", "depth 12"),
                                        (1, ORANGE, "s", "depth 22")):
            ax.plot(xs, [st.mean(by[s][vals]) for s in xs], color=colour, lw=2.2,
                    marker=mark, ms=6, markerfacecolor=colour,
                    markeredgecolor=colour, label=lbl, zorder=3)
        # Shade where the arms separate, so the reader is not invited to read
        # the low-S wobble as signal.
        ax.axvspan(900, 2000, color=MUTED, alpha=0.10, linewidth=0, zorder=0)
        # Retrieval panels get their chance floor drawn. Without it the
        # cross-task panel's axis spans a hundredth of an r and turns noise
        # around chance into an apparent trend -- the reader cannot tell that
        # every point there is at the floor.
        if kind == "retr":
            f = next((RAW / f"{prefix}_{c}_epprb.json"
                      for c in json.loads(D12_SEL.read_text())
                      if (RAW / f"{prefix}_{c}_epprb.json").exists()), None)
            if f is not None:
                n_pool = json.loads(f.read_text())["levels"]["time"]["n_vision_pool_N"]
                ax.axhline(1.0 / n_pool, color=INK_2, lw=1.2, ls=(0, (3, 2)), zorder=2)
                ax.annotate(f"chance (1/{n_pool})", xy=(10.5, 1.0 / n_pool),
                            va="bottom", color=INK_2, fontsize=8.5)
                lo, hi = ax.get_ylim()
                ax.set_ylim(min(lo, 1.0 / n_pool * 0.96), hi)
        style(ax, "S — pretraining subjects (log scale)", ylab, title)
    axes[0, 0].annotate("shaded: S≥1000,\nwhere the arms separate",
                        xy=(0.60, 0.06), xycoords="axes fraction",
                        color=MUTED, fontsize=8.5)
    handles = [Line2D([], [], color=c, lw=2.2, marker=m, ms=6, markerfacecolor=c,
                      markeredgecolor=c, label=l)
               for c, m, l in ((PINK, "D", "depth 12, from scratch"),
                               (ORANGE, "s", "depth 22, from scratch"))]
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
                     fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle("Depth at matched pool, matched budget, each cell at its own optimum",
                 color=INK, fontsize=11.5, y=0.995, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    return fig


def fig_paired():
    """Where the evidence lives: every pair, not the means."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=SURFACE)
    labels, ys = [], []
    for i, (title, prefix, kind, _) in enumerate(READOUTS):
        ps = pairs(prefix, kind, s_filter=HI)
        ds = [a - b for _, _, a, b in ps]
        y = len(READOUTS) - 1 - i
        ys.append(y); labels.append(title)
        # Retrieval and probe live on different scales, so each row is shown as
        # a fraction of that readout's own paired sd -- otherwise the retrieval
        # row would dwarf the probe rows and the comparison would be about units.
        sd = st.stdev(ds)
        for d in ds:
            ax.plot(d / sd, y, marker="o", ms=7, lw=0,
                    markerfacecolor=PINK if d > 0 else ORANGE,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        m = st.mean(ds) / sd
        ax.plot([m], [y], marker="|", ms=26, mew=2.4, color=INK, zorder=4)
        # Placed beyond the widest point of ANY row -- at a fixed 4.5 the
        # within-retrieval row's own points ran through the text.
        ax.annotate(f"{st.mean(ds):+.4f} r   {sum(1 for d in ds if d>0)}/{len(ds)}",
                    xy=(7.0, y), va="center", color=INK_2, fontsize=9)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(READOUTS) - 0.4)
    ax.set_xlim(-4.2, 10.6)
    style(ax, "paired difference, in units of that readout's own paired sd", "",
          "Every pair at S≥1000 — depth-12 minus depth-22", logx=False)
    ax.annotate("favours depth 22", xy=(-0.25, len(READOUTS)-0.55), ha="right",
                color=ORANGE, fontsize=9)
    ax.annotate("favours depth 12", xy=(0.25, len(READOUTS)-0.55), ha="left",
                color=PINK, fontsize=9)
    ax.annotate("| = mean", xy=(0.25, -0.45), ha="left", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    return fig


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    for name, fn in (("depth_arms", fig_arms), ("depth_paired", fig_paired)):
        fig = fn()
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"{name}.{ext}", dpi=200, facecolor=SURFACE,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/{name}.png / .pdf")


if __name__ == "__main__":
    main()
