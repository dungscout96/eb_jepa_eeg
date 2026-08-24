"""Figures for the initialisation comparison at each cell's own optimum.

Three questions, three figures:

  1. What does the comparison look like when neither arm is handicapped by the
     other's stopping rule? (both arms, both protocols, both readouts)
  2. How much of the reported gap was protocol rather than initialisation?
  3. Why -- where does each arm's optimum actually sit as data grows?

Palette is the one the published figures already use, revalidated for this
chart: two hues, worst all-pairs CVD deltaE 24.7 against a floor of 8. Protocol
is encoded by LINE STYLE and marker fill, not by a third hue, so identity never
rests on colour alone and the arms stay the colours a reader already associates
with them.

Usage:
    uv run --group eeg python experiments/snr_scaling/src/plot_optimum_comparison.py
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

# Same tokens as the published figures (plot_add_val_set.py).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#fcfcfb"

WARM_SEL = RAW / "addval_selection_probe.json"
SCRATCH_SEL = RAW / "fromscratch_optimum_selection.json"
ARM = {"warm": dict(color=BLUE, mark="o", label="warm start (REVE)"),
       "scratch": dict(color=ORANGE, mark="s", label="from scratch")}
S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]


def s_of(c): return int(c.split("_")[1][1:])
def fixed_slug(c): return c.replace("_ep800", "")


def mean_r(p):
    if not p.exists(): return None
    f = json.loads(p.read_text())["features"]
    return sum(v["pearson_r"] for v in f.values()) / len(f)


def e2v1(p):
    if not p.exists(): return None
    return json.loads(p.read_text())["levels"]["time"]["e2v_top_k"]["1"]


def cells(sel):
    out = {}
    for c in json.loads(sel.read_text()):
        out.setdefault(s_of(c), []).append(c)
    return out


def series(cell_map, prefix, suffix, reader, slug_fn=lambda c: c):
    xs, ys = [], []
    for S in S_AXIS:
        vals = [reader(RAW / f"{prefix}_{slug_fn(c)}{suffix}.json")
                for c in cell_map.get(S, [])]
        vals = [v for v in vals if v is not None]
        if vals:
            xs.append(S); ys.append(st.mean(vals))
    return xs, ys


def style_axes(ax, xlab, ylab, title):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=RULE, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    ax.set_xscale("log")
    ax.set_xticks(S_AXIS)
    ax.set_xticklabels([str(s) for s in S_AXIS], rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_xlabel(xlab, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


def fig_arms(warm, scratch):
    """Both arms, both protocols, both readouts."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor=SURFACE)
    panels = [
        (axes[0], "e04_tt_test", mean_r, "mean(12) Pearson r",
         "Within-task probe"),
        (axes[1], "e04_retr_test", e2v1, "time-pool e→v@1",
         "Within-task retrieval"),
    ]
    for ax, prefix, reader, ylab, title in panels:
        for name, cmap, slug_fn in (("warm", warm, lambda c: c),
                                    ("scratch", scratch, fixed_slug)):
            spec = ARM[name]
            xf, yf = series(cmap, prefix, "_ep325", reader, slug_fn)
            xo, yo = series(cmap, prefix, "_epprb", reader)
            # Fixed epoch: dashed, hollow. Own optimum: solid, filled. Protocol
            # never rides on hue -- the two arms keep the colours the published
            # figures gave them.
            ax.plot(xf, yf, color=spec["color"], lw=1.6, ls=(0, (4, 2)),
                    marker=spec["mark"], ms=5.5, markerfacecolor=SURFACE,
                    markeredgecolor=spec["color"], zorder=2)
            ax.plot(xo, yo, color=spec["color"], lw=2.2, marker=spec["mark"],
                    ms=6.5, markerfacecolor=spec["color"],
                    markeredgecolor=spec["color"], zorder=3)
        style_axes(ax, "S — pretraining subjects (log scale)", ylab, title)
    handles = [Line2D([], [], color=ARM[a]["color"], lw=2.2, marker=ARM[a]["mark"],
                      ms=6.5, markerfacecolor=ARM[a]["color"],
                      markeredgecolor=ARM[a]["color"], label=ARM[a]["label"])
               for a in ("warm", "scratch")]
    handles += [
        Line2D([], [], color=MUTED, lw=2.2, marker="o", ms=6.5,
               markerfacecolor=MUTED, markeredgecolor=MUTED,
               label="at each cell's own optimum"),
        Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 2)), marker="o", ms=5.5,
               markerfacecolor=SURFACE, markeredgecolor=MUTED,
               label="at fixed epoch 325 (as published)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                     fontsize=9, bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle("Initialisation comparison, with neither arm held to the other's "
                 "stopping rule", color=INK, fontsize=11.5, y=0.99, x=0.008,
                 ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def fig_gap(warm, scratch):
    """The gap itself, and how much of it was protocol."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    for prefix, reader, mark, lbl in (
            ("e04_tt_test", mean_r, "o", "probe — mean(12) r"),
            ("e04_retr_test", e2v1, "s", "retrieval — e→v@1")):
        xf, wf = series(warm, prefix, "_ep325", reader)
        _, sf = series(scratch, prefix, "_ep325", reader, fixed_slug)
        xo, wo = series(warm, prefix, "_epprb", reader)
        _, so = series(scratch, prefix, "_epprb", reader)
        gap_f = [a - b for a, b in zip(wf, sf)]
        gap_o = [a - b for a, b in zip(wo, so)]
        c = BLUE if mark == "o" else ORANGE
        ax.plot(xf, gap_f, color=c, lw=1.6, ls=(0, (4, 2)), marker=mark, ms=5.5,
                markerfacecolor=SURFACE, markeredgecolor=c, zorder=2)
        ax.plot(xo, gap_o, color=c, lw=2.2, marker=mark, ms=6.5,
                markerfacecolor=c, markeredgecolor=c, zorder=3,
                label=lbl)
        ax.fill_between(xf, gap_o, gap_f, color=c, alpha=0.12, linewidth=0,
                        zorder=1)
    ax.axhline(0, color=RULE, lw=1)
    style_axes(ax, "S — pretraining subjects (log scale)",
               "warm − from scratch", "The warm-start advantage, before and after")
    ax.annotate("shaded = the part that was\nthe stopping rule, not the init",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                color=MUTED, fontsize=8.5)
    # Upper-left: the lower-right corner is where both optimum series run at
    # high S, and a legend there sits on top of the orange line.
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left",
                    bbox_to_anchor=(0.03, 0.82))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    return fig


def fig_optima(warm, scratch):
    """Where each arm's optimum sits -- the mechanism behind the correction."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    for name, sel in (("warm", WARM_SEL), ("scratch", SCRATCH_SEL)):
        spec = ARM[name]
        d = json.loads(sel.read_text())
        by = {}
        for c, v in d.items():
            by.setdefault(s_of(c), []).append(v["selected_epoch"])
        xs = sorted(by)
        ax.plot(xs, [st.median(by[s]) for s in xs], color=spec["color"], lw=2.2,
                marker=spec["mark"], ms=6.5, markerfacecolor=spec["color"],
                markeredgecolor=spec["color"], label=spec["label"], zorder=3)
        for s in xs:
            ax.plot([s] * len(by[s]), by[s], color=spec["color"], lw=0,
                    marker=spec["mark"], ms=4, alpha=0.35, zorder=2)
    ax.axhline(375, color=MUTED, lw=1.2, ls=(0, (2, 2)))
    ax.annotate("last checkpoint of the original budget", xy=(11, 388),
                color=MUTED, fontsize=8.5)
    ax.axhline(325, color=RULE, lw=1)
    ax.annotate("the fixed epoch", xy=(11, 300), color=MUTED, fontsize=8.5)
    style_axes(ax, "S — pretraining subjects (log scale)",
               "probe-selected epoch",
               "Where the optimum sits, and why one fixed epoch cannot serve both")
    ax.set_yscale("linear")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    return fig


def main() -> None:
    warm, scratch = cells(WARM_SEL), cells(SCRATCH_SEL)
    FIGS.mkdir(exist_ok=True)
    for name, fn in (("optimum_arms", fig_arms),
                     ("optimum_gap", fig_gap),
                     ("optimum_epochs", fig_optima)):
        fig = fn(warm, scratch)
        for ext in ("png", "pdf"):
            out = FIGS / f"{name}.{ext}"
            fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/{name}.png / .pdf")


if __name__ == "__main__":
    main()
