"""Figures for E0.6 -- was the depth-22 drop at S=1863 real, or one draw?

Two figures, because the result has two separable claims.

**Figure 1, `addval_subject_scaling`** -- the curves themselves, four readouts as
small multiples. Mean line with a shaded +/- 1 sd band across draws.

  The load-bearing graphical decision: **single-draw points are drawn as hollow
  markers with no band, and annotated `n=1`.** Both arms have exactly one such
  point -- e04 at S=1863 and addval at S=2156 -- because each is that arm's
  whole pool, so only one draw exists. That is the entire subject of this
  experiment, and a figure that plotted them identically to the 3-draw points
  would hide the one thing a reader must see. Nothing is imputed for them: no
  band is drawn where no spread was measured.

  Small multiples rather than one axis with four series, because the four
  readouts have incompatible units (Pearson r vs top-1 accuracy). A dual y-axis
  would be the wrong answer to that problem.

**Figure 2, `addval_drop_decomposition`** -- the argument. For each readout, the
SAME step in S (1400 -> 1863) drawn twice: once inside e04, where it rests on
one draw, and once inside addval, where it rests on three. The direction of the
segment is the finding -- down in every readout with one draw, up in most with
three. Raw units per panel, again as small multiples, so no rescaling is needed
to compare the two arrows within a panel (which is the only comparison asked
for).

Colour follows the ENTITY across both figures -- blue is always the e04 arm
(pool 1863), orange always addval (pool 2156) -- so the legend is learned once.
Validated with the dataviz validator, `--mode light --surface #fcfcfb`:
lightness band, chroma floor, CVD separation (worst adjacent dE 24.7 protan),
normal-vision floor (dE 33.6) and contrast vs surface all PASS. Identity is
also carried by marker shape and by direct labels, so it is never colour-alone.

Light mode only, deliberately: print figures for a LaTeX paper. The accessible
table view is `RESULTS_add_val_set.md`, which carries every plotted value --
including the per-draw table, so a reader can see the cells behind each band.

Reads its data by importing `write_results_add_val_set` by filesystem path, so
the figures and the published tables cannot drift apart: same ROWS, same
FAILED_CELLS exclusion, same artifact loader.

    uv run --group eeg python experiments/snr_scaling/src/plot_add_val_set.py
"""

import importlib.util
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = ROOT / "figures"

BLUE = "#2a78d6"      # slot 1 -- e04, pool 1863
ORANGE = "#eb6834"    # slot 2 -- addval, pool 2156
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
RULE = "#c3c2b7"
SURFACE = "#fcfcfb"

ARM_COLOR = {"e04": BLUE, "addval": ORANGE}
ARM_MARK = {"e04": "o", "addval": "s"}
ARM_NAME = {"e04": "e04 — pool 1863 (R1–R4, R7–R10)",
            "addval": "addval — pool 2156 (+ R5)"}


def _load_results_module():
    path = HERE / "write_results_add_val_set.py"
    spec = importlib.util.spec_from_file_location("_wrav", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load_results_module()

# (title, y-label, getter) -- getter returns one value per CELL, so a stdev
# across them is a between-draw sd.
PANELS = [
    ("Within-task probe (ThePresent)", "mean Pearson r over 12 features",
     lambda sl: W.probe_mean12("e04_tt", sl)),
    ("Within-task retrieval, time pool", "e→v top-1 accuracy",
     lambda sl: W.retr_e2v1("e04_retr", "time", sl)),
    ("Cross-task probe (DespicableMe)", "mean Pearson r over 12 features",
     lambda sl: W.probe_mean12("xtask_tt_DM", sl)),
    ("Cross-task retrieval, scene pool", "e→v top-1 accuracy",
     lambda sl: W.retr_e2v1("xtask_retr_DM", "scene", sl)),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=RULE, linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)


def series(getter):
    """-> {arm: [(S, mean, sd_or_None, n), ...]} using the same exclusions as
    the published tables."""
    out: dict[str, list] = {}
    for arm, s, slugs in W.ROWS:
        vals = getter(slugs)                     # FAILED_CELLS already dropped
        if not vals:
            continue
        sd = st.stdev(vals) if len(vals) > 1 else None
        out.setdefault(arm, []).append((s, st.fmean(vals), sd, len(vals)))
    return {a: sorted(v) for a, v in out.items()}


def fig_curves() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), facecolor=SURFACE)
    for ax, (title, ylab, getter) in zip(axes.ravel(), PANELS):
        data = series(getter)
        for arm, pts in data.items():
            colour = ARM_COLOR[arm]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # The mean line spans every point, but the BAND is drawn only over
            # the contiguous run of multi-draw points -- a band cannot be
            # invented for an n=1 cell.
            band = [(x, y, sd) for x, y, sd, n in pts if sd is not None]
            if len(band) > 1:
                ax.fill_between([b[0] for b in band],
                                [b[1] - b[2] for b in band],
                                [b[1] + b[2] for b in band],
                                color=colour, alpha=0.18, linewidth=0, zorder=1)
            elif len(band) == 1:
                # One multi-draw point: show the spread as a bar, not a band.
                x, y, sd = band[0]
                ax.errorbar([x], [y], yerr=sd, color=colour, elinewidth=1.4,
                            capsize=3, zorder=2, fmt="none")
            ax.plot(xs, ys, color=colour, lw=2, zorder=3, solid_capstyle="round")
            for x, y, sd, n in pts:
                single = sd is None
                ax.plot([x], [y], marker=ARM_MARK[arm], ms=8,
                        markerfacecolor=SURFACE if single else colour,
                        markeredgecolor=colour,
                        markeredgewidth=2.0 if single else 1.4, zorder=4)
                if single:
                    ax.annotate("n=1", xy=(x, y), xytext=(4, -13),
                                textcoords="offset points", fontsize=8,
                                color=colour, fontweight="bold")
                elif n != 3:
                    ax.annotate(f"n={n}", xy=(x, y), xytext=(4, 6),
                                textcoords="offset points", fontsize=7.5,
                                color=MUTED)
        ax.set_xscale("log")
        ax.set_xticks([400, 701, 1000, 1400, 1863, 2156])
        ax.set_xticklabels(["400", "701", "1000", "1400", "1863", "2156"],
                           fontsize=8, rotation=30, ha="right",
                           rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_xlabel("S — pretraining subjects (log scale)", color=INK_2,
                      fontsize=9.5)
        ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
        _style(ax)

    handles = [
        Line2D([], [], color=ARM_COLOR[a], lw=2, marker=ARM_MARK[a], ms=8,
               markerfacecolor=ARM_COLOR[a], markeredgecolor=ARM_COLOR[a],
               label=ARM_NAME[a])
        for a in ("e04", "addval")
    ] + [
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=8,
               markerfacecolor=SURFACE, markeredgecolor=MUTED,
               markeredgewidth=2.0,
               label="hollow = whole pool, n=1 draw (no spread measurable)"),
        Line2D([], [], color=MUTED, lw=6, alpha=0.3,
               label="shaded band = mean ± 1 sd across draws"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
                     frameon=False, bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle(
        "E0.6 — folding R5 into the pool makes S=1863 drawable: the drop was one draw\n"
        "depth-22, step-matched (4400 steps), test split (R6) only, addval at fixed epoch 375",
        color=INK, fontsize=11.5, y=0.978, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.075, 1, 0.945))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"addval_subject_scaling.{ext}", dpi=200,
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {FIGS}/addval_subject_scaling.{{png,pdf}}")
    for title, _y, getter in PANELS:
        d = series(getter)
        bits = []
        for arm in ("e04", "addval"):
            for s, m, sd, n in d.get(arm, []):
                if s in (1400, 1863, 2156):
                    bits.append(f"{arm} S={s} {m:.4f} (n={n})")
        print(f"  {title}: " + "; ".join(bits))


def fig_decomposition() -> None:
    """The same step in S, one draw vs three."""
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.2), facecolor=SURFACE)
    for ax, (title, ylab, getter) in zip(axes.ravel(), PANELS):
        data = series(getter)
        lo, hi = [], []
        for i, arm in enumerate(("e04", "addval")):
            pts = {s: (m, sd, n) for s, m, sd, n in data.get(arm, [])}
            if 1400 not in pts or 1863 not in pts:
                continue
            y0, _sd0, n0 = pts[1400]
            y1, _sd1, n1 = pts[1863]
            colour = ARM_COLOR[arm]
            x = i * 1.0
            ax.annotate("", xy=(x, y1), xytext=(x, y0),
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                        lw=2.2, shrinkA=0, shrinkB=0,
                                        mutation_scale=16))
            ax.plot([x], [y0], marker=ARM_MARK[arm], ms=8,
                    markerfacecolor=SURFACE, markeredgecolor=colour,
                    markeredgewidth=1.8, zorder=4)
            ax.plot([x], [y1], marker=ARM_MARK[arm], ms=8,
                    markerfacecolor=colour, markeredgecolor=SURFACE,
                    markeredgewidth=1.4, zorder=4)
            ax.annotate(f"{y1 - y0:+.4f}",
                        xy=(x, max(y0, y1)), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=colour, fontweight="bold")
            # Pinned to the axes, not to the data -- otherwise the label lands
            # on the arrow whenever a panel's two arms sit at different levels.
            ax.annotate(f"{n1} draw{'s' if n1 > 1 else ''}",
                        xy=(x, 0.02), xycoords=("data", "axes fraction"),
                        ha="center", fontsize=8.5, color=INK_2)
            lo += [y0, y1]
            hi += [y0, y1]
        if lo:
            pad = (max(hi) - min(lo)) * 0.45 or 0.01
            ax.set_ylim(min(lo) - pad, max(hi) + pad)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["e04\n(pool 1863)", "addval\n(pool 2156)"],
                           fontsize=9)
        ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
        _style(ax)
        ax.grid(False, axis="x")

    fig.suptitle(
        "E0.6 — the same step in S (1400 → 1863), measured with one draw and with three\n"
        "hollow marker = S=1400, filled = S=1863; the arrow direction is the finding",
        color=INK, fontsize=11.5, y=0.965, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.01, 1, 0.885))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"addval_drop_decomposition.{ext}", dpi=200,
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {FIGS}/addval_drop_decomposition.{{png,pdf}}")




# ---------------------------------------------------------------------------
# The official curve: addval only, full axis, one pool, one epoch
# ---------------------------------------------------------------------------

OFFICIAL_EPOCH_SUFFIX = "_ep325"

OFFICIAL_PANELS = [
    ("Within-task probe (ThePresent)", "mean Pearson r over 12 features",
     lambda sl: W.probe_mean12("e04_tt", sl, OFFICIAL_EPOCH_SUFFIX)),
    ("Within-task retrieval, time pool", "e→v top-1 accuracy",
     lambda sl: W.retr_e2v1("e04_retr", "time", sl, OFFICIAL_EPOCH_SUFFIX)),
    ("Within-task retrieval, scene pool", "e→v top-1 accuracy",
     lambda sl: W.retr_e2v1("e04_retr", "scene", sl, OFFICIAL_EPOCH_SUFFIX)),
    ("Cross-task probe (DespicableMe)", "mean Pearson r over 12 features",
     lambda sl: W.probe_mean12("xtask_tt_DM", sl, OFFICIAL_EPOCH_SUFFIX)),
]


def official_series(getter):
    """addval only. -> [(S, mean, sd_or_None, n), ...]"""
    out = []
    for arm, s, slugs in W.ROWS:
        if arm != "addval":
            continue
        vals = getter(slugs)
        if not vals:
            continue
        out.append((s, st.fmean(vals),
                    st.stdev(vals) if len(vals) > 1 else None, len(vals)))
    return sorted(out)


def fig_official() -> None:
    """The official subject-scaling curve.

    One pool (2156, R1-R5 + R7-R10) and one fixed epoch (325) at every S, so
    there is no splice seam and no selection seam anywhere on the axis. That is
    the whole reason the low-S cells were retrained from this pool rather than
    reusing e04's: at matched S the pool offset on time-pool retrieval is
    +0.0068, which is 7.5x the real S-effect across S=1000->1400 and of the
    OPPOSITE sign, so a spliced curve would have shown a rise where the
    single-pool data is flat.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), facecolor=SURFACE)
    printed = []
    for ax, (title, ylab, getter) in zip(axes.ravel(), OFFICIAL_PANELS):
        pts = official_series(getter)
        if not pts:
            ax.text(0.5, 0.5, "not yet measured", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED, fontsize=10)
            ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
            _style(ax)
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        band = [(x, y, sd) for x, y, sd, n in pts if sd is not None]
        if len(band) > 1:
            ax.fill_between([b[0] for b in band],
                            [b[1] - b[2] for b in band],
                            [b[1] + b[2] for b in band],
                            color=ORANGE, alpha=0.20, linewidth=0, zorder=1)
        ax.plot(xs, ys, color=ORANGE, lw=2.2, zorder=3, solid_capstyle="round")
        for x, y, sd, n in pts:
            single = sd is None
            ax.plot([x], [y], marker="s", ms=7,
                    markerfacecolor=SURFACE if single else ORANGE,
                    markeredgecolor=ORANGE,
                    markeredgewidth=2.0 if single else 1.4, zorder=4)
            # Any point not backed by the full 3 draws is labelled. Without
            # this an n=2 point is indistinguishable from an n=3 one while the
            # legend claims three, which is a false statement about the data,
            # not a cosmetic gap.
            if n != 3:
                ax.annotate(f"n={n}", xy=(x, y), xytext=(-2, -14),
                            textcoords="offset points", fontsize=8,
                            color=ORANGE, fontweight="bold", ha="right")
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 400, 701, 1400, 2156])
        ax.set_xticklabels(["10", "20", "50", "100", "200", "400", "701",
                            "1400", "2156"], fontsize=8, rotation=30,
                           ha="right", rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_xlabel("S — pretraining subjects (log scale)", color=INK_2,
                      fontsize=9.5)
        ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
        _style(ax)
        printed.append((title, pts))

    handles = [
        Line2D([], [], color=ORANGE, lw=2.2, marker="s", ms=7,
               markerfacecolor=ORANGE, markeredgecolor=ORANGE,
               label="S subjects drawn from the 2156-recording pool "
                     "(R1–R5, R7–R10)"),
        Line2D([], [], color=ORANGE, lw=6, alpha=0.3,
               label="shaded band = mean ± 1 sd over nested draws "
                     "(3 unless labelled)"),
        Line2D([], [], color=MUTED, lw=0, marker="s", ms=7,
               markerfacecolor=SURFACE, markeredgecolor=MUTED,
               markeredgewidth=2.0,
               label="hollow = whole pool, n=1 draw (no spread measurable)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=9,
                     frameon=False, bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle(
        "Subject scaling, depth-22 — one pool and one epoch at every S\n"
        "2156-recording pool, step-matched (4400 steps), fixed epoch 325, test split (R6)",
        color=INK, fontsize=11.5, y=0.978, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.10, 1, 0.945))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"official_subject_scaling.{ext}", dpi=200,
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {FIGS}/official_subject_scaling.{{png,pdf}}")
    for title, pts in printed:
        print(f"  {title}: " + ", ".join(
            f"S={s}:{m:.4f}" + ("" if sd is None else f"±{sd:.4f}")
            for s, m, sd, n in pts))

# ---------------------------------------------------------------------------
# Depth-12 vs depth-22 -- does capacity change the subject-scaling exponent?
# ---------------------------------------------------------------------------

def _e03_slug(S: int) -> str:
    """e03's full-pool cell carries no draw suffix; every other S is _d11."""
    return ("e03_s1863_a101_nd" if S == 1863
            else f"e03_s{S}_a101_nd_d11")


def e03_series(prefix: str, getter):
    out = []
    for S in [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]:
        d = W.load(W.RAW / f"{W.tag(prefix)}_{_e03_slug(S)}.json")
        if d is not None:
            out.append((S, getter(d)))
    return out


# The from-scratch depth-22 arm (`e05fs_`, pool 2156, no warm start) -- the
# same 31 cells as the official curve, trained identically MINUS the REVE
# initialisation, and evaluated at the same fixed epoch 325. This supersedes
# the three-point stopgap that lived in `raw_results/_invalid_e05_fromscratch`
# (the accidental first sweep, S>=1400 only, at epoch 375).
#
# Reading the real arm rather than the stopgap matters for more than coverage:
# the stopgap was at a different epoch and had no low-S points, so it could not
# show where the two curves CROSS -- which is the part of the comparison that
# says the warm start is worthless at small cohorts.
FROMSCRATCH_SUFFIX = "_ep325"
FROMSCRATCH_S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]


def fromscratch_series(prefix: str, getter):
    """-> [(S, mean, sd_or_None, n), ...] for the e05fs arm, full axis."""
    out = []
    for S in FROMSCRATCH_S_AXIS + [2156]:
        slugs = ([f"e05fs_s{S}_a101_av_d{d}" for d in (11, 22, 33)]
                 if S != 2156 else ["e05fs_s2156_a101_av"])
        vals = []
        for sl in slugs:
            f = W.RAW / f"{W.tag(prefix)}_{sl}{FROMSCRATCH_SUFFIX}.json"
            if f.exists():
                vals.append(getter(W.load(f)))
        if vals:
            out.append((S, st.fmean(vals),
                        st.stdev(vals) if len(vals) > 1 else None, len(vals)))
    return out


GREEN = "#00a08a"   # slot 3 -- depth-22 from scratch. Validated with the
                    # dataviz validator at --pairs all: chroma floor, lightness
                    # band, contrast, CVD (worst 11.6 protan / 6.3 tritan) and
                    # normal-vision floor (18.6) all PASS. The tritan 6.3 sits
                    # in the band that is legal ONLY with secondary encoding --
                    # which is why linestyle also carries initialisation below.

DEPTH_PANELS = [
    ("Within-task probe (ThePresent)", "mean Pearson r over 12 features",
     "e03_tt", "e04_tt",
     lambda d: st.fmean([d["features"][f]["pearson_r"]
                         for f in W.FEATURES if f in d["features"]]),
     lambda sl: W.probe_mean12("e04_tt", sl, OFFICIAL_EPOCH_SUFFIX)),
    ("Within-task retrieval, time pool", "e→v top-1 accuracy",
     "e03_retr", "e04_retr",
     lambda d: d["levels"]["time"]["e2v_top_k"]["1"],
     lambda sl: W.retr_e2v1("e04_retr", "time", sl, OFFICIAL_EPOCH_SUFFIX)),
]


def fig_depth() -> None:
    """What the REVE warm start buys, and why the depth comparison is confounded.

    THE CORRECTION THIS FIGURE ENCODES. A depth-12-vs-depth-22 plot invites the
    reading "capacity changes the subject-scaling curve". It does not survive
    contact with the initialisation axis:

      - e03 (depth-12) trains FROM SCRATCH -- verified in its wandb args, which
        carry no `--meta.encoder_init_from`.
      - e04/e05 (depth-22) WARM-START from reve_base_eet_init.pth.tar.

      At matched from-scratch init and S=1400 the two depths are 0.1918
      (depth-12) vs 0.1853 +/- 0.0025 (depth-22) -- indistinguishable, with the
      DEEPER one marginally lower. The warm start is worth +0.111 at the same S,
      roughly 15x the depth difference. So the gap in a naive depth plot is the
      initialisation, not the capacity.

    Encoding: colour is the arm, LINESTYLE is the initialisation (solid = warm
    start, dashed = from scratch). That redundancy is required, not decorative --
    the three-colour palette's worst tritan separation is 6.3, which the
    validator permits only with secondary encoding.

    Remaining confound, stated rather than hidden: depth-12 also uses
    patch_size=400/overlap=0 against depth-22's 200/20, so even the matched-init
    comparison is depth+patchification. It happens not to matter for the
    conclusion here, because that comparison comes out null.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), facecolor=SURFACE)
    for ax, (title, ylab, e03_prefix, av_prefix, e03_get, av_get) in zip(
            axes, DEPTH_PANELS):
        # depth-22, warm start, full axis
        pts = official_series(av_get)
        band = [(x, y, sd) for x, y, sd, n in pts if sd is not None]
        if len(band) > 1:
            ax.fill_between([b[0] for b in band],
                            [b[1] - b[2] for b in band],
                            [b[1] + b[2] for b in band],
                            color=ORANGE, alpha=0.20, linewidth=0, zorder=1)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ORANGE, lw=2.2,
                marker="s", ms=7, markerfacecolor=ORANGE,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)

        # depth-22, from scratch, top of the axis only
        fs = fromscratch_series(av_prefix, e03_get)
        if not fs:
            raise RuntimeError(
                "no e05fs_*_ep325 artifacts found; the legend would claim a "
                "series that is not drawn")
        fsb = [(x, y, sd) for x, y, sd, n in fs if sd is not None]
        if len(fsb) > 1:
            ax.fill_between([b[0] for b in fsb],
                            [b[1] - b[2] for b in fsb],
                            [b[1] + b[2] for b in fsb],
                            color=GREEN, alpha=0.20, linewidth=0, zorder=1)
        ax.plot([p[0] for p in fs], [p[1] for p in fs], color=GREEN, lw=2.0,
                linestyle="--", marker="D", ms=6, markerfacecolor=GREEN,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)

        # depth-12, from scratch, full axis, single draw
        e3 = e03_series(e03_prefix, e03_get)
        if not e3:
            raise RuntimeError(
                f"no depth-12 artifacts matched prefix {e03_prefix!r}")
        ax.plot([x for x, _ in e3], [y for _, y in e3], color=BLUE, lw=1.8,
                linestyle="--", marker="o", ms=6, markerfacecolor=SURFACE,
                markeredgecolor=BLUE, markeredgewidth=1.6, zorder=3)

        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 400, 701, 1400, 2156])
        ax.set_xticklabels(["10", "20", "50", "100", "200", "400", "701",
                            "1400", "2156"], fontsize=8, rotation=30,
                           ha="right", rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_xlabel("S — pretraining subjects (log scale)", color=INK_2,
                      fontsize=9.5)
        ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
        _style(ax)

    handles = [
        Line2D([], [], color=ORANGE, lw=2.2, marker="s", ms=7,
               markerfacecolor=ORANGE, markeredgecolor=ORANGE,
               label="depth-22, REVE WARM START — pool 2156, 3 draws"),
        Line2D([], [], color=GREEN, lw=2.0, linestyle="--", marker="D", ms=6,
               markerfacecolor=GREEN, markeredgecolor=GREEN,
               label="depth-22, FROM SCRATCH — pool 2156, 3 draws"),
        Line2D([], [], color=BLUE, lw=1.8, linestyle="--", marker="o", ms=6,
               markerfacecolor=SURFACE, markeredgecolor=BLUE,
               label="depth-12, FROM SCRATCH — pool 1863, single draw, no band"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=9,
                     frameon=False, bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle(
        "The REVE warm start, not depth, drives the gap — at matched init the two depths coincide\n"
        "fixed epoch 325 (from-scratch depth-22 at 375; difference negligible at high S), "
        "step-matched, test split",
        color=INK, fontsize=11, y=0.975, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.17, 1, 0.90))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"warmstart_vs_fromscratch_subject_scaling.{ext}",
                    dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {FIGS}/warmstart_vs_fromscratch_subject_scaling.{{png,pdf}}")
    for title, _y, e03_prefix, av_prefix, e03_get, av_get in DEPTH_PANELS:
        e3 = dict(e03_series(e03_prefix, e03_get))
        fs = {x: y for x, y, _s, _n in fromscratch_series(av_prefix, e03_get)}
        wm = {x: y for x, y, _s, _n in official_series(av_get)}
        print(f"  {title} @S=1400: d12-scratch {e3.get(1400, float('nan')):.4f} | "
              f"d22-scratch {fs.get(1400, float('nan')):.4f} | "
              f"d22-warm {wm.get(1400, float('nan')):.4f}")


# ---------------------------------------------------------------------------
# The paper figure: both of the above on one canvas
# ---------------------------------------------------------------------------

def _probe_raw(d):
    return st.fmean([d["features"][f]["pearson_r"]
                     for f in W.FEATURES if f in d["features"]])


def _time_raw(d):
    return d["levels"]["time"]["e2v_top_k"]["1"]


# Rows are the TASK, columns are the readout -- so reading down a column asks
# "does this readout transfer", and reading across a row asks "do two readouts
# that share no fitted parameters agree". Probe and time pool only: the shot and
# scene pools carry the same shape at coarser resolution, so they are appendix
# material rather than a third and fourth column here.
COMBINED_PANELS = [
    # (title, ylab, e03_prefix, av_prefix, raw_getter, warm_getter, native)
    ("Within-task probe (ThePresent)", "mean Pearson r over 12 features",
     "e03_tt", "e04_tt", _probe_raw,
     lambda sl: W.probe_mean12("e04_tt", sl, OFFICIAL_EPOCH_SUFFIX), False),
    ("Within-task retrieval, time pool", "e→v top-1 accuracy",
     "e03_retr", "e04_retr", _time_raw,
     lambda sl: W.retr_e2v1("e04_retr", "time", sl, OFFICIAL_EPOCH_SUFFIX), False),
    ("Cross-task probe (DespicableMe)", "mean Pearson r over 12 features",
     "xtask_tt_DM", "xtask_tt_DM", _probe_raw,
     lambda sl: W.probe_mean12("xtask_tt_DM", sl, OFFICIAL_EPOCH_SUFFIX), True),
    ("Cross-task retrieval, time pool", "e→v top-1 accuracy",
     "xtask_retr_DM", "xtask_retr_DM", _time_raw,
     lambda sl: W.retr_e2v1("xtask_retr_DM", "time", sl, OFFICIAL_EPOCH_SUFFIX), True),
]

# The DespicableMe-native anchor: a model trained ON DespicableMe with the
# identical recipe, so it is the "what would you get without transferring at
# all" reference for the two cross-task panels.
#
# THREE THINGS THIS IS NOT, all of which the label has to carry:
#   1. It is not at S=2156. Only two native cells were ever trained, S=701 and
#      S=1841; 1841 is the largest and is what is drawn.
#   2. It is depth-12 FROM SCRATCH -- the blue arm's configuration, not the
#      orange one's. So it is the fair ceiling for blue and NOT for orange.
#   3. It is therefore not an upper bound in general: on the probe the orange
#      depth-22 warm-started transfer passes straight through it.
# Drawn as a neutral dotted rule rather than a fourth coloured series, because
# it is a reference level, not another point on the subject axis.
NATIVE_SLUG = "e03_s1841_a85_dm"
NATIVE_S = 1841


def native_value(prefix: str, raw_get):
    d = W.load(W.RAW / f"{W.tag(prefix)}_{NATIVE_SLUG}.json")
    return None if d is None else raw_get(d)


def _draw_three_arms(ax, e03_prefix, av_prefix, raw_get, warm_get) -> bool:
    """The three initialisation/depth arms on one axis. -> did anything draw?"""
    drew = False

    pts = official_series(warm_get)          # depth-22, warm start, full axis
    if pts:
        drew = True
        band = [(x, y, sd) for x, y, sd, n in pts if sd is not None]
        if len(band) > 1:
            ax.fill_between([b[0] for b in band],
                            [b[1] - b[2] for b in band],
                            [b[1] + b[2] for b in band],
                            color=ORANGE, alpha=0.20, linewidth=0, zorder=1)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ORANGE, lw=2.2,
                zorder=4, solid_capstyle="round")
        for x, y, sd, n in pts:
            single = sd is None
            ax.plot([x], [y], marker="s", ms=7,
                    markerfacecolor=SURFACE if single else ORANGE,
                    markeredgecolor=ORANGE,
                    markeredgewidth=2.0 if single else 1.2, zorder=5)
            if n != 3:
                ax.annotate(f"n={n}", xy=(x, y), xytext=(-2, -14),
                            textcoords="offset points", fontsize=7.5,
                            color=ORANGE, fontweight="bold", ha="right")

    fs = fromscratch_series(av_prefix, raw_get)   # depth-22, from scratch
    if fs:
        drew = True
        fsb = [(x, y, sd) for x, y, sd, n in fs if sd is not None]
        if len(fsb) > 1:
            ax.fill_between([b[0] for b in fsb],
                            [b[1] - b[2] for b in fsb],
                            [b[1] + b[2] for b in fsb],
                            color=GREEN, alpha=0.20, linewidth=0, zorder=1)
        ax.plot([p[0] for p in fs], [p[1] for p in fs], color=GREEN, lw=2.0,
                linestyle="--", marker="D", ms=6, markerfacecolor=GREEN,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)

    e3 = e03_series(e03_prefix, raw_get)          # depth-12, from scratch
    if e3:
        drew = True
        ax.plot([x for x, _ in e3], [y for _, y in e3], color=BLUE, lw=1.8,
                linestyle="--", marker="o", ms=6, markerfacecolor=SURFACE,
                markeredgecolor=BLUE, markeredgewidth=1.6, zorder=3)
    return drew


def fig_combined() -> None:
    """The paper's subject-scaling figure: four readouts x three arms.

    Merges what were two figures -- the official warm-start curve and the
    warm-start-vs-from-scratch comparison -- because they share an x-axis and a
    conclusion. Keeping them apart made the reader hold the warm arm's shape in
    memory while looking at the second figure to see what it is being compared
    against.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), facecolor=SURFACE)
    for ax, (title, ylab, e03_prefix, av_prefix, raw_get, warm_get,
             show_native) in zip(axes.ravel(), COMBINED_PANELS):
        if not _draw_three_arms(ax, e03_prefix, av_prefix, raw_get, warm_get):
            ax.text(0.5, 0.5, "not yet measured", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED, fontsize=10)
            ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
            _style(ax)
            continue
        if show_native:
            nat = native_value(e03_prefix, raw_get)
            if nat is not None:
                ax.axhline(nat, color=INK_2, lw=1.4, linestyle=":", zorder=2)
                ax.annotate(f"DM-native (d-12 scratch, S={NATIVE_S}): {nat:.3f}",
                            xy=(0.015, nat), xycoords=("axes fraction", "data"),
                            xytext=(0, 4), textcoords="offset points",
                            fontsize=8, color=INK_2)
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200, 400, 701, 1400, 2156])
        ax.set_xticklabels(["10", "20", "50", "100", "200", "400", "701",
                            "1400", "2156"], fontsize=8, rotation=30,
                           ha="right", rotation_mode="anchor")
        ax.minorticks_off()
        ax.set_xlabel("S — pretraining subjects (log scale)", color=INK_2,
                      fontsize=9.5)
        ax.set_ylabel(ylab, color=INK_2, fontsize=9.5)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
        _style(ax)

    handles = [
        Line2D([], [], color=ORANGE, lw=2.2, marker="s", ms=7,
               markerfacecolor=ORANGE, markeredgecolor=ORANGE,
               label="depth-22, REVE WARM START — pool 2156, 3 draws"),
        Line2D([], [], color=GREEN, lw=2.0, linestyle="--", marker="D", ms=6,
               markerfacecolor=GREEN, markeredgecolor=GREEN,
               label="depth-22, FROM SCRATCH — pool 2156, 3 draws"),
        Line2D([], [], color=BLUE, lw=1.8, linestyle="--", marker="o", ms=6,
               markerfacecolor=SURFACE, markeredgecolor=BLUE,
               label="depth-12, FROM SCRATCH — pool 1863, single draw, no band"),
        Line2D([], [], color=MUTED, lw=0, marker="s", ms=7,
               markerfacecolor=SURFACE, markeredgecolor=MUTED,
               markeredgewidth=2.0,
               label="hollow = whole pool, n=1 draw (no spread measurable)"),
        Line2D([], [], color=INK_2, lw=1.4, linestyle=":",
               label=f"DespicableMe-NATIVE anchor, depth-12 from scratch, "
                     f"S={NATIVE_S} (no native cell at S=2156)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
                     frameon=False, bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle(
        "Subject scaling — top row within-task, bottom row cross-task; probe and time-pool retrieval\n"
        "2156-recording pool, step-matched (4400 steps), fixed epoch 325 "
        "(from-scratch depth-22 at 375), test split (R6)",
        color=INK, fontsize=11.5, y=0.978, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.085, 1, 0.945))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"subject_scaling_combined.{ext}", dpi=200,
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {FIGS}/subject_scaling_combined.{{png,pdf}}")
    for title, _y, e03_prefix, av_prefix, raw_get, warm_get, show_native in \
            COMBINED_PANELS:
        e3 = dict(e03_series(e03_prefix, raw_get))
        fs = {x: y for x, y, _s, _n in fromscratch_series(av_prefix, raw_get)}
        wm = {x: y for x, y, _s, _n in official_series(warm_get)}
        line = (f"  {title} @S=1400: d12-scratch {e3.get(1400, float('nan')):.4f} | "
                f"d22-scratch {fs.get(1400, float('nan')):.4f} | "
                f"d22-warm {wm.get(1400, float('nan')):.4f}")
        if show_native:
            nat = native_value(e03_prefix, raw_get)
            top = wm.get(2156)
            # Printed because the sign of this comparison differs between the
            # two cross-task panels, and that is the finding, not a detail.
            line += (f"\n      native(S={NATIVE_S}) {nat:.4f} vs d22-warm@2156 "
                     f"{top:.4f} -> transfer is "
                     f"{'ABOVE' if top > nat else 'below'} native")
        print(line)


def main() -> None:
    if W.FAILED_CELLS:
        print("excluded from every mean/band (see RESULTS_add_val_set.md 2b):")
        for slug in W.FAILED_CELLS:
            print(f"  {slug}")
    fig_curves()
    fig_decomposition()
    fig_official()
    fig_depth()
    fig_combined()

if __name__ == "__main__":
    main()
