"""Figure for E0.3 -- the (anchors x subjects) scaling surface.

Two panels, because the result has two independent claims:

  (a) The two axes behave differently. Delta r-squared vs FRACTION of the
      available data on each axis, so subjects and anchors -- which have
      incompatible units -- land on one comparable x. Log-log, so a power law
      is a straight line and saturation is a visible bend. The anchor curve
      flattens in its last doubling; the subject curve does not.

  (b) The model-free version. Four iso-budget pairs: the same number of
      (subject x anchor) pairs, spent on subjects vs on anchors. A dumbbell,
      because the claim is a paired comparison within each budget, not four
      independent magnitudes.

Colour follows the ENTITY across both panels -- blue is always the subject
axis, orange always the anchor axis -- so the legend is learned once. Validated
with scripts/validate_palette.js --mode light: CVD dE 24.7, normal-vision 33.6,
contrast >= 3:1, all six checks PASS.

Error bars are jul7's measured seed noise (sigma = 0.0010, 3-seed std,
RESULTS_jul7.md 3.2), NOT a within-run error. E0.3 ran n=1 seed per cell, so
this is the honest way to show how much of the anchor flattening is resolvable:
the A=50->101 step is ~3.4 sigma, and the figure should let a reader see that
rather than take the exponent on faith.

Light mode only, deliberately: a print figure for a LaTeX paper. The accessible
table view is RESULTS.md section 2.10, which carries every plotted value.

    uv run --group eeg python experiments/snr_scaling/plot_e03.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

BLUE = "#2a78d6"      # slot 1 -- subjects, the axis that keeps paying
ORANGE = "#eb6834"    # slot 2 -- anchors, the axis that saturates
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
RULE = "#c3c2b7"
SURFACE = "#fcfcfb"

SEED_SIGMA = 0.0010   # jul7 3-seed std on delta r2
FULL_S, FULL_A = 701, 101


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


def main() -> None:
    d = json.loads((HERE / "e03_surface.json").read_text())
    cells = d["cells"]

    s_axis = [50, 100, 200, 400, FULL_S]
    a_axis = [13, 25, 50, FULL_A]
    s_y = [cells[f"S{s}_A{FULL_A}"]["d_r2"] for s in s_axis]
    a_y = [cells[f"S{FULL_S}_A{a}"]["d_r2"] for a in a_axis]
    s_x = [s / FULL_S for s in s_axis]
    a_x = [a / FULL_A for a in a_axis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.5), facecolor=SURFACE)

    # ------------------- (a) the two axes, log-log ----------------------
    # Plotted RELATIVE to each axis's own full-data value. Absolute delta r2
    # would put anchors above subjects everywhere -- at A=13 the model still
    # has all 701 subjects -- and a reader would take the higher line for the
    # better axis. The claim is about SLOPE, so normalise the level away and
    # let the shapes be compared directly. Both curves therefore reach 1.0 by
    # construction; that is stated on the panel.
    corner = cells[f"S{FULL_S}_A{FULL_A}"]["d_r2"]
    for x, y, colour, mark, label in (
        (s_x, s_y, BLUE, "o", f"subjects (S, up to {FULL_S})"),
        (a_x, a_y, ORANGE, "s", f"anchors (A, up to {FULL_A})"),
    ):
        yn = [v / corner for v in y]
        ax1.errorbar(x, yn, yerr=SEED_SIGMA / corner, color=colour, lw=2,
                     marker=mark, ms=7, markerfacecolor=colour,
                     markeredgecolor=SURFACE, markeredgewidth=1.4, capsize=3,
                     elinewidth=1.2, ecolor=colour, zorder=3, label=label)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks([0.1, 0.2, 0.5, 1.0])
    ax1.set_xticklabels(["10%", "20%", "50%", "100%"])
    ax1.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0])
    ax1.set_yticklabels(["0.2", "0.3", "0.5", "0.7", "1.0"])
    ax1.set_ylim(0.12, 1.55)
    ax1.minorticks_off()          # log minor labels fight the explicit ticks
    ax1.set_xlabel("fraction of the available data on that axis",
                   color=INK_2, fontsize=10)
    ax1.set_ylabel(r"$\Delta r^2$ relative to full data on that axis",
                   color=INK_2, fontsize=10)
    ax1.set_title("(a)  Subjects keep paying; anchors saturate",
                  color=INK, fontsize=11, loc="left", pad=10)
    _style(ax1)
    leg = ax1.legend(loc="lower right", fontsize=8.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # The whole claim is the slope in the LAST doubling. Label only that.
    ax1.annotate(f"+{d['local_exponent_A_at_corner']:.2f}",
                 xy=(0.615, 1.10), fontsize=10.5, color=ORANGE, fontweight="bold")
    ax1.annotate(f"+{d['local_exponent_S_at_corner']:.2f}",
                 xy=(0.615, 0.70), fontsize=10.5, color=BLUE, fontweight="bold")
    ax1.annotate("local exponent over\nthe final doubling",
                 xy=(0.635, 0.48), fontsize=7.5, color=MUTED, ha="left",
                 va="center", style="italic")
    # Both caveats live top-left, the one region no curve enters.
    ax1.annotate("both reach 1.0 by construction —\ncompare the shapes, not the ends",
                 xy=(0.066, 1.44), fontsize=7.5, color=MUTED, style="italic",
                 ha="left", va="center")
    ax1.annotate(f"error bars: seed noise, $\\sigma$={SEED_SIGMA:.4f}, n=1 per cell",
                 xy=(0.066, 1.20), fontsize=7.5, color=MUTED, style="italic",
                 ha="left", va="center")

    # ------------------- (b) iso-budget dumbbell ------------------------
    iso = d["iso_budget"]
    ys = list(range(len(iso)))[::-1]
    for y, row in zip(ys, iso):
        lo, hi = row["d_r2_anchor_heavy"], row["d_r2_subject_heavy"]
        ax2.plot([lo, hi], [y, y], color=RULE, lw=2, zorder=1)
        ax2.plot([lo], [y], marker="s", ms=9, color=ORANGE,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax2.plot([hi], [y], marker="o", ms=9.5, color=BLUE,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax2.annotate(f"{row['gain']:.2f}×", xy=(hi, y), xytext=(9, -3.5),
                     textcoords="offset points", fontsize=9.5, color=BLUE,
                     fontweight="bold")
    ax2.set_yticks(ys)
    ax2.set_yticklabels(
        [f"{r['subject_heavy'].replace('_', ' ')}\nvs {r['anchor_heavy'].replace('_', ' ')}"
         for r in iso], fontsize=8)
    ax2.set_xlim(0, 0.075)        # headroom so gain labels clear the legend
    ax2.set_ylim(-0.6, len(iso) - 0.35)
    ax2.set_xlabel(r"$\Delta r^2$ over a random encoder", color=INK_2, fontsize=10)
    ax2.set_title("(b)  Same data budget, spent on subjects vs anchors",
                  color=INK, fontsize=11, loc="left", pad=10)
    _style(ax2)
    ax2.grid(axis="y", visible=False)
    # Legend by proxy handles -- the marks carry identity, not the axis labels.
    h_sub = plt.Line2D([], [], marker="o", ms=8, color=BLUE, ls="none",
                       markeredgecolor=SURFACE, markeredgewidth=1.4)
    h_anc = plt.Line2D([], [], marker="s", ms=7.5, color=ORANGE, ls="none",
                       markeredgecolor=SURFACE, markeredgewidth=1.4)
    leg2 = ax2.legend([h_sub, h_anc], ["spent on subjects", "spent on anchors"],
                      loc="upper right", fontsize=8.5, frameon=False)
    for t in leg2.get_texts():
        t.set_color(INK_2)
    ax2.annotate("matched (subjects × anchors) pairs within each row",
                 xy=(0.5, -0.5), fontsize=7.5, color=MUTED, ha="center")

    fig.suptitle(
        "E0.3 — (anchors × subjects) scaling, 11 step-matched cells "
        "(4400 steps each, identical full val set)",
        color=INK, fontsize=11.5, y=0.985, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"e03_scaling.{ext}", dpi=200, facecolor=SURFACE)
    print(f"Wrote {HERE}/e03_scaling.{{png,pdf}}")
    print(f"  (a) S {s_y[0]:.4f} -> {s_y[-1]:.4f} (corner exp "
          f"{d['local_exponent_S_at_corner']:+.3f});  "
          f"A {a_y[0]:.4f} -> {a_y[-1]:.4f} (corner exp "
          f"{d['local_exponent_A_at_corner']:+.3f})")
    print(f"  (b) subject-heavy wins {sum(r['gain'] > 1 for r in iso)}/{len(iso)}, "
          f"gains {', '.join(f'{r['gain']:.2f}x' for r in iso)}")


if __name__ == "__main__":
    main()
