"""Figure for E0.2 -- the K-subject averaging curve.

Two panels, because the result has two independent claims:

  (a) Spearman-Brown is VALID here. Measured R(K) -- correlation of two disjoint
      K-subject averages, computed without the formula -- against the formula's
      prediction. Anchored at the MEASURED single-subject reliability R(1), not
      at the CorrCA rho1 from E0.1: those are different observables (mean
      per-dimension embedding reliability vs an optimally-weighted projection of
      raw EEG channels), and anchoring across them would compare apples to
      oranges. Anchoring here makes K=1 agree by construction, so the panel is a
      test of SHAPE, which is the only thing at issue.

  (b) Aggregation is a large lever, and its aggregation POINT matters. Probe r
      vs K for embedding-space vs signal-space averaging, against the ceiling
      sqrt(R(K)) implied by the E0.1 CorrCA reliability -- the bound the probe
      actually lives under.

Palette: slots 1-2 of the reference categorical palette, validated with
scripts/validate_palette.js --mode light (CVD dE 24.7, normal-vision 33.6,
contrast >= 3:1, all checks PASS). The ceiling is drawn as a muted dashed
REFERENCE line rather than a third categorical hue -- it is a bound, not a
series.

Light mode only, deliberately: this is a print figure for a LaTeX paper. Hover
and dark-mode variants do not apply to a static PDF. The accessible table view
is RESULTS.md section 2.8, which carries every plotted value.

    uv run --group eeg python experiments/snr_scaling/plot_k_averaging.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# Reference categorical palette, light mode (see references/palette.md).
BLUE = "#2a78d6"      # slot 1 -- the measured / headline curve
ORANGE = "#eb6834"    # slot 2 -- the comparison curve
INK = "#0b0b0b"       # text primary
INK_2 = "#52514e"     # text secondary
MUTED = "#898781"     # axis / labels
RULE = "#c3c2b7"      # baseline / axis / grid
SURFACE = "#fcfcfb"


def spearman_brown(rho1: float, k: float) -> float:
    return k * rho1 / (1.0 + (k - 1) * rho1)


def _style(ax):
    """Recessive grid and axes; the data carries the ink."""
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
    d = json.loads((HERE / "k_averaging_val.json").read_text())
    ks = d["ks"]
    emp = d["empirical_reliability"]
    emb = d["probe_r_embedding_space"]
    sig = d["probe_r_signal_space"]
    ceil = d["predicted_ceiling"]

    # Panel (a): anchor Spearman-Brown at the MEASURED R(1) -- see docstring.
    r1 = emp["1"]["mean"]
    k_meas = [k for k in ks if str(k) in emp]
    y_meas = [emp[str(k)]["mean"] for k in k_meas]
    y_pred = [spearman_brown(r1, k) for k in k_meas]

    k_emb = [k for k in ks if str(k) in emb]
    y_emb = [emb[str(k)]["mean_r"] for k in k_emb]
    k_sig = [k for k in ks if str(k) in sig]
    y_sig = [sig[str(k)]["mean_r"] for k in k_sig]
    k_ceil = [k for k in ks if str(k) in ceil]
    y_ceil = [float(ceil[str(k)]) for k in k_ceil]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3), facecolor=SURFACE)

    # ---------------- (a) Spearman-Brown validation --------------------
    ax1.plot(k_meas, y_pred, color=ORANGE, lw=2, ls=(0, (5, 3)), zorder=2,
             label="Spearman–Brown, anchored at measured $R(1)$")
    ax1.plot(k_meas, y_meas, color=BLUE, lw=2, marker="o", ms=6.5,
             markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.4,
             zorder=3, label="measured (two disjoint $K$-subject averages)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(k_meas)
    ax1.set_xticklabels([str(k) for k in k_meas])
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("K = subjects averaged", color=INK_2, fontsize=10)
    ax1.set_ylabel("reliability $R(K)$", color=INK_2, fontsize=10)
    ax1.set_title("(a)  Spearman–Brown holds, erring conservatively",
                  color=INK, fontsize=11, loc="left", pad=10)
    _style(ax1)
    leg1 = ax1.legend(loc="upper left", fontsize=8.5, frameon=False)
    for t in leg1.get_texts():
        t.set_color(INK_2)
    # Selective direct label: the largest deviation, not a number per point.
    kd = max(k_meas, key=lambda k: emp[str(k)]["mean"] - spearman_brown(r1, k))
    ax1.annotate(f"measured runs above\nprediction (max +{(emp[str(kd)]['mean'] - spearman_brown(r1, kd)):.3f})",
                 xy=(kd, emp[str(kd)]["mean"]), xytext=(kd * 1.25, emp[str(kd)]["mean"] - 0.26),
                 fontsize=8, color=MUTED,
                 arrowprops=dict(arrowstyle="-", color=RULE, lw=1))
    # A reader must not mistake the K=1 agreement for evidence: the prediction
    # is anchored there, so the panel tests SHAPE only.
    ax1.annotate("anchored here\n(agrees by construction)",
                 xy=(k_meas[0], r1), xytext=(k_meas[0] * 1.12, r1 + 0.30),
                 fontsize=7.5, color=MUTED, style="italic",
                 arrowprops=dict(arrowstyle="-", color=RULE, lw=1))

    # ---------------- (b) probe r vs K ---------------------------------
    ax2.plot(k_ceil, y_ceil, color=MUTED, lw=1.6, ls=(0, (2, 2)), zorder=1,
             label="ceiling $\\sqrt{R(K)}$ (E0.1)")
    ax2.plot(k_emb, y_emb, color=BLUE, lw=2, marker="o", ms=6.5,
             markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.4,
             zorder=3, label="average embeddings (after encoder)")
    ax2.plot(k_sig, y_sig, color=ORANGE, lw=2, marker="s", ms=6,
             markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.4,
             zorder=2, label="average signals (before encoder)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(k_emb)
    ax2.set_xticklabels([str(k) for k in k_emb])
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("K = subjects averaged", color=INK_2, fontsize=10)
    ax2.set_ylabel("probe Pearson $r$  (mean over 12 features)",
                   color=INK_2, fontsize=10)
    ax2.set_title("(b)  Aggregation works — after the encoder",
                  color=INK, fontsize=11, loc="left", pad=10)
    _style(ax2)
    leg2 = ax2.legend(loc="upper left", fontsize=8.5, frameon=False)
    for t in leg2.get_texts():
        t.set_color(INK_2)
    # Direct-label the two endpoints only: the headline gain and the gap.
    ax2.annotate(f"{y_emb[-1]:.3f}", xy=(k_emb[-1], y_emb[-1]),
                 xytext=(-4, 9), textcoords="offset points",
                 fontsize=9, color=BLUE, fontweight="bold", ha="right")
    ax2.annotate(f"{y_sig[-1]:.3f}", xy=(k_sig[-1], y_sig[-1]),
                 xytext=(-4, -16), textcoords="offset points",
                 fontsize=9, color=ORANGE, fontweight="bold", ha="right")
    ax2.annotate(f"{y_emb[0]:.3f}", xy=(k_emb[0], y_emb[0]),
                 xytext=(9, -13), textcoords="offset points",
                 fontsize=9, color=BLUE)
    # Name the headline gain once, rather than a number on every point.
    ax2.annotate(f"{y_emb[-1] / y_emb[0]:.1f}× from subject averaging",
                 xy=(0.5, 0.045), xycoords="axes fraction",
                 fontsize=8.5, color=MUTED, ha="center")

    fig.suptitle(
        f"Cross-subject averaging, R5 val ({d['n_unique_subjects']} subjects, "
        f"{d['n_anchors']} movie anchors, {d['n_draws']} draws per K)",
        color=INK, fontsize=11.5, y=0.99, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"k_averaging.{ext}", dpi=200, facecolor=SURFACE)
    print(f"Wrote {HERE}/k_averaging.{{png,pdf}}")
    print(f"  (a) measured R(1)={r1:.4f}; measured/predicted ratio at K={k_meas[-1]}: "
          f"{y_meas[-1] / y_pred[-1]:.3f}")
    print(f"  (b) embedding {y_emb[0]:.3f} -> {y_emb[-1]:.3f} "
          f"({y_emb[-1] / y_emb[0]:.2f}x);  signal {y_sig[0]:.3f} -> {y_sig[-1]:.3f}")


if __name__ == "__main__":
    main()
