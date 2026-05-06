"""Generate the main baseline-comparison figure for the Cine-JEPA paper.

Three panels:
  A. Bar chart of mean stim regression Pearson r (across 4 stim features), per method,
     sorted descending. Colored by method category.
  B. Bar chart of subject anti-target metrics (age_corr + sex_auc - 0.5 averaged),
     same method order. Lower is better.
  C. Trade-off scatter: x = mean stim r (↑), y = subject-leak score (↓).
     Pareto frontier highlighted; ideal corner = upper-left.

Numbers are hard-coded from `docs/subject_movie_metrics_2026-05-06.md` and the
canonical baselines doc. All at nw=2 ws=4, R6 test split, 5 encoder seeds.

Output:
  docs/figures/baseline_comparison.pdf
  docs/figures/baseline_comparison.png  (preview)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Each row: (method_label, category, [lum, cont, pos, narr] mean ± sd, age_corr, sex_auc)
# Categories: "ours", "trivial", "linear", "supervised", "fm_frozen", "fm_ft", "ablation", "ref"
ROWS = [
    # (label, category, lum, cont, pos, narr, age, sex_auc)
    # Ours
    ("Cine-JEPA + Ridge",        "ours",       0.226, 0.223, 0.226, 0.155, 0.387, 0.726),
    # Trivial
    ("Raw 129-ch stats",         "trivial",   -0.016,-0.008, 0.006,-0.006, 0.107, 0.688),
    ("CorrCA stats (35-d)",      "trivial",    0.167, 0.177, 0.158, 0.040, 0.488, 0.724),
    # Linear / supervised
    ("mTRF (CorrCA-5)",          "linear",     0.117, 0.052, 0.071, 0.038, np.nan, np.nan),
    ("mTRF (raw 129-ch)",        "linear",     0.052, 0.044, 0.027, 0.032, np.nan, np.nan),
    ("Linear ceiling raw_corrca_64", "linear", 0.292, 0.217, 0.262, 0.232, 0.130, 0.564),
    # Supervised end-to-end (Tier 2 native)
    ("Sup. ShallowFBCSPNet",     "supervised", 0.111, 0.057, 0.042, 0.027, np.nan, np.nan),
    ("Sup. Deep4Net",            "supervised", 0.191, 0.142, 0.160, 0.024, np.nan, np.nan),
    ("Sup. EEGNetv4",            "supervised", 0.311, 0.228, 0.273, 0.177, np.nan, np.nan),
    ("Sup. EEGNeX",              "supervised", 0.334, 0.277, 0.310, 0.178, np.nan, np.nan),
    # FM frozen + Ridge
    ("BENDR (frozen)",           "fm_frozen",  0.021, 0.039, 0.023, 0.013,-0.008, 0.480),
    ("BIOT (frozen)",            "fm_frozen",  0.127, 0.107, 0.107, 0.003, 0.474, 0.642),
    ("LaBraM (frozen)",          "fm_frozen",  0.136, 0.132, 0.092, 0.018, 0.525, 0.758),
    ("Luna (frozen)",            "fm_frozen",  0.182, 0.158, 0.171, 0.053, 0.458, 0.687),
    ("CBraMod (frozen)",         "fm_frozen",  0.086, 0.051, 0.072, 0.068, 0.547, 0.734),
    ("EEGPT (frozen)",           "fm_frozen",  0.109, 0.094, 0.100, 0.042, 0.273, 0.606),
    ("REVE (frozen)",            "fm_frozen",  0.199, 0.170, 0.195, 0.119, 0.614, 0.867),
    # FM full FT
    ("BENDR (full FT)",          "fm_ft",      0.087, 0.100, 0.128,-0.001, 0.038, 0.520),
    ("BIOT (full FT)",           "fm_ft",      0.085, 0.074, 0.109, 0.013, 0.558, 0.676),
    ("CBraMod (full FT)",        "fm_ft",      0.223, 0.116, 0.127, 0.094, 0.269, 0.663),
    ("LaBraM (full FT)",         "fm_ft",      0.126, 0.089, 0.070, 0.069, 0.300, 0.690),
    ("Luna (full FT)",           "fm_ft",      0.192, 0.169, 0.184, 0.095, 0.392, 0.647),
    ("EEGPT (full FT)",          "fm_ft",      0.144, 0.139, 0.154, 0.027, 0.420, 0.631),
    ("REVE (full FT)",           "fm_ft",      0.283, 0.284, 0.311, 0.266, 0.655, 0.896),
]

CAT_COLORS = {
    "ours":       "#1f77b4",  # tab:blue
    "trivial":    "#7f7f7f",  # gray
    "linear":     "#ff7f0e",  # orange
    "supervised": "#d62728",  # red
    "fm_frozen":  "#2ca02c",  # green
    "fm_ft":      "#9467bd",  # purple
}
CAT_LABELS = {
    "ours":       "Cine-JEPA (ours)",
    "trivial":    "Trivial baselines",
    "linear":     "Linear / mTRF",
    "supervised": "Supervised end-to-end",
    "fm_frozen":  "FM frozen + linear probe",
    "fm_ft":      "FM full finetune",
}

# Compute per-row composite stim score and subject-leak score
def stim_mean(row):
    return float(np.mean(row[2:6]))

def subj_leak(age_corr, sex_auc):
    """Combine into a single ↓ score: mean(|age_corr|, sex_auc - 0.5).
    For comparison we use mean of the two normalized leaks."""
    if np.isnan(age_corr) or np.isnan(sex_auc):
        return np.nan
    return 0.5 * (abs(age_corr) + (sex_auc - 0.5))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

fig = plt.figure(figsize=(13, 7), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], width_ratios=[1.6, 1])

# ---------- Panel A: stim regression Pearson r ----------
axA = fig.add_subplot(gs[0, :])

# Sort methods by mean stim r (descending) so the bar chart reads left-to-right
rows_sorted = sorted(ROWS, key=stim_mean, reverse=True)
labels = [r[0] for r in rows_sorted]
cats   = [r[1] for r in rows_sorted]
lums   = [r[2] for r in rows_sorted]
conts  = [r[3] for r in rows_sorted]
poss   = [r[4] for r in rows_sorted]
narrs  = [r[5] for r in rows_sorted]

x = np.arange(len(rows_sorted))
w = 0.2
axA.bar(x - 1.5*w, lums,  width=w, label="luminance", color="#fdd49e", edgecolor="k", linewidth=0.4)
axA.bar(x - 0.5*w, conts, width=w, label="contrast",  color="#fdbb84", edgecolor="k", linewidth=0.4)
axA.bar(x + 0.5*w, poss,  width=w, label="position",  color="#fc8d59", edgecolor="k", linewidth=0.4)
axA.bar(x + 1.5*w, narrs, width=w, label="narrative", color="#d7301f", edgecolor="k", linewidth=0.4)
axA.set_xticks(x)
axA.set_xticklabels(labels, rotation=42, ha="right", fontsize=8)
axA.axhline(0, color="k", linewidth=0.4)
axA.set_ylabel("Stim regression  Pearson $r$  (test, ↑)", fontsize=9)
axA.set_title("A. Per-method stim-feature regression (sorted by mean $r$ across 4 features)",
              fontsize=10, loc="left", pad=4)
axA.legend(loc="upper right", fontsize=8, frameon=False, ncol=4)

# Color the x tick labels by category
for tick, cat in zip(axA.get_xticklabels(), cats):
    tick.set_color(CAT_COLORS.get(cat, "k"))

# ---------- Panel B: subject anti-target ----------
axB = fig.add_subplot(gs[1, 0])

# Same x order as A so the visual link is preserved
ages   = [r[6] for r in rows_sorted]
sexes  = [r[7] for r in rows_sorted]
xs_subj = np.arange(len(rows_sorted))

# Two paired bars
axB.bar(xs_subj - w/2, [a if not np.isnan(a) else 0 for a in ages],
        width=w, label="age $r$",  color="#a6bddb", edgecolor="k", linewidth=0.4)
axB.bar(xs_subj + w/2, [s - 0.5 if not np.isnan(s) else 0 for s in sexes],
        width=w, label="sex AUC − 0.5", color="#74a9cf", edgecolor="k", linewidth=0.4)

# Mark n/a methods
for i, (a, s) in enumerate(zip(ages, sexes)):
    if np.isnan(a) or np.isnan(s):
        axB.text(i, 0.02, "n/a", ha="center", fontsize=7, color="#7f7f7f", rotation=90)

axB.axhline(0, color="k", linewidth=0.4)
axB.set_xticks(xs_subj)
axB.set_xticklabels(labels, rotation=42, ha="right", fontsize=7)
axB.set_ylabel("Subject leak  (↓ better)", fontsize=9)
axB.set_title("B. Subject anti-target  (age $r$ vs $0$;  sex AUC vs $0.5$)",
              fontsize=10, loc="left", pad=4)
axB.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

# Color tick labels by category
for tick, cat in zip(axB.get_xticklabels(), cats):
    tick.set_color(CAT_COLORS.get(cat, "k"))

# ---------- Panel C: trade-off scatter (the headline) ----------
axC = fig.add_subplot(gs[1, 1])

# x = mean stim r ↑, y = subject leak ↓
for r in ROWS:
    cat = r[1]
    sx = stim_mean(r)
    sy = subj_leak(r[6], r[7])
    if np.isnan(sy):
        # Plot at y=0 with hollow marker for methods without subject heads
        axC.scatter(sx, -0.02, s=44, marker="o", facecolors="none",
                    edgecolors=CAT_COLORS.get(cat, "k"), linewidths=1.0)
        axC.annotate(r[0], (sx, -0.02), xytext=(3, -8), textcoords="offset points",
                     fontsize=6.5, color=CAT_COLORS.get(cat, "k"))
        continue
    axC.scatter(sx, sy, s=46, color=CAT_COLORS.get(cat, "k"), edgecolors="k",
                linewidths=0.4, zorder=3)
    # Bigger marker for ours
    if cat == "ours":
        axC.scatter(sx, sy, s=180, facecolors="none", edgecolors="#1f77b4",
                    linewidths=1.5, zorder=4)
    axC.annotate(r[0], (sx, sy), xytext=(4, 3), textcoords="offset points",
                 fontsize=7, color=CAT_COLORS.get(cat, "k"))

axC.set_xlabel("Mean stim regression $r$  (↑ better)", fontsize=9)
axC.set_ylabel("Subject leak  $\\frac{1}{2}(|r_{age}| + AUC_{sex}{-}0.5)$  (↓ better)", fontsize=9)
axC.set_title("C. Trade-off:  stim signal  vs  subject-identity leak",
              fontsize=10, loc="left", pad=4)
axC.axhline(0, color="#bbb", linewidth=0.5, zorder=1)
axC.axvline(0, color="#bbb", linewidth=0.5, zorder=1)
# Highlight the ideal (bottom-right) corner — high stim, low leak
axC.text(0.97, 0.04, "ideal corner\n(high stim, low leak)", transform=axC.transAxes,
         ha="right", va="bottom", fontsize=7, color="#444",
         bbox=dict(facecolor="#f4f8e4", edgecolor="#7a9c00", boxstyle="round,pad=0.3", alpha=0.9))
# Shade the ideal quadrant lightly
xlim, ylim = axC.get_xlim(), axC.get_ylim()
axC.axvspan(0.20, xlim[1], ymin=0, ymax=0.35, color="#cdde87", alpha=0.18, zorder=0)

# Category legend (shared)
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CAT_COLORS[c], markeredgecolor="k", markersize=8,
               label=CAT_LABELS[c])
    for c in ["ours", "trivial", "linear", "supervised", "fm_frozen", "fm_ft"]
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

# Save
out = Path(__file__).resolve().parents[1] / "docs/figures"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "baseline_comparison.pdf", bbox_inches="tight")
fig.savefig(out / "baseline_comparison.png", dpi=180, bbox_inches="tight")
print(f"saved: {out / 'baseline_comparison.pdf'}")
print(f"saved: {out / 'baseline_comparison.png'}")
