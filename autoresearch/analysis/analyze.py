"""Compute Spearman correlations between SSL/sanity metrics and probe corrs.

Generates metric_correlation_report.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS_DIR = Path("/Users/dtyoung/Documents/Research/eb_jepa_eeg/autoresearch/analysis")
CSV_PATH = ANALYSIS_DIR / "runs_metrics.csv"
REPORT_PATH = ANALYSIS_DIR / "metric_correlation_report.md"

TARGETS = [
    "val/reg_position_in_movie_corr",
    "val/reg_contrast_rms_corr",
    "val/reg_luminance_mean_corr",
    "val/reg_narrative_event_score_corr",
    "val_corr_weighted",
]

# Group predictors so we can be explicit about contamination
SANITY_PREDICTORS = [
    "sanity/embedding_variance_mean",
    "sanity/embedding_variance_min",
    "sanity/embedding_variance_max",
    "sanity/embedding_variance_std",
    "sanity/embedding_l2_mean",
    "sanity/cosim_random_pairs_mean",
    "sanity/cosim_random_pairs_max",
    "sanity/loss_trend",
    "sanity/loss_rolling_mean",
    "sanity/pred_loss_short",
    "sanity/pred_loss_long",
    "sanity/grad_norm",
    "sanity/linear_probe_acc",
]

TRAIN_PREDICTORS = [
    "train_step/jepa_loss",
    "train_step/vc_loss",
    "train_step/pred_loss",
    "train_step/reg_loss",
    "train_step/cls_loss",
]

PREDICTORS = SANITY_PREDICTORS + TRAIN_PREDICTORS


def pairwise_spearman(df: pd.DataFrame, predictor: str, target: str):
    sub = df[[predictor, target]].dropna()
    if len(sub) < 5:
        return float("nan"), float("nan"), len(sub)
    rho, p = spearmanr(sub[predictor], sub[target])
    return float(rho), float(p), len(sub)


def corr_table(df: pd.DataFrame, targets: list[str], predictors: list[str]) -> pd.DataFrame:
    rows = []
    for p in predictors:
        row = {"predictor": p}
        for t in targets:
            rho, pv, n = pairwise_spearman(df, p, t)
            row[f"rho::{t}"] = rho
            row[f"n::{t}"] = n
            row[f"p::{t}"] = pv
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_float(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:+.3f}"


def topk_table(table: pd.DataFrame, target: str, k=10) -> pd.DataFrame:
    col = f"rho::{target}"
    sub = table[["predictor", col, f"n::{target}", f"p::{target}"]].copy()
    sub = sub.dropna(subset=[col])
    sub["abs_rho"] = sub[col].abs()
    sub = sub.sort_values("abs_rho", ascending=False).head(k)
    return sub.drop(columns=["abs_rho"])


def render_top_md(top: pd.DataFrame, target: str) -> str:
    lines = [
        "| Rank | Predictor | Spearman ρ | n | p-value |",
        "|------|-----------|-----------:|--:|--------:|",
    ]
    for i, (_, row) in enumerate(top.iterrows(), 1):
        rho = row[f"rho::{target}"]
        n = int(row[f"n::{target}"])
        pv = row[f"p::{target}"]
        pv_s = f"{pv:.2e}" if pv == pv else "n/a"
        lines.append(f"| {i} | `{row['predictor']}` | {fmt_float(rho)} | {n} | {pv_s} |")
    return "\n".join(lines)


def render_per_target_for_predictor(table: pd.DataFrame, predictor: str) -> str:
    row = table[table["predictor"] == predictor]
    if row.empty:
        return f"`{predictor}` not found in table."
    row = row.iloc[0]
    lines = [
        "| Target | Spearman ρ | n | p-value |",
        "|--------|-----------:|--:|--------:|",
    ]
    for t in TARGETS:
        rho = row[f"rho::{t}"]
        n = row[f"n::{t}"]
        pv = row[f"p::{t}"]
        n_s = f"{int(n)}" if n == n else "n/a"
        pv_s = f"{pv:.2e}" if isinstance(pv, float) and pv == pv else "n/a"
        lines.append(f"| `{t}` | {fmt_float(rho)} | {n_s} | {pv_s} |")
    return "\n".join(lines)


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} runs.")

    # Stratification keys
    norm_col = "cfg.data.norm_mode"
    reg_col = "cfg.loss.regularizer"

    norm_counts = df[norm_col].fillna("(missing)").value_counts()
    reg_counts = df[reg_col].fillna("(missing)").value_counts()

    overall = corr_table(df, TARGETS, PREDICTORS)

    # Stratified tables (only when subgroup ≥10 runs with weighted target)
    strata: dict[str, pd.DataFrame] = {}
    for nm, _ in norm_counts.items():
        if nm == "(missing)":
            continue
        sub = df[df[norm_col] == nm]
        if (sub["val_corr_weighted"].notnull()).sum() >= 10:
            strata[f"norm_mode={nm}"] = corr_table(sub, TARGETS, PREDICTORS)
    for rg, _ in reg_counts.items():
        if rg == "(missing)":
            continue
        sub = df[df[reg_col] == rg]
        if (sub["val_corr_weighted"].notnull()).sum() >= 10:
            strata[f"regularizer={rg}"] = corr_table(sub, TARGETS, PREDICTORS)

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    L: list[str] = []
    L += [
        "# SSL Metric → Probe Correlation Report",
        "",
        f"**Sample size:** {len(df)} EEG-JEPA runs from W&B project `eb_jepa` ",
        "(past ~120 days, finished/running, ≥1 `val/reg_*_corr` summary value, "
        "≥1 `sanity/*` key). Re-fetched with `api.run(id)` to populate config.",
        "",
        "**Caveat:** only **71/150** runs have `val/reg_position_in_movie_corr` and "
        "`val/reg_narrative_event_score_corr` logged in summary — the older runs only "
        "have `luminance` and `contrast`. So `val_corr_weighted` is computed on n=71. "
        "Per-target correlations against `luminance`/`contrast` use the full n=150.",
        "",
        "**Stratification subgroup sizes (with `val_corr_weighted` non-null):**",
    ]
    for nm, c in norm_counts.items():
        sub = df[df[norm_col].fillna("(missing)") == nm]
        n_w = sub["val_corr_weighted"].notnull().sum()
        L.append(f"- `norm_mode={nm}`: {c} runs total, {n_w} with weighted target")
    for rg, c in reg_counts.items():
        sub = df[df[reg_col].fillna("(missing)") == rg]
        n_w = sub["val_corr_weighted"].notnull().sum()
        L.append(f"- `regularizer={rg}`: {c} runs total, {n_w} with weighted target")
    L += ["",
          "All correlations are Spearman ρ (rank-based, robust). Targets are summary "
          "(final) values of `val/reg_*_corr` and the weighted average ",
          "`val_corr_weighted = 0.3·position + 0.3·contrast + 0.3·luminance + 0.1·narrative`.",
          ""]

    # 1. Top 10 overall
    L += ["## 1. Top 10 predictors by |ρ| with `val_corr_weighted` (overall)", ""]
    top10 = topk_table(overall, "val_corr_weighted", k=10)
    L += [render_top_md(top10, "val_corr_weighted"), ""]
    L += [
        "### Headline (read this first)",
        "",
        "Three findings, in decreasing order of robustness:",
        "",
        "1. **The strongest cross-run predictors (`train_step/reg_loss` ρ=−0.61, "
        "`train_step/cls_loss` ρ=−0.54) are circular** — they are training-time supervised "
        "feature-prediction losses sharing a target distribution with the val/reg eval. "
        "Selecting on them is exactly the \"probe optimization confounds encoder quality\" "
        "concern the user flagged. **Do not use.**",
        "",
        "2. **`sanity/loss_trend` (+0.52) and `pred_loss_short` (+0.33) ride heavily on "
        "the position-leak target** (ρ=+0.48 / +0.22 on `position_in_movie`) and have "
        "**wrong-signed** correlations (loss going UP → corrs UP). Treat as survivorship "
        "indicators, not quality signals.",
        "",
        "3. **On the genuinely-non-leaked targets (`luminance_mean`, `contrast_rms`, "
        "n=150), the only sanity metric with a clear positive association is "
        "`sanity/linear_probe_acc` (ρ=+0.36 luminance, +0.27 contrast).** The unsupervised "
        "collapse markers (`embedding_variance_*`, `cosim_random_pairs_*`) are *correctly "
        "signed* on `val_corr_weighted` (ρ=+0.21 to +0.29 / −0.28 to −0.29) but mostly "
        "**weaken to near-zero within a fixed regulariser** — meaning their cross-run signal "
        "is partly between-regulariser scale, not within-config quality.",
        "",
        "**Bottom line:** on this dataset there is **no cheap online SSL metric with strong "
        "(|ρ|>0.4) and robustly-signed correlation to the non-leaked val targets**. The "
        "best available cheap signal is `sanity/linear_probe_acc`, with the variance/cosim "
        "collapse markers as label-free cross-checks. See §6 for the recommended composite.",
        "",
    ]

    # 1b: Sanity-only ranking (the actually-cheap predictors, no circular contamination)
    L += ["## 2. Top sanity-only predictors (no train_step/* — these are the cheap, non-circular signals)", ""]
    sanity_only = overall[overall["predictor"].isin(SANITY_PREDICTORS)].copy()
    sanity_only["abs"] = sanity_only["rho::val_corr_weighted"].abs()
    sanity_top = sanity_only.sort_values("abs", ascending=False).drop(columns=["abs"])
    L += [render_top_md(sanity_top, "val_corr_weighted"), ""]

    # 2. Stratified
    for stratum, tbl in strata.items():
        n_w = int(df[df[stratum.split('=')[0].replace('norm_mode', norm_col).replace('regularizer', reg_col) if False else (norm_col if stratum.startswith('norm_mode') else reg_col)] == stratum.split('=')[1]]["val_corr_weighted"].notnull().sum())
        L += [f"## 3. Stratified — `{stratum}` (weighted-target n={n_w})", ""]
        L += [render_top_md(topk_table(tbl, "val_corr_weighted", k=10), "val_corr_weighted"), ""]

    # 3. Per-target breakdown for top sanity-only metrics
    L += ["## 4. Per-feature breakdown for top 5 sanity-only predictors", "",
          "Do these metrics predict different val targets equally well?", ""]
    top5_sanity = sanity_top.head(5)["predictor"].tolist()
    for p in top5_sanity:
        L += [f"### `{p}`", "", render_per_target_for_predictor(overall, p), ""]

    L += [
        "### Reading these tables — the position-leak issue",
        "",
        "- `position_in_movie` and `narrative_event_score` only exist for n=71 runs; "
        "`luminance_mean` and `contrast_rms` for n=150.",
        "- The position correlation is partially **leaked through Fourier positional encoding** "
        "(per project memory) — so positive ρ on position is a weaker signal than positive ρ "
        "on contrast/luminance, which require genuine stimulus-feature decoding.",
        "- A predictor that does well on luminance/contrast but poorly on position is "
        "actually a *better* SSL-quality signal than one that wins on position.",
        "",
        "**Key observation:** `sanity/loss_trend` and `sanity/pred_loss_short` get most of "
        "their headline ρ from the leaked **position** target (ρ=+0.48 / +0.22 on position; "
        "ρ≤+0.25 on contrast/luminance, ρ≈0 on narrative). When you re-rank against the "
        "non-leaked targets only, **`sanity/linear_probe_acc` is the strongest "
        "predictor on luminance (ρ=+0.36, n=150) and contrast (ρ=+0.27, n=149)** — the "
        "two targets that genuinely require stimulus-feature decoding. The collapse markers "
        "(`embedding_variance_*`, `cosim_random_pairs_*`) are weak-but-correctly-signed on "
        "these targets.",
        "",
    ]

    # 4. Sign / collapse hypothesis
    L += ["## 5. Sign / collapse-hypothesis check", "",
          "Under the **collapse hypothesis** we expect:",
          "- `cosim_random_pairs_*` ↑ ⇒ collapsed ⇒ probe corrs ↓ ⇒ ρ should be **negative**.",
          "- `embedding_variance_*` ↑ ⇒ spread ⇒ probe corrs ↑ ⇒ ρ should be **positive**.",
          "- `loss_rolling_mean`/`pred_loss_*` ↑ ⇒ poor predictive fit ⇒ ρ should be **negative**.",
          "- `linear_probe_acc` ↑ ⇒ semantic separability ⇒ ρ should be **positive**.",
          ""]
    sign_predictors = SANITY_PREDICTORS
    expected_sign = {
        "sanity/cosim_random_pairs_mean": "−",
        "sanity/cosim_random_pairs_max": "−",
        "sanity/embedding_variance_mean": "+",
        "sanity/embedding_variance_min": "+",
        "sanity/embedding_variance_max": "+",
        "sanity/embedding_variance_std": "?",
        "sanity/embedding_l2_mean": "?",
        "sanity/loss_rolling_mean": "−",
        "sanity/loss_trend": "−",
        "sanity/pred_loss_short": "−",
        "sanity/pred_loss_long": "−",
        "sanity/grad_norm": "?",
        "sanity/linear_probe_acc": "+",
    }
    L += ["| Predictor | ρ vs val_corr_weighted | n | Expected sign | Match? |",
          "|-----------|----------------------:|--:|---------------|--------|"]
    for p in sign_predictors:
        srow = overall[overall["predictor"] == p]
        if srow.empty:
            continue
        rho = srow["rho::val_corr_weighted"].iloc[0]
        n = srow["n::val_corr_weighted"].iloc[0]
        exp = expected_sign[p]
        if math.isnan(rho):
            match = "n/a"
        elif exp == "?":
            match = "—"
        elif exp == "+":
            match = "yes" if rho > 0 else "**no**"
        else:
            match = "yes" if rho < 0 else "**no**"
        n_s = f"{int(n)}" if n == n else "n/a"
        L.append(f"| `{p}` | {fmt_float(rho)} | {n_s} | {exp} | {match} |")

    L += ["",
          "**Findings:**",
          "",
          "- `cosim_random_pairs_mean` ρ=−0.29 ✓, `cosim_random_pairs_max` ρ=−0.28 ✓, "
          "`embedding_variance_mean` ρ=+0.26 ✓, `embedding_variance_min` ρ=+0.21 ✓, "
          "`linear_probe_acc` ρ=+0.25 ✓ — all sign-consistent with the collapse hypothesis.",
          "- `loss_rolling_mean` ρ=+0.33 and `pred_loss_short` ρ=+0.33 are **wrong-signed** "
          "(higher pred loss → higher corrs in this dataset). This is because most failed runs "
          "are not collapsing to high-loss; they collapse to *low* loss with degenerate "
          "embeddings. The runs with high pred loss are still actively trying to fit. So "
          "**absolute pred-loss magnitude is misleading as a quality signal across runs** — "
          "it conflates 'didn't train' (high loss → bad rep) with 'still working hard' "
          "(high loss → good rep). Avoid as a standalone signal.",
          "- `loss_trend` ρ=+0.52 — same caveat, plus an extra wrinkle: the runs that survive "
          "to log val/reg_position (the n=71 subset) are biased toward longer/healthier "
          "training, so `loss_trend` may be picking up survivorship rather than quality.",
          ""]

    # 5. Recommendations
    L += ["## 6. Recommendations", "",
          "**Goal:** a per-run online signal that is (a) cheaply computed by the existing "
          "`SanityCheckHook` (no full validation pass), (b) directionally interpretable, "
          "(c) not circular with the eval probe.",
          "",
          "### Recommended primary signal: `sanity/linear_probe_acc`", "",
          "This is the single best-aligned cheap signal on the **non-leaked** targets:",
          "- ρ = +0.36 on `luminance_mean` (n=150, p=5e-6)",
          "- ρ = +0.27 on `contrast_rms` (n=149, p=8e-4)",
          "- ρ = +0.25 on `val_corr_weighted` (n=71)",
          "",
          "It IS technically a probe, BUT: (i) it's a single linear layer trained for 30 "
          "SGD steps on a 512-sample rolling buffer, (ii) it uses *subject metadata* "
          "(age>median or sex) or *luminance binarisation* as labels — these are the "
          "**only** sanity metric whose label distribution overlaps with the eval features, "
          "and even then the overlap is partial (binarised luminance, not regression). "
          "Cost is negligible compared to the encoder forward pass. The user's concern about "
          "\"probe head contamination\" applies most strongly to high-capacity probes; a "
          "single linear layer on frozen embeddings is closer to a representation-quality "
          "diagnostic than to a confounding optimisation co-target.",
          "",
          "### Recommended secondary signal: anti-collapse score", "",
          "Combine the two sign-correct, scale-aware collapse markers:",
          "",
          "```",
          "anti_collapse = z(sanity/embedding_variance_mean) - z(sanity/cosim_random_pairs_mean)",
          "```",
          "",
          "where `z(·)` standardises across the architecture-search batch. Both terms have "
          "ρ ≈ 0.26–0.29 sign-correct on val_corr_weighted, are bounded in interpretation, "
          "and capture representation spread vs collapse from two angles. They DON'T require "
          "any labels at all — pure unsupervised. Their per-target ρ on contrast/luminance "
          "is small (≤0.18), so they are weaker than `linear_probe_acc` on the genuinely "
          "non-leaked targets, but they fill the role of a label-free cross-check.",
          "",
          "### Proposed composite", "",
          "```",
          "rep_score = z(sanity/linear_probe_acc) + 0.5 * anti_collapse",
          "         = z(sanity/linear_probe_acc)",
          "         + 0.5 * z(sanity/embedding_variance_mean)",
          "         - 0.5 * z(sanity/cosim_random_pairs_mean)",
          "```",
          "",
          "where standardisation is done *across the candidate architectures in the current "
          "search batch* (not across all-time runs). Weights give the probe a primary role "
          "(empirically the strongest non-circular predictor) with the two unsupervised "
          "collapse signals as a label-free cross-check. All three are produced by "
          "`SanityCheckHook` with no extra changes.",
          "",
          "### Fallback / sanity floor", "",
          "Independently of `rep_score`, **gate** on:",
          "",
          "- `sanity/embedding_variance_min > epsilon` (e.g. > 0.01) — a hard collapse "
          "detector. If the smallest variance dim collapses to ~0, the run is degenerate "
          "regardless of other metrics.",
          "- `sanity/cosim_random_pairs_max < 0.99` — the max-cos hits 1.0 only in pathological "
          "collapse; useful as a binary kill switch.",
          "",
          "These gates are cheap and unambiguous — runs that fail them should be auto-rejected "
          "without bothering with the full `rep_score`.",
          "",
          "### Stratification caveats", "",
          "- The `norm_mode` strata are tiny (n=10 and n=11). Don't read individual ρ values "
          "— only the broad pattern.",
          "- Across `regularizer={vc, sigreg}` strata (n=35 / n=36), the **embedding-variance "
          "and cosim-collapse signs are unstable**: in `regularizer=vc` runs, "
          "`embedding_variance_min` ρ flips to −0.21 and `embedding_l2_mean` to −0.43, "
          "while `regularizer=sigreg` shows `cosim_random_pairs_mean` ρ=+0.17 (wrong sign).",
          "- This means the collapse markers are **partly capturing between-regulariser "
          "scale differences** rather than within-config representation quality. The "
          "autoresearch loop fixes the regulariser, so within-run-batch standardisation "
          "(`z()`) should largely cancel this.",
          "- `sanity/linear_probe_acc` weakens dramatically within-stratum (vc: ρ=−0.05, "
          "sigreg: ρ=+0.09 on weighted target — both effectively zero). So **most of its "
          "overall ρ=+0.25 is between-regulariser variance** too: sigreg runs tend to have "
          "both higher probe acc and higher val corrs than vc runs. This is a real concern: "
          "within a fixed regulariser the probe-acc signal collapses. Mitigation: use it "
          "across the autoresearch search batch (which mixes architectures within a fixed "
          "regulariser), and rely on the per-target luminance/contrast ρ (n=150, computed "
          "across regularisers, ρ=+0.36/+0.27) as the main empirical evidence.",
          ""]

    # 6. What to NOT use
    L += ["## 7. Metrics to AVOID as the primary signal", "",
          "- **`train_step/reg_loss` (ρ=−0.61) and `train_step/cls_loss` (ρ=−0.54)** look the "
          "strongest, but they are **not unsupervised**: they are the losses of online "
          "supervised feature-prediction heads that share a target distribution with the "
          "val/reg_* eval. Selecting on them is essentially selecting on \"how well did the "
          "training-time probe head fit the same features the val probe will fit\" — which "
          "the user has explicitly flagged as a confound. Do not use as the autoresearch "
          "decision metric.",
          "- **`train_step/jepa_loss`, `pred_loss`, `loss_rolling_mean`, `pred_loss_short/long`** "
          "have **wrong-signed** correlations across the run population (ρ ≈ +0.25 to +0.43): "
          "absolute pred-loss magnitude conflates 'didn't train at all' with 'still optimising'. "
          "Across runs with different std_coeff/cov_coeff/regularizer settings, the loss is "
          "not on a comparable scale. Use *within* a fixed loss configuration only, never "
          "for cross-architecture ranking.",
          "- **`sanity/loss_trend` (ρ=+0.52)** is intriguing but: (i) sign is opposite of "
          "what 'loss going down = healthy' would predict, (ii) only computed on runs that "
          "produced 20+ rolling-window samples, biasing toward longer survivors. The sign "
          "anomaly suggests it's measuring training stage / survivorship rather than "
          "representation quality. Treat as a diagnostic, not a decision signal.",
          "- **`train_step/vc_loss` (variance/covariance regulariser)** correlates with "
          "anything `std_coeff`/`cov_coeff` is engaged on — proxy for 'regulariser is on' "
          "rather than 'representation is good'. Avoid for cross-config comparison.",
          ""]

    # 7. Appendix
    L += ["## 8. Appendix: full Spearman table (overall, all predictors × all targets)", ""]
    header = "| Predictor | " + " | ".join(t.replace("val/reg_", "").replace("_corr", "") for t in TARGETS) + " | n_min |"
    sep = "|-----------|" + "|".join(["---:"] * (len(TARGETS) + 1)) + "|"
    L += [header, sep]
    for _, row in overall.iterrows():
        cells = [f"`{row['predictor']}`"]
        ns = []
        for t in TARGETS:
            cells.append(fmt_float(row[f"rho::{t}"]))
            n = row[f"n::{t}"]
            if not (isinstance(n, float) and math.isnan(n)):
                ns.append(int(n))
        cells.append(str(min(ns)) if ns else "n/a")
        L.append("| " + " | ".join(cells) + " |")
    L += [""]

    REPORT_PATH.write_text("\n".join(L))
    print(f"Wrote report → {REPORT_PATH}")

    print("\nTOP-5 sanity-only (|ρ| with val_corr_weighted):")
    print(sanity_top.head(5)[["predictor", "rho::val_corr_weighted", "n::val_corr_weighted"]].to_string(index=False))


if __name__ == "__main__":
    main()
