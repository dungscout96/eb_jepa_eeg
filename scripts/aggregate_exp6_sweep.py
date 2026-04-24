"""Aggregate Exp 6 sweep probe_eval outputs into a mean±std table.

Reads probe_eval metrics from SLURM out-files matching logs/exp6sw_*.out,
groups by (std_coeff, pred_dim), averages across seeds, and reports
top configs by stimulus probe corr (mean of pos + lum + cont).

Usage on Delta:
  python scripts/aggregate_exp6_sweep.py \
      --logs_dir logs \
      --prefix exp6sw_ \
      --out sweep_results.md
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


METRICS = [
    "reg_position_in_movie_corr",
    "reg_luminance_mean_corr",
    "reg_contrast_rms_corr",
    "reg_narrative_event_score_corr",
    "cls_position_in_movie_auc",
    "cls_luminance_mean_auc",
    "cls_contrast_rms_auc",
    "cls_narrative_event_score_auc",
    "subject/age_reg/corr",
    "subject/sex/auc",
]


def _parse_config(cfg_line: str):
    m = re.search(r"std=([\d\.]+)\s+pd=(\d+)\s+seed=(\d+)", cfg_line)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2)), int(m.group(3))


def _extract_metric(text: str, key: str) -> "float | None":
    # Match the key as a full path prefix (subject ones have '/')
    for line in text.splitlines():
        if f"probe_eval/test/{key}:" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                return None
    return None


def aggregate(logs_dir: Path, prefix: str):
    cells = defaultdict(list)
    for out_file in sorted(logs_dir.glob(f"{prefix}*.out")):
        text = out_file.read_text(errors="ignore")
        cfg_match = re.search(r"cfg: std=([\d\.]+)\s+pd=(\d+)\s+seed=(\d+)", text)
        if not cfg_match:
            continue
        std, pd, seed = float(cfg_match.group(1)), int(cfg_match.group(2)), int(cfg_match.group(3))
        metrics = {k: _extract_metric(text, k) for k in METRICS}
        if metrics.get("reg_position_in_movie_corr") is None:
            continue  # incomplete / failed
        cells[(std, pd)].append({"seed": seed, "file": out_file.name, **metrics})
    return cells


def summarize(cells):
    rows = []
    for (std, pd), runs in cells.items():
        summary = {"std": std, "pd": pd, "n_seeds": len(runs)}
        for k in METRICS:
            vals = [r[k] for r in runs if r[k] is not None]
            if not vals:
                summary[k] = (None, None)
            else:
                import statistics
                mean = statistics.mean(vals)
                stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
                summary[k] = (mean, stdev)
        # Composite: mean of 3 top stimulus corrs
        cor_keys = ["reg_position_in_movie_corr", "reg_luminance_mean_corr", "reg_contrast_rms_corr"]
        cor_means = [summary[k][0] for k in cor_keys if summary[k][0] is not None]
        summary["stim_corr_mean"] = sum(cor_means) / len(cor_means) if cor_means else None
        rows.append(summary)
    rows.sort(key=lambda r: r.get("stim_corr_mean") or -1, reverse=True)
    return rows


def format_markdown(rows, baseline_5seed=None):
    md = ["# Exp 6 sweep — aggregated results\n"]
    md.append("Rows sorted by composite stimulus probe corr (mean of pos+lum+cont). Values = mean ± std across seeds.\n\n")
    if baseline_5seed:
        md.append("## Exp 6 baseline (5-seed, for reference)\n")
        for k, v in baseline_5seed.items():
            md.append(f"- {k}: {v}\n")
        md.append("\n")
    md.append("## Sweep cells\n")
    md.append("|std|pd|n|pos corr|lum corr|cont corr|narr corr|pos AUC|lum AUC|cont AUC|narr AUC|age corr|sex AUC|composite|\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    def fmt(pair):
        m, s = pair
        if m is None:
            return "—"
        return f"{m:+.3f}±{s:.3f}"
    for r in rows:
        md.append(
            f"|{r['std']}|{r['pd']}|{r['n_seeds']}|"
            f"{fmt(r['reg_position_in_movie_corr'])}|"
            f"{fmt(r['reg_luminance_mean_corr'])}|"
            f"{fmt(r['reg_contrast_rms_corr'])}|"
            f"{fmt(r['reg_narrative_event_score_corr'])}|"
            f"{fmt(r['cls_position_in_movie_auc'])}|"
            f"{fmt(r['cls_luminance_mean_auc'])}|"
            f"{fmt(r['cls_contrast_rms_auc'])}|"
            f"{fmt(r['cls_narrative_event_score_auc'])}|"
            f"{fmt(r['subject/age_reg/corr'])}|"
            f"{fmt(r['subject/sex/auc'])}|"
            f"{r['stim_corr_mean']:+.3f}|\n"
        )
    return "".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--prefix", default="exp6sw_")
    ap.add_argument("--out", default="sweep_results.md")
    args = ap.parse_args()

    cells = aggregate(Path(args.logs_dir), args.prefix)
    print(f"Found {len(cells)} distinct (std, pd) cells, {sum(len(v) for v in cells.values())} runs")
    rows = summarize(cells)
    baseline = {
        "position corr": "0.176 ± 0.048",
        "luminance corr": "0.168 ± 0.059",
        "contrast corr": "0.115 ± 0.053",
        "narrative corr": "-0.003 ± 0.042",
        "position AUC": "0.580 ± 0.025",
        "luminance AUC": "0.567 ± 0.021",
        "contrast AUC": "0.553 ± 0.032",
        "sex AUC": "0.618 ± 0.007",
        "age corr": "0.325 ± 0.030",
    }
    md = format_markdown(rows, baseline_5seed=baseline)
    Path(args.out).write_text(md)
    print(f"Wrote {args.out}")
    # also dump json
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        [{k: v for k, v in r.items()} for r in rows], indent=2, default=str
    ))


if __name__ == "__main__":
    main()
