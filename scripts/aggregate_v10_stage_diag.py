"""Aggregate v10 stage-diagnostic SLURM out-files into a markdown table.

Parses logs/v10stage_*.out into a table grouped by (config, stage), and
runs a 5-encoder-seed t-test against chance for each metric. Mirrors the
View-2 (per-encoder-seed-mean) view from docs/significance_analysis_*.md.

Run on Delta from the v10 worktree:
  python scripts/aggregate_v10_stage_diag.py \
      --logs_dir logs --out docs/v10_stage_diagnostic.md
"""

import argparse
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

# probe_eval prints, e.g.:
#   probe_eval/test/reg_narrative_event_score_corr: 0.011
#   probe_eval/test/cls_narrative_event_score_auc: 0.522
#   probe_eval/test/subject/age_reg/corr: 0.345
METRIC_KEYS = [
    "reg_position_in_movie_corr",
    "reg_luminance_mean_corr",
    "reg_contrast_rms_corr",
    "reg_narrative_event_score_corr",
    "cls_position_in_movie_auc",
    "cls_luminance_mean_auc",
    "cls_contrast_rms_auc",
    "cls_narrative_event_score_auc",
    "subject/age_reg/corr",
    "subject/age_binary/auc",
    "subject/sex/auc",
]

# Chance levels for one-sample t-test against null (no signal).
# Correlations: chance = 0. AUCs: chance = 0.5.
def _chance_for(key: str) -> float:
    return 0.5 if key.endswith("auc") else 0.0


# v10 sbatch prints:
#   Cfg: ckpt=...  nw=4 ws=2 bs=64 probe_seed=7 stage=context_enc tag=nw4ws2_baseline_enc42_p7_context_enc
CFG_RE = re.compile(
    r"Cfg:\s*ckpt=\S+\s+nw=(\d+)\s+ws=(\d+)\s+bs=(\d+)\s+probe_seed=(\d+)\s+"
    r"stage=(\w+)\s+tag=(\S+)"
)
# tag pattern: <prefix>_enc<seed>_p<seed>_<stage>
TAG_RE = re.compile(r"^(?P<prefix>.+?)_enc(?P<enc>\d+)_p(?P<probe>\d+)_(?P<stage>\w+)$")


def _extract_metric(text: str, split: str, key: str):
    needle = f"probe_eval/{split}/{key}:"
    for line in text.splitlines():
        if needle in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                return None
    return None


def parse_logs(logs_dir: Path, split: str = "test"):
    rows = []
    for f in sorted(logs_dir.glob("v10stage_*.out")):
        text = f.read_text(errors="ignore")
        m = CFG_RE.search(text)
        if not m:
            continue
        nw, ws, bs, probe_seed, stage, tag = m.groups()
        tag_m = TAG_RE.match(tag)
        if not tag_m:
            continue
        prefix = tag_m.group("prefix")
        enc_seed = int(tag_m.group("enc"))
        metrics = {k: _extract_metric(text, split, k) for k in METRIC_KEYS}
        if metrics["reg_position_in_movie_corr"] is None:
            continue
        rows.append({
            "file": f.name,
            "config": prefix,
            "stage": stage,
            "enc_seed": enc_seed,
            "probe_seed": int(probe_seed),
            **metrics,
        })
    return rows


def _t_test(values, mu0):
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    if se == 0:
        return mean, sd, float("nan")
    t = (mean - mu0) / se
    # two-sided t-test p-value approximation via normal — for n=5 this
    # over-estimates significance, but we keep it for sortable output.
    # Caller can re-run with scipy.stats if needed.
    try:
        from scipy.stats import t as scipy_t
        p = 2 * (1 - scipy_t.cdf(abs(t), df=n - 1))
    except ImportError:
        # crude fallback: normal approximation
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return mean, sd, p


def aggregate(rows):
    """Group rows by (config, stage). Compute encoder-seed means (avg over
    probe seeds), then mean ± σ across encoder seeds + t-test vs chance."""
    by_group = defaultdict(list)
    for r in rows:
        key = (r["config"], r["stage"])
        by_group[key].append(r)

    summary = {}
    for (config, stage), group in by_group.items():
        # First average over probe seeds within each encoder seed
        by_enc = defaultdict(list)
        for r in group:
            by_enc[r["enc_seed"]].append(r)

        per_enc_means = {}  # enc_seed → {metric → mean over probe seeds}
        for enc_seed, recs in by_enc.items():
            metric_means = {}
            for key in METRIC_KEYS:
                vals = [r[key] for r in recs if r[key] is not None]
                if vals:
                    metric_means[key] = statistics.mean(vals)
            per_enc_means[enc_seed] = metric_means

        # Then t-test across encoder-seed means
        enc_summary = {}
        for key in METRIC_KEYS:
            vals = [m[key] for m in per_enc_means.values() if key in m]
            mean, sd, p = _t_test(vals, _chance_for(key))
            enc_summary[key] = {"mean": mean, "sd": sd, "p": p, "n_enc": len(vals),
                                 "n_runs": len(group)}
        summary[(config, stage)] = enc_summary
    return summary


def render(summary, split, focus_keys=None):
    if focus_keys is None:
        focus_keys = METRIC_KEYS
    lines = []
    lines.append(f"# v10 stage diagnostic — split: {split}")
    lines.append("")
    lines.append("Per-encoder-seed-mean t-test against chance (n=5 enc seeds, "
                 "5 probe seeds each averaged within enc).")
    lines.append("")
    # Group by config — one table per config with stage as column.
    configs = sorted({c for (c, _) in summary})
    stages = sorted({s for (_, s) in summary})
    for cfg in configs:
        lines.append(f"## {cfg}")
        lines.append("")
        header = "| Probe |" + "".join(f" {s} mean ± σ (p) |" for s in stages)
        sep = "|---|" + "---|" * len(stages)
        lines.append(header)
        lines.append(sep)
        for key in focus_keys:
            cells = [f"`{key}`"]
            for s in stages:
                stat = summary.get((cfg, s), {}).get(key)
                if stat is None or math.isnan(stat["mean"]):
                    cells.append("—")
                    continue
                mu = stat["mean"]
                sd = stat["sd"]
                p = stat["p"]
                star = "✓" if (not math.isnan(p) and p < 0.05) else " "
                cells.append(f"{mu:+.4f} ± {sd:.4f} (p={p:.2g}) {star}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", type=Path, default=Path("logs"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("docs/v10_stage_diagnostic.md"))
    args = ap.parse_args()

    rows = parse_logs(args.logs_dir, split=args.split)
    if not rows:
        raise SystemExit(f"No v10stage_*.out logs parsed under {args.logs_dir}")
    summary = aggregate(rows)
    md = render(summary, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"Parsed {len(rows)} runs across {len(summary)} (config,stage) groups")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
