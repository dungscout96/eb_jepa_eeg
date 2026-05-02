"""Aggregate Phase 1 diagnostic sweep logs into a per-condition table.

Reads logs/p1diag_*.out, parses the trailing metric block + "Phase 1 diag <tag>
probe_seed=<seed> done." marker, and emits:
  - JSON dict: cond -> seed -> metric -> value
  - Markdown summary tables (mean ± std across the 5 probe seeds per condition)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

LOG_DIR = Path("logs")
OUT_JSON = Path("docs/phase1_results.json")
OUT_MD = Path("docs/phase1_diagnostics.md")

# Regexes
RE_DONE = re.compile(r"^Phase 1 diag (\S+) probe_seed=(\d+) done\.")
RE_METRIC = re.compile(r"probe_eval/(val|test)/([\w/_]+):\s+([-+0-9.eE]+|nan|inf)")

# Metrics we care about (subset of probe outputs); only test split.
WANTED = [
    "reg_narrative_event_score_corr",
    "reg_position_in_movie_corr",
    "reg_luminance_mean_corr",
    "reg_contrast_rms_corr",
    "cls_position_in_movie_auc",
    "cls_narrative_event_score_auc",
    "cls_luminance_mean_auc",
    "cls_contrast_rms_auc",
    "movie_id/top1_acc",
    "movie_id/top5_acc",
    "subject/age_reg/corr",
    "subject/age_cls/auc",
    "subject/sex/auc",
]


def parse_log(path: Path):
    tag = None
    seed = None
    metrics: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            m = RE_DONE.search(line)
            if m:
                tag, seed = m.group(1), int(m.group(2))
            m = RE_METRIC.search(line)
            if m:
                split, key, val = m.group(1), m.group(2), m.group(3)
                if split != "test":
                    continue
                try:
                    v = float(val)
                except ValueError:
                    continue
                metrics[key] = v
    return tag, seed, metrics


def main():
    by_cond_seed: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    bad = []
    for log in sorted(LOG_DIR.glob("p1diag_*.out")):
        tag, seed, metrics = parse_log(log)
        if tag is None or seed is None:
            bad.append(str(log))
            continue
        by_cond_seed[tag][seed] = metrics

    # JSON dump
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(by_cond_seed, f, indent=2)
    print(f"Wrote {OUT_JSON} ({len(by_cond_seed)} conditions, "
          f"{sum(len(v) for v in by_cond_seed.values())} runs total)")
    if bad:
        print(f"WARN {len(bad)} logs lacked a 'done.' marker:")
        for b in bad:
            print(f"  - {b}")

    # Aggregate
    rows = []  # (cond, metric, mean, std, n)
    for cond, seed_metrics in by_cond_seed.items():
        for metric in WANTED:
            vals = [m[metric] for m in seed_metrics.values() if metric in m]
            if not vals:
                continue
            rows.append((cond, metric, mean(vals), pstdev(vals), len(vals)))

    # Build per-metric tables (rows = conditions, columns = mean±std)
    metric_to_rows: dict[str, list[tuple[str, float, float, int]]] = defaultdict(list)
    for cond, metric, mu, sd, n in rows:
        metric_to_rows[metric].append((cond, mu, sd, n))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MD.open("w") as f:
        f.write("# Phase 1 — Encoder Diagnostic Sweep Results\n\n")
        f.write("Anchor checkpoint: `phaseD_nw4ws2_baseline_s42` (single enc seed, "
                "5 probe seeds per condition).  All metrics on the **test** split.\n\n")
        f.write("Conditions are coded `<layer>_<tower>_<routing>` where:\n")
        f.write("- `layer` ∈ {patch_embed, block0, final}\n")
        f.write("- `tower` ∈ {stu (student / context encoder), tea (EMA target encoder)}\n")
        f.write("- `routing` ∈ {mp (mean-pool, default), kc (--keep_channels)}\n")
        f.write("- `chN` = single-channel attribution (channel N, mean-pool)\n")
        f.write("- `prepred` = project final-layer tokens through predictor.input_proj (24-d)\n\n")
        for metric in WANTED:
            if metric not in metric_to_rows:
                continue
            f.write(f"## {metric}\n\n")
            f.write("| Condition | mean ± std | n |\n|---|---|---|\n")
            for cond, mu, sd, n in sorted(
                metric_to_rows[metric], key=lambda r: -r[1]
            ):
                f.write(f"| `{cond}` | {mu:+.4f} ± {sd:.4f} | {n} |\n")
            f.write("\n")

    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
