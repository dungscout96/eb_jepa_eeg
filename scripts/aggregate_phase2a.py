"""Aggregate Phase 2A OFAT probe_eval outputs into a per-knob table.

Reads stdout files from ``logs/p2a_*.out``, parses the Phase 2A cfg header
line written by ``train_phase2a.sbatch``, and reports per-cell metrics
relative to the Phase 1 anchor (std=0.25, pd=16, seed=42) baseline.

Usage on Delta:
  python scripts/aggregate_phase2a.py --logs_dir logs --out phase2a_results.md
"""

import argparse
import json
import re
import statistics
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

# Phase 1 5-seed baseline (from experiments.md) — compared against in the
# "delta vs baseline" column.
PHASE1_BASELINE = {
    "reg_position_in_movie_corr": (0.176, 0.048),
    "reg_luminance_mean_corr": (0.168, 0.059),
    "reg_contrast_rms_corr": (0.115, 0.053),
    "reg_narrative_event_score_corr": (-0.003, 0.042),
    "cls_position_in_movie_auc": (0.580, 0.025),
    "cls_luminance_mean_auc": (0.567, 0.021),
    "cls_contrast_rms_auc": (0.553, 0.032),
    "cls_narrative_event_score_auc": (0.528, 0.025),
    "subject/age_reg/corr": (0.325, 0.030),
    "subject/sex/auc": (0.618, 0.007),
}
STIM_KEYS = ["reg_position_in_movie_corr", "reg_luminance_mean_corr", "reg_contrast_rms_corr"]
PHASE1_COMPOSITE = sum(PHASE1_BASELINE[k][0] for k in STIM_KEYS) / len(STIM_KEYS)


# Matches the cfg line written by train_phase2a.sbatch:
#   cfg: knob=patch_size seed=42 std=0.25 cov=0.25 pd=16 enc_depth=2 pred_depth=2 patch_size=25 ...
_CFG_RE = re.compile(r"cfg:\s+knob=(\S+)\s+seed=(\d+)\s+(.*)")


def _parse_cfg_header(text: str) -> "dict | None":
    for line in text.splitlines():
        m = _CFG_RE.search(line)
        if not m:
            continue
        knob, seed, rest = m.group(1), int(m.group(2)), m.group(3)
        kv = {}
        for tok in rest.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
        kv["knob"] = knob
        kv["seed"] = seed
        return kv
    return None


def _extract_metric(text: str, key: str) -> "float | None":
    for line in text.splitlines():
        if f"probe_eval/test/{key}:" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                return None
    return None


def _split_segments(text: str):
    """Split a log file into one segment per (cfg header, metrics) pair.

    Handles single-cell training sbatches and multi-cell eval-only sbatches
    uniformly: a new segment begins at each "cfg: knob=..." header line.
    """
    segments = []
    cur = []
    for line in text.splitlines():
        if _CFG_RE.search(line) and cur:
            segments.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        segments.append("\n".join(cur))
    return segments


def aggregate(logs_dir: Path, prefix: str):
    cells = defaultdict(list)
    KNOB_TO_CFGKEY = {
        "patch_size": "patch_size",
        "patch_overlap": "patch_overlap",
        "n_masks_long": "n_masks_long",
        "long_patch": "long_patch",
        "pred_depth": "pred_depth",
        "enc_depth": "enc_depth",
        "lr": "lr",
        "ema_end": "ema_end",
        "corrca": "corrca",
    }
    for out_file in sorted(logs_dir.glob(f"{prefix}*.out")):
        text = out_file.read_text(errors="ignore")
        for seg in _split_segments(text):
            cfg = _parse_cfg_header(seg)
            if cfg is None:
                continue
            metrics = {k: _extract_metric(seg, k) for k in METRICS}
            if metrics.get("reg_position_in_movie_corr") is None:
                continue  # segment didn't reach eval
            knob = cfg["knob"]
            value = cfg.get(KNOB_TO_CFGKEY.get(knob, knob), "?")
            cell_key = (knob, value)
            cells[cell_key].append({"seed": cfg["seed"], "file": out_file.name, **metrics})
    return cells


def summarize(cells):
    rows = []
    for (knob, value), runs in cells.items():
        summary = {"knob": knob, "value": value, "n_seeds": len(runs)}
        for k in METRICS:
            vals = [r[k] for r in runs if r[k] is not None]
            if not vals:
                summary[k] = (None, None)
            else:
                mean = statistics.mean(vals)
                stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
                summary[k] = (mean, stdev)
        cor_means = [summary[k][0] for k in STIM_KEYS if summary[k][0] is not None]
        summary["stim_corr_mean"] = sum(cor_means) / len(cor_means) if cor_means else None
        rows.append(summary)

    # Sort within each knob by composite, desc. Then by knob alphabetical.
    rows.sort(key=lambda r: (r["knob"], -(r.get("stim_corr_mean") or -1)))
    return rows


def format_markdown(rows):
    md = ["# Phase 2A — OFAT screening results\n\n"]
    md.append("Anchor = Phase 1 best cell (std=cov=0.25, pd=16, seed=42). Each row changes one knob from that anchor.\n\n")
    md.append(f"Phase 1 5-seed baseline composite stimulus corr: **{PHASE1_COMPOSITE:+.3f}**. ")
    md.append("Detection threshold at 1 seed: |Δ composite| ≥ ~0.10 (≈2σ) to survive to Phase 2B.\n\n")
    md.append("## Per-cell metrics\n\n")
    md.append("|knob|value|n|pos corr|lum corr|cont corr|narr corr|pos AUC|lum AUC|cont AUC|narr AUC|age corr|sex AUC|composite|Δvs baseline|\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    def fmt(pair):
        m, s = pair
        if m is None:
            return "—"
        return f"{m:+.3f}±{s:.3f}" if s > 0 else f"{m:+.3f}"

    for r in rows:
        composite = r["stim_corr_mean"]
        delta_str = "—"
        if composite is not None:
            delta_str = f"{composite - PHASE1_COMPOSITE:+.3f}"
        md.append(
            f"|{r['knob']}|{r['value']}|{r['n_seeds']}|"
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
            f"{composite:+.3f}|{delta_str}|\n"
        )
    return "".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--prefix", default="p2a_")
    ap.add_argument("--out", default="phase2a_results.md")
    args = ap.parse_args()

    cells = aggregate(Path(args.logs_dir), args.prefix)
    n_runs = sum(len(v) for v in cells.values())
    print(f"Found {len(cells)} distinct cells, {n_runs} runs")

    rows = summarize(cells)
    md = format_markdown(rows)
    Path(args.out).write_text(md)
    print(f"Wrote {args.out}")

    Path(args.out).with_suffix(".json").write_text(json.dumps(
        [{k: v for k, v in r.items()} for r in rows], indent=2, default=str
    ))


if __name__ == "__main__":
    main()
