"""Aggregate issue #8 SLURM out-files into markdown comparison tables.

Pulls probe_eval metrics from:
  - logs/issue8_*.out         — Phase A (no-CorrCA ablation, 5 seeds) + Phase B (temporal sweep)
  - logs/mlp_probe_*.out      — MLP-vs-linear probe runs on Exp 6 baseline checkpoints

Run on Delta from the repo root:
  python scripts/aggregate_issue8.py --logs_dir logs --out issue8_results.md

Pure stdlib; no wandb dependency. Falls back to wandb only if the user
passes --wandb (not implemented yet — TODO if needed).
"""

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

# probe_eval prints lines like:
#   probe_eval/test/reg_position_in_movie_corr: 0.176
#   probe_eval/test/cls_position_in_movie_auc: 0.580
#   probe_eval/test/subject/age_reg/corr: 0.325
#   probe_eval/test/subject/sex/auc: 0.618
#   probe_eval/test/movie_id/top1_acc: 0.123
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
    "movie_id/top1_acc",
    "movie_id/top5_acc",
]

ISSUE8_CFG_RE = re.compile(
    r"Cfg:\s*nw=(\d+)\s+ws=(\d+)\s+seed=(\d+)\s+bs=(\d+)\s+corrca=([01])\s+tag=(\S+)"
)
MLP_CFG_RE = re.compile(
    r"Cfg:\s*ckpt=\S*seed(\d+)/\S*\s+probe=(\w+)\s+hidden=(\d+)\s+movie_hidden=(\d+)\s+tag=(\S+)"
)


def _extract_metric(text: str, split: str, key: str):
    needle = f"probe_eval/{split}/{key}:"
    for line in text.splitlines():
        if needle in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                return None
    return None


def parse_issue8_logs(logs_dir: Path, split: str = "test"):
    """Return list of dicts, one per Phase A/B run."""
    rows = []
    for f in sorted(logs_dir.glob("issue8_*.out")):
        text = f.read_text(errors="ignore")
        m = ISSUE8_CFG_RE.search(text)
        if not m:
            continue
        nw, ws, seed, bs, corrca, tag = m.groups()
        metrics = {k: _extract_metric(text, split, k) for k in METRIC_KEYS}
        if metrics["reg_position_in_movie_corr"] is None:
            continue  # incomplete or failed
        rows.append({
            "file": f.name,
            "tag": tag,
            "nw": int(nw), "ws": int(ws), "seed": int(seed),
            "bs": int(bs), "corrca": int(corrca),
            **metrics,
        })
    return rows


def parse_mlp_probe_logs(logs_dir: Path, split: str = "test"):
    """Return list of dicts, one per MLP-probe run."""
    rows = []
    for f in sorted(logs_dir.glob("mlp_probe_*.out")):
        text = f.read_text(errors="ignore")
        m = MLP_CFG_RE.search(text)
        if not m:
            continue
        seed, probe_type, hidden, movie_hidden, tag = m.groups()
        metrics = {k: _extract_metric(text, split, k) for k in METRIC_KEYS}
        if metrics["reg_position_in_movie_corr"] is None:
            continue
        rows.append({
            "file": f.name,
            "tag": tag,
            "seed": int(seed),
            "probe_type": probe_type,
            "hidden_dim": int(hidden),
            "movie_hidden": int(movie_hidden),
            **metrics,
        })
    return rows


def _fmt(v):
    return f"{v:+.3f}" if v is not None else "  —  "


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def render_phase_a(rows, exp6_baseline=None):
    """Phase A: per-seed no-CorrCA at nw4_ws2 + paired delta vs Exp 6 baseline.

    exp6_baseline: optional dict seed -> {metric: value} for paired delta.
    """
    if not rows:
        return "## Phase A — no rows\n"
    rows_a = [r for r in rows if r["nw"] == 4 and r["ws"] == 2 and r["corrca"] == 0]
    if not rows_a:
        return "## Phase A — no nw4_ws2 no-CorrCA rows\n"
    rows_a.sort(key=lambda r: r["seed"])

    md = ["## Phase A — CorrCA ablation (5 seeds, nw4_ws2, NO CorrCA)\n"]
    headline = ("seed", "pos_corr", "lum_corr", "cont_corr", "narr_corr",
                "pos_auc", "lum_auc", "cont_auc", "narr_auc",
                "age_corr", "sex_auc")
    md.append("| " + " | ".join(headline) + " |")
    md.append("|" + "|".join(["---"] * len(headline)) + "|")
    keymap = {
        "pos_corr": "reg_position_in_movie_corr",
        "lum_corr": "reg_luminance_mean_corr",
        "cont_corr": "reg_contrast_rms_corr",
        "narr_corr": "reg_narrative_event_score_corr",
        "pos_auc": "cls_position_in_movie_auc",
        "lum_auc": "cls_luminance_mean_auc",
        "cont_auc": "cls_contrast_rms_auc",
        "narr_auc": "cls_narrative_event_score_auc",
        "age_corr": "subject/age_reg/corr",
        "sex_auc": "subject/sex/auc",
    }
    for r in rows_a:
        cells = [str(r["seed"])] + [_fmt(r[keymap[k]]) for k in headline[1:]]
        md.append("| " + " | ".join(cells) + " |")
    # mean/std
    means = ["mean±std"]
    for k in headline[1:]:
        m, s = _mean_std([r[keymap[k]] for r in rows_a])
        means.append(f"{m:+.3f}±{s:.3f}" if m is not None else "—")
    md.append("| " + " | ".join(means) + " |")
    md.append("")

    if exp6_baseline:
        md.append("### Paired Δ vs Exp 6 (matched seed, with CorrCA)\n")
        md.append("| " + " | ".join(headline) + " |")
        md.append("|" + "|".join(["---"] * len(headline)) + "|")
        deltas = defaultdict(list)
        for r in rows_a:
            base = exp6_baseline.get(r["seed"])
            if not base:
                continue
            cells = [str(r["seed"])]
            for k in headline[1:]:
                v_new = r[keymap[k]]
                v_base = base.get(keymap[k])
                if v_new is None or v_base is None:
                    cells.append("—")
                else:
                    d = v_new - v_base
                    deltas[k].append(d)
                    cells.append(f"{d:+.3f}")
            md.append("| " + " | ".join(cells) + " |")
        means = ["mean Δ"]
        for k in headline[1:]:
            if deltas[k]:
                m, s = _mean_std(deltas[k])
                means.append(f"{m:+.3f}±{s:.3f}")
            else:
                means.append("—")
        md.append("| " + " | ".join(means) + " |")
        md.append("")
    return "\n".join(md)


def render_phase_b(rows):
    rows_b = [r for r in rows if not (r["nw"] == 4 and r["ws"] == 2)]
    if not rows_b:
        return ""
    rows_b.sort(key=lambda r: (r["nw"], r["ws"]))
    md = ["## Phase B — temporal context sweep (single seed)\n"]
    md.append("| nw | ws | total_ctx | corrca | seed | pos_corr | lum_corr | cont_corr | pos_auc | sex_auc |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows_b:
        ctx = r["nw"] * r["ws"]
        md.append(
            f"| {r['nw']} | {r['ws']} | {ctx}s | {r['corrca']} | {r['seed']} | "
            f"{_fmt(r['reg_position_in_movie_corr'])} | "
            f"{_fmt(r['reg_luminance_mean_corr'])} | "
            f"{_fmt(r['reg_contrast_rms_corr'])} | "
            f"{_fmt(r['cls_position_in_movie_auc'])} | "
            f"{_fmt(r['subject/sex/auc'])} |"
        )
    md.append("")
    return "\n".join(md)


def render_mlp_probe(rows):
    if not rows:
        return ""
    rows.sort(key=lambda r: (r["seed"], r["probe_type"], r["hidden_dim"]))
    md = ["## MLP probe vs linear probe (Exp 6 std=0.25 pd=24 +CorrCA)\n"]
    headline = ("seed", "probe", "hidden", "pos_corr", "lum_corr", "cont_corr",
                "pos_auc", "age_corr", "sex_auc", "movie_id_top1")
    md.append("| " + " | ".join(headline) + " |")
    md.append("|" + "|".join(["---"] * len(headline)) + "|")
    for r in rows:
        ph = "linear" if r["probe_type"] == "linear" else f"mlp(h={r['hidden_dim']},mh={r['movie_hidden']})"
        md.append(
            f"| {r['seed']} | {r['probe_type']} | {r['hidden_dim']} | "
            f"{_fmt(r['reg_position_in_movie_corr'])} | "
            f"{_fmt(r['reg_luminance_mean_corr'])} | "
            f"{_fmt(r['reg_contrast_rms_corr'])} | "
            f"{_fmt(r['cls_position_in_movie_auc'])} | "
            f"{_fmt(r['subject/age_reg/corr'])} | "
            f"{_fmt(r['subject/sex/auc'])} | "
            f"{_fmt(r['movie_id/top1_acc'])} |"
        )
    md.append("")
    # Paired delta MLP - linear, per metric, averaged over seeds
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][(r["probe_type"], r["hidden_dim"])] = r
    keymap = {
        "pos_corr": "reg_position_in_movie_corr",
        "lum_corr": "reg_luminance_mean_corr",
        "cont_corr": "reg_contrast_rms_corr",
        "pos_auc": "cls_position_in_movie_auc",
        "age_corr": "subject/age_reg/corr",
        "sex_auc": "subject/sex/auc",
        "movie_id_top1": "movie_id/top1_acc",
    }
    md.append("### Paired Δ (mlp − linear), mean over seeds\n")
    md.append("| variant | " + " | ".join(keymap.keys()) + " |")
    md.append("|" + "|".join(["---"] * (len(keymap) + 1)) + "|")
    variants = sorted({(r["probe_type"], r["hidden_dim"]) for r in rows if r["probe_type"] == "mlp"})
    for var in variants:
        deltas = defaultdict(list)
        for seed, by_var in by_seed.items():
            base = by_var.get(("linear", 128))
            cur = by_var.get(var)
            if not base or not cur:
                continue
            for k, raw in keymap.items():
                if base[raw] is None or cur[raw] is None:
                    continue
                deltas[k].append(cur[raw] - base[raw])
        cells = [f"mlp(h={var[1]})"]
        for k in keymap:
            if deltas[k]:
                m, s = _mean_std(deltas[k])
                cells.append(f"{m:+.3f}±{s:.3f}")
            else:
                cells.append("—")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", type=Path, default=Path("logs"))
    ap.add_argument("--out", type=Path, default=Path("issue8_results.md"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    issue8_rows = parse_issue8_logs(args.logs_dir, split=args.split)
    mlp_rows = parse_mlp_probe_logs(args.logs_dir, split=args.split)

    md = [f"# Issue #8 results ({args.split} split)\n"]
    md.append(f"_Parsed {len(issue8_rows)} issue8_*.out + {len(mlp_rows)} mlp_probe_*.out files._\n")
    md.append(render_phase_a(issue8_rows))
    md.append(render_phase_b(issue8_rows))
    md.append(render_mlp_probe(mlp_rows))

    args.out.write_text("\n".join(md))
    print(f"Wrote {args.out} ({sum(len(s) for s in md)} chars)")


if __name__ == "__main__":
    main()
