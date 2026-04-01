"""Pull W&B run metrics and produce a summary for the sweep.

Usage
-----
# Summarize a single run:
uv run python scripts/pull_wandb.py t6iialwu

# Summarize multiple runs:
uv run python scripts/pull_wandb.py t6iialwu jrepxrw9

# Summarize all runs in the project:
uv run python scripts/pull_wandb.py --all

# Print a results.tsv-compatible row (for copy-pasting):
uv run python scripts/pull_wandb.py t6iialwu --tsv

# Compare runs side by side:
uv run python scripts/pull_wandb.py t6iialwu jrepxrw9 --compare
"""

import argparse
import sys

import wandb

# ── Config ──────────────────────────────────────────────────────────────────
ENTITY = "sccn"
PROJECT = "eb_jepa"

# Key metrics to extract (final values)
SUMMARY_KEYS = [
    # Pretraining
    "train_step/pred_loss",
    "train/pred_loss",
    "train/vc_loss",
    "train/loss",
    # Sanity
    "sanity/cosim_random_pairs_mean",
    "sanity/embedding_variance_mean",
    "sanity/embedding_variance_min",
    "sanity/grad_norm",
    "sanity/linear_probe_acc",
    "sanity/loss_trend",
    "sanity/pred_loss_short",
    "sanity/pred_loss_long",
    # Validation probes
    "val/reg_loss",
    "val/cls_loss",
    "val/reg_contrast_rms_corr",
    "val/reg_entropy_corr",
    "val/reg_luminance_mean_corr",
    "val/reg_scene_natural_score_corr",
    "val/cls_contrast_rms_auc",
    "val/cls_entropy_auc",
    "val/cls_luminance_mean_auc",
    "val/cls_scene_natural_score_auc",
]

# Mapping from results.tsv columns to wandb summary keys
TSV_MAPPING = {
    "pred_loss": "train/pred_loss",
    "probe_acc": "sanity/linear_probe_acc",
    "cosim": "sanity/cosim_random_pairs_mean",
    "embed_std": "sanity/embedding_variance_mean",
    "val_reg": "val/reg_loss",
    "val_cls": "val/cls_loss",
}


def get_api():
    return wandb.Api()


def fetch_run(api, run_id: str):
    return api.run(f"{ENTITY}/{PROJECT}/{run_id}")


def print_run_summary(run):
    """Print a human-readable summary of a single run."""
    print(f"\n{'=' * 70}")
    print(f"Run: {run.name}  (id: {run.id})")
    print(f"State: {run.state}")
    print(f"Created: {run.created_at}")
    tags = run.tags or []
    print(f"Tags: {', '.join(tags)}")

    # Config highlights
    cfg = run.config
    print(f"\n── Config ──")
    config_keys = [
        "optim.lr", "optim.epochs", "data.batch_size",
        "loss.std_coeff", "loss.cov_coeff",
        "model.encoder_embed_dim", "model.encoder_depth",
        "masking.min_context_fraction",
    ]
    for k in config_keys:
        # Config might be nested or flat
        val = _get_nested(cfg, k)
        if val is not None:
            print(f"  {k}: {val}")

    # Summary metrics
    summary = run.summary
    print(f"\n── Final Metrics ──")
    for key in SUMMARY_KEYS:
        val = summary.get(key)
        if val is not None:
            print(f"  {key}: {_fmt(val)}")

    # Epoch count
    epoch = summary.get("epoch") or summary.get("_step")
    if epoch is not None:
        print(f"  epoch: {epoch}")

    print(f"{'=' * 70}")


def print_tsv_row(run):
    """Print a results.tsv-compatible row."""
    summary = run.summary
    cfg = run.config

    # Try to get commit from config or tags
    commit = "unknown"
    if "git_commit" in cfg:
        commit = cfg["git_commit"][:7]

    pid = run.id
    vals = {}
    for tsv_col, wb_key in TSV_MAPPING.items():
        v = summary.get(wb_key)
        vals[tsv_col] = f"{v:.4f}" if v is not None else "0.000"

    status = "keep" if run.state == "finished" else "crash"
    desc = run.name or ""

    row = "\t".join([
        commit, pid,
        vals["pred_loss"], vals["probe_acc"], vals["cosim"],
        vals["embed_std"], vals["val_reg"], vals["val_cls"],
        status, desc,
    ])
    print("\n── results.tsv row ──")
    print("commit\tpid\tpred_loss\tprobe_acc\tcosim\tembed_std\tval_reg\tval_cls\tstatus\tdescription")
    print(row)


def print_comparison(runs):
    """Print a side-by-side comparison table."""
    # Collect all metrics
    keys_to_compare = [
        "train/pred_loss", "train/vc_loss",
        "sanity/cosim_random_pairs_mean", "sanity/embedding_variance_mean",
        "sanity/linear_probe_acc", "sanity/grad_norm",
        "val/reg_loss", "val/cls_loss",
        "val/reg_contrast_rms_corr", "val/reg_entropy_corr",
        "val/cls_contrast_rms_auc", "val/cls_entropy_auc",
    ]

    # Header
    name_width = 38
    col_width = 16
    header = f"{'metric':<{name_width}}"
    for r in runs:
        label = r.id
        header += f"{label:>{col_width}}"
    print(f"\n{header}")
    print("─" * (name_width + col_width * len(runs)))

    for key in keys_to_compare:
        row = f"{key:<{name_width}}"
        for r in runs:
            val = r.summary.get(key)
            row += f"{_fmt(val):>{col_width}}"
        print(row)

    # Config diffs
    print(f"\n── Config Differences ──")
    config_keys = [
        "optim.lr", "optim.epochs", "data.batch_size",
        "loss.std_coeff", "loss.cov_coeff",
        "model.encoder_embed_dim", "model.encoder_depth",
    ]
    for k in config_keys:
        vals = [_get_nested(r.config, k) for r in runs]
        if len(set(str(v) for v in vals)) > 1:  # only show diffs
            row = f"{k:<{name_width}}"
            for v in vals:
                row += f"{str(v):>{col_width}}"
            print(row)


def print_history_trend(run, keys=None):
    """Print epoch-by-epoch trend for key metrics.

    Extracts epoch-boundary rows (where train/pred_loss exists) for a clean
    per-epoch view, plus a step-level sample for finer-grained trends.
    """
    # Epoch-boundary keys (logged once per epoch)
    epoch_keys = [
        "train/pred_loss", "train/vc_loss",
        "sanity/cosim_random_pairs_mean", "sanity/embedding_variance_mean",
        "sanity/linear_probe_acc", "sanity/grad_norm",
        "val/reg_loss", "val/cls_loss",
    ]
    # Step-level keys (logged every step)
    step_keys = [
        "train_step/pred_loss", "sanity/cosim_random_pairs_mean",
        "sanity/embedding_variance_mean",
    ]

    # Collect epoch-boundary rows
    epoch_rows = []
    step_rows = []
    for row in run.scan_history():
        if row.get("epoch") is not None and row.get("train/pred_loss") is not None:
            epoch_rows.append(row)
        elif row.get("train_step/pred_loss") is not None:
            step_rows.append(row)

    # Print epoch-level table
    if epoch_rows:
        display_keys = [k for k in epoch_keys if any(r.get(k) is not None for r in epoch_rows)]
        print(f"\n── Epoch Summary ({run.id}) ──")
        col_width = 14
        header = f"{'epoch':>6}"
        for k in display_keys:
            short = k.split("/")[-1][:12]
            header += f"{short:>{col_width}}"
        print(header)
        print("─" * (6 + col_width * len(display_keys)))

        for row in epoch_rows:
            epoch = row.get("epoch", "?")
            epoch_str = str(int(epoch)) if isinstance(epoch, (int, float)) else "?"
            line = f"{epoch_str:>6}"
            for k in display_keys:
                val = row.get(k)
                line += f"{_fmt(val):>{col_width}}"
            print(line)
    else:
        print("  (no epoch-level data found)")

    # Print step-level sample (sparse)
    if step_rows and len(step_rows) > 5:
        n = len(step_rows)
        # Sample ~15 evenly spaced points
        step_size = max(1, n // 15)
        sampled = step_rows[::step_size]
        if step_rows[-1] not in sampled:
            sampled.append(step_rows[-1])

        display_keys = [k for k in step_keys if any(r.get(k) is not None for r in sampled)]
        print(f"\n── Step-level Sample ({n} total steps) ──")
        col_width = 14
        header = f"{'step':>6}"
        for k in display_keys:
            short = k.split("/")[-1][:12]
            header += f"{short:>{col_width}}"
        print(header)
        print("─" * (6 + col_width * len(display_keys)))

        for row in sampled:
            step = row.get("_step", "?")
            step_str = str(int(step)) if isinstance(step, (int, float)) else "?"
            line = f"{step_str:>6}"
            for k in display_keys:
                val = row.get(k)
                line += f"{_fmt(val):>{col_width}}"
            print(line)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_nested(d: dict, dotted_key: str):
    """Get a value from a possibly-nested dict using dot notation."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _fmt(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pull and analyze W&B run metrics")
    parser.add_argument("run_ids", nargs="*", help="W&B run IDs to analyze")
    parser.add_argument("--all", action="store_true", help="Fetch all runs in the project")
    parser.add_argument("--tsv", action="store_true", help="Print results.tsv-compatible rows")
    parser.add_argument("--compare", action="store_true", help="Side-by-side comparison")
    parser.add_argument("--trend", action="store_true", help="Show epoch-by-epoch trend")
    parser.add_argument("--last", type=int, default=0, help="Fetch the N most recent runs")
    args = parser.parse_args()

    if not args.run_ids and not args.all and not args.last:
        parser.print_help()
        sys.exit(1)

    api = get_api()

    if args.all or args.last:
        runs_iter = api.runs(f"{ENTITY}/{PROJECT}", order="-created_at")
        if args.last:
            runs = [r for _, r in zip(range(args.last), runs_iter)]
        else:
            runs = list(runs_iter)
        print(f"Found {len(runs)} runs in {ENTITY}/{PROJECT}")
    else:
        runs = [fetch_run(api, rid) for rid in args.run_ids]

    if args.compare and len(runs) >= 2:
        print_comparison(runs)
    else:
        for run in runs:
            print_run_summary(run)
            if args.trend:
                print_history_trend(run)

    if args.tsv:
        for run in runs:
            print_tsv_row(run)


if __name__ == "__main__":
    main()
