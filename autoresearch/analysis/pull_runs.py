"""Pull EEG-JEPA wandb runs and dump cross-run metrics table."""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import wandb

OUT_DIR = Path("/Users/dtyoung/Documents/Research/eb_jepa_eeg/autoresearch/analysis")
OUT_CSV = OUT_DIR / "runs_metrics.csv"

# Metrics we want to extract from run.summary
TARGET_METRICS = [
    "val/reg_position_in_movie_corr",
    "val/reg_contrast_rms_corr",
    "val/reg_luminance_mean_corr",
    "val/reg_narrative_event_score_corr",
    "val/reg_loss",
]

SANITY_METRICS = [
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

TRAIN_METRICS = [
    "train_step/jepa_loss",
    "train_step/vc_loss",
    "train_step/pred_loss",
    "train_step/reg_loss",
    "train_step/cls_loss",
]

ALL_METRICS = TARGET_METRICS + SANITY_METRICS + TRAIN_METRICS

CONFIG_KEYS = [
    "data.norm_mode",
    "data.n_windows",
    "data.window_size_seconds",
    "data.batch_size",
    "loss.regularizer",
    "loss.std_coeff",
    "loss.cov_coeff",
    "loss.use_projector",
    "loss.pred_loss_type",
    "model.encoder_depth",
    "model.encoder_embed_dim",
    "model.patch_size",
    "model.predictor_depth",
    "optim.lr",
    "optim.epochs",
    "optim.warmup_epochs",
    "sanity_checks.enabled",
    "logging.wandb_group",
]


def _flatten(d, prefix=""):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _safe_float(x):
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return float("nan")
        return f
    except (TypeError, ValueError):
        return float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    print("Fetching runs from project 'eb_jepa'...")
    runs = api.runs("eb_jepa", per_page=500, order="-created_at")
    cutoff = datetime.now(timezone.utc) - timedelta(days=120)

    rows = []
    n_seen = 0
    n_kept = 0
    for run in runs:
        n_seen += 1
        if n_seen > 800:
            break  # hard cap
        # Filter by state
        if run.state in ("crashed", "failed"):
            continue
        # Filter by created_at
        try:
            created = run.created_at
            if isinstance(created, str):
                # wandb returns ISO string
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            else:
                created_dt = created
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            if created_dt < cutoff:
                continue
        except Exception:
            pass

        summary = dict(run.summary)  # copy
        # Need at least one val/reg_*_corr key
        has_target = any(k in summary for k in TARGET_METRICS[:4])
        if not has_target:
            continue
        # Need at least one sanity_* key (drop runs that lack ALL)
        has_any_sanity = any(k in summary for k in SANITY_METRICS)
        if not has_any_sanity:
            continue

        # Re-fetch run object to populate config (iterator returns shallow runs)
        try:
            full_run = api.run(f"eb_jepa/{run.id}")
            config_flat = _flatten(dict(full_run.config))
        except Exception:
            config_flat = _flatten(dict(run.config))

        row = {
            "run_id": run.id,
            "run_name": run.name,
            "created_at": str(run.created_at),
            "state": run.state,
        }
        for ck in CONFIG_KEYS:
            row[f"cfg.{ck}"] = config_flat.get(ck, None)  # keep as-is (str/bool/num)

        for m in ALL_METRICS:
            row[m] = _safe_float(summary.get(m))

        # weighted target
        pos = row["val/reg_position_in_movie_corr"]
        con = row["val/reg_contrast_rms_corr"]
        lum = row["val/reg_luminance_mean_corr"]
        nar = row["val/reg_narrative_event_score_corr"]

        def _ok(x):
            return isinstance(x, float) and not math.isnan(x)

        if all(_ok(x) for x in (pos, con, lum, nar)):
            row["val_corr_weighted"] = 0.3 * pos + 0.3 * con + 0.3 * lum + 0.1 * nar
        else:
            row["val_corr_weighted"] = float("nan")

        rows.append(row)
        n_kept += 1
        if n_kept >= 150:
            break

    print(f"Seen {n_seen} runs, kept {n_kept}.")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with shape {df.shape}")


if __name__ == "__main__":
    main()
