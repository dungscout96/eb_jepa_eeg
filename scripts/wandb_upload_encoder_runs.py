"""Re-upload pretraining runs from log files to a new W&B project.

Used to push the Phase-D nw4_ws2 baseline runs (5 enc seeds) to sccn/eb_jepa
so the lab account has the canonical pre-training trajectories.

The original W&B runs (if they exist on the user's personal account) are
NOT touched; this creates fresh runs in sccn/eb_jepa using metrics parsed
from the SLURM log file. The intent is that the synthetic runs reproduce
the train/val curves we already have on disk, not that they replace any
authoritative W&B record.

Usage (from repo root, on Delta):
    python scripts/wandb_upload_encoder_runs.py \
        --logs logs/issue8D_17927885.out logs/issue8D_17927886.out ... \
        --entity sccn --project eb_jepa --group phaseD_nw4ws2_baseline
"""
import argparse
import re
from pathlib import Path

import wandb

EPOCH_PAT = re.compile(
    r"Epoch (\d+)/100.*?loss=([\d.]+).*?vc=([\d.]+).*?pred=([\d.]+).*?val_reg=([\d.]+)",
    re.DOTALL,
)
TAG_PAT = re.compile(r"tag=([\w]+)")
SEED_PAT = re.compile(r"seed=(\d+)")
NW_PAT = re.compile(r"nw=(\d+)")
WS_PAT = re.compile(r"ws=(\d+)")
LR_PAT = re.compile(r"lr=([\d.eE+-]+)")
EPOCHS_PAT = re.compile(r"epochs=(\d+)")

VAL_KEYS = (
    "reg_narrative_event_score_corr",
    "reg_position_in_movie_corr",
    "reg_luminance_mean_corr",
    "reg_contrast_rms_corr",
    "cls_narrative_event_score_auc",
    "cls_position_in_movie_auc",
    "cls_luminance_mean_auc",
    "cls_contrast_rms_auc",
)


def parse_log(log_path: Path):
    text = log_path.read_text()
    cfg_line = next((l for l in text.splitlines() if l.startswith("Cfg:")), "")
    tag = (TAG_PAT.search(cfg_line) or [None, "unknown"])
    tag = tag.group(1) if tag else "unknown"
    seed = SEED_PAT.search(cfg_line)
    config = {
        "tag": tag,
        "seed": int(seed.group(1)) if seed else None,
        "n_windows": int(NW_PAT.search(cfg_line).group(1)) if NW_PAT.search(cfg_line) else None,
        "window_size_seconds": int(WS_PAT.search(cfg_line).group(1)) if WS_PAT.search(cfg_line) else None,
        "lr": float(LR_PAT.search(cfg_line).group(1)) if LR_PAT.search(cfg_line) else None,
        "epochs": int(EPOCHS_PAT.search(cfg_line).group(1)) if EPOCHS_PAT.search(cfg_line) else 100,
        "source_log": str(log_path),
        "source_jobid": log_path.stem.split("_")[-1],
    }

    rows = EPOCH_PAT.findall(text)
    val_metric_lists: dict[str, list[float]] = {k: [] for k in VAL_KEYS}
    for k in VAL_KEYS:
        pat = re.compile(rf"val/{re.escape(k)}: (-?[\d.]+)")
        val_metric_lists[k] = [float(m) for m in pat.findall(text)]

    epoch_metrics: list[dict] = []
    for i, (ep, total, vc, pred, val_reg) in enumerate(rows):
        d = {
            "epoch": int(ep),
            "train/total_loss": float(total),
            "train/vc_loss": float(vc),
            "train/pred_loss": float(pred),
            "val/reg_loss": float(val_reg),
        }
        for k in VAL_KEYS:
            if i < len(val_metric_lists[k]):
                d[f"val/{k}"] = val_metric_lists[k][i]
        epoch_metrics.append(d)

    return tag, config, epoch_metrics


def upload(log_path: Path, entity: str, project: str, group: str):
    tag, config, epoch_metrics = parse_log(log_path)
    if not epoch_metrics:
        print(f"  ! no epoch metrics found in {log_path}, skipping")
        return None

    run = wandb.init(
        entity=entity,
        project=project,
        name=tag,
        group=group,
        config=config,
        tags=["phase_d_baseline", "uploaded_from_log"],
        reinit=True,
    )
    for row in epoch_metrics:
        wandb.log(row, step=row["epoch"])
    print(f"  ✓ uploaded {tag} ({len(epoch_metrics)} epochs) → {run.url}")
    wandb.finish()
    return run.url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="Training log paths")
    ap.add_argument("--entity", default="sccn")
    ap.add_argument("--project", default="eb_jepa")
    ap.add_argument("--group", default="phaseD_nw4ws2_baseline")
    args = ap.parse_args()

    print(f"Uploading {len(args.logs)} runs to {args.entity}/{args.project} (group={args.group})")
    for log in args.logs:
        p = Path(log)
        if not p.exists():
            print(f"  ! missing: {p}")
            continue
        upload(p, args.entity, args.project, args.group)


if __name__ == "__main__":
    main()
