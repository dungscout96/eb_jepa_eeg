"""Per-epoch probe + retrieval curve for a sweep of cells, from cached inputs.

WHY THIS EXISTS. Each `e04_reve_scaling` cell currently picks its checkpoint by
smoothed argmax of `val/clip_scene_auc` -- an AUC over 29 windows
(`eval.val_recording_fraction: 0.1`) with sd ~0.10 and adjacent-epoch swings
~0.09. That is noise around a slow trend, which is why it needs a window-25
smoother, and an unsmoothed version of it once manufactured a finding that
reversed under smoothing (experiments/snr_scaling/RESULTS.md 2.10). A probe fit
on held-out recordings is a far lower-variance signal and measures the quantity
the paper reports.

WHY IT IS CHEAP. `probe_traintest.py` re-reads 1971 FIF files and re-encodes
~199k windows per invocation. But the EEG windows, the regression targets and
the V-JEPA-2 vectors are all CHECKPOINT-INDEPENDENT -- only the forward pass
changes across an epoch curve. This script reads a small fixed recording set
ONCE, holds it in memory, and re-encodes it per checkpoint. Per-checkpoint cost
drops from ~40 min to a few seconds, so a 28-cell x 15-epoch curve fits in under
a GPU-hour instead of ~280.

WHICH RECORDINGS, AND THE SCOPE LIMIT. The `val` split (R5) is disjoint from
every e04 cohort at every S -- e04 pretrains on [R1..R4, R7..R10] -- so
selection never touches pretraining data, and the test split R6 is never read.
The zero-overlap head-fit check (config_probe_TP_headfit_R5.yaml) already
measured that a 293-recording head-fit pool reproduces the S-curve shape at
r = 0.9989, which is the evidence that a small pool preserves the RANKING a
selector needs. **This is available for the e04 arm only.** The e05_addval and
e05_fromscratch arms trained on R5, so they have no held-out selection data at
all and their fixed epoch stays fixed by necessity.

TWO METRICS, ONE FORWARD PASS.
  probe      mean Pearson r over the 12 scalar features, ridge at a FIXED alpha
             (--alpha). Fixed, because letting RidgeCV re-pick alpha per epoch
             stacks a second noisy selection on top of the one being measured.
             Fit on the first --n-fit recordings, scored on the rest.
  retrieval  time/shot/scene e->v@1 through the trained clip_head. Fits nothing,
             so it uses all recordings -- and because it is never selected on,
             it is the independent check that probe-based selection is not
             merely selecting the probe.

Usage (on Delta, where the checkpoints are mounted):
    uv run --group eeg python eb_jepa/evaluation/clip_probe/epoch_curve.py \\
        --ckpt-root /work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling \\
        --cells-glob 'e04_s*_a101_nd*' \\
        --config experiments/snr_scaling/config/config_probe_TP_e04.yaml \\
        --output experiments/snr_scaling/raw_results/e04_epoch_curve.json

Pick the fixed alpha once, then hard-code it into the submitter:
    ... --calibrate-alpha --cells e04_s701_a101_nd_d22 --epochs 200
"""
import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from eb_jepa.architectures import MovieCLIPHead
from eb_jepa.evaluation.clip_probe.probe import (
    SCALAR_FEATURES_DEFAULT,
    build_dataset,
    encode_windows,
    load_recording_windows,
)
from eb_jepa.evaluation.clip_probe.retrieval import (
    _evaluate_all_levels,
    _load_clip_ckpt,
)
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_config

# Alpha for the selection ridge. Calibrated once with --calibrate-alpha and then
# held fixed across every cell and every epoch: the curve must move because the
# encoder moved, not because the head's regularisation was re-tuned under it.
DEFAULT_ALPHA = 100.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-root", required=True,
                    help="Directory holding one subdirectory per cell.")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="Explicit cell names. Mutually exclusive with --cells-glob.")
    ap.add_argument("--cells-glob", default=None,
                    help="Discover cells by glob under --ckpt-root. Skips "
                         "*FAILED* and *smoke* like select_e04.py does.")
    ap.add_argument("--config", required=True,
                    help="Probe-eval config yaml (architecture + data).")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="Split to select on. Default val (R5), the only split "
                         "disjoint from every e04 cohort. NEVER pass test.")
    ap.add_argument("--epochs", nargs="+", type=int, default=None,
                    help="Epochs to evaluate. Default: every epoch_*.pth.tar in "
                         "the cell directory.")
    ap.add_argument("--n-recordings", type=int, default=80,
                    help="Size of the fixed selection set drawn from --split.")
    ap.add_argument("--n-fit", type=int, default=60,
                    help="How many of those fit the ridge head; the rest score "
                         "it. Split BY RECORDING, so the scored half is unseen "
                         "subjects.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the recording draw. Fixed across cells and "
                         "epochs -- every checkpoint sees the same data.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--calibrate-alpha", action="store_true",
                    help="Fit RidgeCV instead and report the chosen alpha per "
                         "feature, to set DEFAULT_ALPHA. Does not write a curve.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--half", action="store_true",
                    help="Hold the cached windows in fp16 (halves the ~1.7 GB "
                         "footprint); batches are cast back on the way to GPU.")
    ap.add_argument("--topks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--t-bucket-s", type=float, default=0.1)
    ap.add_argument("--output", default="epoch_curve.json")
    args = ap.parse_args()
    if bool(args.cells) == bool(args.cells_glob):
        raise SystemExit("Pass exactly one of --cells / --cells-glob.")
    if args.n_fit >= args.n_recordings:
        raise SystemExit("--n-fit must leave recordings over to score on.")
    return args


def discover_cells(ckpt_root: str, cells_glob: str) -> list[str]:
    """Cell names under ckpt_root, excluding superseded and smoke runs.

    Mirrors select_e04.py's exclusions so the two selectors are computed over
    the same cell set -- a curve that silently included a *_FAILED_* directory
    would disagree with e04_selection.json for a reason that is not the metric.
    """
    return sorted(
        (Path(d).name for d in glob.glob(f"{ckpt_root}/{cells_glob}")
         if "FAILED" not in d and "smoke" not in d),
        key=lambda n: (int(re.search(r"_s(\d+)_", n).group(1)), n),
    )


def checkpoints_for(cell_dir: Path, epochs: list[int] | None) -> list[tuple[int, Path]]:
    saved = sorted(
        (int(p.stem.split("_")[1].split(".")[0]), p)
        for p in cell_dir.glob("epoch_*.pth.tar")
    )
    if epochs is not None:
        want = set(epochs)
        saved = [(e, p) for e, p in saved if e in want]
    return saved


def build_cache(dataset, args):
    """Materialise the checkpoint-independent half of the evaluation, once.

    Everything here -- windows, targets, movie-time, shot/scene ids, V-JEPA-2
    vectors -- is a property of the DATA, so it is read once and reused by every
    checkpoint of every cell.
    """
    n_avail = len(dataset)
    if args.n_recordings > n_avail:
        raise SystemExit(
            f"--n-recordings {args.n_recordings} exceeds the {args.split} split's "
            f"{n_avail} recordings.")
    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(n_avail)[:args.n_recordings]
    fit_recs = set(chosen[:args.n_fit].tolist())

    windows, Ys, is_fit = [], [], []
    t_starts, task_ids, embeds, shot_ids, scene_ids = [], [], [], [], []
    t0 = time.time()
    for i, rec_idx in enumerate(chosen.tolist()):
        eeg_in = load_recording_windows(dataset, rec_idx)     # [n_win, 1, C, T]
        if args.half:
            eeg_in = eeg_in.half()
        n_win = len(eeg_in)
        windows.append(eeg_in)
        Ys.append(dataset.feature_recordings[rec_idx].numpy())
        is_fit.append(np.full(n_win, rec_idx in fit_recs, dtype=bool))
        t_starts.append(dataset.t_start_recordings[rec_idx].numpy())
        task = dataset._recording_tasks[rec_idx]
        task_ids.append(np.full(n_win, dataset._task_to_idx[task], dtype=np.int64))
        embeds.append(dataset.embedding_recordings[rec_idx].numpy())
        # shot/scene ids are absent unless the dataset was built in recipe_mode;
        # -1 makes _eval_group_level drop those anchors rather than mis-pool them.
        shot_ids.append(dataset.shot_id_recordings[rec_idx].numpy().astype(np.int64)
                        if dataset.shot_id_recordings
                        else np.full(n_win, -1, dtype=np.int64))
        scene_ids.append(dataset.scene_id_recordings[rec_idx].numpy().astype(np.int64)
                         if dataset.scene_id_recordings
                         else np.full(n_win, -1, dtype=np.int64))
        if (i + 1) % 20 == 0:
            print(f"  cached {i+1}/{len(chosen)} recordings")

    cache = {
        "eeg": torch.cat(windows, dim=0),
        "Y": np.concatenate(Ys, axis=0),
        "is_fit": np.concatenate(is_fit, axis=0),
        "t_starts": np.concatenate(t_starts, axis=0).astype(np.float32),
        "task_ids": np.concatenate(task_ids, axis=0),
        "embeds": np.concatenate(embeds, axis=0).astype(np.float32),
        "shot_ids": np.concatenate(shot_ids, axis=0),
        "scene_ids": np.concatenate(scene_ids, axis=0),
        "rec_ids": np.concatenate(
            [np.full(len(w), r, dtype=np.int64) for w, r in zip(windows, chosen.tolist())]),
        "n_recordings": int(args.n_recordings),
        "n_fit_recordings": int(args.n_fit),
    }
    M = len(cache["eeg"])
    gb = cache["eeg"].element_size() * cache["eeg"].nelement() / 1e9
    print(f"  cache: M={M} windows ({cache['is_fit'].sum()} fit / "
          f"{(~cache['is_fit']).sum()} score), {gb:.2f} GB, {time.time()-t0:.0f}s")
    return cache


def _mask_groups(Y):
    """Group feature indices by identical NaN mask.

    Ridge on a shared mask is a multi-target solve, so 12 features usually cost
    one factorisation rather than 12. Grouping (rather than intersecting the
    masks) keeps this exactly equivalent to fitting each feature on its own
    valid rows, which is what probe_traintest.py does.
    """
    groups: dict[bytes, list[int]] = {}
    for i in range(Y.shape[1]):
        key = (~np.isnan(Y[:, i])).tobytes()
        groups.setdefault(key, []).append(i)
    return groups


def probe_score(X, cache, alpha, calibrate=False):
    """Mean Pearson r over the 12 features, head fit on the fit recordings."""
    Y, is_fit = cache["Y"], cache["is_fit"]
    scaler = StandardScaler().fit(X[is_fit])
    Xn = scaler.transform(X)
    per_feature, alphas = {}, {}
    for _key, feat_idx in _mask_groups(Y).items():
        valid = ~np.isnan(Y[:, feat_idx[0]])
        tr = valid & is_fit
        te = valid & ~is_fit
        if tr.sum() < 100 or te.sum() < 10:
            continue
        cols = np.array(feat_idx)
        y_tr = Y[np.ix_(tr, cols)]
        if calibrate:
            for j, c in enumerate(cols):
                reg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xn[tr], y_tr[:, j])
                alphas[SCALAR_FEATURES_DEFAULT[c]] = float(reg.alpha_)
                pred = reg.predict(Xn[te])
                per_feature[SCALAR_FEATURES_DEFAULT[c]] = _safe_r(pred, Y[te, c])
            continue
        reg = Ridge(alpha=alpha).fit(Xn[tr], y_tr)
        pred = np.atleast_2d(reg.predict(Xn[te]))
        if pred.shape[0] != te.sum():                 # single-target -> [n]
            pred = pred.reshape(te.sum(), -1)
        for j, c in enumerate(cols):
            per_feature[SCALAR_FEATURES_DEFAULT[c]] = _safe_r(pred[:, j], Y[te, c])
    rs = [v for v in per_feature.values() if not np.isnan(v)]
    out = {"probe_mean_r": float(np.mean(rs)) if rs else float("nan"),
           "probe_per_feature": per_feature}
    if calibrate:
        out["alphas"] = alphas
    return out


def _safe_r(pred, y):
    if np.std(pred) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(pred, y).statistic)


@torch.no_grad()
def retrieval_score(X, cache, clip_head, device, topks, t_bucket_s):
    """time/shot/scene e->v@1 through the trained head. Fits nothing."""
    eeg_batched = torch.from_numpy(X).to(device).view(-1, X.shape[1], 1, 1, 1)
    z_eeg = clip_head.project_eeg(eeg_batched).float().cpu().numpy()
    z_vis = clip_head.project_vision(
        torch.from_numpy(cache["embeds"]).to(device)).float().cpu().numpy()
    levels = _evaluate_all_levels(
        z_eeg, z_vis, cache["task_ids"], cache["t_starts"],
        cache["shot_ids"], cache["scene_ids"], topks, t_bucket_s,
    )
    return {f"e2v1_{lvl}": float(res["e2v_top_k"][1]) for lvl, res in levels.items()}


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_config(args.config)

    if args.split == "test":
        raise SystemExit(
            "Refusing to select on the test split -- that is the readout the "
            "paper reports. Use --split val (R5).")

    cells = args.cells or discover_cells(args.ckpt_root, args.cells_glob)
    if not cells:
        raise SystemExit(f"No cells matched under {args.ckpt_root}")
    print(f"{len(cells)} cell(s): {cells[0]} ... {cells[-1]}")

    print(f"Building {args.split.upper()} split (task={cfg.data.task}) ...")
    dataset = build_dataset(cfg, args.split, SCALAR_FEATURES_DEFAULT)
    print(f"  n_recordings={len(dataset)}, n_chans={dataset.n_chans}")

    print(f"Caching {args.n_recordings} recordings ...")
    cache = build_cache(dataset, args)

    encoder = build_encoder(
        cfg, n_chans=dataset.n_chans, n_times=dataset.n_times,
        chs_info=dataset.get_chs_info(), n_windows=cfg.data.n_windows,
    )
    clip_head = MovieCLIPHead(
        eeg_in_dim=cfg.model.encoder_embed_dim,
        vision_in_dim=dataset.frame_embedding_dim,
        proj_dim=int(cfg.loss.proj_dim),
        temperature=float(cfg.loss.temperature),
        drop_proj=float(cfg.loss.get("drop_proj", 0.5)),
        vision_passthrough=bool(cfg.loss.get("vision_passthrough", True)),
        n_residual_blocks=int(cfg.loss.get("n_residual_blocks", 1)),
    )
    encoder.to(device).eval()
    clip_head.to(device).eval()

    out = {
        "_meta": {
            "ckpt_root": args.ckpt_root, "config": args.config,
            "split": args.split, "seed": args.seed,
            "n_recordings": cache["n_recordings"],
            "n_fit_recordings": cache["n_fit_recordings"],
            "n_windows": int(len(cache["eeg"])),
            "alpha": None if args.calibrate_alpha else args.alpha,
            "features": SCALAR_FEATURES_DEFAULT,
        },
        "cells": {},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        cell_dir = Path(args.ckpt_root) / cell
        ckpts = checkpoints_for(cell_dir, args.epochs)
        if not ckpts:
            print(f"  {cell}: no matching epoch_*.pth.tar, skipped")
            continue
        out["cells"][cell] = {}
        for epoch, path in ckpts:
            t0 = time.time()
            enc_sd, head_sd = _load_clip_ckpt(str(path))
            encoder.load_state_dict(enc_sd, strict=False)
            clip_head.load_state_dict(head_sd, strict=False)
            eeg = cache["eeg"].float() if args.half else cache["eeg"]
            X = encode_windows(encoder, eeg, device, args.encode_batch)
            rec = probe_score(X, cache, args.alpha, calibrate=args.calibrate_alpha)
            rec.update(retrieval_score(X, cache, clip_head, device,
                                       args.topks, args.t_bucket_s))
            rec["seconds"] = round(time.time() - t0, 1)
            out["cells"][cell][str(epoch)] = rec
            print(f"  {cell} ep{epoch:<4} probe_r={rec['probe_mean_r']:+.4f}  "
                  f"e2v1_time={rec['e2v1_time']:.4f}  "
                  f"e2v1_scene={rec['e2v1_scene']:.4f}  ({rec['seconds']}s)")
            if args.calibrate_alpha:
                print(f"    alphas: {rec['alphas']}")
        # Written per cell, so a walltime kill keeps everything already computed.
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nWrote {args.output} "
          f"({len(out['cells'])} cell(s), "
          f"{sum(len(v) for v in out['cells'].values())} checkpoint(s))")


if __name__ == "__main__":
    main()
