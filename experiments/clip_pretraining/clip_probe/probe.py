"""Per-window probe: does the EEG encoder capture the time-varying visual content?

Each (recording, window_t) becomes one datapoint. The encoder embeds the EEG
window; the regression target is the scalar movie feature at that window's
time (luminance, motion_energy, etc.). Cross-validation splits BY RECORDING,
so the held-out fold contains unseen subjects.

This corrects the per-recording-averaged probe, where averaging over random
crops washed out the time-varying signal we actually want to detect.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/clip_pretraining/clip_probe/probe.py \\
        --checkpoint /path/to/latest.pth.tar \\
        --config experiments/clip_pretraining/clip_probe/configs/qxhl9rfl.yaml \\
        --device cuda --output probe_results/qxhl9rfl_trained.json

Use --random-baseline to skip the checkpoint load and probe a fresh-init
encoder of the same architecture (the null/noise floor).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_config

SCALAR_FEATURES_DEFAULT = [
    "luminance_mean", "contrast_rms", "edge_density", "saturation_mean",
    "entropy", "motion_energy", "n_faces", "face_area_frac",
    "depth_mean", "scene_natural_score",
    "position_in_movie", "narrative_event_score",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Path to .pth.tar; omit and use --random-baseline for the null")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", nargs="+", default=SCALAR_FEATURES_DEFAULT)
    ap.add_argument("--cv-splits", type=int, default=5,
                    help="GroupKFold splits (groups = recordings)")
    ap.add_argument("--encode-batch", type=int, default=64,
                    help="Windows per encoder forward pass within a recording")
    ap.add_argument("--max-recordings", type=int, default=None,
                    help="Cap n recordings for quick smoke runs")
    ap.add_argument("--output", default="probe_results.json")
    ap.add_argument("--random-baseline", action="store_true")
    return ap.parse_args()


def build_dataset(cfg, split, features):
    return JEPAMovieDataset(
        split=split,
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        task=cfg.data.task,
        temporal_stride=cfg.data.get("temporal_stride", 1),
        feature_names=features,  # populates dataset.feature_recordings
        cfg=cfg.data,
        preprocessed=cfg.data.preprocessed,
        preprocessed_dir=cfg.data.get("preprocessed_dir", None),
        visual_processing_delay_s=0.0,
    )


def load_encoder_state(encoder, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    print(f"  loaded {len(enc_sd)} encoder tensors  missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]: print(f"  missing (first 3): {missing[:3]}")
    if unexpected[:3]: print(f"  unexpected (first 3): {unexpected[:3]}")


@torch.no_grad()
def embed_recording_all_windows(encoder, dataset, rec_idx, device, batch_size):
    """Encode every window in one recording. Returns (X[n_win, D], Y[n_win, n_features])."""
    crop_inds = dataset._crop_inds[rec_idx]
    fif_path = dataset._fif_paths[rec_idx]
    raw = _read_raw_windows(fif_path, crop_inds)            # [n_win, C, T] numpy float32
    eeg = torch.from_numpy(raw)
    # Apply per-recording norm using stats from ALL of this recording's windows
    # (matches the training-time per_recording norm when n_windows=1 — at training
    # time __getitem__ also computes per-recording stats from the sampled windows).
    if dataset._norm_mode == "per_recording":
        rec_mean = eeg.mean(dim=(0, 2), keepdim=True)
        rec_std = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
        eeg = (eeg - rec_mean) / rec_std
    else:
        eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std

    # Treat each window as its own "trial" with n_windows=1 (matches training shape).
    eeg_in = eeg.unsqueeze(1)                                # [n_win, 1, C, T]
    embs = []
    for start in range(0, len(eeg_in), batch_size):
        batch = eeg_in[start:start + batch_size].to(device)
        tokens = encoder.encode_tokens(batch, mask=None)
        pooled = encoder.pool_to_windows(tokens)             # [B, D, 1, 1, 1]
        emb = pooled.squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy()  # [B, D]
        embs.append(emb)
    X = np.concatenate(embs, axis=0)                          # [n_win, D]
    Y = dataset.feature_recordings[rec_idx].numpy()           # [n_win, n_features]
    return X, Y


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = load_config(args.config)
    print(f"Loading {args.split} split for task={cfg.data.task} ...")
    dataset = build_dataset(cfg, args.split, SCALAR_FEATURES_DEFAULT)
    print(f"  n_recordings={len(dataset)}, n_chans={dataset.n_chans}, n_times={dataset.n_times}")

    encoder = build_encoder(
        cfg, n_chans=dataset.n_chans, n_times=dataset.n_times,
        chs_info=dataset.get_chs_info(), n_windows=cfg.data.n_windows,
    )
    if not args.random_baseline:
        assert args.checkpoint, "Provide --checkpoint or use --random-baseline"
        print(f"Loading encoder weights from {args.checkpoint}")
        load_encoder_state(encoder, args.checkpoint)
    else:
        print("Using RANDOM-INIT encoder (no checkpoint loaded)")
    encoder = encoder.to(device).eval()

    n_rec = len(dataset) if args.max_recordings is None else min(args.max_recordings, len(dataset))
    print(f"\nEmbedding ALL windows of {n_rec} recordings (per-window probe) ...")
    Xs, Ys, groups = [], [], []
    for rec_idx in range(n_rec):
        X_rec, Y_rec = embed_recording_all_windows(
            encoder, dataset, rec_idx, device, args.encode_batch,
        )
        Xs.append(X_rec); Ys.append(Y_rec); groups.append(np.full(len(X_rec), rec_idx))
        if (rec_idx + 1) % 25 == 0:
            print(f"  {rec_idx+1}/{n_rec} done  (total windows so far: {sum(len(x) for x in Xs)})")
    X = np.concatenate(Xs, axis=0)        # [N_windows_total, D]
    Y = np.concatenate(Ys, axis=0)        # [N_windows_total, n_features]
    g = np.concatenate(groups, axis=0)    # [N_windows_total] recording id

    print(f"\nFinal: X={X.shape}, Y={Y.shape}, n_recordings={len(np.unique(g))}")
    print(f"Per-feature target variance (across windows):")
    for i, feat in enumerate(SCALAR_FEATURES_DEFAULT):
        if feat in args.features:
            valid = ~np.isnan(Y[:, i])
            print(f"  {feat:<26}  std={Y[valid, i].std():.4f}  (n={valid.sum()})")

    Xs_norm = StandardScaler().fit_transform(X)
    results = {"random_baseline": args.random_baseline, "split": args.split,
               "n_recordings": int(len(np.unique(g))), "n_windows_total": int(len(X)),
               "encoder_dim": int(X.shape[1]),
               "features": {}}

    print(f"\n{'feature':<26}  {'R^2 mean':>10}  {'R^2 std':>10}  {'n':>7}")
    print("-" * 67)
    cv = GroupKFold(n_splits=args.cv_splits)
    for i, feat in enumerate(SCALAR_FEATURES_DEFAULT):
        if feat not in args.features:
            continue
        y = Y[:, i]
        valid = ~np.isnan(y)
        if valid.sum() < 100 or y[valid].std() < 1e-9:
            print(f"{feat:<26}  [insufficient]")
            continue
        model = RidgeCV(alphas=np.logspace(-2, 4, 13))
        scores = cross_val_score(
            model, Xs_norm[valid], y[valid],
            groups=g[valid], cv=cv, scoring="r2", n_jobs=1,
        )
        r2_mean, r2_std = float(scores.mean()), float(scores.std())
        print(f"{feat:<26}  {r2_mean:>10.4f}  {r2_std:>10.4f}  {int(valid.sum()):>7}")
        results["features"][feat] = {"r2_mean": r2_mean, "r2_std": r2_std,
                                     "n": int(valid.sum())}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
