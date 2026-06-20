"""Did the EEG encoder actually learn anything during CLIP pretraining?

Loads the encoder weights from a CLIPPretrain (or MaskedJEPA) checkpoint, encodes
every val-split recording into a per-recording embedding (mean over n random
window passes), then runs ridge regression against scalar movie features and
reports cross-validated R^2. Compares against a fresh-random-init encoder of
the same architecture so we can tell whether the trained encoder actually beat
the noise floor.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/clip_probe/probe.py \\
        --checkpoint /path/to/latest.pth.tar \\
        --config config/clip_pretrain.yaml \\
        --split val --n-passes 10 --device cuda

Use --random-baseline to skip the load and probe a freshly initialised encoder
(same config, no weights) so you can compare the two side by side.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from eb_jepa.datasets.hbn import JEPAMovieDataset
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
    ap.add_argument("--checkpoint", required=False, default=None,
                    help="Path to .pth.tar checkpoint; omit + use --random-baseline")
    ap.add_argument("--config", required=True, help="YAML used to train the run")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n-passes", type=int, default=10,
                    help="Random window crops to average per recording (probe_eval uses 20)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", nargs="+", default=SCALAR_FEATURES_DEFAULT)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--output", default="probe_results.json")
    ap.add_argument("--random-baseline", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def build_dataset(cfg, split):
    return JEPAMovieDataset(
        split=split,
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        task=cfg.data.task,
        temporal_stride=cfg.data.get("temporal_stride", 1),
        feature_names=SCALAR_FEATURES_DEFAULT,  # we want them in the per-window labels
        cfg=cfg.data,
        preprocessed=cfg.data.preprocessed,
        preprocessed_dir=cfg.data.get("preprocessed_dir", None),
        visual_processing_delay_s=0.0,
    )


def load_encoder_state(encoder, ckpt_path):
    """Load encoder.* keys from a checkpoint into the encoder. Supports both
    MaskedJEPA and CLIPPretrain checkpoints — both store encoder weights under
    the encoder.* prefix."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    print(f"  loaded {len(enc_sd)} encoder tensors  missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"  (missing keys, first 5): {missing[:5]}")
        print(f"  (unexpected, first 5): {unexpected[:5]}")


@torch.no_grad()
def embed_recording(encoder, dataset, rec_idx, n_passes, device, seed):
    """Encode one recording n_passes times with different random crops; mean-pool."""
    embeds, scalars = [], []
    for p in range(n_passes):
        torch.manual_seed(seed * 1_000_003 + p * 7919 + rec_idx)
        eeg, feats, *_ = dataset[rec_idx]            # eeg: [n_windows, C, T]
        eeg = eeg.unsqueeze(0).to(device)            # [1, n_windows, C, T]
        tokens = encoder.encode_tokens(eeg, mask=None)
        pooled = encoder.pool_to_windows(tokens)     # [1, D, T, 1, 1]
        emb = pooled.squeeze(-1).squeeze(-1).mean(dim=2).squeeze(0).cpu().numpy()  # [D]
        embeds.append(emb)
        scalars.append(feats.mean(dim=0).numpy())   # [n_features]
    return np.stack(embeds).mean(0), np.stack(scalars).mean(0)


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = load_config(args.config)
    print(f"Loading {args.split} split for task={cfg.data.task} ...")
    dataset = build_dataset(cfg, split=args.split)
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

    print(f"\nEmbedding {len(dataset)} recordings with {args.n_passes} passes each ...")
    Xs, Ys = [], []
    for rec_idx in range(len(dataset)):
        emb, scal = embed_recording(encoder, dataset, rec_idx, args.n_passes, device, seed=args.seed)
        Xs.append(emb); Ys.append(scal)
        if (rec_idx + 1) % 25 == 0:
            print(f"  {rec_idx+1}/{len(dataset)} done")
    X = np.stack(Xs)               # [n_rec, D]
    Y = np.stack(Ys)               # [n_rec, n_features]
    print(f"X shape={X.shape}, Y shape={Y.shape}")

    Xs_norm = StandardScaler().fit_transform(X)
    results = {"random_baseline": args.random_baseline, "n_passes": args.n_passes,
               "split": args.split, "n_recordings": len(dataset),
               "features": {}}

    print(f"\n{'feature':<26}  {'R^2 mean':>10}  {'R^2 std':>10}  {'n':>5}")
    print("-" * 65)
    cv = KFold(n_splits=args.cv_splits, shuffle=True, random_state=0)
    for i, feat in enumerate(SCALAR_FEATURES_DEFAULT):
        if feat not in args.features:
            continue
        y = Y[:, i]
        valid = ~np.isnan(y)
        if valid.sum() < args.cv_splits + 1 or y[valid].std() < 1e-9:
            print(f"{feat:<26}  [insufficient]")
            continue
        model = RidgeCV(alphas=np.logspace(-2, 4, 13))
        scores = cross_val_score(model, Xs_norm[valid], y[valid], cv=cv, scoring="r2")
        r2_mean, r2_std = float(scores.mean()), float(scores.std())
        print(f"{feat:<26}  {r2_mean:>10.4f}  {r2_std:>10.4f}  {int(valid.sum()):>5}")
        results["features"][feat] = {"r2_mean": r2_mean, "r2_std": r2_std,
                                     "n": int(valid.sum())}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
