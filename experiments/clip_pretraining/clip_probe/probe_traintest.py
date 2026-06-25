"""ImageNet-style linear probe: fit on TRAIN, evaluate on TEST.

Differs from `probe.py` (which does GroupKFold-by-recording within a single
split) by using the train release(s) entirely for fitting the linear head and
the test release for evaluation only. Matches the SSL literature's convention
(SimCLR/MoCo/CLIP/DINOv2 linear-eval protocol on ImageNet) and removes the
small-fold variance that biases clip_probe's test estimates downward.

Loads CLIP checkpoints (encoder weights only) like probe.py, so it works on
both CLIPPretrain and SceneCLIPPretrain runs.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/clip_pretraining/clip_probe/probe_traintest.py \\
        --checkpoint /abs/path/latest.pth.tar \\
        --config experiments/clip_pretraining/clip_probe/configs/recipe_depth4.yaml \\
        --device cuda \\
        --output probe_results/probe_traintest_<name>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from experiments.clip_pretraining.clip_probe.probe import (
    SCALAR_FEATURES_DEFAULT,
    build_dataset,
    embed_recording_all_windows,
    load_encoder_state,
)
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Path to .pth.tar; omit and use --random-baseline for the null")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", nargs="+", default=SCALAR_FEATURES_DEFAULT)
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--max-train-recordings", type=int, default=None,
                    help="Cap n train recordings for smoke runs")
    ap.add_argument("--output", default="probe_traintest.json")
    ap.add_argument("--random-baseline", action="store_true")
    return ap.parse_args()


def embed_split(encoder, dataset, device, batch_size, max_recordings=None):
    """Embed every window of every recording in a dataset."""
    n_rec = len(dataset) if max_recordings is None else min(max_recordings, len(dataset))
    Xs, Ys = [], []
    for rec_idx in range(n_rec):
        X_rec, Y_rec = embed_recording_all_windows(
            encoder, dataset, rec_idx, device, batch_size,
        )
        Xs.append(X_rec)
        Ys.append(Y_rec)
        if (rec_idx + 1) % 50 == 0:
            print(f"  {rec_idx+1}/{n_rec} done  (windows so far: {sum(len(x) for x in Xs)})")
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = load_config(args.config)

    print(f"Building TRAIN split (task={cfg.data.task}) ...")
    train_set = build_dataset(cfg, "train", SCALAR_FEATURES_DEFAULT)
    print(f"  n_recordings={len(train_set)}, n_chans={train_set.n_chans}")

    print(f"Building TEST split ...")
    test_set = build_dataset(cfg, "test", SCALAR_FEATURES_DEFAULT)
    print(f"  n_recordings={len(test_set)}")

    encoder = build_encoder(
        cfg, n_chans=train_set.n_chans, n_times=train_set.n_times,
        chs_info=train_set.get_chs_info(), n_windows=cfg.data.n_windows,
    )
    if not args.random_baseline:
        assert args.checkpoint, "Provide --checkpoint or use --random-baseline"
        print(f"Loading encoder weights from {args.checkpoint}")
        load_encoder_state(encoder, args.checkpoint)
    else:
        print("Using RANDOM-INIT encoder (no checkpoint loaded)")
    encoder = encoder.to(device).eval()

    print(f"\nEmbedding TRAIN recordings ({len(train_set) if not args.max_train_recordings else args.max_train_recordings}) ...")
    X_tr, Y_tr = embed_split(encoder, train_set, device, args.encode_batch, args.max_train_recordings)
    print(f"  TRAIN  X={X_tr.shape}  Y={Y_tr.shape}")

    print(f"\nEmbedding TEST recordings ({len(test_set)}) ...")
    X_te, Y_te = embed_split(encoder, test_set, device, args.encode_batch)
    print(f"  TEST   X={X_te.shape}  Y={Y_te.shape}")

    # Standardize features using train stats only (no test leakage).
    scaler = StandardScaler().fit(X_tr)
    X_tr_n = scaler.transform(X_tr)
    X_te_n = scaler.transform(X_te)

    results = {
        "random_baseline": args.random_baseline,
        "n_train_recordings": int(len(train_set) if not args.max_train_recordings else args.max_train_recordings),
        "n_test_recordings": int(len(test_set)),
        "n_train_windows": int(X_tr.shape[0]),
        "n_test_windows": int(X_te.shape[0]),
        "encoder_dim": int(X_tr.shape[1]),
        "features": {},
    }

    print(f"\n{'feature':<26}  {'R² (test)':>10}  {'pearson r':>10}  {'alpha':>10}")
    print('-' * 64)
    for i, feat in enumerate(SCALAR_FEATURES_DEFAULT):
        if feat not in args.features:
            continue
        y_tr = Y_tr[:, i]
        y_te = Y_te[:, i]
        valid_tr = ~np.isnan(y_tr)
        valid_te = ~np.isnan(y_te)
        if valid_tr.sum() < 100 or y_tr[valid_tr].std() < 1e-9:
            print(f"{feat:<26}  [insufficient train]")
            continue
        if valid_te.sum() < 10 or y_te[valid_te].std() < 1e-9:
            print(f"{feat:<26}  [insufficient test]")
            continue
        # RidgeCV picks the best alpha by internal LOO-style CV on the train fit
        reg = RidgeCV(alphas=np.logspace(-2, 4, 13))
        reg.fit(X_tr_n[valid_tr], y_tr[valid_tr])
        pred_te = reg.predict(X_te_n[valid_te])
        r2 = float(r2_score(y_te[valid_te], pred_te))
        try:
            r = float(pearsonr(pred_te, y_te[valid_te]).statistic)
        except Exception:
            r = float("nan")
        print(f"{feat:<26}  {r2:+10.4f}  {r:+10.4f}  {float(reg.alpha_):>10.3g}")
        results["features"][feat] = {
            "r2": r2, "pearson_r": r, "alpha": float(reg.alpha_),
            "n_train": int(valid_tr.sum()), "n_test": int(valid_te.sum()),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
