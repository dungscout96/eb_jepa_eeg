"""Diagnose whether the movie_id failure is probe-side or encoder-side.

Loads one JEPA encoder seed, extracts per-clip embeddings on train/val/test
with --keep_channels=True, and runs three movie_id probes:

  Test A — sklearn LogisticRegression(L-BFGS, multinomial, C swept on val)
  Test B — torch 2-layer MLP head with AdamW + WD, 200 epochs, lr swept on val
  Test C — current production probe (single Linear, Adam lr=1e-3, 20 epochs)
           for sanity reference

Reports test top1/top5 for each. If A or B beats C by >= 0.05 absolute,
the failure was probe-side. If all three agree at chance, the encoder
genuinely doesn't have segment-discriminative info.
"""

import argparse
import math
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import EEGEncoderTokens
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config, setup_device, setup_seed
from experiments.eeg_jepa.main import resolve_preprocessed_dir

logger = get_logger(__name__)


def _build_encoder(ckpt_path: str, train_set: JEPAMovieDataset, cfg, device) -> EEGEncoderTokens:
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = sd.get("model_state_dict", sd)
    enc = EEGEncoderTokens(
        n_chans=train_set.n_chans,
        n_times=train_set.n_times,
        embed_dim=cfg.model.encoder_embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=cfg.data.n_windows,
        patch_size=cfg.model.get("patch_size", 50),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=train_set.get_chs_info(),
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
    ).to(device)
    ce_sd = {k[len("context_encoder."):]: v for k, v in sd.items()
             if k.startswith("context_encoder.")}
    enc.load_state_dict(ce_sd, strict=False)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _per_clip_embed(loader, encoder, device, pos_idx) -> tuple[np.ndarray, np.ndarray]:
    """Return per-clip embeddings (mean of windows) and per-clip positions.

    Inlines keep_channels=True pooling: encode_tokens → mean over patches →
    concat C into D → mean over windows → [D' = C*D].
    """
    xs, pos = [], []
    for eeg, features, _ in tqdm(loader, desc="embedding", leave=False):
        eeg = eeg.to(device)
        with torch.no_grad():
            tokens = encoder.encode_tokens(eeg, mask=None)  # [B, C*T*P, D]
            B = tokens.shape[0]
            C, T, P, D = encoder.n_chans, encoder.n_windows, encoder.n_patches_per_window, encoder.embed_dim
            x_tok = tokens.view(B, C, T, P, D)
            pooled = x_tok.mean(dim=3)                      # [B, C, T, D]
            pooled = pooled.permute(0, 2, 1, 3).reshape(B, T, C * D)
            per_clip = pooled.mean(dim=1)                   # [B, C*D] mean over windows
        xs.append(per_clip.cpu().numpy())
        pos.append(features[:, :, pos_idx].mean(dim=1).numpy())
    return np.concatenate(xs), np.concatenate(pos)


def run(
    checkpoint: str,
    seed: int = 42,
    n_bins: int = 20,
    fname: str = "experiments/eeg_jepa/cfgs/default.yaml",
    out_json: str = "",
):
    setup_seed(seed)
    device = setup_device("auto")

    overrides = {
        "data.n_windows": 4, "data.window_size_seconds": 2,
        "data.batch_size": 64, "data.num_workers": 4,
        "data.norm_mode": "per_recording",
        "data.corrca_filters": "corrca_filters.npz",
    }
    cfg = load_config(fname, overrides)
    pre_dir = resolve_preprocessed_dir(cfg.data.get("preprocessed_dir", None))
    pre = cfg.data.get("preprocessed", False)
    feature_names = list(cfg.data.get("feature_names", JEPAMovieDataset.DEFAULT_FEATURES))
    pos_idx = feature_names.index("position_in_movie")

    logger.info("Loading datasets...")
    train_set = JEPAMovieDataset(
        split="train", n_windows=4, window_size_seconds=2,
        feature_names=feature_names, cfg=cfg.data,
        preprocessed=pre, preprocessed_dir=pre_dir,
    )
    eval_sets = {}
    for split in ("val", "test"):
        eval_sets[split] = JEPAMovieDataset(
            split=split, n_windows=4, window_size_seconds=2,
            feature_names=feature_names,
            eeg_norm_stats=train_set.get_eeg_norm_stats(),
            cfg=cfg.data, preprocessed=pre, preprocessed_dir=pre_dir,
        )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=4)
    val_loader = DataLoader(eval_sets["val"], batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(eval_sets["test"], batch_size=64, shuffle=False, num_workers=4)

    encoder = _build_encoder(checkpoint, train_set, cfg, device)
    logger.info("Encoder loaded; D = %d (= C*embed_dim under keep_channels)",
                encoder.n_chans * encoder.embed_dim)

    # Train embed: 1 random clip per recording per epoch × 5 passes for stability
    logger.info("Embedding train (5 random-clip passes)...")
    Xs, Ys = [], []
    for _ in range(5):
        x, y = _per_clip_embed(train_loader, encoder, device, pos_idx)
        Xs.append(x); Ys.append(y)
    X_train = np.concatenate(Xs); y_train_pos = np.concatenate(Ys)
    logger.info("Embedding val...")
    X_val, y_val_pos = _per_clip_embed(val_loader, encoder, device, pos_idx)
    logger.info("Embedding test...")
    X_test, y_test_pos = _per_clip_embed(test_loader, encoder, device, pos_idx)

    # Bin edges from train positions (matches probe_eval.py's _train_movie_id_probe)
    bin_edges = np.linspace(y_train_pos.min(), y_train_pos.max() + 1e-8, n_bins + 1)
    y_train = np.clip(np.digitize(y_train_pos, bin_edges) - 1, 0, n_bins - 1)
    y_val = np.clip(np.digitize(y_val_pos, bin_edges) - 1, 0, n_bins - 1)
    y_test = np.clip(np.digitize(y_test_pos, bin_edges) - 1, 0, n_bins - 1)

    # Standardize features (helps L-BFGS + Adam alike)
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True).clip(min=1e-8)
    Xtr_n = (X_train - mu) / sd
    Xv_n = (X_val - mu) / sd
    Xte_n = (X_test - mu) / sd

    def _topk(logits, y, k):
        topk_idx = np.argsort(-logits, axis=1)[:, :k]
        return float((topk_idx == y[:, None]).any(axis=1).mean())

    results = {}

    # ---- Test A: sklearn LogReg with C swept on val ----
    logger.info("=== Test A: sklearn LogReg, C-sweep on val ===")
    best_C, best_val_top1, best_logits_test = None, -1.0, None
    for C in (1e-4, 1e-2, 1, 1e2, 1e4):
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000).fit(Xtr_n, y_train)
        val_logits = clf.decision_function(Xv_n)
        if val_logits.ndim == 1:  # binary edge case
            val_logits = np.column_stack([-val_logits, val_logits])
        v1 = _topk(val_logits, y_val, 1)
        logger.info("  C=%.0e: val_top1=%.4f", C, v1)
        if v1 > best_val_top1:
            best_val_top1, best_C = v1, C
            best_logits_test = clf.decision_function(Xte_n)
    if best_logits_test.ndim == 1:
        best_logits_test = np.column_stack([-best_logits_test, best_logits_test])
    results["A_logreg"] = {
        "best_C": best_C,
        "test_top1": _topk(best_logits_test, y_test, 1),
        "test_top5": _topk(best_logits_test, y_test, 5),
    }

    # ---- Test B: torch 2-layer MLP, lr-sweep on val ----
    logger.info("=== Test B: 2-layer MLP head, lr+wd sweep on val ===")
    D = X_train.shape[1]
    Xtr_t = torch.from_numpy(Xtr_n).float().to(device)
    ytr_t = torch.from_numpy(y_train).long().to(device)
    Xv_t = torch.from_numpy(Xv_n).float().to(device)
    yv_t = torch.from_numpy(y_val).long().to(device)
    Xte_t = torch.from_numpy(Xte_n).float().to(device)
    yte_t = torch.from_numpy(y_test).long().to(device)
    best_test_top1 = -1.0
    best_cfg = None
    best_logits_B = None
    for lr in (1e-3, 3e-4, 1e-4):
        for wd in (1e-4, 1e-3, 1e-2):
            torch.manual_seed(seed)
            head = nn.Sequential(
                nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, n_bins),
            ).to(device)
            opt = AdamW(head.parameters(), lr=lr, weight_decay=wd)
            best_v_top1 = -1
            for ep in range(200):
                head.train()
                perm = torch.randperm(len(Xtr_t), device=device)
                for s in range(0, len(perm), 512):
                    idx = perm[s:s+512]
                    opt.zero_grad()
                    loss = F.cross_entropy(head(Xtr_t[idx]), ytr_t[idx])
                    loss.backward()
                    opt.step()
                if (ep + 1) % 20 == 0:
                    head.eval()
                    with torch.no_grad():
                        vl = head(Xv_t).cpu().numpy()
                    v1 = _topk(vl, y_val, 1)
                    if v1 > best_v_top1:
                        best_v_top1 = v1
                        with torch.no_grad():
                            tl = head(Xte_t).cpu().numpy()
                        if v1 > best_test_top1:
                            best_test_top1 = v1
                            best_cfg = (lr, wd, ep + 1)
                            best_logits_B = tl
            logger.info("  lr=%.0e wd=%.0e: best_val_top1=%.4f", lr, wd, best_v_top1)
    results["B_mlp"] = {
        "best_cfg": best_cfg,
        "test_top1": _topk(best_logits_B, y_test, 1),
        "test_top5": _topk(best_logits_B, y_test, 5),
    }

    # ---- Test C: production single Linear, Adam lr=1e-3, 20 epochs (no sweep) ----
    logger.info("=== Test C: production single Linear, Adam lr=1e-3, 20 epochs ===")
    torch.manual_seed(seed)
    head = nn.Linear(D, n_bins).to(device)
    opt = Adam(head.parameters(), lr=1e-3)
    for ep in range(20):
        head.train()
        opt.zero_grad()
        loss = F.cross_entropy(head(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        tl = head(Xte_t).cpu().numpy()
    results["C_production_linear"] = {
        "test_top1": _topk(tl, y_test, 1),
        "test_top5": _topk(tl, y_test, 5),
    }

    print("\n=== movie_id diagnostic results (test top1 / top5) ===")
    print(f"{'probe':<30} {'top1':>10} {'top5':>10}  chance: 0.050 / 0.250")
    for name, r in results.items():
        print(f"{name:<30} {r['test_top1']:>10.4f} {r['test_top5']:>10.4f}")

    if out_json:
        import json
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    fire.Fire(run)
