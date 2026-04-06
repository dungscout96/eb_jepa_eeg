"""Diagnose encoder embedding quality: collapse detection, UMAP, temporal structure.

Usage (on Delta):
    PYTHONPATH=. .venv/bin/python scripts/diagnose_embeddings.py \
        --ckpt checkpoints/eeg_jepa/dev_2026-04-05_20-14/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed2025/latest.pth.tar
"""

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.architectures import EEGEncoderTokens
from eb_jepa.datasets.hbn import JEPAMovieDataset
from eb_jepa.training_utils import load_config


def load_encoder(ckpt_path, cfg, chs_info, n_chans, n_times):
    """Reconstruct encoder and load weights from checkpoint."""
    embed_dim = cfg.model.encoder_embed_dim
    encoder = EEGEncoderTokens(
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=embed_dim,
        depth=cfg.model.encoder_depth,
        heads=cfg.model.encoder_heads,
        head_dim=cfg.model.encoder_head_dim,
        n_windows=cfg.data.n_windows,
        patch_size=cfg.model.get("patch_size", 200),
        patch_overlap=cfg.model.get("patch_overlap", 20),
        freqs=cfg.model.get("freqs", 4),
        chs_info=chs_info,
        mlp_dim_ratio=cfg.model.get("mlp_dim_ratio", 2.66),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Extract context_encoder weights
    sd = {}
    prefix = "context_encoder."
    for k, v in ckpt["model_state_dict"].items():
        if k.startswith(prefix):
            sd[k[len(prefix):]] = v
    encoder.load_state_dict(sd, strict=False)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_embeddings(encoder, dataset, n_samples=200):
    """Extract embeddings and metadata from dataset."""
    all_embeddings = []
    all_features = []
    all_rec_ids = []

    n = min(n_samples, len(dataset))
    for i in range(n):
        eeg, features, *rest = dataset[i]
        # eeg: [n_windows, C, W]
        tokens = encoder.encode_tokens(eeg.unsqueeze(0))  # [1, n_tokens, embed_dim]
        all_embeddings.append(tokens.squeeze(0))  # [n_tokens, embed_dim]
        all_features.append(features)
        all_rec_ids.append(i)

    embeddings = torch.cat(all_embeddings, dim=0)  # [total_tokens, embed_dim]
    return embeddings, all_features, all_rec_ids


def collapse_diagnostics(embeddings, out_dir):
    """Check for representation collapse."""
    print("\n=== COLLAPSE DIAGNOSTICS ===")
    E = embeddings.float()

    # 1. Per-dimension variance
    var = E.var(dim=0)
    print(f"Embedding shape: {E.shape}")
    print(f"Per-dim variance: mean={var.mean():.4f}, min={var.min():.4f}, max={var.max():.4f}")

    # 2. Effective dimensionality (participation ratio)
    cov = torch.cov(E.T)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=0)
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    print(f"Effective dimensionality (participation ratio): {pr:.1f} / {E.shape[1]}")

    # 3. Cosine similarity distribution
    n_pairs = min(5000, len(E) * (len(E) - 1) // 2)
    idx1 = torch.randint(0, len(E), (n_pairs,))
    idx2 = torch.randint(0, len(E), (n_pairs,))
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    cosim = torch.nn.functional.cosine_similarity(E[idx1], E[idx2], dim=1)
    print(f"Cosine similarity: mean={cosim.mean():.4f}, std={cosim.std():.4f}, "
          f"min={cosim.min():.4f}, max={cosim.max():.4f}")

    # 4. Embedding norm distribution
    norms = E.norm(dim=1)
    print(f"Embedding norms: mean={norms.mean():.4f}, std={norms.std():.4f}")

    # Verdict
    if cosim.mean() > 0.9:
        print("\n** COLLAPSE DETECTED: cosim mean > 0.9 — embeddings are nearly identical **")
    elif cosim.mean() > 0.7:
        print("\n** WARNING: High cosim mean > 0.7 — possible partial collapse **")
    elif pr < E.shape[1] / 4:
        print(f"\n** WARNING: Low effective dim ({pr:.0f}/{E.shape[1]}) — dimensional collapse **")
    else:
        print("\n** OK: No obvious collapse detected **")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(range(len(var)), var.numpy())
    axes[0].set_title("Per-dimension variance")
    axes[0].set_xlabel("Dimension")

    axes[1].hist(cosim.numpy(), bins=50, edgecolor="black")
    axes[1].axvline(cosim.mean(), color="red", linestyle="--", label=f"mean={cosim.mean():.3f}")
    axes[1].set_title("Cosine similarity distribution")
    axes[1].legend()

    axes[2].bar(range(len(eigenvalues)), eigenvalues.flip(0).numpy())
    axes[2].set_title(f"Eigenvalue spectrum (PR={pr:.1f})")
    axes[2].set_xlabel("Component")

    plt.tight_layout()
    plt.savefig(out_dir / "collapse_diagnostics.png", dpi=150)
    print(f"Saved: {out_dir / 'collapse_diagnostics.png'}")


def umap_visualization(embeddings, all_features, out_dir):
    """UMAP colored by movie features and recording ID."""
    try:
        from umap import UMAP
    except ImportError:
        print("umap-learn not installed, skipping UMAP. Install with: pip install umap-learn")
        return

    print("\n=== UMAP VISUALIZATION ===")
    # Subsample if too many
    E = embeddings.float().numpy()
    if len(E) > 5000:
        idx = np.random.choice(len(E), 5000, replace=False)
        E = E[idx]

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    coords = reducer.fit_transform(E)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by token index (proxy for temporal position)
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=np.arange(len(E)), cmap="viridis", s=3, alpha=0.5)
    axes[0].set_title("UMAP colored by token index (time proxy)")
    plt.colorbar(sc, ax=axes[0])

    # Color by embedding norm (collapse indicator)
    norms = np.linalg.norm(E, axis=1)
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=norms, cmap="plasma", s=3, alpha=0.5)
    axes[1].set_title("UMAP colored by embedding norm")
    plt.colorbar(sc, ax=axes[1])

    plt.tight_layout()
    plt.savefig(out_dir / "umap_embeddings.png", dpi=150)
    print(f"Saved: {out_dir / 'umap_embeddings.png'}")


def temporal_analysis(encoder, dataset, out_dir, n_recordings=5):
    """Analyze temporal structure within individual recordings."""
    print("\n=== TEMPORAL ANALYSIS ===")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    all_autocorrs = []

    for rec_idx in range(min(n_recordings, len(dataset))):
        eeg, features, *rest = dataset[rec_idx]
        with torch.no_grad():
            tokens = encoder.encode_tokens(eeg.unsqueeze(0))  # [1, n_tokens, D]
        # Average tokens per window position
        emb = tokens.squeeze(0)  # [n_tokens, D]
        # PCA to 3D for trajectory
        emb_centered = emb - emb.mean(0)
        U, S, V = torch.svd(emb_centered)
        pca3 = (emb_centered @ V[:, :3]).numpy()

        axes[0].plot(pca3[:, 0], label=f"Rec {rec_idx}" if rec_idx < 3 else None, alpha=0.7)

        # Autocorrelation of first PC
        pc1 = pca3[:, 0]
        pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)
        autocorr = np.correlate(pc1, pc1, mode="full")
        autocorr = autocorr[len(pc1)-1:] / len(pc1)
        all_autocorrs.append(autocorr[:min(20, len(autocorr))])

    axes[0].set_title("PC1 over time (per recording)")
    axes[0].set_xlabel("Token position (time)")
    axes[0].set_ylabel("PC1 value")
    axes[0].legend()

    # Average autocorrelation
    min_len = min(len(a) for a in all_autocorrs)
    avg_autocorr = np.mean([a[:min_len] for a in all_autocorrs], axis=0)
    axes[1].bar(range(min_len), avg_autocorr)
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set_title("Avg autocorrelation of PC1 (across recordings)")
    axes[1].set_xlabel("Lag (tokens)")
    axes[1].set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.savefig(out_dir / "temporal_analysis.png", dpi=150)
    print(f"Saved: {out_dir / 'temporal_analysis.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--config", default="experiments/eeg_jepa/cfgs/default.yaml")
    parser.add_argument("--out-dir", default=None, help="Output directory for plots")
    parser.add_argument("--n-samples", type=int, default=100, help="Recordings to analyze")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if args.out_dir is None:
        out_dir = ckpt_path.parent / "diagnostics"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config with same overrides as training
    cfg = load_config(args.config, {
        "model.encoder_depth": 2,
        "optim.lr": 5e-4,
        "loss.std_coeff": 0.25,
        "loss.cov_coeff": 0.25,
    })

    print("Loading validation dataset...")
    val_set = JEPAMovieDataset(
        split="val",
        n_windows=cfg.data.n_windows,
        window_size_seconds=cfg.data.window_size_seconds,
        cfg=cfg.data,
        preprocessed=cfg.data.get("preprocessed", False),
    )

    print("Loading encoder from checkpoint...")
    encoder = load_encoder(
        ckpt_path, cfg,
        chs_info=val_set.get_chs_info(),
        n_chans=val_set.n_chans,
        n_times=val_set.n_times,
    )

    print(f"Extracting embeddings from {min(args.n_samples, len(val_set))} recordings...")
    embeddings, all_features, all_rec_ids = extract_embeddings(encoder, val_set, args.n_samples)

    collapse_diagnostics(embeddings, out_dir)
    umap_visualization(embeddings, all_features, out_dir)
    temporal_analysis(encoder, val_set, out_dir)

    print(f"\nAll diagnostics saved to: {out_dir}")


if __name__ == "__main__":
    main()
