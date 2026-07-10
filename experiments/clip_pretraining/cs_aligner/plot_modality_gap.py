"""Figure-1-style modality-gap visualization for EEG ↔ V-JEPA-2 CLIP.

Reproduces the two-panel t-SNE from CS-Aligner (Yin et al., 2025,
arXiv:2502.17028, Fig. 1): each panel projects the EEG-window embeddings and
their matched V-JEPA-2 target vectors into a shared space via ``MovieCLIPHead``,
then runs t-SNE on the concatenation and colors points by modality. A
persistent modality gap shows up as two spatially separated clusters; a well-
aligned model shows interleaved / overlapping clusters.

Panel (a) = "before" reference (random-init encoder + head by default, or an
externally supplied baseline checkpoint). Panel (b) = the model under test.

Default panel (b) points at the best from-scratch checkpoint per
``experiments/clip_pretraining/soft_target_clip/RESULTS_jul7.md`` §3.4:
TP-only soft τ=0.05 400 ep, seed=2026. See ``_submit.py`` for the wired-up
Delta job.

Usage (from repo root, on the cluster where the checkpoint lives):
    CKPT_DIR=/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip/jul7-tp_soft_target_clip_seed2026
    PYTHONPATH=. uv run --group eeg python \\
        experiments/clip_pretraining/cs_aligner/plot_modality_gap.py \\
        --config $CKPT_DIR/config_TP.yaml \\
        --checkpoint-b $CKPT_DIR/latest.pth.tar \\
        --label-b "Soft-Target CLIP (jul7 best)" \\
        --split val --n-recordings 30 \\
        --output experiments/clip_pretraining/cs_aligner/modality_gap.png

Compare two trained checkpoints (e.g., soft-target vs CS-Aligner once it lands):
    ... --checkpoint-a A.pth.tar --label-a "Soft-Target" \\
        --checkpoint-b B.pth.tar --label-b "CS-Aligner"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from eb_jepa.architectures import MovieCLIPHead
from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows

# Continuous per-window movie features to store alongside embeddings so
# downstream plots can color by them. Uses JEPAMovieDataset.DEFAULT_FEATURES.
FEATURE_NAMES = list(JEPAMovieDataset.DEFAULT_FEATURES)
from eb_jepa.evaluation.clip_probe.probe import load_encoder_state
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Training config yaml (defines encoder, head, data).")
    ap.add_argument("--checkpoint-a", default=None,
                    help="Baseline checkpoint for panel (a). Omit for random init.")
    ap.add_argument("--checkpoint-b", default=None,
                    help="Trained checkpoint for panel (b). Omit for random init.")
    ap.add_argument("--label-a", default="Random init")
    ap.add_argument("--label-b", default="Trained CLIP")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-recordings", type=int, default=20,
                    help="Recordings to draw windows from.")
    ap.add_argument("--windows-per-recording", type=int, default=None,
                    help="Cap windows/recording (uniform stride). None → all.")
    ap.add_argument("--max-windows-per-panel", type=int, default=2000,
                    help="Post-collection subsample for t-SNE tractability.")
    ap.add_argument("--tsne-perplexity", type=float, default=30.0)
    ap.add_argument("--tsne-seed", type=int, default=0)
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task override for cfg.data.task, "
                         "e.g. 'ThePresent,DespicableMe'.")
    ap.add_argument("--save-npz", default=None,
                    help="If set, save panel-(b) embeddings + metadata + t-SNE "
                         "coords here. Reused by plot_embedding_structure.py.")
    ap.add_argument("--output", default="modality_gap.png",
                    help="Output image path. A companion .pdf is written alongside.")
    return ap.parse_args()


def build_dataset(cfg, split, task_override=None):
    """Force recipe_mode + per_window + mean_center so V-JEPA-2 targets match
    what the training loop projected against. ``task_override`` (list[str]) lets
    the caller include additional movies not in the training config, e.g. adding
    DespicableMe to a TP-only-trained checkpoint for cross-movie visualization."""
    data_cfg = cfg.data
    task = task_override if task_override is not None else data_cfg.task
    return JEPAMovieDataset(
        split=split,
        n_windows=data_cfg.n_windows,
        window_size_seconds=data_cfg.window_size_seconds,
        task=task,
        temporal_stride=data_cfg.get("temporal_stride", 1),
        feature_names=FEATURE_NAMES,
        cfg=data_cfg,
        preprocessed=data_cfg.preprocessed,
        preprocessed_dir=data_cfg.get("preprocessed_dir", None),
        visual_processing_delay_s=0.0,
        recipe_mode=True,
        recipe_target_kind="per_window",
        recipe_mean_center=True,
        recipe_require_shots=False,
    )


def load_clip_head_state(clip_head: MovieCLIPHead, ckpt_path: str) -> None:
    """Load ``clip_head.*`` tensors from a training checkpoint. Silently skips
    keys the head doesn't own; warns if nothing loads."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    head_sd = {k[len("clip_head."):]: v for k, v in sd.items() if k.startswith("clip_head.")}
    missing, unexpected = clip_head.load_state_dict(head_sd, strict=False)
    print(f"  loaded {len(head_sd)} clip_head tensors  "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if not head_sd:
        print("  WARNING: no clip_head.* keys found — checkpoint may be encoder-only.")


@torch.no_grad()
def embed_windows(
    encoder,
    clip_head: MovieCLIPHead,
    dataset: JEPAMovieDataset,
    device: torch.device,
    n_recordings: int,
    windows_per_recording: int | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(Z_eeg, Z_vis, meta)``. Both embedding arrays are ``[N, P]``,
    L2-normalized in the shared space. ``meta`` is a dict of per-window arrays
    (task, shot_id, scene_id, t_start, rec_id) with the SAME row order as the
    embeddings — row ``i`` is one window whose EEG and V-JEPA-2 vectors sit at
    ``Z_eeg[i]`` and ``Z_vis[i]``."""
    n_rec = min(n_recordings, len(dataset))
    # Deterministic shuffle so multi-movie datasets get a mix of tasks (not
    # just whichever the loader lists first). Fixed seed → both panels of a
    # modality-gap figure see the same recording set.
    rec_order = np.random.default_rng(0).permutation(len(dataset))[:n_rec]
    z_eeg_all, z_vis_all = [], []
    task_all, shot_all, scene_all, tstart_all, rec_all = [], [], [], [], []
    feats_all = []
    for step, rec_idx in enumerate(rec_order):
        rec_idx = int(rec_idx)
        crop_inds = dataset._crop_inds[rec_idx]
        fif_path = dataset._fif_paths[rec_idx]
        vis_targets = dataset.embedding_recordings[rec_idx]           # [n_win, D_vjepa]
        rec_task = dataset._recording_tasks[rec_idx]
        shot_ids = dataset.shot_id_recordings[rec_idx].numpy()        # [n_win]
        scene_ids = dataset.scene_id_recordings[rec_idx].numpy()      # [n_win]
        t_starts = dataset.t_start_recordings[rec_idx].numpy()        # [n_win]
        feats = dataset.feature_recordings[rec_idx].numpy()           # [n_win, n_features]

        n_win = len(crop_inds)
        if windows_per_recording is not None and n_win > windows_per_recording:
            stride = max(1, n_win // windows_per_recording)
            keep = np.arange(0, n_win, stride)[:windows_per_recording]
            crop_inds = crop_inds[keep]
            vis_targets = vis_targets[keep]
            shot_ids = shot_ids[keep]
            scene_ids = scene_ids[keep]
            t_starts = t_starts[keep]
            feats = feats[keep]

        raw = _read_raw_windows(fif_path, crop_inds)          # [n_win, C, T]
        eeg = torch.from_numpy(raw)
        if dataset._norm_mode == "per_recording":
            rec_mean = eeg.mean(dim=(0, 2), keepdim=True)
            rec_std = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            eeg = (eeg - rec_mean) / rec_std
        else:
            eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
        eeg_in = eeg.unsqueeze(1)                              # [n_win, 1, C, T]

        z_eeg_rec = []
        for start in range(0, len(eeg_in), batch_size):
            batch = eeg_in[start:start + batch_size].to(device)
            tokens = encoder.encode_tokens(batch, mask=None)
            pooled = encoder.pool_to_windows(tokens)           # [B, D, 1, 1, 1]
            z = clip_head.project_eeg(pooled)                  # [B, P], L2-norm
            z_eeg_rec.append(z.cpu().numpy())
        z_eeg_rec = np.concatenate(z_eeg_rec, axis=0)          # [n_win, P]

        vis_t = vis_targets.to(device=device, dtype=torch.float32)
        z_vis_rec = clip_head.project_vision(vis_t).cpu().numpy()  # [n_win, P]

        n = len(z_eeg_rec)
        z_eeg_all.append(z_eeg_rec)
        z_vis_all.append(z_vis_rec)
        task_all.append(np.array([rec_task] * n))
        shot_all.append(shot_ids.astype(np.int64))
        scene_all.append(scene_ids.astype(np.int64))
        tstart_all.append(t_starts.astype(np.float32))
        rec_all.append(np.full(n, rec_idx, dtype=np.int64))
        feats_all.append(feats.astype(np.float32))
        if (step + 1) % 10 == 0:
            print(f"  {step + 1}/{n_rec} recordings embedded  "
                  f"(windows so far: {sum(len(x) for x in z_eeg_all)})")

    feats_stack = np.concatenate(feats_all, axis=0)                # [N, n_features]
    meta = {
        "task": np.concatenate(task_all, axis=0),
        "shot_id": np.concatenate(shot_all, axis=0),
        "scene_id": np.concatenate(scene_all, axis=0),
        "t_start": np.concatenate(tstart_all, axis=0),
        "rec_id": np.concatenate(rec_all, axis=0),
    }
    for i, fname in enumerate(FEATURE_NAMES):
        meta[f"feat_{fname}"] = feats_stack[:, i]
    return (np.concatenate(z_eeg_all, axis=0),
            np.concatenate(z_vis_all, axis=0),
            meta)


def coherent_subsample(
    n_windows: int, max_n: int, rng: np.random.Generator
) -> np.ndarray:
    """Return an index array into the paired (EEG, V-JEPA-2) rows so both
    modalities keep the same underlying windows (metadata stays coherent)."""
    if n_windows <= max_n:
        return np.arange(n_windows)
    return rng.choice(n_windows, size=max_n, replace=False)


def compute_panel(
    cfg,
    dataset,
    device,
    checkpoint: str | None,
    args,
    rng: np.random.Generator,
    save_npz_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fresh encoder + head, load the checkpoint (or leave random),
    embed, subsample, and t-SNE. Returns ``(xy, is_vision)`` with ``xy`` shape
    ``[2N, 2]`` and ``is_vision`` a bool mask marking the V-JEPA-2 rows.

    When ``save_npz_path`` is set, also writes the raw shared-space embeddings,
    the t-SNE coords, and per-window metadata (task, shot_id, scene_id, t_start,
    rec_id) so downstream plots can recolor without re-encoding."""
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
    if checkpoint:
        print(f"Loading encoder from {checkpoint}")
        load_encoder_state(encoder, checkpoint)
        print(f"Loading clip_head from {checkpoint}")
        load_clip_head_state(clip_head, checkpoint)
    else:
        print("Using RANDOM-INIT encoder + clip_head")
    encoder = encoder.to(device).eval()
    clip_head = clip_head.to(device).eval()

    z_eeg, z_vis, meta = embed_windows(
        encoder, clip_head, dataset, device,
        args.n_recordings, args.windows_per_recording, args.encode_batch,
    )
    idx = coherent_subsample(len(z_eeg), args.max_windows_per_panel, rng)
    z_eeg, z_vis = z_eeg[idx], z_vis[idx]
    meta = {k: v[idx] for k, v in meta.items()}
    print(f"  panel counts: eeg={len(z_eeg)}, vis={len(z_vis)}, dim={z_eeg.shape[1]}")

    X = np.concatenate([z_eeg, z_vis], axis=0).astype(np.float32)
    is_vision = np.concatenate([
        np.zeros(len(z_eeg), dtype=bool),
        np.ones(len(z_vis), dtype=bool),
    ])
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.tsne_seed,
    )
    xy = tsne.fit_transform(X)

    if save_npz_path:
        # Metadata is per-window; tile to per-point (EEG rows then V-JEPA-2 rows)
        # so it aligns with X / xy / is_vision.
        out = {
            "z_shared": X,               # [2N, P] raw shared-space embeddings
            "xy": xy,                    # [2N, 2] t-SNE coords
            "is_vision": is_vision,      # [2N] bool
            **{k: np.concatenate([v, v], axis=0) for k, v in meta.items()},
        }
        Path(save_npz_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_npz_path, **out)
        print(f"  saved embeddings + metadata to {save_npz_path}")

    return xy, is_vision


def plot_panel(ax, xy, is_vision, title):
    ax.scatter(
        xy[~is_vision, 0], xy[~is_vision, 1],
        s=8, alpha=0.55, c="#1f77b4", edgecolors="none", label="EEG",
    )
    ax.scatter(
        xy[is_vision, 0], xy[is_vision, 1],
        s=8, alpha=0.55, c="#d62728", edgecolors="none", label="V-JEPA-2",
    )
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#888")
        spine.set_linewidth(0.6)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    cfg = load_config(args.config)
    task_override = args.tasks.split(",") if args.tasks else None
    if task_override is not None:
        print(f"Overriding cfg.data.task → {task_override}")
    print(f"Loading {args.split} split ...")
    dataset = build_dataset(cfg, args.split, task_override=task_override)
    print(f"  n_recordings={len(dataset)}, n_chans={dataset.n_chans}, "
          f"n_times={dataset.n_times}, vjepa_dim={dataset.frame_embedding_dim}")

    rng = np.random.default_rng(args.tsne_seed)

    print(f"\n=== Panel (a): {args.label_a} ===")
    xy_a, mask_a = compute_panel(
        cfg, dataset, device, args.checkpoint_a, args, rng, save_npz_path=None,
    )
    print(f"\n=== Panel (b): {args.label_b} ===")
    xy_b, mask_b = compute_panel(
        cfg, dataset, device, args.checkpoint_b, args, rng,
        save_npz_path=args.save_npz,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    plot_panel(axes[0], xy_a, mask_a, f"(a) {args.label_a}")
    plot_panel(axes[1], xy_b, mask_b, f"(b) {args.label_b}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=11,
    )
    fig.suptitle(
        "t-SNE of EEG and V-JEPA-2 embeddings in the shared CLIP space",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nSaved {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
