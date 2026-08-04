"""Build the `z_shared` npz that ``plot_split_modality.py`` consumes, from a
JEPA (LeJEPA / Laya) pretraining checkpoint.

The visualization script does **independent** t-SNE per modality, so we don't
need a truly-shared projected space (which JEPA lacks — no ``clip_head``).
We just need one npz row per window with:
  - the JEPA encoder's per-window pooled embedding (for EEG rows)
  - the V-JEPA-2 target feature at the same window (for V-JEPA-2 rows)
plus the metadata (task, shot_id, feat_*) that plot_split_modality colors by.

Encoder embeddings live at dim ``encoder_embed_dim`` (e.g. 384 for Laya, 512
for LeJEPA); V-JEPA-2 targets live at dim 1408. We zero-pad the smaller side
so both fit into a single ``z_shared`` matrix. The per-modality t-SNE slices
by ``is_vision``, so the zero-padded columns contribute nothing to distances
within a modality — the padding is only a storage convenience.

Usage:
    PYTHONPATH=. uv run --group eeg python \\
        experiments/jepa_pretraining/build_split_modality_npz.py \\
        --config /path/to/run/config.yaml \\
        --checkpoint /path/to/run/latest.pth.tar \\
        --output /path/to/run/split_modality.npz \\
        --n-recordings 30 --split val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from eb_jepa.datasets.hbn import JEPAMovieDataset, _read_raw_windows
from eb_jepa.evaluation.clip_probe.probe import load_encoder_state
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_config

FEATURE_NAMES = list(JEPAMovieDataset.DEFAULT_FEATURES)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="JEPA checkpoint; omit for random-init encoder.")
    ap.add_argument("--output", required=True, help="Output .npz path.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-recordings", type=int, default=30)
    ap.add_argument("--windows-per-recording", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=2000,
                    help="Post-collection cap per modality (t-SNE tractability).")
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task override, e.g. 'ThePresent,DespicableMe'.")
    return ap.parse_args()


def build_recipe_dataset(cfg, split, task_override=None):
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
        recipe_mean_center=False,
        recipe_require_shots=False,
    )


@torch.no_grad()
def collect(encoder, dataset, device, n_recordings, windows_per_recording, batch_size):
    n_rec = min(n_recordings, len(dataset))
    rec_order = np.random.default_rng(0).permutation(len(dataset))[:n_rec]
    z_eeg_all, z_vis_all = [], []
    task_all, shot_all, feats_all = [], [], []

    for step, rec_idx in enumerate(rec_order):
        rec_idx = int(rec_idx)
        crop_inds = dataset._crop_inds[rec_idx]
        fif_path = dataset._fif_paths[rec_idx]
        vis_targets = dataset.embedding_recordings[rec_idx]      # [n_win, D_vjepa]
        rec_task = dataset._recording_tasks[rec_idx]
        shot_ids = dataset.shot_id_recordings[rec_idx].numpy()
        feats = dataset.feature_recordings[rec_idx].numpy()      # [n_win, n_feat]

        n_win = len(crop_inds)
        if windows_per_recording is not None and n_win > windows_per_recording:
            stride = max(1, n_win // windows_per_recording)
            keep = np.arange(0, n_win, stride)[:windows_per_recording]
            crop_inds = crop_inds[keep]
            vis_targets = vis_targets[keep]
            shot_ids = shot_ids[keep]
            feats = feats[keep]

        raw = _read_raw_windows(fif_path, crop_inds)             # [n_win, C, T]
        eeg = torch.from_numpy(raw)
        if dataset._norm_mode == "per_recording":
            rec_mean = eeg.mean(dim=(0, 2), keepdim=True)
            rec_std = eeg.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            eeg = (eeg - rec_mean) / rec_std
        else:
            eeg = (eeg - dataset._eeg_mean) / dataset._eeg_std
        eeg_in = eeg.unsqueeze(1)                                # [n_win, 1, C, T]

        z_eeg_rec = []
        for start in range(0, len(eeg_in), batch_size):
            batch = eeg_in[start:start + batch_size].to(device)
            tokens = encoder.encode_tokens(batch, mask=None)     # [B, n_ctx, D]
            pooled = encoder.pool_to_windows(tokens)             # [B, D, 1, 1, 1]
            z = pooled.squeeze(-1).squeeze(-1).squeeze(-1)       # [B, D]
            z_eeg_rec.append(z.cpu().numpy().astype(np.float32))
        z_eeg_rec = np.concatenate(z_eeg_rec, axis=0)            # [n_win, D_eeg]
        z_vis_rec = vis_targets.numpy().astype(np.float32)        # [n_win, D_vjepa]

        n = len(z_eeg_rec)
        z_eeg_all.append(z_eeg_rec)
        z_vis_all.append(z_vis_rec)
        task_all.append(np.array([rec_task] * n))
        shot_all.append(shot_ids.astype(np.int64))
        feats_all.append(feats.astype(np.float32))
        if (step + 1) % 10 == 0:
            print(f"  {step + 1}/{n_rec} recordings embedded  "
                  f"(windows so far: {sum(len(x) for x in z_eeg_all)})")

    z_eeg = np.concatenate(z_eeg_all, axis=0)
    z_vis = np.concatenate(z_vis_all, axis=0)
    task = np.concatenate(task_all, axis=0)
    shot = np.concatenate(shot_all, axis=0)
    feats = np.concatenate(feats_all, axis=0)
    return z_eeg, z_vis, task, shot, feats


def main():
    args = parse_args()
    cfg = load_config(args.config)
    task_override = args.tasks.split(",") if args.tasks else None
    device = torch.device(args.device)

    print(f"Building dataset (split={args.split}, task={task_override or cfg.data.task}) ...")
    dataset = build_recipe_dataset(cfg, args.split, task_override)

    encoder = build_encoder(
        cfg, n_chans=dataset.n_chans, n_times=dataset.n_times,
        chs_info=dataset.get_chs_info(), n_windows=cfg.data.n_windows,
    )
    if args.checkpoint:
        print(f"Loading encoder from {args.checkpoint}")
        load_encoder_state(encoder, args.checkpoint)
    else:
        print("Random-init encoder (no --checkpoint given)")
    encoder = encoder.to(device).eval()

    z_eeg, z_vis, task, shot, feats = collect(
        encoder, dataset, device,
        args.n_recordings, args.windows_per_recording, args.encode_batch,
    )

    # Cap per-modality window count (t-SNE cost is O(N^2) memory-wise).
    rng = np.random.default_rng(0)
    if len(z_eeg) > args.max_windows:
        idx = rng.choice(len(z_eeg), size=args.max_windows, replace=False)
        z_eeg = z_eeg[idx]; z_vis = z_vis[idx]
        task = task[idx]; shot = shot[idx]; feats = feats[idx]
    print(f"Final counts: eeg={len(z_eeg)}  vis={len(z_vis)}  "
          f"(dim_eeg={z_eeg.shape[1]}, dim_vis={z_vis.shape[1]})")

    # Pad the smaller of (EEG, V-JEPA-2) with zeros so both fit into one
    # [2N, D_max] matrix. Independent per-modality t-SNE means the padded
    # columns contribute nothing to intra-modality distances.
    D_max = max(z_eeg.shape[1], z_vis.shape[1])
    def _pad(x, D):
        if x.shape[1] == D: return x
        pad = np.zeros((len(x), D - x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)
    z_eeg_p = _pad(z_eeg, D_max)
    z_vis_p = _pad(z_vis, D_max)
    z_shared = np.concatenate([z_eeg_p, z_vis_p], axis=0).astype(np.float32)
    is_vision = np.concatenate([
        np.zeros(len(z_eeg_p), dtype=bool),
        np.ones(len(z_vis_p), dtype=bool),
    ])

    # Metadata tiled to per-point (EEG rows then V-JEPA-2 rows).
    out = {
        "z_shared": z_shared,
        "is_vision": is_vision,
        "task": np.concatenate([task, task], axis=0),
        "shot_id": np.concatenate([shot, shot], axis=0),
    }
    for i, fname in enumerate(FEATURE_NAMES):
        v = feats[:, i]
        out[f"feat_{fname}"] = np.concatenate([v, v], axis=0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"Wrote {out_path}  ({z_shared.shape[0]} rows, D={z_shared.shape[1]})")


if __name__ == "__main__":
    main()
