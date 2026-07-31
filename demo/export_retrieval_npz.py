"""Export everything the e->v retrieval demo needs into a single npz.

Runs on the cluster (checkpoints + HBN FIFs live there). Projects every EEG
window of a split into the shared CLIP space along with its paired V-JEPA-2
target, and tacks on raw EEG for a subset of "showcase" windows so the demo
page can draw real traces.

The output deliberately uses the ``z_shared`` / ``is_vision`` layout that
``eb_jepa/evaluation/clip_probe/retrieval.py --from-npz`` already reads, so the
export can be validated in one line without any new code::

    PYTHONPATH=. uv run --group eeg python \\
        eb_jepa/evaluation/clip_probe/retrieval.py \\
        --from-npz demo_test.npz --t-bucket-s 0.5 --output check.json

Usage (from repo root, on Delta)::

    CKPT_DIR=/work/hdd/bbnv/dtyoung/eb_jepa/scene_clip_from_checkpoint/jul22_warmstart_lr3e4_ep299
    PYTHONPATH=. uv run --group eeg python demo/export_retrieval_npz.py \\
        --config $CKPT_DIR/config.yaml \\
        --checkpoint $CKPT_DIR/latest.pth.tar \\
        --split test --output demo_test.npz

See ``demo/_submit_export.py`` for the wired-up job and ``demo/README.md`` for
the full pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eb_jepa.architectures import MovieCLIPHead
from eb_jepa.datasets.hbn import _read_raw_windows
from eb_jepa.evaluation.clip_probe.probe import (
    SCALAR_FEATURES_DEFAULT,
    build_dataset,
    load_encoder_state,
)
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_config

# The dataset comes from probe.build_dataset — the SAME call retrieval.py makes,
# so the export reproduces the published §3.10 numbers exactly. Note this leaves
# the V-JEPA-2 targets raw (no recipe_mode, so no mean-centering), which differs
# from what scene_clip trained against; the published metric was computed this
# way, and matching it is what lets the demo assert against the committed JSON.
# The per-recording encode loop is borrowed from the modality-gap figure.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.clip_pretraining.cs_aligner.plot_modality_gap import (  # noqa: E402
    embed_windows,
    load_clip_head_state,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Training config yaml saved NEXT TO the checkpoint.")
    ap.add_argument("--checkpoint", required=True,
                    help="CLIP checkpoint (needs both encoder.* and clip_head.*).")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated override for cfg.data.task.")
    ap.add_argument("--max-recordings", type=int, default=None,
                    help="Cap recordings (debugging). None -> the whole split.")
    ap.add_argument("--n-showcase", type=int, default=120,
                    help="Windows to ship raw EEG for, stratified over scene x "
                         "recording. 120 windows ~ 12 MB at 129ch/400 samples.")
    ap.add_argument("--showcase-recordings", default=None,
                    help="Comma-separated rec_id list. Ships EVERY window of "
                         "these recordings instead of the stratified sample — "
                         "what the scrubbing timeline demo needs. ~10 MB per "
                         "recording (101 windows x 129ch x 400).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, help="Destination .npz")
    return ap.parse_args()


def pick_showcase(meta: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Choose ~n row indices spread over scenes and recordings.

    Round-robins over scenes (so the demo can cover the whole movie) and, within
    a scene, prefers rows from recordings not yet drawn from. Rows with
    scene_id == -1 are skipped: they cannot be scored at the scene level.
    """
    scene_ids = meta["scene_id"]
    rec_ids = meta["rec_id"]
    eligible = np.flatnonzero(scene_ids >= 0)
    if len(eligible) <= n:
        return eligible

    by_scene: dict[int, list[int]] = {}
    for i in eligible:
        by_scene.setdefault(int(scene_ids[i]), []).append(int(i))
    for rows in by_scene.values():
        rng.shuffle(rows)

    scenes = sorted(by_scene)
    picked: list[int] = []
    used_recs: set[int] = set()
    # Two passes: the first spreads across recordings, the second backfills.
    for prefer_new_rec in (True, False):
        cursor = {s: 0 for s in scenes}
        while len(picked) < n:
            progressed = False
            for s in scenes:
                rows = by_scene[s]
                while cursor[s] < len(rows):
                    row = rows[cursor[s]]
                    cursor[s] += 1
                    if prefer_new_rec and int(rec_ids[row]) in used_recs:
                        continue
                    picked.append(row)
                    used_recs.add(int(rec_ids[row]))
                    progressed = True
                    break
                if len(picked) >= n:
                    break
            if not progressed:
                break
    return np.array(sorted(set(picked))[:n], dtype=np.int64)


def read_showcase_eeg(dataset, meta: dict, rows: np.ndarray) -> np.ndarray:
    """Raw EEG for the chosen rows, normalized exactly as ``embed_windows`` did.

    ``embed_windows`` walks recordings in a fixed order and appends all of a
    recording's windows contiguously, so a row's position inside its recording
    is its offset from that recording's first row.
    """
    rec_ids = meta["rec_id"]
    first_row = {}
    for i, r in enumerate(rec_ids):
        first_row.setdefault(int(r), i)

    out = np.zeros((len(rows), dataset.n_chans, dataset.n_times), dtype=np.float16)
    by_rec: dict[int, list[int]] = {}
    for slot, row in enumerate(rows):
        by_rec.setdefault(int(rec_ids[row]), []).append(slot)

    for rec_idx, slots in by_rec.items():
        crop_inds = dataset._crop_inds[rec_idx]
        raw = torch.from_numpy(_read_raw_windows(dataset._fif_paths[rec_idx], crop_inds))
        # Same normalization as embed_windows: stats over ALL windows of the rec.
        if dataset._norm_mode == "per_recording":
            mean = raw.mean(dim=(0, 2), keepdim=True)
            std = raw.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)
            raw = (raw - mean) / std
        else:
            raw = (raw - dataset._eeg_mean) / dataset._eeg_std
        for slot in slots:
            win = int(rows[slot]) - first_row[rec_idx]
            out[slot] = raw[win].numpy().astype(np.float16)
    return out


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    if args.tasks:
        cfg.data.task = args.tasks.split(",")
    dataset = build_dataset(cfg, args.split, SCALAR_FEATURES_DEFAULT)
    print(f"{args.split}: {len(dataset)} recordings, "
          f"n_chans={dataset.n_chans}, n_times={dataset.n_times}")

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
    load_encoder_state(encoder, args.checkpoint)
    load_clip_head_state(clip_head, args.checkpoint)
    encoder = encoder.to(device).eval()
    clip_head = clip_head.to(device).eval()

    n_rec = args.max_recordings or len(dataset)
    z_eeg, z_vis, meta = embed_windows(
        encoder, clip_head, dataset, device,
        n_recordings=n_rec, windows_per_recording=None,
        batch_size=args.encode_batch,
    )
    print(f"embedded M={len(z_eeg)} windows, P={z_eeg.shape[1]}")

    # embed_windows labels its feat_* columns from its own 4-name list, but this
    # dataset carries the probe's 12 features — the names would not line up.
    # Nothing downstream reads them, so drop them rather than ship wrong labels.
    meta = {k: v for k, v in meta.items() if not k.startswith("feat_")}

    rng = np.random.default_rng(args.seed)
    if args.showcase_recordings:
        want = {int(x) for x in args.showcase_recordings.split(",")}
        showcase_idx = np.flatnonzero(np.isin(meta["rec_id"], list(want)))
        missing = want - set(meta["rec_id"][showcase_idx].tolist())
        if missing:
            raise SystemExit(f"rec_id(s) not in this split: {sorted(missing)}")
        print(f"showcase = every window of recordings {sorted(want)}")
    else:
        showcase_idx = pick_showcase(meta, args.n_showcase, rng)
    showcase_eeg = read_showcase_eeg(dataset, meta, showcase_idx)
    print(f"showcase: {len(showcase_idx)} windows "
          f"({showcase_eeg.nbytes / 1e6:.1f} MB raw EEG)")

    # z_shared / is_vision layout — EEG rows first, then the row-aligned vision
    # rows, with per-window metadata doubled to match. This is exactly what
    # retrieval.py --from-npz expects (_load_from_npz).
    # float32, not float16: the local build asserts its recomputed Top-K against
    # the committed RESULTS json, and fp16 round-off can flip near-ties.
    z_shared = np.concatenate([z_eeg, z_vis], axis=0).astype(np.float32)
    is_vision = np.concatenate([
        np.zeros(len(z_eeg), dtype=bool), np.ones(len(z_vis), dtype=bool),
    ])

    chs = dataset.get_chs_info()
    ch_names = [c["ch_name"] for c in chs]
    ch_pos = np.array([c["loc"][:3] for c in chs], dtype=np.float32)
    provenance = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "split": args.split,
        "n_recordings": int(n_rec),
        "n_eeg_windows_M": int(len(z_eeg)),
        "proj_dim": int(z_eeg.shape[1]),
        "n_chans": int(dataset.n_chans),
        "sfreq": float(getattr(dataset, "sfreq", 200.0)),
        "window_size_seconds": float(cfg.data.window_size_seconds),
        "norm_mode": str(dataset._norm_mode),
    }

    out = {
        "z_shared": z_shared,
        "is_vision": is_vision,
        **{k: np.concatenate([v, v], axis=0) for k, v in meta.items()},
        "showcase_idx": showcase_idx,
        "showcase_eeg": showcase_eeg,
        "showcase_fif": np.array(
            [Path(dataset._fif_paths[int(meta["rec_id"][i])]).name for i in showcase_idx]
        ),
        "ch_names": np.array(ch_names),
        "ch_pos": ch_pos,
        "provenance_json": np.array(json.dumps(provenance)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"wrote {args.output}  ({size_mb:.1f} MB)")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
