"""Top-K retrieval evaluation for EEG ↔ V-JEPA-2 CLIP-style checkpoints.

Implements the SSL-literature-standard retrieval evaluation used by
EEG2Video, EEG-CLIP, and NICE-EEG. For each EEG window in the eval split
(anchor), the trained encoder + `MovieCLIPHead` projects to `z_eeg`; each
unique (task, movie-time) in the eval split becomes one V-JEPA-2 candidate
projected to `z_vis`. Cosine similarity `z_eeg @ z_vis.T` produces the
retrieval matrix.

Two directions reported (both are standard in the SSL literature):

- **e→v** (EEG anchor → find paired vision): for each EEG window, is the
  correct (task, movie-time) V-JEPA-2 target in the top-K by cosine? This
  is "identify the frame you're watching."
- **v→e** (vision anchor → find matching EEG): for each unique (task,
  movie-time) V-JEPA-2 target, is any EEG window at that same (task,
  movie-time) in the top-K by cosine? This is "given a frame, find someone
  watching it." Multi-positive by construction because multiple subjects
  view the same time.

Chance level for e→v Top-1 is `1 / n_pool`. Report both raw accuracy and
"chance-relative" accuracy (raw / chance) for interpretability across pool
sizes.

Usage:
    PYTHONPATH=. .venv/bin/python eb_jepa/evaluation/clip_probe/retrieval.py \\
        --checkpoint /path/to/latest.pth.tar \\
        --config /path/to/config_TP.yaml \\
        --split test --topks 1 5 10 \\
        --output retrieval_test_ckpt.json

Use --random-baseline to skip the checkpoint load and evaluate a fresh-init
encoder + head of the same architecture (the null/noise floor).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eb_jepa.architectures import MovieCLIPHead
from eb_jepa.evaluation.clip_probe.probe import (
    build_dataset,
    embed_recording_all_windows,
    SCALAR_FEATURES_DEFAULT,
)
from eb_jepa.training.builder import build_encoder
from eb_jepa.training_utils import load_config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-npz", default=None,
                    help="Skip encoder+head loading; read shared-space embeddings "
                         "from a .npz file produced by "
                         "experiments/clip_pretraining/cs_aligner/plot_modality_gap.py "
                         "(--save-npz). Fields required: z_shared [2N, P], "
                         "is_vision [2N] bool, task [2N], t_start [2N]. Ignores "
                         "--checkpoint / --config / --split when set.")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to .pth.tar; omit and use --random-baseline for the null")
    ap.add_argument("--config", default=None,
                    help="Training config yaml. Required unless --from-npz is set.")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--max-recordings", type=int, default=None,
                    help="Cap n recordings for quick smoke runs")
    ap.add_argument("--topks", nargs="+", type=int, default=[1, 5, 10],
                    help="K values for Top-K accuracy")
    ap.add_argument("--t-bucket-s", type=float, default=0.1,
                    help="Round t_start to this granularity (seconds) when "
                         "deduplicating the vision pool. Matches V-JEPA-2's ~2 Hz "
                         "clip rate; smaller = more unique buckets but same targets.")
    ap.add_argument("--output", default="retrieval.json")
    ap.add_argument("--random-baseline", action="store_true")
    return ap.parse_args()


def _load_clip_ckpt(ckpt_path):
    """Load state dict, strip torch.compile prefix, split into encoder + head."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    head_sd = {k[len("clip_head."):]: v for k, v in sd.items() if k.startswith("clip_head.")}
    return enc_sd, head_sd


@torch.no_grad()
def _embed_recording_and_meta(encoder, dataset, rec_idx, device, batch_size):
    """Encode all EEG windows in a recording; return (X [n_win, D], t_start [n_win], vjepa2 [n_win, D_v])."""
    X, _Y = embed_recording_all_windows(encoder, dataset, rec_idx, device, batch_size)
    t_starts = dataset.t_start_recordings[rec_idx].numpy()
    embeds = dataset.embedding_recordings[rec_idx].numpy()
    return X, t_starts, embeds


def _build_vision_pool(all_task_ids, all_t_starts, all_embeds, t_bucket_s):
    """Dedupe V-JEPA-2 targets by (task, round(t_start / t_bucket_s)).

    Returns:
        pool_embeds:   [N_pool, D_v] — one V-JEPA-2 vector per unique key
        anchor_to_pool: [M] int64 — for each anchor row in all_task_ids, the pool index
        pool_keys:     list of (task_id, t_bucket) tuples, len == N_pool
    """
    buckets = np.round(all_t_starts / t_bucket_s).astype(np.int64)
    keys = list(zip(all_task_ids.tolist(), buckets.tolist()))
    key_to_idx: dict[tuple[int, int], int] = {}
    pool_embeds_list: list[np.ndarray] = []
    pool_keys: list[tuple[int, int]] = []
    anchor_to_pool = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        if k not in key_to_idx:
            key_to_idx[k] = len(key_to_idx)
            pool_embeds_list.append(all_embeds[i])
            pool_keys.append(k)
        anchor_to_pool[i] = key_to_idx[k]
    pool_embeds = np.stack(pool_embeds_list, axis=0)
    return pool_embeds, anchor_to_pool, pool_keys


def _topk_e2v(similarity: np.ndarray, correct_idx: np.ndarray, ks: list[int]) -> dict[int, float]:
    """For each anchor row, is the correct pool index in the top-K?

    similarity: [M, N_pool] float
    correct_idx: [M] int64 — the true pool index for each anchor
    """
    k_max = max(ks)
    # argpartition returns unsorted; get exact ordering only for the top k_max.
    top = np.argpartition(-similarity, kth=k_max - 1, axis=1)[:, :k_max]
    # Sort those k_max by descending similarity for stable Top-K
    row_idx = np.arange(similarity.shape[0])[:, None]
    top_sorted = top[row_idx, np.argsort(-similarity[row_idx, top], axis=1)]
    hits = (top_sorted == correct_idx[:, None])  # [M, k_max] bool
    return {k: float(hits[:, :k].any(axis=1).mean()) for k in ks}


def _topk_v2e(similarity: np.ndarray, anchor_to_pool: np.ndarray, n_pool: int,
              ks: list[int]) -> dict[int, float]:
    """For each pool row (vision anchor), is any EEG anchor with matching pool index in top-K?

    similarity: [M, N_pool] float — rows are EEG anchors, cols are vision pool.
      We transpose so rows become vision anchors: [N_pool, M].
    anchor_to_pool: [M] int64 — pool index each EEG anchor belongs to.
    """
    sim_v2e = similarity.T  # [N_pool, M]
    k_max = max(ks)
    top = np.argpartition(-sim_v2e, kth=k_max - 1, axis=1)[:, :k_max]
    row_idx = np.arange(n_pool)[:, None]
    top_sorted = top[row_idx, np.argsort(-sim_v2e[row_idx, top], axis=1)]
    # For pool index j, "correct" means the EEG anchor at top position has anchor_to_pool == j.
    correct_labels = anchor_to_pool[top_sorted]  # [N_pool, k_max]
    j_col = np.arange(n_pool)[:, None]           # [N_pool, 1]
    hits = (correct_labels == j_col)             # [N_pool, k_max]
    return {k: float(hits[:, :k].any(axis=1).mean()) for k in ks}


def _load_from_npz(path: str):
    """Load pre-computed shared-space embeddings from `plot_modality_gap.py`.

    Returns:
        z_eeg [M, P] float32 — EEG anchors (rows with is_vision=False)
        z_vis_all [M, P] float32 — paired V-JEPA-2 vectors (rows with is_vision=True),
            aligned 1:1 with z_eeg by row order (coherent_subsample preserves pairing)
        t_starts [M] float32 — movie time per (EEG, vision) pair
        task_ids [M] int64 — integer task IDs (built from unique task strings)
    """
    d = np.load(path, allow_pickle=False)
    required = ["z_shared", "is_vision", "task", "t_start"]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")
    z_shared = d["z_shared"].astype(np.float32)
    is_vision = d["is_vision"].astype(bool)
    task = d["task"]
    t_start = d["t_start"].astype(np.float32)
    z_eeg = z_shared[~is_vision]
    z_vis = z_shared[is_vision]
    t_starts = t_start[~is_vision]
    task_strs = task[~is_vision]
    # plot_modality_gap.py concatenates EEG then vision from the same paired
    # source rows, so is_vision=False and is_vision=True are already row-aligned
    # by index. Sanity: check vision-side task/t_start match EEG-side to catch
    # any format change.
    t_start_vis = t_start[is_vision]
    task_vis = task[is_vision]
    if not (np.array_equal(t_starts, t_start_vis) and np.array_equal(task_strs, task_vis)):
        raise ValueError(
            f"{path}: is_vision=False and is_vision=True rows are not row-paired "
            "by (task, t_start). File format may have changed."
        )
    unique_tasks = sorted(set(task_strs.tolist()))
    task_to_id = {t: i for i, t in enumerate(unique_tasks)}
    task_ids = np.array([task_to_id[t] for t in task_strs.tolist()], dtype=np.int64)
    return z_eeg, z_vis, t_starts, task_ids, unique_tasks


def main():
    args = parse_args()

    if args.from_npz:
        print(f"Loading pre-computed shared-space embeddings from {args.from_npz} ...")
        z_eeg, z_vis_paired, t_starts, task_ids, unique_tasks = _load_from_npz(args.from_npz)
        M = z_eeg.shape[0]
        print(f"  M={M} paired (EEG, vision) rows, tasks={unique_tasks}")

        pool_embeds_placeholder, anchor_to_pool, pool_keys = _build_vision_pool(
            task_ids, t_starts, z_vis_paired, args.t_bucket_s,
        )
        # The .npz already carries head-projected z_vis; the pool builder used
        # z_vis_paired as the "embeddings" so the dedup logic naturally selects
        # one projected vector per unique (task, t_bucket). No further
        # projection needed.
        z_vis = pool_embeds_placeholder.astype(np.float32)
        N_pool = z_vis.shape[0]
        print(f"  N_pool={N_pool} unique (task, t_bucket={args.t_bucket_s}s) vision targets")

        S = z_eeg @ z_vis.T
        e2v_topk = _topk_e2v(S, anchor_to_pool, args.topks)
        v2e_topk = _topk_v2e(S, anchor_to_pool, N_pool, args.topks)

        _report(args, results_meta={
            "source": "npz",
            "npz_path": args.from_npz,
            "n_eeg_anchors_M": M,
            "n_vision_pool_N": N_pool,
        }, e2v_topk=e2v_topk, v2e_topk=v2e_topk, N_pool=N_pool, M=M)
        return

    if args.config is None:
        raise SystemExit("--config is required unless --from-npz is set.")

    device = torch.device(args.device)
    cfg = load_config(args.config)
    print(f"Loading {args.split} split for task={cfg.data.task} ...")
    dataset = build_dataset(cfg, args.split, SCALAR_FEATURES_DEFAULT)
    print(f"  n_recordings={len(dataset)}, n_chans={dataset.n_chans}, n_times={dataset.n_times}")

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

    if not args.random_baseline:
        assert args.checkpoint, "Provide --checkpoint or use --random-baseline"
        enc_sd, head_sd = _load_clip_ckpt(args.checkpoint)
        miss_e, unex_e = encoder.load_state_dict(enc_sd, strict=False)
        miss_h, unex_h = clip_head.load_state_dict(head_sd, strict=False)
        print(f"  encoder: loaded {len(enc_sd)}  missing={len(miss_e)}  unexpected={len(unex_e)}")
        print(f"  clip_head: loaded {len(head_sd)}  missing={len(miss_h)}  unexpected={len(unex_h)}")
    else:
        print("  random-baseline: fresh-init encoder + head")

    encoder.to(device).eval()
    clip_head.to(device).eval()

    # ------------------------------------------------------------------
    # Encode every window in every recording; collect (X_eeg, t_start,
    # task_id, V-JEPA-2 target).
    # ------------------------------------------------------------------
    n_rec = len(dataset) if args.max_recordings is None else min(args.max_recordings, len(dataset))
    all_X_eeg, all_t_starts, all_task_ids, all_embeds = [], [], [], []
    for rec_idx in range(n_rec):
        X, t_starts, embeds = _embed_recording_and_meta(
            encoder, dataset, rec_idx, device, args.encode_batch,
        )
        task = dataset._recording_tasks[rec_idx]
        task_id = dataset._task_to_idx[task]
        all_X_eeg.append(X)
        all_t_starts.append(t_starts)
        all_task_ids.append(np.full(len(X), task_id, dtype=np.int64))
        all_embeds.append(embeds)
        if (rec_idx + 1) % 20 == 0:
            print(f"  encoded {rec_idx + 1}/{n_rec} recordings")

    X_eeg = np.concatenate(all_X_eeg, axis=0).astype(np.float32)     # [M, D_enc]
    t_starts = np.concatenate(all_t_starts, axis=0).astype(np.float32)  # [M]
    task_ids = np.concatenate(all_task_ids, axis=0).astype(np.int64)    # [M]
    embeds = np.concatenate(all_embeds, axis=0).astype(np.float32)      # [M, D_v]
    M = X_eeg.shape[0]
    print(f"Anchor total: M={M} EEG windows across {n_rec} recordings")

    # ------------------------------------------------------------------
    # Project through the clip_head and build the vision pool.
    # ------------------------------------------------------------------
    # project_eeg expects [B, D, T, 1, 1] and treats T as the token axis. Our
    # per-window vectors from embed_recording_all_windows have already been
    # pooled to one token per window (T=1 after pool_to_windows), so we add
    # trailing dims to match the head's expected shape.
    with torch.no_grad():
        eeg_batched = (
            torch.from_numpy(X_eeg).to(device).view(-1, X_eeg.shape[1], 1, 1, 1)
        )
        z_eeg = clip_head.project_eeg(eeg_batched).float().cpu().numpy()

    pool_embeds, anchor_to_pool, pool_keys = _build_vision_pool(
        task_ids, t_starts, embeds, args.t_bucket_s,
    )
    with torch.no_grad():
        z_vis = clip_head.project_vision(
            torch.from_numpy(pool_embeds).float().to(device)
        ).float().cpu().numpy()
    N_pool = z_vis.shape[0]
    print(f"Pool: N={N_pool} unique (task, t_bucket) V-JEPA-2 targets "
          f"(bucketed at {args.t_bucket_s}s)")

    # ------------------------------------------------------------------
    # Retrieval matrix + Top-K.
    # ------------------------------------------------------------------
    # z_eeg [M, P], z_vis [N_pool, P] — cosine (they are already L2-normalized by the head).
    S = z_eeg @ z_vis.T                                   # [M, N_pool]

    e2v_topk = _topk_e2v(S, anchor_to_pool, args.topks)
    v2e_topk = _topk_v2e(S, anchor_to_pool, N_pool, args.topks)

    _report(args, results_meta={
        "source": "checkpoint",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_recordings": n_rec,
        "n_eeg_anchors_M": M,
        "n_vision_pool_N": N_pool,
    }, e2v_topk=e2v_topk, v2e_topk=v2e_topk, N_pool=N_pool, M=M)


def _report(args, results_meta: dict, e2v_topk: dict[int, float],
            v2e_topk: dict[int, float], N_pool: int, M: int):
    """Assemble the results JSON + stdout summary shared by both entry paths."""
    chance_e2v = {k: k / N_pool for k in args.topks}
    chance_v2e = {k: k / M for k in args.topks}

    results = {
        **results_meta,
        "random_baseline": args.random_baseline,
        "t_bucket_s": args.t_bucket_s,
        "topks": args.topks,
        "e2v_top_k": {str(k): e2v_topk[k] for k in args.topks},
        "e2v_chance": {str(k): chance_e2v[k] for k in args.topks},
        "e2v_relative": {str(k): e2v_topk[k] / max(chance_e2v[k], 1e-12) for k in args.topks},
        "v2e_top_k": {str(k): v2e_topk[k] for k in args.topks},
        "v2e_chance": {str(k): chance_v2e[k] for k in args.topks},
        "v2e_relative": {str(k): v2e_topk[k] / max(chance_v2e[k], 1e-12) for k in args.topks},
    }

    print()
    print(f"e→v Top-K (retrieve V-JEPA-2 target for each EEG anchor, N_pool={N_pool}):")
    for k in args.topks:
        print(f"  Top-{k}: {e2v_topk[k]:.4f}  (chance {chance_e2v[k]:.4f}, "
              f"{e2v_topk[k]/max(chance_e2v[k], 1e-12):.1f}× above chance)")
    print(f"v→e Top-K (retrieve EEG anchor for each vision pool entry, M={M}):")
    for k in args.topks:
        print(f"  Top-{k}: {v2e_topk[k]:.4f}  (chance {chance_v2e[k]:.4f}, "
              f"{v2e_topk[k]/max(chance_v2e[k], 1e-12):.1f}× above chance)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
