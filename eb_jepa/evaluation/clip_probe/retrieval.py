"""Top-K retrieval evaluation for EEG ↔ V-JEPA-2 CLIP-style checkpoints.

Implements the SSL-literature-standard retrieval evaluation used by
EEG2Video, EEG-CLIP, and NICE-EEG. Three pool granularities computed in one
pass:

- **time** (finest): each unique (task, round(t_start / t_bucket_s)) is one
  candidate. V-JEPA-2 targets at the same movie-time are identical, so the
  pool entry is the first occurrence's projected vector. Retrieval means
  "identify the exact movie moment you're watching." Largest pool → highest
  chance level.
- **shot**: each unique (task, shot_id) is one candidate. Pool entry is the
  L2-normalized centroid of projected V-JEPA-2 vectors within that shot.
  Retrieval means "identify the shot you're watching." Smaller pool.
- **scene**: each unique (task, scene_id) is one candidate. Pool entry is
  the L2-normalized centroid across all windows in that scene. Retrieval
  means "identify the scene you're watching." Smallest pool.

Windows without shot / scene coverage (id = -1) are dropped from those
levels but retained for the time-level metrics.

Two directions per level:

- **e→v** (EEG anchor → find paired vision): for each EEG window, is the
  correct pool entry (its own time / shot / scene) in the top-K by cosine?
- **v→e** (vision anchor → find matching EEG): for each pool entry, is any
  EEG window belonging to that pool entry in the top-K by cosine? Multi-
  positive at all levels because multiple subject-windows map to the same
  pool entry.

Chance level for e→v Top-K is `K / N_pool`. Report both raw accuracy and
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


def _build_pool_first_occurrence(keys: list, embeds: np.ndarray):
    """Dedupe by exact key equality; take the first occurrence's embedding as
    the pool entry. Suitable for time-bucket retrieval where V-JEPA-2 targets
    at the same movie-time are identical (up to window-boundary rounding).

    Args:
        keys:   list[hashable] of length M — one key per anchor (e.g. tuples).
        embeds: [M, D] float — per-anchor embeddings (V-JEPA-2 raw, or z_vis).

    Returns:
        pool_embeds:    [N_pool, D] float
        anchor_to_pool: [M] int64 — for each anchor row, its pool index
        valid_mask:     [M] bool  — all True for this variant
    """
    key_to_idx: dict = {}
    pool_embeds_list: list[np.ndarray] = []
    anchor_to_pool = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        if k not in key_to_idx:
            key_to_idx[k] = len(key_to_idx)
            pool_embeds_list.append(embeds[i])
        anchor_to_pool[i] = key_to_idx[k]
    pool_embeds = np.stack(pool_embeds_list, axis=0)
    valid_mask = np.ones(len(keys), dtype=bool)
    return pool_embeds, anchor_to_pool, valid_mask


def _build_pool_centroid(keys: list, embeds: np.ndarray):
    """Dedupe by key equality; take the L2-normalized MEAN of all embeddings
    at each key as the pool entry (centroid). Suitable for shot/scene
    retrieval where the "pool entry" semantically means the group centroid
    in the projected (L2-normalized) space.

    Callers should pre-filter to remove invalid keys (e.g. shot_id=-1) —
    this function assumes all keys in ``keys`` are valid pool members.

    Args:
        keys:   list[hashable] of length M — pre-filtered per-anchor keys.
        embeds: [M, D] float — per-anchor embeddings (typically L2-normalized z_vis).

    Returns:
        pool_embeds:    [N_pool, D] float — one L2-normalized centroid per unique key
        anchor_to_pool: [M] int64 — pool index for each anchor
        valid_mask:     [M] bool — all True (kept for API parity with the other builder)
    """
    if len(keys) == 0:
        raise ValueError("No valid anchors for pool centroid — empty keys list.")
    key_to_idx: dict = {}
    anchor_to_pool = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        if k not in key_to_idx:
            key_to_idx[k] = len(key_to_idx)
        anchor_to_pool[i] = key_to_idx[k]
    N_pool = len(key_to_idx)
    D = embeds.shape[1]
    sums = np.zeros((N_pool, D), dtype=np.float64)
    counts = np.zeros(N_pool, dtype=np.int64)
    np.add.at(sums, anchor_to_pool, embeds)
    np.add.at(counts, anchor_to_pool, 1)
    centroids = (sums / counts[:, None]).astype(np.float32)
    # Re-normalize because the mean of L2-normalized vectors is not itself unit-length.
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.clip(norms, 1e-8, None)
    valid_mask = np.ones(len(keys), dtype=bool)
    return centroids, anchor_to_pool, valid_mask


def _build_time_pool(task_ids, t_starts, embeds, t_bucket_s):
    """Time-bucket pool: key = (task, round(t_start / t_bucket_s))."""
    buckets = np.round(t_starts / t_bucket_s).astype(np.int64)
    keys = list(zip(task_ids.tolist(), buckets.tolist()))
    return _build_pool_first_occurrence(keys, embeds)


def _build_group_pool(task_ids, group_ids, embeds):
    """Shot/scene pool: key = (task, group_id). Pool entry = L2-normalized
    centroid over anchors sharing the key. Anchors with group_id=-1 are
    filtered out before centroid computation.

    Returns:
        pool_embeds:    [N_pool, D] float32
        anchor_to_pool: [M_valid] int64 — pool index for each retained anchor
        valid_mask:     [M] bool — which anchors survived (group_id >= 0)
    """
    valid_mask = group_ids >= 0
    if not valid_mask.any():
        return (
            np.zeros((0, embeds.shape[1]), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            valid_mask,
        )
    task_v = task_ids[valid_mask]
    grp_v = group_ids[valid_mask]
    embeds_v = embeds[valid_mask]
    keys = list(zip(task_v.tolist(), grp_v.tolist()))
    pool, a2p, _ = _build_pool_centroid(keys, embeds_v)
    return pool, a2p, valid_mask


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
        z_vis_paired [M, P] float32 — paired V-JEPA-2 vectors (rows with is_vision=True),
            aligned 1:1 with z_eeg by row order (coherent_subsample preserves pairing)
        t_starts [M] float32 — movie time per (EEG, vision) pair
        shot_ids [M] int64 — shot id per pair (-1 = no shot boundary)
        scene_ids [M] int64 — scene id per pair (-1 = no scene)
        task_ids [M] int64 — integer task IDs (built from unique task strings)
        unique_tasks: list[str] — task label strings, indexed by task_ids
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
    # Optional shot / scene columns — present in newer npz files.
    shot_ids = (
        d["shot_id"][~is_vision].astype(np.int64) if "shot_id" in d.files
        else np.full(len(t_starts), -1, dtype=np.int64)
    )
    scene_ids = (
        d["scene_id"][~is_vision].astype(np.int64) if "scene_id" in d.files
        else np.full(len(t_starts), -1, dtype=np.int64)
    )
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
    return z_eeg, z_vis, t_starts, shot_ids, scene_ids, task_ids, unique_tasks


def _eval_time_level(z_eeg, z_vis_paired, task_ids, t_starts, topks, t_bucket_s):
    """Time-bucket-level retrieval. Each unique (task, round(t/dt)) is one candidate.
    Uses first-occurrence deduplication (V-JEPA-2 targets at the same movie-time
    are identical up to window-boundary rounding, so any occurrence works)."""
    pool, a2p, _ = _build_time_pool(task_ids, t_starts, z_vis_paired, t_bucket_s)
    N = pool.shape[0]
    M = z_eeg.shape[0]
    S = z_eeg @ pool.T
    return {
        "n_eeg_anchors_M": int(M),
        "n_vision_pool_N": int(N),
        "e2v_top_k": _topk_e2v(S, a2p, topks),
        "v2e_top_k": _topk_v2e(S, a2p, N, topks),
    }


def _eval_group_level(z_eeg, z_vis_paired, task_ids, group_ids, topks, label):
    """Shot- or scene-level retrieval. Pool entries are L2-normalized centroids
    of z_vis over all anchors sharing (task, group_id). Anchors with
    group_id=-1 are filtered from both anchors and the pool."""
    pool, a2p, valid_mask = _build_group_pool(task_ids, group_ids, z_vis_paired)
    if pool.shape[0] == 0:
        return {
            "n_eeg_anchors_M": 0,
            "n_vision_pool_N": 0,
            "e2v_top_k": {k: 0.0 for k in topks},
            "v2e_top_k": {k: 0.0 for k in topks},
            "note": f"no valid {label} labels; all anchors had group_id=-1",
        }
    z_eeg_v = z_eeg[valid_mask]
    N = pool.shape[0]
    M = z_eeg_v.shape[0]
    S = z_eeg_v @ pool.T
    return {
        "n_eeg_anchors_M": int(M),
        "n_vision_pool_N": int(N),
        "e2v_top_k": _topk_e2v(S, a2p, topks),
        "v2e_top_k": _topk_v2e(S, a2p, N, topks),
    }


def _evaluate_all_levels(z_eeg, z_vis_paired, task_ids, t_starts, shot_ids,
                         scene_ids, topks, t_bucket_s):
    """Compute retrieval metrics at all three pool granularities."""
    return {
        "time": _eval_time_level(z_eeg, z_vis_paired, task_ids, t_starts, topks, t_bucket_s),
        "shot": _eval_group_level(z_eeg, z_vis_paired, task_ids, shot_ids, topks, "shot"),
        "scene": _eval_group_level(z_eeg, z_vis_paired, task_ids, scene_ids, topks, "scene"),
    }


def main():
    args = parse_args()

    if args.from_npz:
        print(f"Loading pre-computed shared-space embeddings from {args.from_npz} ...")
        z_eeg, z_vis_paired, t_starts, shot_ids, scene_ids, task_ids, unique_tasks = (
            _load_from_npz(args.from_npz)
        )
        M = z_eeg.shape[0]
        print(f"  M={M} paired (EEG, vision) rows, tasks={unique_tasks}")
        print(f"  shot coverage: {(shot_ids >= 0).sum()}/{M}   "
              f"scene coverage: {(scene_ids >= 0).sum()}/{M}")

        levels = _evaluate_all_levels(
            z_eeg, z_vis_paired, task_ids, t_starts, shot_ids, scene_ids,
            args.topks, args.t_bucket_s,
        )
        _report(args, {
            "source": "npz",
            "npz_path": args.from_npz,
            "n_eeg_anchors_total_M": int(M),
        }, levels)
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
    # Encode every window in every recording; also collect per-anchor
    # (t_start, shot_id, scene_id, V-JEPA-2 target).
    # ------------------------------------------------------------------
    n_rec = len(dataset) if args.max_recordings is None else min(args.max_recordings, len(dataset))
    all_X_eeg, all_t_starts, all_task_ids, all_embeds = [], [], [], []
    all_shot_ids, all_scene_ids = [], []
    for rec_idx in range(n_rec):
        X, t_starts_rec, embeds_rec = _embed_recording_and_meta(
            encoder, dataset, rec_idx, device, args.encode_batch,
        )
        task = dataset._recording_tasks[rec_idx]
        task_id = dataset._task_to_idx[task]
        all_X_eeg.append(X)
        all_t_starts.append(t_starts_rec)
        all_task_ids.append(np.full(len(X), task_id, dtype=np.int64))
        all_embeds.append(embeds_rec)
        # shot_id_recordings / scene_id_recordings may be empty when the
        # dataset wasn't in recipe_mode; fall back to -1.
        if dataset.shot_id_recordings:
            all_shot_ids.append(dataset.shot_id_recordings[rec_idx].numpy().astype(np.int64))
        else:
            all_shot_ids.append(np.full(len(X), -1, dtype=np.int64))
        if dataset.scene_id_recordings:
            all_scene_ids.append(dataset.scene_id_recordings[rec_idx].numpy().astype(np.int64))
        else:
            all_scene_ids.append(np.full(len(X), -1, dtype=np.int64))
        if (rec_idx + 1) % 20 == 0:
            print(f"  encoded {rec_idx + 1}/{n_rec} recordings")

    X_eeg = np.concatenate(all_X_eeg, axis=0).astype(np.float32)     # [M, D_enc]
    t_starts = np.concatenate(all_t_starts, axis=0).astype(np.float32)  # [M]
    task_ids = np.concatenate(all_task_ids, axis=0).astype(np.int64)    # [M]
    embeds = np.concatenate(all_embeds, axis=0).astype(np.float32)      # [M, D_v]
    shot_ids = np.concatenate(all_shot_ids, axis=0).astype(np.int64)    # [M]
    scene_ids = np.concatenate(all_scene_ids, axis=0).astype(np.int64)  # [M]
    M = X_eeg.shape[0]
    print(f"Anchor total: M={M} EEG windows across {n_rec} recordings")
    print(f"  shot coverage: {(shot_ids >= 0).sum()}/{M}   "
          f"scene coverage: {(scene_ids >= 0).sum()}/{M}")

    # ------------------------------------------------------------------
    # Project through the clip_head. project_eeg expects [B, D, T, 1, 1] and
    # treats T as the token axis. Our per-window vectors have already been
    # pooled to one token per window (T=1 after pool_to_windows), so we add
    # trailing dims to match the head's expected shape.
    # ------------------------------------------------------------------
    with torch.no_grad():
        eeg_batched = (
            torch.from_numpy(X_eeg).to(device).view(-1, X_eeg.shape[1], 1, 1, 1)
        )
        z_eeg = clip_head.project_eeg(eeg_batched).float().cpu().numpy()
        z_vis_paired = clip_head.project_vision(
            torch.from_numpy(embeds).float().to(device)
        ).float().cpu().numpy()

    levels = _evaluate_all_levels(
        z_eeg, z_vis_paired, task_ids, t_starts, shot_ids, scene_ids,
        args.topks, args.t_bucket_s,
    )
    _report(args, {
        "source": "checkpoint",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_recordings": int(n_rec),
        "n_eeg_anchors_total_M": int(M),
    }, levels)


def _report(args, results_meta: dict, levels: dict):
    """Assemble the results JSON + stdout summary shared by both entry paths.

    ``levels`` is {'time': {...}, 'shot': {...}, 'scene': {...}} where each
    inner dict has n_eeg_anchors_M, n_vision_pool_N, e2v_top_k, v2e_top_k.
    """
    ks = args.topks

    def _add_chance_relative(level_res: dict) -> dict:
        M = level_res["n_eeg_anchors_M"]
        N = level_res["n_vision_pool_N"]
        e2v = level_res["e2v_top_k"]
        v2e = level_res["v2e_top_k"]
        chance_e2v = {k: (k / N if N else 0.0) for k in ks}
        chance_v2e = {k: (k / M if M else 0.0) for k in ks}
        return {
            **level_res,
            "e2v_top_k": {str(k): float(e2v[k]) for k in ks},
            "v2e_top_k": {str(k): float(v2e[k]) for k in ks},
            "e2v_chance": {str(k): chance_e2v[k] for k in ks},
            "v2e_chance": {str(k): chance_v2e[k] for k in ks},
            "e2v_relative": {str(k): (e2v[k] / max(chance_e2v[k], 1e-12)) for k in ks},
            "v2e_relative": {str(k): (v2e[k] / max(chance_v2e[k], 1e-12)) for k in ks},
        }

    levels_out = {name: _add_chance_relative(lvl) for name, lvl in levels.items()}
    results = {
        **results_meta,
        "random_baseline": args.random_baseline,
        "t_bucket_s": args.t_bucket_s,
        "topks": ks,
        "levels": levels_out,
    }

    for name, lvl in levels.items():
        M = lvl["n_eeg_anchors_M"]
        N = lvl["n_vision_pool_N"]
        if N == 0:
            print(f"\n[{name}] skipped ({lvl.get('note', 'no valid labels')})")
            continue
        e2v = lvl["e2v_top_k"]
        v2e = lvl["v2e_top_k"]
        print(f"\n[{name}] pool N={N} (candidates), anchors M={M}")
        print(f"  e→v Top-K (identify {name} from EEG):")
        for k in ks:
            ch = k / N
            print(f"    Top-{k}: {e2v[k]:.4f}  (chance {ch:.4f}, {e2v[k]/max(ch, 1e-12):.1f}× above chance)")
        print(f"  v→e Top-K (find EEG in {name}):")
        for k in ks:
            ch = k / M
            print(f"    Top-{k}: {v2e[k]:.4f}  (chance {ch:.4f}, {v2e[k]/max(ch, 1e-12):.1f}× above chance)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
