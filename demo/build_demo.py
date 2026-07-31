"""Turn a demo export npz into a self-contained e->v retrieval page.

Runs locally. Recomputes Top-K retrieval from the exported shared-space
embeddings using the *same* helpers that produced the numbers in
``experiments/clip_pretraining/scene_clip_from_checkpoint/RESULTS.md`` §3.10,
asserts they still match, then renders the qualitative page: for a handful of
EEG windows, the top-5 movie candidates the model retrieves, with the correct
one marked.

Usage::

    PYTHONPATH=. .venv/bin/python demo/build_demo.py \\
        --npz demo/data/demo_test.npz \\
        --reference experiments/clip_pretraining/scene_clip_from_checkpoint/\\
probe_results/retrieval_warmstart_lr3e4_retrain_jul22_ep299_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eb_jepa.evaluation.clip_probe.retrieval import (
    _build_group_pool,
    _build_time_pool,
    _evaluate_all_levels,
    _load_from_npz,
)

from demo import movie_assets as ma
from demo.render import render_page

LEVELS = ("time", "shot", "scene")
TOPKS = [1, 5, 10]
T_BUCKET_S = 0.5  # every published number used 0.5; the retrieval.py CLI
                  # default of 0.1 would give a different pool entirely.
N_CANDIDATES = 5


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Export from export_retrieval_npz.py")
    ap.add_argument("--reference", default=None,
                    help="Committed retrieval JSON to assert against. Strongly "
                         "recommended — without it the page's numbers are unverified.")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--n-rows", type=int, default=12, help="Showcase rows on the page.")
    ap.add_argument("--n-misses", type=int, default=3,
                    help="How many of those rows should be failures.")
    ap.add_argument("--gifs", action=argparse.BooleanOptionalAction, default=True,
                    help="Animate the ground-truth candidate of each row.")
    ap.add_argument("--out-dir", default="demo", help="Where index.html + assets/ go.")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


def build_level(level: str, z_vis, task_ids, t_starts, shot_ids, scene_ids):
    """Pool for one granularity, plus the (task, group) key of each pool row.

    ``retrieval.py``'s builders assign pool indices by first occurrence but do
    not hand back the key map, so reconstruct it the same way — the demo needs
    to know *which* movie moment each pool row stands for.
    """
    if level == "time":
        pool, a2p, valid = _build_time_pool(task_ids, t_starts, z_vis, T_BUCKET_S)
        groups = np.round(t_starts / T_BUCKET_S).astype(np.int64)
    else:
        group_ids = shot_ids if level == "shot" else scene_ids
        pool, a2p, valid = _build_group_pool(task_ids, group_ids, z_vis)
        groups = group_ids

    keys, seen = [], {}
    for task, grp in zip(task_ids[valid].tolist(), groups[valid].tolist()):
        k = (task, grp)
        if k not in seen:
            seen[k] = len(seen)
            keys.append(k)
    assert len(keys) == pool.shape[0], f"{level}: key reconstruction disagrees with pool"

    # Anchor row index -> row index within the valid (filtered) similarity matrix.
    pos = np.full(len(valid), -1, dtype=np.int64)
    pos[valid] = np.arange(int(valid.sum()))
    return pool, a2p, valid, pos, keys


def check_against_reference(levels: dict, ref_path: str, tol: float) -> None:
    ref = json.loads(Path(ref_path).read_text())
    problems = []
    for level in LEVELS:
        got, want = levels[level], ref["levels"][level]
        if got["n_vision_pool_N"] != want["n_vision_pool_N"]:
            problems.append(f"{level}: pool N {got['n_vision_pool_N']} != {want['n_vision_pool_N']}")
        for k in TOPKS:
            for direction in ("e2v_top_k", "v2e_top_k"):
                a = got[direction][k]
                b = want[direction][str(k)]
                if abs(a - b) > tol:
                    problems.append(f"{level} {direction}@{k}: {a:.4f} != {b:.4f}")
    if problems:
        raise SystemExit(
            "Recomputed retrieval does not match the committed results:\n  "
            + "\n  ".join(problems)
            + "\n\nThe export is wrong — refusing to build a page with numbers that "
              "do not reproduce RESULTS.md §3.10."
        )
    print(f"✓ retrieval reproduces {Path(ref_path).name} within {tol}")


# ---------------------------------------------------------------------------
# Showcase selection
# ---------------------------------------------------------------------------


def rank_of_truth(sim_row: np.ndarray, correct: int) -> int:
    """1-based rank of the correct pool entry in a similarity row."""
    return int((sim_row > sim_row[correct]).sum()) + 1


def blank_scenes(indexes, tasks, task_ids, scene_ids, rel_floor=0.5):
    """(task_i, scene) keys whose representative still is visually near-empty.

    *The Present* opens on ~10 s of black title card. The model does rank it
    first for a lot of windows — that stays visible in the candidate strips —
    but asking the reader "which scene was this?" about a black rectangle
    demonstrates nothing, so these are excluded from being the *question*.

    The cut is "far below the bulk", not a quantile: real scenes here all sit at
    entropy 7.2–7.8 while the title card is 1.4, so anything under half the
    median is unambiguously blank. A quantile would throw away ordinary scenes.
    """
    ent: dict[tuple, float] = {}
    for t_i, task in enumerate(tasks):
        mi = indexes.get(task)
        if mi is None:
            continue
        for s in np.unique(scene_ids[(task_ids == t_i) & (scene_ids >= 0)]).tolist():
            span = mi.scene_span(int(s))
            if span is not None:
                ent[(t_i, int(s))] = mi.representative_entropy(span)
    if not ent:
        return set()
    cutoff = rel_floor * float(np.median(list(ent.values())))
    return {k for k, v in ent.items() if v < cutoff}


def select_rows(showcase_idx, z_eeg, level_data, scene_ids, task_ids,
                n_rows, n_misses, skip_keys=frozenset()):
    """Pick page rows: mostly hits, a few honest failures, spread over scenes.

    Selection is on the SCENE level (the demo's headline granularity); the same
    windows are then shown at every level so the toggle compares like for like.
    """
    pool, a2p, valid, pos, _ = level_data["scene"]
    cand = [i for i in showcase_idx
            if valid[i] and (int(task_ids[i]), int(scene_ids[i])) not in skip_keys]
    sims = z_eeg[cand] @ pool.T
    ranks = np.array([rank_of_truth(sims[r], a2p[pos[i]]) for r, i in enumerate(cand)])

    hits = [(i, rk) for i, rk in zip(cand, ranks) if rk <= N_CANDIDATES]
    misses = [(i, rk) for i, rk in zip(cand, ranks) if rk > N_CANDIDATES]
    n_hits = max(0, n_rows - n_misses)

    def spread(pairs, k):
        """Greedily take items from scenes not yet represented, best rank first."""
        pairs = sorted(pairs, key=lambda p: (p[1], int(scene_ids[p[0]])))
        picked, used = [], set()
        for want_new in (True, False):
            for i, rk in pairs:
                if len(picked) >= k:
                    break
                s = int(scene_ids[i])
                if want_new and s in used:
                    continue
                if (i, rk) in picked:
                    continue
                picked.append((i, rk))
                used.add(s)
        return picked[:k]

    chosen = spread(hits, n_hits) + spread(misses, n_misses)
    # Present in movie order so the page reads as a walk through the film.
    return [i for i, _ in sorted(chosen, key=lambda p: int(scene_ids[p[0]]))]


def top1_concentration(level_data, z_eeg):
    """The single pool entry the model names most often, and how often it is right.

    Aggregate Top-K hides collapse: a model can score well while dumping a large
    share of its first guesses onto one attractor. On this checkpoint that is
    very visible on the page, so measure it rather than let the reader wonder
    why the same thumbnail keeps appearing at rank 1.
    """
    pool, a2p, valid, pos, keys = level_data
    top1 = (z_eeg[valid] @ pool.T).argmax(1)
    counts = np.bincount(top1, minlength=len(keys))
    j = int(counts.argmax())
    m = len(top1)
    return {
        "group": int(keys[j][1]),
        "task_i": int(keys[j][0]),
        "predicted_share": float(counts[j] / m),
        "true_share": float((a2p == j).sum() / m),
        "uniform_share": 1.0 / len(keys),
    }


def row_candidates(row: int, level_data, z_eeg, k: int = N_CANDIDATES):
    """Top-k pool entries for one EEG window, plus where the truth landed."""
    pool, a2p, valid, pos, keys = level_data
    if not valid[row]:
        return None
    sim = z_eeg[row] @ pool.T
    correct = int(a2p[pos[row]])
    order = np.argsort(-sim)[:k]
    return {
        "correct_key": keys[correct],
        "truth_rank": rank_of_truth(sim, correct),
        "n_pool": int(pool.shape[0]),
        "candidates": [
            {"key": keys[j], "score": float(sim[j]), "is_truth": int(j) == correct}
            for j in order.tolist()
        ],
    }


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


def build_indexes(tasks):
    indexes = {t: ma.MovieIndex(t) for t in tasks if t in ma.MOVIE_METADATA}
    for t in tasks:
        if t not in indexes:
            print(f"  ! no movie metadata for task {t!r} — its candidates get no image")
    return indexes


def build_assets(rows, tasks, indexes, out_dir: Path, want_gifs: bool):
    """Render a still for every candidate shown, and a GIF for each truth."""
    # (task, level, group) -> Span, collected across every row and level.
    spans: dict[tuple, ma.Span] = {}
    gif_keys: set[tuple] = set()
    for row in rows:
        for level, res in row["levels"].items():
            if res is None:
                continue
            # The truth needs an image even when it misses the top-5 — that is
            # exactly the row where the page shows "here is what it should have
            # picked".
            wanted = [(c["key"], c["is_truth"]) for c in res["candidates"]]
            wanted.append((res["correct_key"], True))
            for (task_i, grp), is_truth in wanted:
                task = tasks[task_i]
                if task not in indexes:
                    continue
                span = indexes[task].span_for(level, grp, T_BUCKET_S)
                if span is None:
                    continue
                key = (task, level, grp)
                spans[key] = span
                if want_gifs and is_truth:
                    gif_keys.add(key)

    # One sequential decode per movie covers every still and every GIF frame.
    assets: dict[tuple, dict] = {}
    asset_dir = out_dir / "assets"
    for task, mi in indexes.items():
        keys = [k for k in spans if k[0] == task]
        if not keys:
            continue
        times: set[float] = set()
        still_t: dict[tuple, float] = {}
        gif_times: dict[tuple, list[float]] = {}
        for k in keys:
            t = mi.representative_time(spans[k])
            still_t[k] = t
            times.add(t)
            if k in gif_keys:
                gt = ma.gif_frame_times(spans[k], duration=mi.duration, center=t)
                gif_times[k] = gt
                times |= set(gt)
        print(f"  decoding {task}: {len(times)} frames for {len(keys)} candidates "
              f"({len(gif_times)} animated)")
        frames = ma.decode_frames(task, times, mi.fps)

        def at(t: float) -> np.ndarray:
            return frames[int(round(t * mi.fps))]

        for k in keys:
            _, level, grp = k
            stem = f"{task}_{level}_{grp}"
            still = ma.write_still(at(still_t[k]), asset_dir / f"{stem}.jpg")
            entry = {"still": f"assets/{still.name}", "label": spans[k].label,
                     "short": spans[k].short,
                     "t0": spans[k].t0, "t1": spans[k].t1}
            if k in gif_times:
                gif = ma.write_gif([at(t) for t in gif_times[k]], asset_dir / f"{stem}.gif")
                entry["gif"] = f"assets/{gif.name}"
            assets[k] = entry
    return assets


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    npz_path = Path(args.npz)

    # The same loader retrieval.py --from-npz uses, so the demo and the metric
    # script cannot drift apart in how they read the export.
    z_eeg, z_vis, t_starts, shot_ids, scene_ids, task_ids, tasks = _load_from_npz(str(npz_path))
    extra = np.load(npz_path, allow_pickle=False)
    provenance = json.loads(str(extra["provenance_json"]))
    print(f"loaded M={len(z_eeg)} windows, P={z_eeg.shape[1]}, tasks={tasks}")

    metrics = _evaluate_all_levels(
        z_eeg, z_vis, task_ids, t_starts, shot_ids, scene_ids, TOPKS, T_BUCKET_S
    )
    for level in LEVELS:
        m = metrics[level]
        n = m["n_vision_pool_N"]
        e2v = m["e2v_top_k"]
        print(f"  {level:5s} N={n:4d}  e->v top1={e2v[1]:.3f} top5={e2v[5]:.3f} "
              f"top10={e2v[10]:.3f}  (chance {1 / n:.3f}/{5 / n:.3f}/{10 / n:.3f})")

    if args.reference:
        check_against_reference(metrics, args.reference, args.tolerance)
    else:
        print("! no --reference given; page numbers are unverified against RESULTS.md")

    level_data = {
        lv: build_level(lv, z_vis, task_ids, t_starts, shot_ids, scene_ids)
        for lv in LEVELS
    }

    indexes = build_indexes(tasks)
    skip = blank_scenes(indexes, tasks, task_ids, scene_ids)
    if skip:
        print(f"  excluding {len(skip)} visually-empty scene(s) from row selection: "
              + ", ".join(f"{tasks[t]}#{s}" for t, s in sorted(skip)))

    showcase_idx = extra["showcase_idx"]
    chosen = select_rows(showcase_idx, z_eeg, level_data, scene_ids, task_ids,
                         args.n_rows, args.n_misses, skip_keys=skip)
    slot = {int(r): k for k, r in enumerate(showcase_idx.tolist())}

    # Stable, non-identifying participant labels in first-appearance order.
    fifs = extra["showcase_fif"]
    participant, seen_fif = {}, {}
    for r in chosen:
        f = str(fifs[slot[r]])
        if f not in seen_fif:
            seen_fif[f] = f"participant {len(seen_fif) + 1}"
        participant[r] = seen_fif[f]

    rows = []
    for r in chosen:
        rows.append({
            "row": int(r),
            "task": tasks[int(task_ids[r])],
            "t_start": float(t_starts[r]),
            "participant": participant[r],
            "eeg": extra["showcase_eeg"][slot[r]].astype(np.float32),
            "levels": {lv: row_candidates(int(r), level_data[lv], z_eeg) for lv in LEVELS},
        })
    print(f"selected {len(rows)} showcase rows "
          f"({sum(1 for x in rows if x['levels']['scene']['truth_rank'] > N_CANDIDATES)} misses)")

    out_dir = Path(args.out_dir)
    assets = build_assets(rows, tasks, indexes, out_dir, args.gifs)

    concentration = top1_concentration(level_data["scene"], z_eeg)
    conc_label = None
    mi = indexes.get(tasks[concentration["task_i"]])
    if mi is not None:
        span = mi.scene_span(concentration["group"])
        conc_label = span.label if span else None
    print(f"  top-1 collapse: scene {concentration['group']} predicted for "
          f"{concentration['predicted_share']:.1%} of windows, correct for "
          f"{concentration['true_share']:.1%}")

    html_path = render_page(
        rows=rows, assets=assets, metrics=metrics, tasks=tasks,
        provenance=provenance, topks=TOPKS, out_dir=out_dir,
        concentration={**concentration, "label": conc_label},
        ch_pos=extra["ch_pos"] if "ch_pos" in extra else None,
        ch_names=[str(c) for c in extra["ch_names"]] if "ch_names" in extra else [],
        reference=args.reference,
    )
    total = sum(f.stat().st_size for f in (out_dir / "assets").glob("*"))
    print(f"wrote {html_path}  (+ {total / 1e6:.1f} MB of assets)")


if __name__ == "__main__":
    main()
