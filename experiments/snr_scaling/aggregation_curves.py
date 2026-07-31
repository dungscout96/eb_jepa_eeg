"""Empirical aggregation curves on the Top-K retrieval metric.

§3.10 of ``clip_pretraining/scene_clip_from_checkpoint/RESULTS.md`` is a
single-trial number: one 2 s EEG window against N visual centroids. This asks
what happens when the *query* pools more EEG, holding the candidate pool exactly
fixed — same centroids, same N, so chance stays K/N and every row is comparable.

Three regimes, in increasing order of what they assume:

  temporal-k   average k CONSECUTIVE windows within one recording,
               non-overlapping, label = majority group of the block. Assumes
               nothing about where shots or scenes begin — the honest
               "more seconds of EEG" curve.
  oracle-seg   average every window of one (recording, group). Uses ground-truth
               segment boundaries, so it is the upper bound on temporal-k rather
               than a deployable number.
  n-subjects   average over n recordings for the same group (oracle segment).
               One recording == one subject watching the film once.

Input is the shared-space export written by ``demo/export_retrieval_npz.py``
(a data dependency, not a code one — nothing here imports from ``demo/``).

Usage::

    PYTHONPATH=. uv run --group eeg python \\
        experiments/snr_scaling/aggregation_curves.py \\
        --npz demo/data/demo_val.npz \\
        --output experiments/snr_scaling/aggregation_val_ThePresent.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from eb_jepa.evaluation.clip_probe.retrieval import _build_group_pool, _load_from_npz

TOPKS = [1, 5, 10]
LEVELS = ["shot", "scene"]
TEMPORAL_K = [2, 4, 8, 16]
SUBJECT_N = [2, 4, 8, 16]
DRAWS = 20
WINDOW_S = 2.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="Shared-space export from demo/export_retrieval_npz.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None, help="Write results JSON here.")
    return ap.parse_args()


def _norm(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def build_group_level(level, z_vis, task_ids, shot_ids, scene_ids):
    """Pool + per-pool-row (task, group) key, matching retrieval.py's ordering.

    ``_build_group_pool`` assigns pool indices by first occurrence but does not
    return the key map; reconstruct it the same way so the attractor can be
    named.
    """
    groups = shot_ids if level == "shot" else scene_ids
    pool, a2p, valid = _build_group_pool(task_ids, groups, z_vis)
    keys, seen = [], {}
    for k in zip(task_ids[valid].tolist(), groups[valid].tolist()):
        if k not in seen:
            seen[k] = len(seen)
            keys.append(k)
    assert len(keys) == pool.shape[0]
    return pool, a2p, valid, keys


def topk(Q, pool, correct, ks=TOPKS):
    """Top-K accuracy plus the share of queries whose top-1 is the modal answer."""
    S = Q @ pool.T
    kmax = min(max(ks), S.shape[1])
    top = np.argpartition(-S, kth=kmax - 1, axis=1)[:, :kmax]
    r = np.arange(len(S))[:, None]
    top = top[r, np.argsort(-S[r, top], axis=1)]
    hits = top == correct[:, None]
    out = {k: float(hits[:, :k].any(axis=1).mean()) for k in ks}
    j, n = Counter(top[:, 0].tolist()).most_common(1)[0]
    out["modal_pool_idx"] = int(j)
    out["modal_share"] = float(n / len(S))
    return out


def run_level(level, z_eeg, z_vis, task_ids, t_starts, shot_ids, scene_ids,
              rec_ids, rng):
    pool, a2p, valid, keys = build_group_level(
        level, z_vis, task_ids, shot_ids, scene_ids)
    N = pool.shape[0]
    zi, ri, ti = z_eeg[valid], rec_ids[valid], t_starts[valid]
    groups = (shot_ids if level == "shot" else scene_ids)[valid]
    key_to_j = {k: j for j, k in enumerate(keys)}
    corr = np.array([key_to_j[(int(a), int(b))]
                     for a, b in zip(task_ids[valid], groups)])
    rows = {}

    rows["k=1"] = {**topk(zi, pool, corr), "m": int(len(zi))}

    for k in TEMPORAL_K:
        Q, C = [], []
        for r in np.unique(ri):
            m = ri == r
            order = np.argsort(ti[m])
            zr, cr = zi[m][order], corr[m][order]
            for b in range(len(zr) // k):
                sl = slice(b * k, (b + 1) * k)
                Q.append(zr[sl].mean(0))
                C.append(Counter(cr[sl].tolist()).most_common(1)[0][0])
        rows[f"k={k}"] = {**topk(_norm(np.stack(Q)), pool, np.array(C)), "m": len(Q)}

    Q, C = [], []
    for r in np.unique(ri):
        m = ri == r
        for j in np.unique(corr[m]):
            Q.append(zi[m][corr[m] == j].mean(0))
            C.append(j)
    rows["oracle_segment"] = {**topk(_norm(np.stack(Q)), pool, np.array(C)), "m": len(Q)}

    recs = np.unique(ri)
    for n in SUBJECT_N + [len(recs)]:
        if n > len(recs):
            continue
        draws = 1 if n == len(recs) else DRAWS
        acc = {k: [] for k in TOPKS}
        modal = []
        for _ in range(draws):
            sel = recs if n == len(recs) else rng.choice(recs, n, replace=False)
            mask = np.isin(ri, sel)
            Q, C = [], []
            for j in np.unique(corr[mask]):
                Q.append(zi[mask][corr[mask] == j].mean(0))
                C.append(j)
            res = topk(_norm(np.stack(Q)), pool, np.array(C))
            for k in TOPKS:
                acc[k].append(res[k])
            modal.append(res["modal_share"])
        rows[f"subjects={n}"] = {
            **{k: float(np.mean(v)) for k, v in acc.items()},
            "m": int(N), "draws": draws,
            "modal_pool_idx": -1, "modal_share": float(np.mean(modal)),
        }

    return {
        "n_pool_N": int(N),
        "chance": {k: k / N for k in TOPKS},
        "modal_group_id": int(keys[rows["k=1"]["modal_pool_idx"]][1]),
        "modal_group_true_share": float(
            (a2p == rows["k=1"]["modal_pool_idx"]).sum() / len(corr)),
        "rows": rows,
    }


def main():
    args = parse_args()
    z_eeg, z_vis, t, shot, scene, task, tasks = _load_from_npz(args.npz)
    d = np.load(args.npz, allow_pickle=False)
    rec = d["rec_id"][~d["is_vision"]].astype(np.int64)
    prov = json.loads(str(d["provenance_json"])) if "provenance_json" in d.files else {}
    rng = np.random.default_rng(args.seed)

    print(f"{args.npz}: M={len(z_eeg)} windows, {len(np.unique(rec))} recordings")
    out = {"npz": args.npz, "seed": args.seed, "provenance": prov,
           "n_recordings": int(len(np.unique(rec))), "levels": {}}

    for level in LEVELS:
        res = run_level(level, z_eeg, z_vis, task, t, shot, scene, rec, rng)
        out["levels"][level] = res
        N = res["n_pool_N"]
        print(f"\n[{level}] pool N={N}  chance top1={1 / N:.3f}  "
              f"modal answer = {level} {res['modal_group_id']} "
              f"(true {res['modal_group_true_share']:.1%} of windows)")
        for name, r in res["rows"].items():
            accs = " ".join(f"top{k}={r[k]:.3f} ({r[k] / (k / N):.1f}x)" for k in TOPKS)
            print(f"    {name:<16s} m={r['m']:6d}  {accs}   "
                  f"modal top-1 {r['modal_share']:.0%}")

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
