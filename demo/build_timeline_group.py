"""Subject-averaged timeline: the same scrub, but the query is K brains, not one.

At each of the 101 movie moments, average the shared-space EEG embeddings of the
first K participants in a fixed nested order, renormalise, and retrieve against
the unchanged candidate pool. Switching K rebuilds nothing — you watch the same
timeline get better as brains are added, which is the whole
[`snr_scaling`](../experiments/snr_scaling/RESULTS.md) thesis made visible.

The displayed waveform is the matching average of the *raw* EEG, so the trace
visibly settles as K grows: mean |z| falls 0.301 (K=1) to 0.081 (K=106), i.e.
the subject-specific noise averages away and the stimulus-locked component does
not.

Two inputs:
  --npz      the ordinary export (z_eeg for every recording) — retrieval only
  --avg-npz  subject-averaged raw waveforms from export_avg on the cluster,
             built with the SAME channel set and subject order (both are
             recorded inside that file and are checked here)

Dead recordings are excluded up front; see demo/README.md.

    PYTHONPATH=. .venv/bin/python demo/build_timeline_group.py \\
        --npz demo/data/demo_test.npz \\
        --avg-npz demo/data/avg_eeg_test.npz \\
        --amp demo/data/amp_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eb_jepa.evaluation.clip_probe.retrieval import _load_from_npz

from demo import movie_assets as ma
from demo.build_timeline import CLIP_SD, TRACE_HZ, build_time_level, quantize
from demo.render_timeline import render_timeline

T_BUCKET_S = 0.5
N_CANDIDATES = 5
DRAWS = 40  # random subject sets per K, for the stability estimate only


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--avg-npz", required=True)
    ap.add_argument("--amp", default=None,
                    help="Amplitude census JSON; recordings with rel_amp < 0.01 "
                         "are dropped as dead.")
    ap.add_argument("--out-dir", default="demo")
    ap.add_argument("--out-name", default="timeline_group.html")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def topk_rank(Q, pool, correct, ks=(1, 5, 10)):
    S = Q @ pool.T
    rank = (np.argsort(-S, axis=1) == correct[:, None]).argmax(1) + 1
    return {k: float((rank <= k).mean()) for k in ks}, rank, S


def aggregate(z, moment, mask, n_pool):
    """Mean shared-space embedding per movie moment over the selected anchors."""
    acc = np.zeros((n_pool, z.shape[1]))
    cnt = np.zeros(n_pool)
    np.add.at(acc, moment[mask], z[mask])
    np.add.at(cnt, moment[mask], 1)
    ok = cnt > 0
    Q = acc[ok] / cnt[ok, None]
    return Q / np.linalg.norm(Q, axis=1, keepdims=True), np.flatnonzero(ok)


def main():
    args = parse_args()
    z, zv, t, sh, sc, tk, tasks = _load_from_npz(args.npz)
    d = np.load(args.npz, allow_pickle=False)
    prov = json.loads(str(d["provenance_json"]))
    rec = d["rec_id"][~d["is_vision"]].astype(np.int64)
    window_s = float(prov.get("window_size_seconds", 2.0))
    n_chans_total = int(prov.get("n_chans", 129))

    dead = set()
    if args.amp:
        dead = {r["rec"] for r in json.load(open(args.amp)) if r["rel_amp"] < 0.01}
    live = np.array([r for r in np.unique(rec) if r not in dead])
    print(f"{len(live)} live recordings (dropped {sorted(dead)} as dead)")

    pool, a2p, keys = build_time_level(zv, tk, t)
    N = pool.shape[0]

    avg = np.load(args.avg_npz, allow_pickle=False)
    order = avg["order"].astype(np.int64)
    chans = avg["chans"].astype(int).tolist()
    ks = [int(k) for k in avg["ks"]]
    if set(order.tolist()) != set(live.tolist()):
        raise SystemExit("--avg-npz was built over a different recording set than "
                         "--npz + --amp imply; rebuild it with the same order.")
    print(f"K levels {ks} · channels {chans}")

    # Aggregate the same nested prefixes the waveforms were averaged over.
    rng = np.random.default_rng(args.seed)
    entries = []
    for K in ks:
        sel = set(order[:K].tolist())
        Q, moments_present = aggregate(z, a2p, np.isin(rec, list(sel)), N)
        acc, rank, S = topk_rank(Q, pool, moments_present)

        # A single nested prefix is one draw; with only N queries Top-1 carries
        # ~3 points of binomial noise. Quote the multi-draw mean beside it.
        if K < len(live):
            draws = []
            for _ in range(DRAWS):
                s2 = set(rng.choice(live, K, replace=False).tolist())
                Q2, m2 = aggregate(z, a2p, np.isin(rec, list(s2)), N)
                draws.append(topk_rank(Q2, pool, m2)[0])
            mu = {k: float(np.mean([x[k] for x in draws])) for k in (1, 5, 10)}
            sd = {k: float(np.std([x[k] for x in draws])) for k in (1, 5, 10)}
        else:
            mu, sd = acc, {k: 0.0 for k in (1, 5, 10)}

        moments = []
        for i, m in enumerate(moments_present.tolist()):
            sim = S[i]
            top = np.argsort(-sim)[:N_CANDIDATES]
            moments.append({
                "t": float(keys[m][1] * T_BUCKET_S),
                "truth": int(m), "rank": int(rank[i]),
                "cand": [int(j) for j in top.tolist()],
                "score": [round(float(sim[j]), 4) for j in top.tolist()],
            })

        sig = avg[f"avg_{K}"].astype(np.float32)[:, ::max(1, int(round(
            float(prov.get("sfreq", 200.0)) / TRACE_HZ)))]
        entries.append({
            "rec": K, "n": len(moments), "moments": moments,
            "top1": acc[1], "top5": acc[5], "top10": acc[10],
            "eeg_b64": quantize(sig), "eeg_shape": [sig.shape[0], sig.shape[1]],
            "meta": (f"<strong>{K} participant{'s' if K > 1 else ''} averaged</strong> · "
                     f"Top-1 {100 * acc[1]:.1f}% · Top-5 {100 * acc[5]:.1f}% · "
                     f"Top-10 {100 * acc[10]:.1f}% "
                     + (f"· over {DRAWS} random sets of {K}: Top-1 "
                        f"{100 * mu[1]:.1f}±{100 * sd[1]:.1f}%, Top-10 "
                        f"{100 * mu[10]:.1f}±{100 * sd[10]:.1f}%"
                        if K < len(live) else "· all live participants, one set")),
        })
        print(f"  K={K:4d}  this set: {acc[1]:.3f}/{acc[5]:.3f}/{acc[10]:.3f}   "
              f"{DRAWS}-draw mean: {mu[1]:.3f}±{sd[1]:.3f} / {mu[10]:.3f}±{sd[10]:.3f}")

    # Frames: identical pool to the single-subject timeline, so the stills are
    # already on disk; regenerate only if missing.
    task = tasks[0]
    mi = ma.MovieIndex(task, window_size_seconds=window_s)
    spans = {j: mi.time_span(int(k[1]), T_BUCKET_S) for j, k in enumerate(keys)}
    out_dir = Path(args.out_dir)
    need = {j: mi.representative_time(sp) for j, sp in spans.items()
            if not (out_dir / "assets" / f"{task}_time_{keys[j][1]}.jpg").exists()}
    if need:
        print(f"  decoding {task}: {len(set(need.values()))} missing frames")
        frames = ma.decode_frames(task, set(need.values()), mi.fps)
        for j, tt in need.items():
            ma.write_still(frames[int(round(tt * mi.fps))],
                           out_dir / "assets" / f"{task}_time_{keys[j][1]}.jpg")
    pool_assets = [{"src": f"assets/{task}_time_{keys[j][1]}.jpg",
                    "label": spans[j].label, "short": spans[j].short,
                    "t": spans[j].t0} for j in range(N)]

    ent = {j: mi.representative_entropy(sp) for j, sp in spans.items()}
    blank = 0.5 * float(np.median(list(ent.values())))
    first_lit = next((i for i, m in enumerate(entries[0]["moments"])
                      if ent.get(m["truth"], 1e9) >= blank), 0)

    single = entries[0]
    full = entries[-1]
    path = render_timeline(
        subjects=entries, pool_assets=pool_assets, out_dir=out_dir,
        out_name=args.out_name, provenance=prov, tasks=tasks, n_pool=N,
        n_chans=n_chans_total, chans=chans, ch_names=[], trace_hz=TRACE_HZ,
        clip_sd=CLIP_SD, window_s=window_s, start_pos=first_lit,
        overall={"top1": full["top1"], "top5": full["top5"], "top10": full["top10"]},
        title="Add brains, watch it sharpen",
        intro_html=(
            f"The same scrub through <em>{task}</em>, but the query is the "
            f"<strong>average</strong> of K participants' EEG at each moment rather "
            f"than one person's. Nothing else changes — same {N} candidates, same "
            f"chance rate. Switch K and watch both the trace settle and the guesses "
            f"improve."),
        control_label="participants averaged",
        labels=[str(k) for k in ks],
        overview_label=f"average of K recordings · {len(chans)} of {n_chans_total} channels",
        footer_items=[
            f"<li><strong>Only the query changes.</strong> The candidate pool is the "
            f"same {N} V-JEPA-2 moments throughout, so chance stays "
            f"{1 / N:.1%} at Top-1 and every K is directly comparable.</li>",
            f"<li><strong>What it buys.</strong> Top-10 goes {100 * single['top10']:.0f}% "
            f"at K=1 to {100 * full['top10']:.0f}% at K={ks[-1]}; Top-1 rises too but "
            f"flattens around K≈32. Averaging brains recovers the stimulus-locked "
            f"component; averaging <em>seconds</em> within one brain does not — see "
            f"<code>snr_scaling</code> §2.6.</li>",
            f"<li><strong>Read the small-K rows with the error bars.</strong> There are "
            f"only {N} queries, so a single set of K participants carries ~3 points of "
            f"binomial noise at Top-1. Each tab also reports the mean ± sd over "
            f"{DRAWS} random sets of that size; the K={ks[-1]} tab is every live "
            f"participant, so it is a single set by construction.</li>",
            "<li><strong>The trace is the same average.</strong> Mean |z| falls from "
            "0.30 at K=1 to 0.08 at K=106 — subject-specific activity cancels, the "
            "shared response does not. That ratio implies a single-trial reliability "
            "of roughly 0.06, in the range measured by CorrCA in "
            "<code>snr_scaling</code> §2.</li>",
            "<li>Dead recordings are excluded (2 of 108 on this split carry no usable "
            "EEG). Single-participant behaviour, including who is strong and who is "
            f"not, is on the <a href=\"timeline.html\">per-participant page</a>.</li>",
        ],
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
