"""Scrubbing timeline: walk one subject's EEG recording, watch what the model
retrieves at each moment.

Time-level retrieval (N=101 candidate moments, one per 2 s of the film) is the
hardest granularity and the one that makes a timeline meaningful — every point
in the recording has exactly one correct answer, so you can scrub and watch the
model succeed or fail moment by moment.

Ships three subjects, not one. Per-subject Top-1 on this checkpoint ranges from
0.000 to 0.178 and the differences are real (split-half r = 0.81), so showing
only the best subject would misrepresent the method. The page defaults to the
strongest and lets you switch to a typical and a failing one.

Needs an export produced with ``--showcase-recordings``, which ships every
window of the named recordings rather than a stratified sample::

    python demo/export_retrieval_npz.py ... --showcase-recordings 3,84,50 \\
        --output demo_test_timeline.npz

Then::

    PYTHONPATH=. .venv/bin/python demo/build_timeline.py \\
        --npz demo/data/demo_test_timeline.npz
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from eb_jepa.evaluation.clip_probe.retrieval import _build_time_pool, _load_from_npz

from demo import movie_assets as ma
from demo.render_timeline import render_timeline

T_BUCKET_S = 0.5
N_CANDIDATES = 5
TRACE_HZ = 50.0   # display rate for the embedded EEG; 2 s -> 100 points
CLIP_SD = 4.0     # int8 quantisation range, in per-recording z units


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="Export written with --showcase-recordings.")
    ap.add_argument("--n-channels", type=int, default=8,
                    help="Channels drawn in the trace (farthest-point sampled).")
    ap.add_argument("--out-dir", default="demo")
    ap.add_argument("--out-name", default="timeline.html")
    return ap.parse_args()


def build_time_level(z_vis, task_ids, t_starts):
    """Time pool plus the (task, bucket) key of each pool row, in retrieval.py's
    first-occurrence order."""
    pool, a2p, valid = _build_time_pool(task_ids, t_starts, z_vis, T_BUCKET_S)
    buckets = np.round(t_starts / T_BUCKET_S).astype(np.int64)
    keys, seen = [], {}
    for k in zip(task_ids.tolist(), buckets.tolist()):
        if k not in seen:
            seen[k] = len(seen)
            keys.append(k)
    assert len(keys) == pool.shape[0]
    return pool, a2p, keys


def quantize(sig: np.ndarray) -> str:
    """[C, T] z-scored EEG -> base64 int8, clipped at +-CLIP_SD.

    Float JSON for 3 subjects x 8 channels x 203 s would be several MB; int8 at
    the display rate is ~80 KB per subject and visually identical.
    """
    q = np.clip(sig / CLIP_SD, -1.0, 1.0)
    return base64.b64encode((q * 127).astype(np.int8).tobytes()).decode("ascii")


def main():
    args = parse_args()
    npz = Path(args.npz)
    z_eeg, z_vis, t_starts, shot_ids, scene_ids, task_ids, tasks = _load_from_npz(str(npz))
    d = np.load(npz, allow_pickle=False)
    prov = json.loads(str(d["provenance_json"]))
    rec_ids = d["rec_id"][~d["is_vision"]].astype(np.int64)
    sfreq = float(prov.get("sfreq", 200.0))
    window_s = float(prov.get("window_size_seconds", 2.0))

    pool, a2p, keys = build_time_level(z_vis, task_ids, t_starts)
    N = pool.shape[0]
    S_all = z_eeg @ pool.T
    rank_all = (np.argsort(-S_all, axis=1) == a2p[:, None]).argmax(1) + 1
    print(f"time pool N={N}  overall top1={(rank_all <= 1).mean():.3f} "
          f"top5={(rank_all <= 5).mean():.3f} top10={(rank_all <= 10).mean():.3f}")

    show_idx = d["showcase_idx"]
    show_eeg = d["showcase_eeg"]
    slot = {int(r): i for i, r in enumerate(show_idx.tolist())}
    recs = sorted(set(rec_ids[show_idx].tolist()))
    if not recs:
        raise SystemExit("export has no showcase recordings; re-run the export "
                         "with --showcase-recordings")

    # Channels: farthest-point over the montage, shared by every subject so the
    # traces stay comparable when you switch.
    from demo.render import pick_channels
    ch_pos = d["ch_pos"] if "ch_pos" in d.files else None
    chans = (pick_channels(ch_pos, args.n_channels) if ch_pos is not None
             else list(range(args.n_channels)))
    ch_names = [str(c) for c in d["ch_names"]] if "ch_names" in d.files else []

    step = max(1, int(round(sfreq / TRACE_HZ)))
    subjects = []
    for rec in recs:
        rows = np.flatnonzero(rec_ids == rec)
        rows = rows[np.argsort(t_starts[rows])]
        top1 = float((rank_all[rows] <= 1).mean())
        top5 = float((rank_all[rows] <= 5).mean())
        top10 = float((rank_all[rows] <= 10).mean())

        moments = []
        for r in rows:
            sim = S_all[r]
            order = np.argsort(-sim)[:N_CANDIDATES]
            moments.append({
                "t": float(t_starts[r]),
                "truth": int(a2p[r]),
                "rank": int(rank_all[r]),
                "cand": [int(j) for j in order.tolist()],
                "score": [round(float(sim[j]), 4) for j in order.tolist()],
            })

        # One continuous trace per channel across the whole recording.
        sig = np.concatenate(
            [show_eeg[slot[int(r)]][chans].astype(np.float32) for r in rows], axis=1
        )[:, ::step]
        subjects.append({
            "rec": int(rec),
            "top1": top1, "top5": top5, "top10": top10,
            "n": len(rows),
            "moments": moments,
            "eeg_b64": quantize(sig),
            "eeg_shape": [len(chans), int(sig.shape[1])],
        })
        print(f"  rec {rec:4d}: {len(rows)} windows  top1={top1:.3f} "
              f"top5={top5:.3f} top10={top10:.3f}  trace {sig.shape}")

    # Rank subjects strongest first, and label by where they sit in the split.
    per_rec_top1 = np.array([
        (rank_all[rec_ids == r] <= 1).mean() for r in np.unique(rec_ids)
    ])
    subjects.sort(key=lambda s: -s["top1"])
    n_tot = len(per_rec_top1)
    for s in subjects:
        pctl = 100 * float((per_rec_top1 < s["top1"]).mean())
        s["percentile"] = pctl
        s["meta"] = (f"<strong>recording {s['rec']}</strong> · Top-1 "
                     f"{100 * s['top1']:.1f}% · Top-5 {100 * s['top5']:.1f}% · "
                     f"Top-10 {100 * s['top10']:.1f}% · {pctl:.0f}th percentile "
                     f"of {n_tot} participants")

    # Stills for every candidate moment, from one sequential decode of the mp4.
    task = tasks[0]
    mi = ma.MovieIndex(task, window_size_seconds=window_s)
    spans = {j: mi.time_span(int(k[1]), T_BUCKET_S) for j, k in enumerate(keys)}
    times = {j: mi.representative_time(sp) for j, sp in spans.items()}
    # Open past the title card. The film starts on ~10 s of black, so index 0
    # would show six black rectangles; the data is unchanged, only where the
    # playhead sits when the page loads.
    ent = {j: mi.representative_entropy(sp) for j, sp in spans.items()}
    blank = 0.5 * float(np.median(list(ent.values())))
    first_lit = next((i for i, m in enumerate(subjects[0]["moments"])
                      if ent.get(m["truth"], 1e9) >= blank), 0)
    print(f"  opening at moment {first_lit} (t={subjects[0]['moments'][first_lit]['t']:.0f}s) "
          f"— {first_lit} blank leading moment(s)")

    print(f"  decoding {task}: {len(set(times.values()))} frames for {N} moments")
    frames = ma.decode_frames(task, set(times.values()), mi.fps)
    out_dir = Path(args.out_dir)
    asset_dir = out_dir / "assets"
    pool_assets = []
    for j in range(N):
        p = ma.write_still(frames[int(round(times[j] * mi.fps))],
                           asset_dir / f"{task}_time_{keys[j][1]}.jpg")
        pool_assets.append({"src": f"assets/{p.name}", "label": spans[j].label,
                            "short": spans[j].short, "t": spans[j].t0})

    path = render_timeline(
        subjects=subjects, pool_assets=pool_assets, out_dir=out_dir,
        out_name=args.out_name, provenance=prov, tasks=tasks,
        n_pool=N, n_chans=int(prov.get("n_chans", 129)),
        chans=chans, ch_names=ch_names, trace_hz=TRACE_HZ, clip_sd=CLIP_SD,
        window_s=window_s, start_pos=first_lit,
        title="Scrub the recording, watch the guesses",
        intro_html=(
            f"One participant's {int(prov.get('n_chans', 129))}-channel EEG while they "
            f"watched <em>{task}</em>, start to finish. Drag anywhere on the trace or "
            f"the ribbon. At every {window_s:g} s step the model ranks all {N} moments "
            f"of the film; you see the moment they were actually watching and the five "
            f"it ranks highest."),
        control_label="participant",
        labels=["strongest", "typical", "weakest"],
        overview_label=f"whole recording · {len(chans)} of "
                       f"{int(prov.get('n_chans', 129))} channels",
        footer_items=[
            f"<li><strong>Three participants, not one.</strong> Per-participant Top-1 "
            f"on this split ranges from {per_rec_top1.min():.3f} to "
            f"{per_rec_top1.max():.3f} across {len(per_rec_top1)} people, median "
            f"{np.median(per_rec_top1):.3f}, and the differences are real — split-half "
            f"reliability r = 0.81. The \"strongest\" tab is the best of "
            f"{len(per_rec_top1)}, so its number is inflated by that selection (its own "
            f"two halves give 0.216 and 0.140). \"Typical\" and \"weakest\" are there "
            f"so the page cannot be read as the method's typical behaviour.</li>",
            f"<li><strong>Time level is the hardest granularity</strong> — {N} "
            f"candidates, one per {window_s:g} s of film, chance Top-1 {1 / N:.1%}. "
            f"Shot and scene level are easier; see the "
            f"<a href=\"index.html\">scene-level page</a>.</li>",
            f"<li>The query is a single {window_s:g} s window encoded on its own — no "
            f"temporal context, no averaging across participants. Averaging across "
            f"participants helps a lot; see the "
            f"<a href=\"timeline_group.html\">subject-averaged page</a>.</li>",
            "<li>The ribbon shows where in the film the model is right: bar height and "
            "opacity encode the rank of the correct moment on a log scale (rank 1 = "
            "full height). Long dark runs are stretches it tracks; pale runs are "
            "stretches it loses.</li>",
            f"<li>Traces are per-recording z-scored, drawn at {TRACE_HZ:g} Hz and "
            f"clipped at ±{CLIP_SD:g} SD for display. Bars under each candidate are "
            f"cosine similarity on a shared scale; only the ordering is used.</li>",
        ],
        overall={"top1": float((rank_all <= 1).mean()),
                 "top5": float((rank_all <= 5).mean()),
                 "top10": float((rank_all <= 10).mean())},
    )
    size = path.stat().st_size / 1e6
    print(f"wrote {path}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
