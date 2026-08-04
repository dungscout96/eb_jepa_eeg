"""Render the e->v retrieval page: EEG trace SVGs + a self-contained index.html.

Colors follow the validated reference palette (blue series-1 for the EEG trace
and score bars, status-good green for a correct retrieval). Correctness is never
carried by color alone — the correct candidate also gets a rank badge and a
"correct" label. Palette validated with the dataviz validator against both
surfaces (#fcfcfb light / #1a1a19 dark): all checks pass.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np

# --- palette (roles, not raw hex, everywhere below) ------------------------
LIGHT = {
    "surface": "#fcfcfb", "plane": "#f9f9f7",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "series": "#2a78d6", "good": "#0ca30c",
    "border": "rgba(11,11,11,0.10)",
}
DARK = {
    "surface": "#1a1a19", "plane": "#0d0d0d",
    "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
    "series": "#3987e5", "good": "#0ca30c",
    "border": "rgba(255,255,255,0.10)",
}

N_TRACE_CHANNELS = 8
TRACE_W, TRACE_H = 232, 104
TRACE_DOWNSAMPLE = 2  # 400 samples -> 200 points; already finer than the pixels
CLIP_SD = 4.0

LEVEL_BLURB = {
    "time": "each candidate is one 0.5 s moment of the film",
    "shot": "each candidate is one camera shot",
    "scene": "each candidate is one scene (a shot or a group of related shots)",
}


# ---------------------------------------------------------------------------
# EEG traces
# ---------------------------------------------------------------------------


def pick_channels(ch_pos: np.ndarray, k: int = N_TRACE_CHANNELS) -> list[int]:
    """Farthest-point sample k channels so the traces span the scalp.

    Deterministic: seeded at the most anterior channel, then greedily taking the
    electrode furthest from everything picked so far.
    """
    pos = np.nan_to_num(np.asarray(ch_pos, dtype=np.float64))
    picked = [int(np.argmax(pos[:, 1]))]
    d = np.linalg.norm(pos - pos[picked[0]], axis=1)
    while len(picked) < min(k, len(pos)):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(pos - pos[nxt], axis=1))
    # Top-of-head down, so the stack reads like a conventional EEG montage.
    return sorted(picked, key=lambda i: -pos[i][1])


def trace_svg(eeg: np.ndarray, chans: list[int], ch_names, scale: float,
              window_s: float) -> str:
    """Stacked single-series EEG traces. One series, so no legend — the caption
    names it. Amplitude scale is shared across every row on the page, otherwise
    a quiet window would look as active as a loud one."""
    n = len(chans)
    pad_l, pad_t, pad_b = 4, 6, 12
    lane = (TRACE_H - pad_t - pad_b) / n
    amp = lane * 0.42
    x_span = TRACE_W - pad_l - 4
    step = max(1, TRACE_DOWNSAMPLE)

    parts = [
        f'<svg viewBox="0 0 {TRACE_W} {TRACE_H}" width="{TRACE_W}" height="{TRACE_H}" '
        f'role="img" aria-label="{n} of {len(ch_names)} EEG channels over '
        f'{window_s:g} seconds" class="eeg">'
    ]
    for row, ci in enumerate(chans):
        y0 = pad_t + lane * (row + 0.5)
        sig = np.clip(np.asarray(eeg[ci], dtype=np.float64)[::step] / scale, -1.0, 1.0)
        xs = pad_l + x_span * np.arange(len(sig)) / max(1, len(sig) - 1)
        ys = y0 - amp * sig
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline class="tr" points="{pts}"/>')
    parts.append(
        f'<line class="ax" x1="{pad_l}" y1="{TRACE_H - pad_b + 2}" '
        f'x2="{pad_l + x_span}" y2="{TRACE_H - pad_b + 2}"/>'
        f'<text class="tick" x="{pad_l}" y="{TRACE_H - 2}">0</text>'
        f'<text class="tick" x="{pad_l + x_span}" y="{TRACE_H - 2}" '
        f'text-anchor="end">{window_s:g} s</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _candidate_html(cand, assets, tasks, score_domain, rank) -> str:
    task = tasks[cand["key"][0]]
    key = (task, cand["_level"], cand["key"][1])
    a = assets.get(key)
    truth = cand["is_truth"]
    src = a["gif"] if (truth and a and "gif" in a) else (a["still"] if a else None)
    label = a["label"] if a else "—"
    short = a.get("short", label) if a else "—"
    # Cosines in this space are mostly negative and tightly clustered, so
    # score/max is meaningless. Encode position within the shared [lo, hi]
    # window of everything shown on the page; the footer states that window and
    # the exact value is printed beside each bar.
    lo, hi = score_domain
    frac = max(0.03, min(1.0, (cand["score"] - lo) / max(hi - lo, 1e-9)))
    img = (f'<img src="{html.escape(src)}" alt="{html.escape(label)}" loading="lazy">'
           if src else '<div class="noimg">no frame</div>')
    badge = '<span class="chk" aria-hidden="true">✓</span>' if truth else ""
    return (
        f'<figure class="cand{" is-truth" if truth else ""}" '
        f'title="rank {rank} · {html.escape(label)} · cosine {cand["score"]:.3f}">'
        f'<div class="thumb">{img}<span class="rank">{rank}</span>{badge}</div>'
        f'<figcaption>'
        f'<span class="bar"><i style="width:{100 * frac:.1f}%"></i></span>'
        f'<span class="score">{cand["score"]:.2f}</span>'
        f'<span class="cap">{html.escape(short)}</span>'
        f'{"<span class=\'truthlab\'>correct</span>" if truth else ""}'
        f"</figcaption></figure>"
    )


def _row_html(row, assets, tasks, score_domain, n_candidates) -> str:
    blocks = []
    for level, res in row["levels"].items():
        if res is None:
            blocks.append(f'<div class="cands" data-level="{level}">'
                          f'<p class="nores">no {level} label for this window</p></div>')
            continue
        cards = []
        for i, c in enumerate(res["candidates"], start=1):
            c = {**c, "_level": level}
            cards.append(_candidate_html(c, assets, tasks, score_domain, i))
        rank = res["truth_rank"]
        hit = rank <= n_candidates
        if hit:
            verdict = (f'<p class="verdict ok"><span aria-hidden="true">✓</span> '
                       f'correct candidate ranked <strong>#{rank}</strong> of '
                       f'{res["n_pool"]}</p>')
        else:
            task = tasks[res["correct_key"][0]]
            a = assets.get((task, level, res["correct_key"][1]))
            shown = (f'<img src="{html.escape(a["still"])}" alt="the correct '
                     f'{level}" loading="lazy">') if a else ""
            verdict = (
                f'<p class="verdict miss">missed — the correct candidate ranked '
                f'<strong>#{rank}</strong> of {res["n_pool"]}'
                f'<span class="actual">{shown}<span>actual</span></span></p>'
            )
        blocks.append(f'<div class="cands" data-level="{level}">'
                      f'<div class="strip">{"".join(cards)}</div>{verdict}</div>')

    return (
        f'<article class="row">'
        f'<div class="query">{row["svg"]}'
        f'<p class="qmeta"><strong>{html.escape(row["participant"])}</strong>'
        f'<span>{html.escape(row["task"])} · {_mmss(row["t_start"])}</span></p></div>'
        f'<div class="results">{"".join(blocks)}</div>'
        f"</article>"
    )


def _tiles_html(metrics, topks) -> str:
    out = []
    for level, m in metrics.items():
        n = m["n_vision_pool_N"]
        tiles = []
        for k in topks:
            acc = m["e2v_top_k"][k]
            chance = k / n
            tiles.append(
                f'<div class="tile"><span class="k">Top-{k}</span>'
                f'<span class="v">{_pct(acc)}</span>'
                f'<span class="s">{acc / chance:.1f}× chance ({_pct(chance)})</span></div>'
            )
        out.append(
            f'<div class="tiles" data-level="{level}">{"".join(tiles)}'
            f'<div class="tile pool"><span class="k">candidates</span>'
            f'<span class="v">{n}</span>'
            f'<span class="s">{html.escape(LEVEL_BLURB[level])}</span></div></div>'
        )
    return "".join(out)


def _css() -> str:
    def block(scope: str, p: dict) -> str:
        body = " ".join(f"--{k}: {v};" for k, v in p.items())
        return f"{scope} {{ {body} }}"

    return f"""
:root {{ color-scheme: light dark; }}
{block(":root", LIGHT)}
@media (prefers-color-scheme: dark) {{
  {block(':root:where(:not([data-theme="light"]))', DARK)}
}}
{block(':root[data-theme="dark"]', DARK)}

* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 0 20px 64px;
  background: var(--plane); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1080px; margin: 0 auto; }}
/* Run names and paths are long unbreakable tokens — let them wrap rather than
   push the page into horizontal scroll on narrow screens. */
code {{ overflow-wrap: anywhere; }}
header {{ padding: 40px 0 8px; }}
h1 {{ font-size: 28px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }}
.sub {{ margin: 0; color: var(--ink2); max-width: 62ch; }}
.prov {{ margin: 12px 0 0; color: var(--muted); font-size: 12.5px; }}
.prov code {{ font-size: 12px; }}

.panel {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin: 20px 0;
}}
.tiles {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.tiles[data-level] {{ display: none; }}
.tile {{
  flex: 1 1 150px; min-width: 140px; padding: 12px 14px;
  border-radius: 10px; background: var(--plane); border: 1px solid var(--border);
}}
.tile .k {{ display: block; font-size: 12px; color: var(--muted);
  text-transform: uppercase; letter-spacing: .06em; }}
.tile .v {{ display: block; font-size: 30px; line-height: 1.1; margin: 2px 0 2px;
  color: var(--ink); }}
.tile .s {{ display: block; font-size: 12.5px; color: var(--ink2); }}
.tile.pool .v {{ color: var(--ink2); }}

.controls {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  margin: 0 0 12px; }}
.controls .lbl {{ font-size: 12.5px; color: var(--muted);
  text-transform: uppercase; letter-spacing: .06em; }}
.seg {{ display: inline-flex; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; background: var(--plane); }}
.seg button {{
  appearance: none; border: 0; background: transparent; cursor: pointer;
  font: inherit; font-size: 13.5px; padding: 7px 16px; color: var(--ink2);
}}
.seg button[aria-pressed="true"] {{ background: var(--series); color: #fff; }}
.seg button + button {{ border-left: 1px solid var(--border); }}

.row {{
  display: grid; grid-template-columns: 248px minmax(0, 1fr); gap: 20px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin: 12px 0;
}}
.query .eeg {{ display: block; width: 100%; height: auto; }}
.eeg .tr {{ fill: none; stroke: var(--series); stroke-width: 1;
  stroke-linejoin: round; opacity: .85; }}
.eeg .ax {{ stroke: var(--axis); stroke-width: 1; }}
.eeg .tick {{ fill: var(--muted); font-size: 8px;
  font-family: system-ui, sans-serif; }}
.qmeta {{ margin: 4px 0 0; font-size: 13px; display: flex;
  flex-direction: column; gap: 1px; }}
.qmeta span {{ color: var(--ink2); font-variant-numeric: tabular-nums; }}

.results {{ min-width: 0; }}
.cands {{ display: none; }}
.strip {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 2px; }}
.cand {{ margin: 0; min-width: 0; }}
.thumb {{ position: relative; border-radius: 6px; overflow: hidden;
  background: var(--plane); aspect-ratio: 16 / 9;
  outline: 2px solid transparent; outline-offset: -2px; }}
.thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.noimg {{ display: grid; place-items: center; height: 100%; font-size: 11px;
  color: var(--muted); }}
.rank {{ position: absolute; top: 4px; left: 4px; min-width: 17px; height: 17px;
  border-radius: 4px; background: rgba(0,0,0,.62); color: #fff;
  font-size: 11px; line-height: 17px; text-align: center;
  font-variant-numeric: tabular-nums; }}
.cand.is-truth .thumb {{ outline-color: var(--good); }}
.chk {{ position: absolute; top: 4px; right: 4px; width: 17px; height: 17px;
  border-radius: 4px; background: var(--good); color: #fff; font-size: 12px;
  line-height: 17px; text-align: center; }}
figcaption {{ padding: 5px 2px 0; }}
.bar {{ display: block; height: 3px; border-radius: 999px; background: var(--grid); }}
.bar i {{ display: block; height: 100%; border-radius: 999px;
  background: var(--series); }}
.score {{ display: block; font-size: 12px; color: var(--ink2); margin-top: 3px;
  font-variant-numeric: tabular-nums; }}
.cap {{ display: block; font-size: 11.5px; color: var(--muted);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.truthlab {{ display: block; font-size: 11.5px; color: var(--good); font-weight: 600; }}

.verdict {{ margin: 10px 0 0; font-size: 13px; color: var(--ink2);
  display: flex; align-items: center; gap: 8px; }}
.verdict strong {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
.verdict.ok span[aria-hidden] {{ color: var(--good); font-weight: 700; }}
.actual {{ display: inline-flex; align-items: center; gap: 6px; margin-left: 4px; }}
.actual img {{ width: 56px; border-radius: 4px; outline: 2px solid var(--good);
  outline-offset: -2px; display: block; }}
.actual span {{ font-size: 11.5px; color: var(--good); font-weight: 600; }}
.nores {{ font-size: 13px; color: var(--muted); margin: 0; }}

footer {{ margin-top: 28px; font-size: 13px; color: var(--ink2); }}
footer h2 {{ font-size: 14px; margin: 0 0 8px; color: var(--ink); }}
footer li {{ margin: 0 0 6px; }}
footer code {{ font-size: 12px; }}

@media (max-width: 760px) {{
  .row {{ grid-template-columns: 1fr; }}
  .query .eeg {{ max-width: 340px; }}
  /* Keep all five candidates — the verdict text refers to ranks, so dropping
     cards would make it lie. The strip scrolls inside itself; the page never
     scrolls sideways. */
  .strip {{ grid-template-columns: none; grid-auto-flow: column;
    grid-auto-columns: 44%; overflow-x: auto; padding-bottom: 4px;
    scroll-snap-type: x proximity; overscroll-behavior-x: contain; }}
  .cand {{ scroll-snap-align: start; }}
  .verdict {{ flex-wrap: wrap; }}
}}
"""


def _js(default_level: str) -> str:
    return f"""
(function () {{
  var root = document.documentElement;
  function apply(level) {{
    document.querySelectorAll('[data-level]').forEach(function (el) {{
      el.style.display = el.dataset.level === level
        ? (el.classList.contains('tiles') ? 'flex' : 'block') : 'none';
    }});
    document.querySelectorAll('.seg button').forEach(function (b) {{
      b.setAttribute('aria-pressed', String(b.dataset.set === level));
    }});
  }}
  document.querySelectorAll('.seg button').forEach(function (b) {{
    b.addEventListener('click', function () {{ apply(b.dataset.set); }});
  }});
  apply('{default_level}');
}})();
"""


def render_page(*, rows, assets, metrics, tasks, provenance, topks, out_dir: Path,
                ch_pos, ch_names, reference: str | None, concentration=None,
                default_level: str = "scene") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_meta = provenance
    window_s = float(npz_meta.get("window_size_seconds", 2.0))
    n_chans = int(npz_meta.get("n_chans") or len(ch_names) or rows[0]["eeg"].shape[0])

    # One amplitude scale for every trace on the page: a robust global bound so
    # rows stay comparable. Same for the score-bar domain — one axis, shared.
    stack = np.stack([r["eeg"] for r in rows])
    # p90, not a near-max: a handful of artifact-heavy channels would otherwise
    # set the scale and flatten every other trace into a straight line.
    # Excursions beyond it clip, which the footer says.
    scale = float(np.percentile(np.abs(stack), 90.0)) or 1.0
    all_scores = [c["score"] for r in rows for res in r["levels"].values() if res
                  for c in res["candidates"]]
    score_domain = (float(min(all_scores)), float(max(all_scores)))

    chans = (pick_channels(ch_pos) if ch_pos is not None and len(ch_pos)
             else list(range(min(N_TRACE_CHANNELS, rows[0]["eeg"].shape[0]))))
    for r in rows:
        r["svg"] = trace_svg(r["eeg"], chans, ch_names, scale, window_s)

    n_cand = max(len(res["candidates"]) for r in rows for res in r["levels"].values() if res)
    body_rows = "".join(_row_html(r, assets, tasks, score_domain, n_cand) for r in rows)

    scene = metrics["scene"]
    head_top1 = scene["e2v_top_k"][1]
    head_top10 = scene["e2v_top_k"][max(topks)]
    split = npz_meta.get("split", "?")
    m_total = npz_meta.get("n_eeg_windows_M", "?")

    # Aggregate Top-K can hide collapse onto one attractor. When it happens it
    # is impossible to miss on the page (the same thumbnail at rank 1, row after
    # row), so name it with numbers instead of leaving the reader to guess.
    conc_note = ""
    if concentration and concentration["predicted_share"] > 2 * concentration["true_share"]:
        c = concentration
        what = c.get("label") or f"scene {c['group']}"
        conc_note = (
            f'<li><strong>The model has a favourite answer.</strong> It names '
            f'{html.escape(str(what))} as its first guess for '
            f'<strong>{_pct(c["predicted_share"])}</strong> of all {m_total} windows, '
            f'though that is the correct answer only {_pct(c["true_share"])} of the time '
            f'(uniform would be {_pct(c["uniform_share"])}). That is why the same '
            f'thumbnail recurs at rank 1 in the rows above. It does not change the Top-K figures '
            f'above — but a large share of first guesses landing on one attractor is '
            f'something those figures alone would not tell you.</li>'
        )

    ref_note = (f"Every number here is recomputed from the exported embeddings and "
                f"checked against <code>{html.escape(Path(reference).name)}</code> "
                f"before this page is written.")if reference else (
                "These numbers were <strong>not</strong> checked against the committed "
                "results — rebuild with <code>--reference</code>.")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reading the movie out of the brain — EEG→video retrieval</title>
<style>{_css()}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Reading the movie out of the brain</h1>
  <p class="sub">Two seconds of {n_chans}-channel EEG from someone
  watching <em>{html.escape(str(rows[0]['task']))}</em> goes in. The model ranks every
  candidate moment of the film by how well it matches. It puts the correct scene first
  <strong>{_pct(head_top1)}</strong> of the time and in the top {max(topks)}
  <strong>{_pct(head_top10)}</strong> of the time — against {scene['n_vision_pool_N']}
  candidates, where chance would be {_pct(1 / scene['n_vision_pool_N'])}.</p>
  <p class="prov">Held-out <strong>{html.escape(str(split))}</strong> split ·
  {m_total} EEG windows · scene-CLIP encoder warm-started from REVE ·
  <code>{html.escape(Path(str(npz_meta.get('checkpoint', ''))).parent.name)}</code></p>
</header>

<section class="panel">
  <div class="controls">
    <span class="lbl">candidate granularity</span>
    <div class="seg" role="group" aria-label="Retrieval granularity">
      <button type="button" data-set="time" aria-pressed="false">time</button>
      <button type="button" data-set="shot" aria-pressed="false">shot</button>
      <button type="button" data-set="scene" aria-pressed="true">scene</button>
    </div>
  </div>
  {_tiles_html(metrics, topks)}
</section>

{body_rows}

<footer>
  <h2>How to read this</h2>
  <ul>
    <li>Each row is one {window_s:g} s EEG window ({len(chans)} of {n_chans} channels
      shown, one shared amplitude scale set to the 90th percentile of |z|, so larger
      excursions clip). To its right are the five candidates the
      model ranks highest, best first. The correct one — when it makes the top five —
      is outlined and marked <span style="color:var(--good);font-weight:600">✓ correct</span>.</li>
    <li>Bars show cosine similarity in the shared EEG–video space, positioned
      within the common range spanned by every candidate on this page
      ({score_domain[0]:+.2f} to {score_domain[1]:+.2f}). Absolute values are
      small, often negative, and cluster tightly — retrieval only uses the
      <em>ordering</em>, and the exact value is printed beside each bar.</li>
    <li>Chance is <code>K / N</code>: with {scene['n_vision_pool_N']} scene
      candidates, a random model gets Top-1 right {_pct(1 / scene['n_vision_pool_N'])}
      of the time. The "× chance" figure is the honest measure of the effect.</li>
    <li>This is <strong>EEG→video</strong> only. The reverse direction (given a scene,
      find the EEG of someone watching it) is far weaker on this checkpoint — Top-1
      about 2× chance, and Top-10 <em>below</em> chance. Don't read this page as
      evidence that both directions work.</li>
    {conc_note}
    <li>Rows are chosen to include successes and failures; they are illustrative,
      not a random sample. The aggregate numbers above are over all {m_total} windows.</li>
    <li>{ref_note} See §3.10 of
      <code>experiments/clip_pretraining/scene_clip_from_checkpoint/RESULTS.md</code>.</li>
  </ul>
</footer>
</div>
<script>{_js(default_level)}</script>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path
