"""The scrubbing timeline page.

Everything is precomputed; the JS only indexes into embedded arrays and redraws
a canvas, so scrubbing costs nothing and the page stays a single static file.

Charts follow the same validated palette as ``render.py`` — blue series-1 for
the EEG and score bars, status-good green for a correct retrieval, and a
single-hue blue ramp for the rank ribbon (sequential = one hue, light to dark).
Correctness never rests on colour alone: the correct candidate also carries a
rank badge and a "correct" label.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from demo.render import DARK, LIGHT, _pct

OVERVIEW_H = 96
DETAIL_H = 132


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def _css() -> str:
    def block(scope, p):
        return f"{scope} {{ " + " ".join(f"--{k}: {v};" for k, v in p.items()) + " }"

    return f"""
:root {{ color-scheme: light dark; }}
{block(":root", LIGHT)}
@media (prefers-color-scheme: dark) {{
  {block(':root:where(:not([data-theme="light"]))', DARK)}
}}
{block(':root[data-theme="dark"]', DARK)}

* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 0 20px 56px; background: var(--plane); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased; }}
.wrap {{ max-width: 1120px; margin: 0 auto; }}
code {{ overflow-wrap: anywhere; font-size: 12px; }}
header {{ padding: 40px 0 4px; }}
h1 {{ font-size: 28px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }}
.sub {{ margin: 0; color: var(--ink2); max-width: 64ch; }}
.prov {{ margin: 12px 0 0; color: var(--muted); font-size: 12.5px; }}

.panel {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin: 16px 0; }}

.controls {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.lbl {{ font-size: 12.5px; color: var(--muted); text-transform: uppercase;
  letter-spacing: .06em; }}
.seg {{ display: inline-flex; border: 1px solid var(--border); border-radius: 999px;
  overflow: hidden; background: var(--plane); }}
.seg button {{ appearance: none; border: 0; background: transparent; cursor: pointer;
  font: inherit; font-size: 13.5px; padding: 7px 14px; color: var(--ink2); }}
.seg button[aria-pressed="true"] {{ background: var(--series); color: #fff; }}
.seg button + button {{ border-left: 1px solid var(--border); }}
.play {{ appearance: none; border: 1px solid var(--border); background: var(--plane);
  color: var(--ink); border-radius: 999px; padding: 7px 16px; cursor: pointer;
  font: inherit; font-size: 13.5px; }}
.clock {{ font-size: 13.5px; color: var(--ink2); font-variant-numeric: tabular-nums; }}

.subjmeta {{ margin: 10px 0 0; font-size: 13px; color: var(--ink2); }}
.subjmeta strong {{ color: var(--ink); }}

canvas {{ display: block; width: 100%; touch-action: none; }}
#overview, #ribbon {{ cursor: pointer; }}
.capt {{ font-size: 12px; color: var(--muted); margin: 4px 0 0;
  display: flex; justify-content: space-between; }}
.tracklabel {{ font-size: 12.5px; color: var(--muted); margin: 0 0 6px;
  text-transform: uppercase; letter-spacing: .06em; }}

.stage {{ display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 20px;
  align-items: start; }}
.truth figure, .cands figure {{ margin: 0; }}
.frame {{ position: relative; border-radius: 8px; overflow: hidden;
  background: var(--plane); aspect-ratio: 16/9;
  outline: 2px solid transparent; outline-offset: -2px; }}
.frame img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.truth .frame {{ outline-color: var(--good); }}
figcaption {{ padding: 5px 2px 0; font-size: 12px; color: var(--ink2); }}

.strip {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 2px; }}
.cand .frame {{ border-radius: 6px; }}
.rank {{ position: absolute; top: 4px; left: 4px; min-width: 17px; height: 17px;
  border-radius: 4px; background: rgba(0,0,0,.62); color: #fff; font-size: 11px;
  line-height: 17px; text-align: center; font-variant-numeric: tabular-nums; }}
.chk {{ position: absolute; top: 4px; right: 4px; width: 17px; height: 17px;
  border-radius: 4px; background: var(--good); color: #fff; font-size: 12px;
  line-height: 17px; text-align: center; }}
.cand.is-truth .frame {{ outline-color: var(--good); }}
.bar {{ display: block; height: 3px; border-radius: 999px; background: var(--grid);
  margin-top: 5px; }}
.bar i {{ display: block; height: 100%; border-radius: 999px; background: var(--series); }}
.sc {{ display: block; font-size: 12px; color: var(--ink2);
  font-variant-numeric: tabular-nums; }}
.cp {{ display: block; font-size: 11.5px; color: var(--muted); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }}
.truthlab {{ display: block; font-size: 11.5px; color: var(--good); font-weight: 600; }}
.verdict {{ margin: 12px 0 0; font-size: 13.5px; color: var(--ink2); }}
.verdict strong {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
.verdict.ok b {{ color: var(--good); }}

footer {{ margin-top: 24px; font-size: 13px; color: var(--ink2); }}
footer h2 {{ font-size: 14px; margin: 0 0 8px; color: var(--ink); }}
footer li {{ margin: 0 0 6px; }}

@media (max-width: 820px) {{
  .stage {{ grid-template-columns: 1fr; }}
  .strip {{ grid-template-columns: none; grid-auto-flow: column;
    grid-auto-columns: 44%; overflow-x: auto; overscroll-behavior-x: contain; }}
}}
"""


def _js() -> str:
    # Braces are doubled only where CSS-like literals appear; this is a plain
    # string, not an f-string, so JS braces stay as written.
    return r"""
(function () {
  var D = window.__TL__;
  var si = 0, pos = 0, playing = null;
  var ov = document.getElementById('overview'), dt = document.getElementById('detail'),
      rb = document.getElementById('ribbon');

  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

  function decode(b64, shape) {
    var bin = atob(b64), a = new Int8Array(bin.length);
    for (var i = 0; i < bin.length; i++) a[i] = (bin.charCodeAt(i) << 24) >> 24;
    var C = shape[0], T = shape[1], out = [];
    for (var c = 0; c < C; c++) out.push(a.subarray(c * T, (c + 1) * T));
    return out;
  }
  D.subjects.forEach(function (s) { s.eeg = decode(s.eeg_b64, s.eeg_shape); });

  function fit(cv, h) {
    var r = window.devicePixelRatio || 1, w = cv.clientWidth;
    cv.width = Math.max(1, Math.round(w * r)); cv.height = Math.round(h * r);
    cv.style.height = h + 'px';
    var g = cv.getContext('2d'); g.setTransform(r, 0, 0, r, 0, 0);
    return { g: g, w: w, h: h };
  }

  function traces(cv, H, chans, i0, i1) {
    var f = fit(cv, H), g = f.g, w = f.w;
    g.clearRect(0, 0, w, H);
    var n = chans.length, lane = (H - 10) / n, amp = lane * 0.42;
    g.lineWidth = 1; g.strokeStyle = css('--series'); g.lineJoin = 'round';
    var span = i1 - i0, stepPx = Math.max(1, Math.ceil(span / w));
    for (var c = 0; c < n; c++) {
      var y0 = 5 + lane * (c + 0.5);
      g.beginPath();
      for (var i = i0, k = 0; i < i1; i += stepPx, k++) {
        var v = chans[c][i] / 127, x = w * (i - i0) / span, y = y0 - amp * v;
        k ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.stroke();
    }
    return f;
  }

  function drawOverview() {
    var s = D.subjects[si], T = s.eeg_shape[1];
    var f = traces(ov, D.overviewH, s.eeg, 0, T);
    var g = f.g, w = f.w, H = D.overviewH;
    var x0 = w * pos / s.n, x1 = w * (pos + 1) / s.n;
    g.fillStyle = css('--series'); g.globalAlpha = 0.16;
    g.fillRect(x0, 0, Math.max(2, x1 - x0), H); g.globalAlpha = 1;
    g.strokeStyle = css('--series'); g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x0, 0); g.lineTo(x0, H); g.stroke();
  }

  function drawDetail() {
    var s = D.subjects[si], T = s.eeg_shape[1], per = T / s.n;
    traces(dt, D.detailH, s.eeg, Math.floor(pos * per), Math.floor((pos + 1) * per));
  }

  function drawRibbon() {
    var s = D.subjects[si], f = fit(rb, D.ribbonH), g = f.g, w = f.w, H = D.ribbonH;
    g.clearRect(0, 0, w, H);
    var bw = w / s.n;
    for (var i = 0; i < s.n; i++) {
      var r = s.moments[i].rank;
      // Sequential single-hue ramp: better rank -> taller and darker. Log, not
      // linear: on a linear scale rank 10 of 101 still fills 90% of the bar and
      // every subject looks equally good.
      var good = 1 - Math.log(r) / Math.log(D.nPool);
      var h = Math.max(2, (H - 2) * good);
      g.fillStyle = i === pos ? css('--ink') : css('--series');
      g.globalAlpha = i === pos ? 1 : (0.25 + 0.75 * good);
      g.fillRect(i * bw, H - h, Math.max(1, bw - 1), h);
    }
    g.globalAlpha = 1;
  }

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  function card(j, rank, score, isTruth) {
    var a = D.pool[j], lo = D.scoreLo, hi = D.scoreHi;
    var frac = Math.max(0.03, Math.min(1, (score - lo) / Math.max(hi - lo, 1e-9)));
    return '<figure class="cand' + (isTruth ? ' is-truth' : '') + '" title="rank ' +
      rank + ' · ' + esc(a.label) + ' · cosine ' + score.toFixed(3) + '">' +
      '<div class="frame"><img src="' + a.src + '" alt="' + esc(a.label) + '">' +
      '<span class="rank">' + rank + '</span>' +
      (isTruth ? '<span class="chk" aria-hidden="true">✓</span>' : '') + '</div>' +
      '<figcaption><span class="bar"><i style="width:' + (100 * frac).toFixed(1) +
      '%"></i></span><span class="sc">' + score.toFixed(2) + '</span>' +
      '<span class="cp">' + esc(a.short) + '</span>' +
      (isTruth ? '<span class="truthlab">correct</span>' : '') +
      '</figcaption></figure>';
  }

  function drawStage() {
    var s = D.subjects[si], m = s.moments[pos], tr = D.pool[m.truth];
    document.getElementById('truthimg').innerHTML =
      '<div class="frame"><img src="' + tr.src + '" alt="' + esc(tr.label) + '"></div>' +
      '<figcaption>what they were watching · ' + esc(tr.short) + '</figcaption>';
    var html = '';
    for (var i = 0; i < m.cand.length; i++)
      html += card(m.cand[i], i + 1, m.score[i], m.cand[i] === m.truth);
    document.getElementById('cands').innerHTML = html;
    var v = document.getElementById('verdict');
    if (m.rank <= m.cand.length) {
      v.className = 'verdict ok';
      v.innerHTML = '<b>✓</b> correct moment ranked <strong>#' + m.rank +
        '</strong> of ' + D.nPool;
    } else {
      v.className = 'verdict';
      v.innerHTML = 'correct moment ranked <strong>#' + m.rank + '</strong> of ' +
        D.nPool + ' — not in the top ' + m.cand.length;
    }
    document.getElementById('clock').textContent =
      D.fmt(m.t) + ' – ' + D.fmt(m.t + D.windowS) + '  (' + (pos + 1) + '/' + s.n + ')';
  }

  function redraw() { drawOverview(); drawDetail(); drawRibbon(); drawStage(); }

  function seek(p) {
    var s = D.subjects[si];
    pos = Math.max(0, Math.min(s.n - 1, p));
    document.getElementById('slider').value = pos;
    redraw();
  }

  function fromEvent(el, e) {
    var r = el.getBoundingClientRect();
    var x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    seek(Math.floor(D.subjects[si].n * x / r.width));
  }
  [ov, rb].forEach(function (el) {
    var down = false;
    el.addEventListener('pointerdown', function (e) {
      down = true; el.setPointerCapture(e.pointerId); fromEvent(el, e); });
    el.addEventListener('pointermove', function (e) { if (down) fromEvent(el, e); });
    el.addEventListener('pointerup', function () { down = false; });
    el.addEventListener('pointercancel', function () { down = false; });
  });

  document.getElementById('slider').addEventListener('input', function (e) {
    seek(parseInt(e.target.value, 10)); });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight') { seek(pos + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { seek(pos - 1); e.preventDefault(); }
    else if (e.key === ' ') { toggle(); e.preventDefault(); }
  });

  function toggle() {
    var b = document.getElementById('play');
    if (playing) { clearInterval(playing); playing = null; b.textContent = '▶ play'; return; }
    b.textContent = '■ pause';
    playing = setInterval(function () {
      var s = D.subjects[si];
      if (pos >= s.n - 1) { toggle(); return; }
      seek(pos + 1);
    }, 700);
  }
  document.getElementById('play').addEventListener('click', toggle);

  document.querySelectorAll('.seg button').forEach(function (b) {
    b.addEventListener('click', function () {
      si = parseInt(b.dataset.i, 10);
      document.querySelectorAll('.seg button').forEach(function (o) {
        o.setAttribute('aria-pressed', String(o === b)); });
      var s = D.subjects[si];
      document.getElementById('slider').max = s.n - 1;
      document.getElementById('subjmeta').innerHTML = s.meta;
      seek(Math.min(pos, s.n - 1));
    });
  });

  window.addEventListener('resize', redraw);
  document.getElementById('subjmeta').innerHTML = D.subjects[0].meta;
  seek(D.startPos || 0);
})();
"""


def render_timeline(*, subjects, pool_assets, out_dir: Path, out_name: str,
                    provenance, tasks, n_pool, n_chans, chans, ch_names,
                    trace_hz, clip_sd, window_s, overall,
                    title: str, intro_html: str, control_label: str,
                    labels: list, footer_items: list, overview_label: str,
                    start_pos: int = 0) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = [s for su in subjects for m in su["moments"] for s in m["score"]]
    lo, hi = float(min(scores)), float(max(scores))

    seg = "".join(
        f'<button type="button" data-i="{i}" '
        f'aria-pressed="{"true" if i == 0 else "false"}">'
        f'{labels[i] if i < len(labels) else "subject " + str(i + 1)}</button>'
        for i in range(len(subjects))
    )

    data = {
        "subjects": [
            {k: v for k, v in s.items() if k != "moments"} | {"moments": s["moments"]}
            for s in subjects
        ],
        "pool": pool_assets,
        "nPool": n_pool,
        "scoreLo": lo, "scoreHi": hi,
        "windowS": window_s,
        "overviewH": OVERVIEW_H, "detailH": DETAIL_H, "ribbonH": 46,
        "startPos": int(start_pos),
    }
    payload = json.dumps(data, separators=(",", ":"))

    footer_html = "\n".join(footer_items)
    split = provenance.get("split", "?")
    ckpt = Path(str(provenance.get("checkpoint", ""))).parent.name

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_css()}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>{html.escape(title)}</h1>
  <p class="sub">{intro_html}</p>
  <p class="prov">Held-out <strong>{html.escape(str(split))}</strong> split ·
  time-level retrieval, {n_pool} candidates · overall Top-1
  {_pct(overall['top1'])} / Top-5 {_pct(overall['top5'])} / Top-10
  {_pct(overall['top10'])} · <code>{html.escape(ckpt)}</code></p>
</header>

<section class="panel">
  <div class="controls">
    <span class="lbl">{html.escape(control_label)}</span>
    <div class="seg" role="group" aria-label="{html.escape(control_label)}">{seg}</div>
    <button class="play" id="play" type="button">▶ play</button>
    <span class="clock" id="clock"></span>
  </div>
  <p class="subjmeta" id="subjmeta"></p>

  <p class="tracklabel" style="margin-top:14px">{overview_label}</p>
  <canvas id="overview" height="{OVERVIEW_H}" aria-label="EEG overview, drag to seek"></canvas>
  <p class="capt"><span>0:00</span><span>drag to seek</span></p>

  <p class="tracklabel" style="margin-top:12px">rank of the correct moment, over time
    — taller and darker is better</p>
  <canvas id="ribbon" height="46" aria-label="Rank of the correct moment over time"></canvas>
  <input type="range" id="slider" min="0" max="{subjects[0]['n'] - 1}" value="0"
    style="width:100%;margin-top:6px" aria-label="Position in recording">

  <p class="tracklabel" style="margin-top:12px">the {window_s:g} s the model is looking at</p>
  <canvas id="detail" height="{DETAIL_H}" aria-label="EEG at the current position"></canvas>
</section>

<section class="panel stage">
  <div class="truth">
    <p class="tracklabel">actually on screen</p>
    <figure id="truthimg"></figure>
  </div>
  <div class="cands">
    <p class="tracklabel">the model's top 5 of {n_pool}</p>
    <div class="strip" id="cands"></div>
    <p class="verdict" id="verdict"></p>
  </div>
</section>

<footer>
  <h2>How to read this</h2>
  <ul>
{footer_html}
  </ul>
</footer>
</div>
<script>window.__TL__ = {payload};
window.__TL__.fmt = function (t) {{
  var m = Math.floor(t / 60), s = Math.floor(t % 60);
  return m + ':' + (s < 10 ? '0' : '') + s;
}};
</script>
<script>{_js()}</script>
</body>
</html>
"""
    path = out_dir / out_name
    path.write_text(doc, encoding="utf-8")
    return path
