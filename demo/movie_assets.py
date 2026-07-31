"""Turn a retrieval pool entry into a picture of the movie.

A pool entry is a (task, group) key at one of three granularities — a 0.5 s time
bucket, a shot, or a scene. Each maps to a span of movie time; this module
resolves that span and renders a still (and optionally a short animation) from
the source mp4.

All frames come out of a single sequential decode pass — random-access seeking
185 times through an mp4 is far slower than reading it once and keeping the
~600 frames we care about.

Paths come from ``eb_jepa.datasets.hbn.MOVIE_METADATA`` / ``_MOVIE_PATHS`` so
this stays in sync with what the dataset actually used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eb_jepa.datasets.hbn import MOVIE_METADATA, PROJECT_ROOT, _MOVIE_PATHS

# The shell exports IMAGEIO_FFMPEG_EXE pointing at a path that no longer
# exists; imageio_ffmpeg ships its own binary, so fall back to that rather than
# requiring a system ffmpeg (there is none on this machine).
_bundled = None
try:
    import imageio_ffmpeg

    _bundled = Path(imageio_ffmpeg.__file__).parent / "binaries"
    _exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if not _exe or not Path(_exe).exists():
        cands = sorted(p for p in _bundled.glob("ffmpeg-*") if p.is_file())
        if cands:
            os.environ["IMAGEIO_FFMPEG_EXE"] = str(cands[-1])
except ImportError:  # pragma: no cover - imageio_ffmpeg is a hard dep here
    pass

import imageio.v2 as imageio  # noqa: E402  (must follow the env fix above)
from PIL import Image  # noqa: E402

# Videos not covered by hbn._MOVIE_PATHS.
_EXTRA_VIDEOS = {
    "DespicableMe": PROJECT_ROOT / "movie_annotation" / "movies" / "despicable_me.mp4",
}

STILL_WIDTH = 256
GIF_WIDTH = 200
GIF_FRAMES = 10
GIF_SPAN_S = 2.0  # matches the EEG window length
GIF_DURATION_S = 0.12  # per frame


def video_path(task: str) -> Path | None:
    p = _MOVIE_PATHS.get(task) or _EXTRA_VIDEOS.get(task)
    return p if p and Path(p).exists() else None


@dataclass(frozen=True)
class Span:
    """A stretch of movie time, plus how the demo should name it.

    ``short`` fits under a thumbnail; ``label`` is the full description and goes
    in the tooltip and alt text.
    """
    task: str
    t0: float
    t1: float
    label: str
    short: str

    @property
    def mid(self) -> float:
        return 0.5 * (self.t0 + self.t1)


class MovieIndex:
    """Shot/scene time spans for one movie, from the same tables training used."""

    def __init__(self, task: str, window_size_seconds: float = 2.0):
        meta = MOVIE_METADATA[task]
        self.task = task
        self.fps = float(meta["fps"])
        self.duration = float(meta["duration"])
        self.window_size_seconds = float(window_size_seconds)

        df = pd.read_parquet(
            meta["feature_parquet"], columns=["timestamp_s", "shot_id", "entropy"]
        ).sort_values("timestamp_s")
        g = df.groupby("shot_id")["timestamp_s"].agg(["min", "max"])
        self.shot_spans = {int(s): (float(r["min"]), float(r["max"])) for s, r in g.iterrows()}
        # Image entropy per frame — used to pick a representative still. Films
        # open and close on black; the midpoint of a span is often a title card
        # or a fade, which says nothing about what the viewer was looking at.
        self._frame_t = df["timestamp_s"].to_numpy(dtype=np.float64)
        self._frame_entropy = df["entropy"].to_numpy(dtype=np.float64)

        scene_map = pd.read_csv(meta["scene_map"])
        self.shot_to_scene = dict(
            zip(scene_map["shot_id"].astype(int), scene_map["scene_id"].astype(int))
        )
        self.scene_shots: dict[int, list[int]] = {}
        for shot, scene in self.shot_to_scene.items():
            self.scene_shots.setdefault(scene, []).append(shot)
        for shots in self.scene_shots.values():
            shots.sort()

    # -- span resolution ---------------------------------------------------

    def time_span(self, bucket: int, t_bucket_s: float) -> Span:
        t0 = bucket * t_bucket_s
        t1 = min(t0 + self.window_size_seconds, self.duration)
        return Span(self.task, t0, t1,
                    f"{_mmss(t0)}–{_mmss(t1)}", _mmss(t0))

    def shot_span(self, shot_id: int) -> Span | None:
        span = self.shot_spans.get(int(shot_id))
        if span is None:
            return None
        t0, t1 = span
        return Span(self.task, t0, t1,
                    f"shot {shot_id} · {_mmss(t0)}–{_mmss(t1)}",
                    f"shot {shot_id} · {_mmss(t0)}")

    def scene_span(self, scene_id: int) -> Span | None:
        """Representative span for a scene = its longest member shot.

        A scene can be a cut-back-and-forth of several shots; the longest one is
        the most recognizable single still, while the label reports the scene's
        full extent so the page stays honest about what the pool entry covers.
        """
        shots = self.scene_shots.get(int(scene_id))
        if not shots:
            return None
        spans = [self.shot_spans[s] for s in shots if s in self.shot_spans]
        if not spans:
            return None
        t0_all = min(s[0] for s in spans)
        t1_all = max(s[1] for s in spans)
        t0, t1 = max(spans, key=lambda s: s[1] - s[0])
        n = len(spans)
        extent = f"{_mmss(t0_all)}–{_mmss(t1_all)}"
        label = f"scene {scene_id} · {extent}" + (f" · {n} shots" if n > 1 else "")
        return Span(self.task, t0, t1, label, f"scene {scene_id} · {_mmss(t0_all)}")

    def span_for(self, level: str, group: int, t_bucket_s: float) -> Span | None:
        if level == "time":
            return self.time_span(int(group), t_bucket_s)
        if level == "shot":
            return self.shot_span(int(group))
        if level == "scene":
            return self.scene_span(int(group))
        raise ValueError(f"unknown level {level!r}")

    def representative_time(self, span: Span, trim: float = 0.15) -> float:
        """The most visually informative moment inside a span.

        Takes the highest-entropy frame in the middle of the span — trimming the
        ends avoids fades and cut boundaries, and maximizing entropy avoids
        black title cards, which would otherwise represent whole scenes as an
        empty rectangle.
        """
        lo, hi = span.t0, span.t1
        pad = (hi - lo) * trim
        i0, i1 = np.searchsorted(self._frame_t, [lo + pad, hi - pad])
        if i1 <= i0:
            i0, i1 = np.searchsorted(self._frame_t, [lo, hi])
        if i1 <= i0:
            return span.mid
        return float(self._frame_t[i0 + int(np.argmax(self._frame_entropy[i0:i1]))])

    def representative_entropy(self, span: Span, trim: float = 0.15) -> float:
        """Image entropy of the still ``representative_time`` would pick.

        Lets callers spot spans that are visually empty — the opening title card
        of *The Present* is 10 s of black, and a retrieval demo built around
        "which scene was this?" should not ask that question about a blank
        screen. Read straight from the feature table; no decoding.
        """
        t = self.representative_time(span, trim)
        i = int(np.searchsorted(self._frame_t, t))
        i = min(max(i, 0), len(self._frame_entropy) - 1)
        return float(self._frame_entropy[i])

    def scene_of_time(self, t: float) -> int | None:
        for shot, (t0, t1) in self.shot_spans.items():
            if t0 <= t <= t1:
                return self.shot_to_scene.get(shot)
        return None


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def gif_frame_times(span: Span, n: int = GIF_FRAMES, window_s: float = GIF_SPAN_S,
                    duration: float | None = None,
                    center: float | None = None) -> list[float]:
    """Evenly spaced sample times inside ``span``, capped to ``window_s`` and
    centered on ``center`` (default: the span midpoint), clamped to the span."""
    length = min(span.t1 - span.t0, window_s)
    if length <= 0:
        return [span.t0]
    mid = span.mid if center is None else center
    start = min(max(mid - length / 2, span.t0), span.t1 - length)
    start = max(0.0, start)
    if duration is not None:
        start = min(start, max(0.0, duration - length))
    return [start + length * i / max(1, n - 1) for i in range(n)]


def decode_frames(task: str, times: set[float], fps: float) -> dict[int, np.ndarray]:
    """Read the movie once, returning {frame_index: RGB array} for the requested
    times. Sequential — cost is one full decode regardless of how many frames."""
    path = video_path(task)
    if path is None:
        raise FileNotFoundError(f"no video on disk for task {task!r}")
    wanted = {int(round(t * fps)) for t in times}
    out: dict[int, np.ndarray] = {}
    reader = imageio.get_reader(str(path))
    try:
        last_needed = max(wanted)
        for i, frame in enumerate(reader):
            if i in wanted:
                out[i] = frame
            if i >= last_needed:
                break
    finally:
        reader.close()
    missing = wanted - out.keys()
    if missing:
        # Past-the-end requests clamp to the final decoded frame.
        fallback = out[max(out)] if out else None
        for i in missing:
            if fallback is not None:
                out[i] = fallback
    return out


def _resize(frame: np.ndarray, width: int) -> Image.Image:
    img = Image.fromarray(frame)
    h = max(1, round(img.height * width / img.width))
    return img.resize((width, h), Image.LANCZOS)


def write_still(frame: np.ndarray, dest: Path, width: int = STILL_WIDTH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _resize(frame, width).save(dest, "JPEG", quality=82, optimize=True)
    return dest


def write_gif(frames: list[np.ndarray], dest: Path, width: int = GIF_WIDTH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    imgs = [_resize(f, width).convert("P", palette=Image.ADAPTIVE, colors=128)
            for f in frames]
    imgs[0].save(
        dest, "GIF", save_all=True, append_images=imgs[1:],
        duration=int(GIF_DURATION_S * 1000), loop=0, optimize=True,
    )
    return dest
