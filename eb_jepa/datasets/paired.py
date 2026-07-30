"""Time-aligned cross-subject pairing on top of :class:`JEPAMovieDataset`.

The cross-subject predictive objective needs, for one movie moment *t*, the EEG
of two *different* subjects watching that same moment. Every other objective in
this repo is within-sample (masked JEPA) or within-batch (the CLIP family reads
``t_starts``/``scene_ids`` that already happen to be in the batch), so nothing
here previously needed a cross-recording *lookup*. This module builds that index.

Movie time comes from ``t_start_recordings[r]`` (``frame_idx / movie_fps``,
populated unconditionally in ``JEPAMovieDataset.__init__``). We join on the
*value*, not on the window index — the same thing
``evaluation/clip_probe/retrieval.py::_build_time_pool`` does when it pools
across recordings.

Note on why the value-join is insurance rather than a bug fix: both places that
drop windows are monotone in movie time, so today they only trim a prefix/suffix
and window index *i* already is the same movie moment in every recording.

  * ``HBNMovieDataset``'s ``keep_mask`` drops iff ``frame_index`` is out of
    range, and ``frame_index`` is monotone in ``window_onset``.
  * ``get_window_frame_embedding`` returns None iff no V-JEPA-2 timestamp lands
    in ``[t_start, t_start + window_size_seconds)``. Both shipped grids are
    uniform 0.5 s with no gaps (ThePresent n=406 over [0.0, 202.5];
    DespicableMe n=341 over [0.0, 170.0]), and 0.5 s is far below any window
    size we use, so this can only fail past the last timestamp.

The index costs ~10 lines and keeps the pairing correct — rather than silently
wrong — if the embedding grid is ever re-extracted at a coarser stride, or a
movie whose grid is not uniform is wired into ``MOVIE_METADATA``.
"""
import logging
from collections import defaultdict

import torch

from eb_jepa.datasets.hbn import JEPAMovieDataset

logger = logging.getLogger(__name__)


class PairedSubjectJEPADataset(JEPAMovieDataset):
    """Yields (subject A, subject B) EEG clips from the same movie window.

    IMPORTANT: unlike every other dataset in this repo, ``__getitem__`` is
    **not** recording-indexed — ``self[i]`` does not return recording *i*. The
    index selects a subject-A slot; the movie time and the partner subject are
    drawn randomly per access. Code that needs per-recording access (the probe
    and retrieval evaluators) must build a plain ``JEPAMovieDataset`` instead;
    they all already do.

    Items are fixed-shape tensors so the default DataLoader collate works —
    no custom ``Sampler`` and no ``collate_fn``:

        eeg      [2, n_windows, C, T]  float32  (row 0 = subject A, row 1 = B)
        t_starts [n_windows]           float32  movie time in seconds
        rec_ids  [2]                   int64    (rec_idx_A, rec_idx_B)

    Args:
        pairs_per_recording: how many times each recording appears as subject A
            per epoch. ``1`` reproduces ``JEPAMovieDataset``'s epoch semantics
            (one clip per recording), which keeps steps/epoch and the LR
            schedule shape comparable to a masked-JEPA baseline.
        pair_min_partners: a movie window is only usable if at least this many
            *other* recordings also have it. ``1`` is the minimum for pairing.
        **kwargs: forwarded verbatim to ``JEPAMovieDataset``.
    """

    #: Guard for anything tempted to treat this as recording-indexed.
    is_recording_indexed = False

    def __init__(self, *args, pairs_per_recording: int = 1,
                 pair_min_partners: int = 1, **kwargs):
        # Validate before super().__init__, which spends minutes loading and
        # windowing the releases — no reason to do that only to reject an arg.
        if pairs_per_recording < 1:
            raise ValueError(
                f"pairs_per_recording must be >= 1, got {pairs_per_recording}"
            )
        if pair_min_partners < 1:
            raise ValueError(
                f"pair_min_partners must be >= 1 (pairing needs a partner), "
                f"got {pair_min_partners}"
            )
        self.pairs_per_recording = int(pairs_per_recording)
        self.pair_min_partners = int(pair_min_partners)
        super().__init__(*args, **kwargs)
        self._build_pair_index()

    # -- index construction -------------------------------------------------

    def _tkey(self, t_start: float) -> int:
        """Quantize movie time to a window ordinal.

        Distinct windows are exactly ``window_size_seconds`` apart, so this
        absorbs up to +-ws/2 of drift with no possibility of merging two
        genuinely different windows. Bucketing by a fixed epsilon (as
        retrieval.py's ``--t-bucket-s`` does) is fine at ws=2 where t_start is
        an exact multiple of 2.0, but drifts for any ws where ``ws * fps`` is
        not an integer (ws=1.5 on DespicableMe moves 0.04 s per window).
        """
        return int(round(float(t_start) / float(self.window_size_seconds)))

    def _build_pair_index(self) -> None:
        """Build the (movie, window-ordinal) -> [recording] index.

        Sets ``_tkey_to_win``, ``_anchor_to_recs``, ``_rec_anchors``, ``_a_pool``.
        """
        n_recs = len(self._fif_paths)
        stride = self.temporal_stride
        n_win_per_clip = self.n_windows

        # (1) per-recording  window-ordinal -> window index
        self._tkey_to_win: list[dict[int, int]] = []
        collisions: list[str] = []
        for r in range(n_recs):
            m: dict[int, int] = {}
            for i, tv in enumerate(self.t_start_recordings[r].tolist()):
                k = self._tkey(tv)
                if k in m:
                    collisions.append(
                        f"{self._fif_paths[r]}: windows {m[k]} and {i} both map to key {k}"
                    )
                m[k] = i
            self._tkey_to_win.append(m)
        if collisions:
            raise ValueError(
                f"{len(collisions)} t_start collision(s) after quantizing by "
                f"window_size_seconds={self.window_size_seconds}. Two distinct "
                "windows mapped to one key, so the movie-time grid is not a clean "
                "multiple of the window size and the quantizer is unsound. "
                f"First few: {collisions[:3]}"
            )

        # (2) anchors. A recording can serve a clip anchored at k0 only if EVERY
        #     window of the clip is present. With the current data this is
        #     trivially satisfied (drops are a suffix trim), so it acts as a
        #     tripwire: it keeps the pairing correct rather than silently wrong
        #     if the embedding grid ever develops an interior hole.
        anchor_to_recs: dict[tuple[str, int], list[int]] = defaultdict(list)
        for r, m in enumerate(self._tkey_to_win):
            movie = self._recording_tasks[r]
            for k0 in m:
                if all((k0 + j * stride) in m for j in range(n_win_per_clip)):
                    anchor_to_recs[(movie, k0)].append(r)

        # (3) prune anchors that cannot yield a distinct-subject pair, then
        #     invert to per-recording anchor lists.
        self._anchor_to_recs: dict[tuple[str, int], list[int]] = {}
        self._rec_anchors: list[list[tuple[str, int]]] = [[] for _ in range(n_recs)]
        for key, recs in anchor_to_recs.items():
            if len(recs) < 1 + self.pair_min_partners:
                continue
            # An anchor is only usable if it spans at least two subjects.
            if len({self._subject_of(r) for r in recs}) < 2:
                continue
            self._anchor_to_recs[key] = recs
            for r in recs:
                self._rec_anchors[r].append(key)

        # (4) subject-A pool = recordings with at least one pairable anchor
        self._a_pool = [r for r in range(n_recs) if self._rec_anchors[r]]
        if not self._a_pool:
            raise ValueError(
                "No pairable (movie, time) anchors found: no movie window is "
                f"shared by >= {1 + self.pair_min_partners} recordings spanning "
                ">= 2 subjects. Check that the split has multiple subjects per "
                "movie and that n_windows/temporal_stride are not too large for "
                "the recordings available."
            )

        partner_counts = sorted(len(v) for v in self._anchor_to_recs.values())
        median_partners = partner_counts[len(partner_counts) // 2]
        logger.info(
            "Cross-subject pair index: %d anchors, median %d recordings/anchor, "
            "subject-A pool %d/%d recordings (len=%d items/epoch)",
            len(self._anchor_to_recs), median_partners,
            len(self._a_pool), n_recs, len(self),
        )
        if len(self._a_pool) < n_recs:
            logger.warning(
                "%d/%d recordings have no pairable anchor and will never be "
                "sampled as subject A. Coverage-driven window drops are expected "
                "to be a pure suffix trim, so any shortfall here means that "
                "assumption no longer holds — investigate the V-JEPA-2 timestamp "
                "grid before trusting the run.",
                n_recs - len(self._a_pool), n_recs,
            )

    def _subject_of(self, rec_idx: int) -> str:
        """Subject identifier for a recording.

        Falls back to a per-recording sentinel so that a recording with missing
        metadata is never treated as the same subject as another one.
        """
        meta = self._recording_metadata[rec_idx] or {}
        subj = meta.get("subject", meta.get("participant_id", None))
        return str(subj) if subj is not None else f"__rec{rec_idx}"

    # -- item access --------------------------------------------------------

    def __len__(self):
        return len(self._a_pool) * self.pairs_per_recording

    def __getitem__(self, idx):
        rec_a = self._a_pool[idx % len(self._a_pool)]
        anchors = self._rec_anchors[rec_a]
        key = anchors[torch.randint(len(anchors), (1,)).item()]
        partners = self._anchor_to_recs[key]

        # Reject-sample a different subject. Bounded retries, then a
        # deterministic scan, so this can never loop forever. Anchors are
        # pre-filtered to span >= 2 subjects, so the scan always succeeds
        # unless rec_a is the anchor's only subject.
        subj_a = self._subject_of(rec_a)
        rec_b = -1
        for _ in range(8):
            cand = partners[torch.randint(len(partners), (1,)).item()]
            if self._subject_of(cand) != subj_a:
                rec_b = cand
                break
        if rec_b < 0:
            rec_b = next(
                (r for r in partners if self._subject_of(r) != subj_a), -1
            )
        if rec_b < 0:
            raise RuntimeError(
                f"Anchor {key} has no partner with a subject different from "
                f"{subj_a!r} (recording {rec_a}), which _build_pair_index should "
                "have excluded. This is a bug in the anchor filter."
            )

        _movie, k0 = key
        stride = self.temporal_stride
        win_a = [self._tkey_to_win[rec_a][k0 + j * stride] for j in range(self.n_windows)]
        win_b = [self._tkey_to_win[rec_b][k0 + j * stride] for j in range(self.n_windows)]

        eeg_a = self._load_clip(rec_a, win_a)   # [n_windows, C, T]
        eeg_b = self._load_clip(rec_b, win_b)

        return (
            torch.stack([eeg_a, eeg_b], dim=0),                  # [2, n_windows, C, T]
            self.t_start_recordings[rec_a][win_a],               # [n_windows]
            torch.tensor([rec_a, rec_b], dtype=torch.long),      # [2]
        )
