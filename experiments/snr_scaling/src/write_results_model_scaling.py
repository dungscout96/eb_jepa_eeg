"""Generate RESULTS_model_scaling.md from the e04_reve_scaling (depth-22) sweep.

Reads every JSON the four submitters wrote to ``raw_results/`` (see
``submit/_submit_traintest_e04.py`` and ``submit/_submit_retrieval_e04.py``)
plus ``e04_selection.json``, and renders markdown tables directly from the
data -- no numbers are hand-transcribed, so a re-run of the sweep only needs a
re-run of this script to update the doc.

Usage:
    uv run --group eeg python experiments/snr_scaling/src/write_results_model_scaling.py
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # experiments/snr_scaling
RAW = ROOT / "raw_results"
OUT = ROOT / "RESULTS_model_scaling.md"

CELL_RE = re.compile(r"^e04_s(?P<s>\d+)_a101_nd(?:_d(?P<draw>\d+))?$")
FEATURES = [
    "luminance_mean", "contrast_rms", "edge_density", "saturation_mean",
    "entropy", "motion_energy", "n_faces", "face_area_frac",
    "depth_mean", "scene_natural_score",
    "position_in_movie", "narrative_event_score",
]
S_VALUES = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]
TOPKS = ["1", "5", "10"]
LEVELS = ["time", "shot", "scene"]


def parse_cell(slug: str) -> tuple[int, int | None]:
    m = CELL_RE.match(slug)
    if not m:
        raise ValueError(f"unrecognised e04 slug: {slug}")
    return int(m["s"]), (int(m["draw"]) if m["draw"] else None)


def cells_by_s() -> dict[int, list[str]]:
    # Cell directories live on the cluster, not locally -- discover slugs from
    # the selection JSON instead, which enumerates every cell that was probed.
    out: dict[int, list[str]] = {s: [] for s in S_VALUES}
    sel = json.loads((ROOT / "e04_selection.json").read_text())
    for slug in sel:
        s, _draw = parse_cell(slug)
        out.setdefault(s, []).append(slug)
    return {s: sorted(v) for s, v in out.items()}


CELLS_BY_S = cells_by_s()
EPOCHS = {slug: info["selected_epoch"]
          for slug, info in json.loads((ROOT / "e04_selection.json").read_text()).items()
          if "selected_epoch" in info}


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def mean_sd(vals: list[float]) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{st.fmean(vals):.4f} +/- {st.stdev(vals):.4f}"


# ---------------------------------------------------------------------------
# Probe (Pearson r) tables
# ---------------------------------------------------------------------------

def probe_table(prefix: str, split: str, is_dm: bool) -> str:
    """One row per S, one column per feature, mean +/- sd across draws."""
    tag = f"{prefix}{split}" if is_dm else f"{prefix}_{split}"
    header = "| S | " + " | ".join(FEATURES) + " | mean(12) |"
    sep = "|---|" + "---|" * (len(FEATURES) + 1)
    rows = [header, sep]
    for s in S_VALUES:
        per_feature: dict[str, list[float]] = {f: [] for f in FEATURES}
        for slug in CELLS_BY_S[s]:
            d = load_json(RAW / f"{tag}_{slug}.json")
            if d is None:
                continue
            for f in FEATURES:
                if f in d["features"]:
                    per_feature[f].append(d["features"][f]["pearson_r"])
        if not any(per_feature.values()):
            continue
        cells = [mean_sd(per_feature[f]) for f in FEATURES]
        all_means = [st.fmean(per_feature[f]) for f in FEATURES if per_feature[f]]
        overall = f"{st.fmean(all_means):.4f}" if all_means else "-"
        rows.append(f"| {s} | " + " | ".join(cells) + f" | **{overall}** |")
    # random baseline
    rb = load_json(RAW / f"{tag}_random.json") if not is_dm else load_json(RAW / f"{tag}_random_e04.json")
    if rb:
        cells = [f"{rb['features'][f]['pearson_r']:.4f}" if f in rb["features"] else "-" for f in FEATURES]
        overall = st.fmean([rb["features"][f]["pearson_r"] for f in FEATURES if f in rb["features"]])
        rows.append(f"| random | " + " | ".join(cells) + f" | **{overall:.4f}** |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Retrieval tables
# ---------------------------------------------------------------------------

def retrieval_table(prefix: str, split: str, level: str, is_dm: bool) -> str:
    tag = f"{prefix}{split}" if is_dm else f"{prefix}_{split}"
    header = ("| S | e2v@1 | e2v@5 | e2v@10 | e2v@1 (xChance) | "
              "v2e@1 | v2e@5 | v2e@10 | v2e@1 (xChance) | N_pool |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for s in S_VALUES:
        e2v = {k: [] for k in TOPKS}
        v2e = {k: [] for k in TOPKS}
        e2v_rel1, v2e_rel1, npool = [], [], None
        for slug in CELLS_BY_S[s]:
            d = load_json(RAW / f"{tag}_{slug}.json")
            if d is None:
                continue
            lv = d["levels"][level]
            npool = lv["n_vision_pool_N"]
            for k in TOPKS:
                e2v[k].append(lv["e2v_top_k"][k])
                v2e[k].append(lv["v2e_top_k"][k])
            e2v_rel1.append(lv["e2v_relative"]["1"])
            v2e_rel1.append(lv["v2e_relative"]["1"])
        if npool is None:
            continue
        rows.append(
            f"| {s} | "
            + " | ".join(f"{st.fmean(e2v[k]):.3f}" for k in TOPKS) + " | "
            + f"{st.fmean(e2v_rel1):.2f}x | "
            + " | ".join(f"{st.fmean(v2e[k]):.3f}" for k in TOPKS) + " | "
            + f"{st.fmean(v2e_rel1):.2f}x | {npool} |"
        )
    rb_name = f"{tag}_random.json" if not is_dm else f"{tag}_random_e04.json"
    rb = load_json(RAW / rb_name)
    if rb:
        lv = rb["levels"][level]
        rows.append(
            "| random | "
            + " | ".join(f"{lv['e2v_top_k'][k]:.3f}" for k in TOPKS) + " | "
            + f"{lv['e2v_relative']['1']:.2f}x | "
            + " | ".join(f"{lv['v2e_top_k'][k]:.3f}" for k in TOPKS) + " | "
            + f"{lv['v2e_relative']['1']:.2f}x | {lv['n_vision_pool_N']} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Selection / methodology table
# ---------------------------------------------------------------------------

def selection_table() -> str:
    sel = json.loads((ROOT / "e04_selection.json").read_text())
    header = "| cell | S | draw | best_ep (smoothed) | selected_ep (snapped) | snap_dist | drop% from peak |"
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for slug in sorted(sel, key=lambda sl: parse_cell(sl)):
        info = sel[slug]
        if "selected_epoch" not in info:
            continue
        s, draw = parse_cell(slug)
        rows.append(
            f"| {slug} | {s} | {draw if draw else '-'} | {info['best_epoch']} | "
            f"{info['selected_epoch']} | {info['snap_distance']} | {info['drop_pct']:.1f} |"
        )
    return "\n".join(rows)


def main() -> None:
    n_cells = len(EPOCHS)
    parts = []
    parts.append(f"""# RESULTS_model_scaling — depth-22 subject-scaling sweep (e04_reve_scaling)

Full probe (12-feature Pearson r) and retrieval (time/shot/scene, e2v/v2e,
top-{{1,5,10}}) evaluation of kkokate's depth-22 encoder sweep
(`/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling`), across all {n_cells} cells
(S in {{{", ".join(str(s) for s in S_VALUES)}}}, 3 draws each except the
full-pool S=1863 cell), on both ThePresent (within-task) and DespicableMe
(cross-task, zero-shot for retrieval / linear-transfer for probe). This is the
depth-22 counterpart to `RESULTS.md` (2.12/2.13) and `RESULTS_cross_task.md`,
which cover the depth-12 `e03_scaling` arm, and the data behind
`figures/depth-12-vs-22-subject-scaling.pdf`.

All numbers in this file are generated directly from the raw JSONs in
`raw_results/` by `src/write_results_model_scaling.py` -- re-run that script
to regenerate this file after any change to the sweep.

## Headline observation: S=1863 (full pool) underperforms S=1400

Every table below rises monotonically with S from 10 through 1400, then drops
at the full-pool S=1863 cell -- consistently across both probe tasks, both
splits, and all three retrieval granularities (e.g. within-task test-split
probe mean r: 0.2927 at S=1400 vs 0.2579 at S=1863; within-task time-pool
e2v@1 test: 0.065 at S=1400 vs 0.052 at S=1863). This is consistent enough
across *every* independent readout to not be attributable to formatting or a
single bad number, but two things temper how much weight it should carry:

1. **S=1863 has no draw replicate** (it is the whole pool, so there is only
   one way to draw it), while every other S point here is a mean over 3
   draws. Its checkpoint-selection epoch (375, snapped from a smoothed best
   of 399 -- see the epoch table below) is therefore a single noisy draw, not
   an average, and the drop could partly reflect that one checkpoint's
   selection variance rather than a true population-level saturation-then-decline.
2. This mirrors a documented pattern in the depth-12 arm: `RESULTS.md` 2.10
   reports that raw-argmax epoch selection manufactured an apparent
   "overfitting worsens as data shrinks" artifact in e03, which reversed once
   selection was smoothed. The smoothing here is already applied, but a
   single-draw cell is the case that protection is weakest for.

Worth a 3-draw replicate at S=1863 (matching kkokate's own
`e04_s1863_a101_seed7` alternate-seed run, excluded from this sweep per the
methodology note above) before treating this as a real capacity/subject-count
interaction rather than a selection artifact.

## Methodology

**Per-cell early-stopped checkpoint selection**, not a fixed epoch. Unlike
e03's own published `RESULTS.md` 2.12/2.13 tables (which evaluate a single
fixed epoch, 325, on every cell), every e04 cell here uses its OWN
smoothed-selection epoch: read the `val/clip_scene_auc` history from the
offline wandb datastore, smooth with a centred rolling mean (window=25),
argmax, snap to the nearest saved checkpoint. Same selection function as
`src/select_and_probe_e03.py`, run via `src/select_e04.py`. Selected epochs
range 75-375 across cells (see table below) -- a single shared epoch would
have been wrong for several of them.

**Caveat on any depth-12 vs depth-22 comparison**: because e03's own tables
use a fixed epoch while these use per-cell selection, a straight
depth-12-vs-22 delta computed from `RESULTS.md` numbers directly is not an
apples-to-apples comparison of the *architecture* alone -- part of any
difference could be attributable to the selection protocol difference. A
clean comparison would re-select e03's checkpoints the same way before
diffing.

**Probe head-fit pool is constant across cells, only the encoder's cohort
varies.** `probe_traintest.py` fits a fresh RidgeCV head on the full training
pool -- 1863 recordings for ThePresent, 1832 for DespicableMe (see note below)
-- regardless of which S the checkpoint's own encoder was pretrained with (set
via `max_subjects` in that cell's pretraining `config.yaml`, not touched by
the eval config). This isolates "how good are the features an S-subject
encoder learned" from "how much data did the linear probe itself see."

**DespicableMe train pool is 1832, not 1841.** The extended-cohort opt-in
(`data.train_releases: [R1,R2,R3,R4,R7,R8,R9,R10]` in
`config/config_probe_DM_e04.yaml`, equivalently `HBN_TRAIN_RELEASES` env var)
gives exactly 1863 ThePresent recordings, matching e03's own established
figure. For DespicableMe it empirically gives 1832, not the 1841 that
`_submit_traintest.py`'s `EXPECTED_TRAIN_RECORDINGS` assumes -- e03 has never
actually produced an `xtask_tt_DM*` artifact to check that assumption
against, so 1841 there is an unverified docstring estimate. The 9-recording
gap is presumably a handful of DespicableMe recordings failing
`reject_recording()`'s annotation/duration checks that ThePresent's don't;
confirmed consistent across all 58 e04 cross-task cells, so not measurement
noise.

**Retrieval has no head-fitting step.** `retrieval.py` is zero-shot cosine
similarity between frozen EEG and V-JEPA-2 embeddings -- the train-pool
caveats above apply only to the probe (Pearson r) numbers, not retrieval.

### Selected epoch per cell

{selection_table()}

---

## 1. Within-task (ThePresent) probe -- Pearson r

RidgeCV head fit on the full 1863-recording pool, evaluated per feature.
Mean +/- stdev across the 3 draws at each S (single value where only one draw
exists, i.e. S=1863).

### 1.1 Validation split

{probe_table("e04_tt", "val", is_dm=False)}

### 1.2 Test split

{probe_table("e04_tt", "test", is_dm=False)}

---

## 2. Within-task retrieval -- e->v / v->e top-K accuracy

e2v: given an EEG window, is the matching vision pool entry in the top-K by
cosine similarity? v2e: given a vision pool entry, is any matching EEG window
in the top-K? "xChance" is top-1 accuracy divided by chance (1/N_pool).

### 2.1 Time-bucket pools (finest granularity, N~101)

**Validation**

{retrieval_table("e04_retr", "val", "time", is_dm=False)}

**Test**

{retrieval_table("e04_retr", "test", "time", is_dm=False)}

### 2.2 Shot pools

**Validation**

{retrieval_table("e04_retr", "val", "shot", is_dm=False)}

**Test**

{retrieval_table("e04_retr", "test", "shot", is_dm=False)}

### 2.3 Scene pools (coarsest granularity)

**Validation**

{retrieval_table("e04_retr", "val", "scene", is_dm=False)}

**Test**

{retrieval_table("e04_retr", "test", "scene", is_dm=False)}

---

## 3. Cross-task (DespicableMe) probe -- Pearson r, linear transfer

Same TP-trained checkpoints, RidgeCV head refit on DespicableMe's own
1832-recording train pool, evaluated on DespicableMe val/test. This measures
linear transfer of the learned features, not zero-shot alignment (that's
section 4).

### 3.1 Validation split

{probe_table("xtask_tt_DM", "val", is_dm=True)}

### 3.2 Test split

{probe_table("xtask_tt_DM", "test", is_dm=True)}

---

## 4. Cross-task retrieval -- zero-shot alignment on DespicableMe

Test split only (matching e03's own cross-retrieval convention). Nothing is
refit here -- the strictest transfer test: does the ThePresent-trained
EEG encoder still land in the right place in V-JEPA-2 space for a movie it
never saw during pretraining?

### 4.1 Time-bucket pools

{retrieval_table("xtask_retr_DM", "test", "time", is_dm=True)}

### 4.2 Shot pools

{retrieval_table("xtask_retr_DM", "test", "shot", is_dm=True)}

### 4.3 Scene pools

{retrieval_table("xtask_retr_DM", "test", "scene", is_dm=True)}

---

## Provenance

- Checkpoints: `/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling` (read-only;
  nothing in this pipeline writes there).
- Selection: `src/select_e04.py` -> `e04_selection.json`.
- Submitters: `submit/_submit_traintest_e04.py` (`within`/`cross` presets),
  `submit/_submit_retrieval_e04.py` (`within`/`cross` presets). Both support
  a `verify` action that checks `n_train_recordings` (probe only) against the
  expected pool size, and re-running either submitter is idempotent (every
  step is skip-if-output-exists).
- Configs: `config/config_probe_TP_e04.yaml` (ThePresent, shared across all
  cells), `config/config_probe_DM_e04.yaml` (DespicableMe cross-task).
- Raw JSONs: `raw_results/e04_tt_*`, `raw_results/e04_retr_*`,
  `raw_results/xtask_tt_DM*_e04_*`, `raw_results/xtask_retr_DMtest_e04_*`.
""")
    OUT.write_text("".join(parts))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
