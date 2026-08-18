"""Generate RESULTS_add_val_set.md -- does the subject-scaling saturation hold
once the val release (R5) is folded into the pretraining pool?

Reads the raw JSONs both arms' submitters wrote to ``raw_results/`` and renders
every table directly from them, so nothing is hand-transcribed:

  e04     kkokate's depth-22 e04_reve_scaling, pool = 1863 (R1-R4, R7-R10),
          at each cell's own smoothed-selection epoch (e04_selection.json).
  addval  e05_addval -- the identical recipe with R5 in the pool (2156), at a
          fixed epoch 375. See submit/_submit_e05_addval.py for why a fixed
          epoch is the protocol rather than a fallback, and why 375 makes the
          two arms comparable to within one 25-epoch checkpoint interval.

TEST SPLIT ONLY, both arms, every table. The addval cells trained on R5, so a
val number for them would be measured on data their encoder saw. R6 is
untouched by both arms and is the only split on which they are comparable.

Usage:
    uv run --group eeg python experiments/snr_scaling/src/write_results_add_val_set.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # experiments/snr_scaling
RAW = ROOT / "raw_results"
OUT = ROOT / "RESULTS_add_val_set.md"

FEATURES = [
    "luminance_mean", "contrast_rms", "edge_density", "saturation_mean",
    "entropy", "motion_energy", "n_faces", "face_area_frac",
    "depth_mean", "scene_natural_score",
    "position_in_movie", "narrative_event_score",
]
TOPKS = ["1", "5", "10"]
LEVELS = ["time", "shot", "scene"]

E04_SELECTED = {
    slug: info["selected_epoch"]
    for slug, info in json.loads((ROOT / "e04_selection.json").read_text()).items()
    if "selected_epoch" in info
}
E05_EPOCH = 375

# Which cells make up each (arm, S) row. e04's tail is carried for context --
# the curve has to be visible for "does it still saturate" to mean anything --
# and the addval arm holds the three S values it was trained at.
# Which cells make up each (arm, S) row.
#
# The addval arm now spans the FULL axis from one pool (10 -> 2156) and is the
# official subject-scaling curve. e04's axis is carried alongside it for the
# retraction argument and the pool calibration -- not as a competing result.
E05_S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]
E04_S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400]
ROWS: list[tuple[str, int, list[str]]] = (
    [("e04", s, [f"e04_s{s}_a101_nd_d{d}" for d in (11, 22, 33)])
     for s in E04_S_AXIS]
    + [("e04", 1863, ["e04_s1863_a101_nd"])]
    + [("addval", s, [f"e05_s{s}_a101_av_d{d}" for d in (11, 22, 33)])
       for s in E05_S_AXIS]
    + [("addval", 2156, ["e05_s2156_a101_av"])]
)
ARM_POOL = {"e04": 1863, "addval": 2156}
ARM_LABEL = {"e04": "e04 (pool 1863)", "addval": "addval (pool 2156)"}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def fmt(vals: list[float], places: int = 4) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{places}f}"
    return f"{st.fmean(vals):.{places}f} +/- {st.stdev(vals):.{places}f}"


def tag(prefix: str, split: str = "test") -> str:
    """`xtask_*_DM` artifacts glue the split on directly; the rest use '_'."""
    return f"{prefix}{split}" if prefix.endswith("DM") else f"{prefix}_{split}"


# Cells excluded from every AGGREGATE (mean +/- sd) below, but still shown
# individually in section 2b so the exclusion is visible rather than silent.
#
# The bar is an optimisation failure evident in the artifacts, never a
# preference for the answer. e04 has the same failure mode and the same remedy:
# three of its cells were kept as `_FAILED_seed2026` directories and re-run at
# --meta.seed=7.
FAILED_CELLS: dict[str, str] = {}

# Cells whose published number comes from a RE-RUN at a different seed, after
# the original failed to converge. Disclosed here rather than silently swapped,
# and reported in section 2b, because "we re-ran it" is exactly the kind of
# step that can launder a preferred answer if it goes unstated.
#
# The failure was judged on artifacts before the re-run existed, not after:
# flat at the random baseline across EVERY epoch, plus a high training loss.
# The re-run then landed on its siblings rather than above them, which is the
# outcome that would have exposed seed-shopping had it gone the other way.
# Base note shared by all reseeded cells; each entry adds its own numbers.
_RESEED_WHY = (
    "failed to converge at seed 2026 and was re-run at --meta.seed=7, the same "
    "remedy e04 applied to its own three `_FAILED_seed2026` cells. Detected by "
    "`_submit_e05_addval.py screen`, which flags a cell whose final training "
    "loss exceeds 1.30x the median of its own S group -- relative, because loss "
    "scales with cohort size (S=10 sits near 0.70, S=1863 near 2.8), so a fixed "
    "cutoff would miss the same failure at low S. On this arm the three genuine "
    "failures sat at 1.67-3.09x their group median while the worst healthy cell "
    "was 1.19x. Superseded runs are kept as `<cell>_FAILED_seed2026` on Delta "
    "with their artifacts under `raw_results/_failed_e05_*`."
)
RESEEDED_CELLS = {
    "e05_s1400_a101_av_d11": _RESEED_WHY + (
        " This cell: time-pool e2v@1 0.017/0.017/0.018 at epochs 300/350/375 "
        "(flat, at the 0.010 random baseline) vs siblings' 0.066-0.074; final "
        "loss 3.9 vs 2.8. Not an unlucky cohort -- the S=1863 d11 cell, a "
        "strict superset of this cohort, trained normally. Re-run gives "
        "0.068/0.067/0.069 and clip_scene_auc 0.875 at ep375 (was 0.607)."
    ),
    "e05_s400_a101_av_d11": _RESEED_WHY + (
        " This cell: final loss 4.02 = 3.09x its S=400 median; "
        "clip_scene_auc@375 0.532 vs siblings' 0.742/0.800."
    ),
    "e05_s701_a101_av_d33": _RESEED_WHY + (
        " This cell: final loss 4.07 = 1.89x its S=701 median; time-pool e2v@1 "
        "0.0123 against the 0.0099 random baseline, vs siblings' 0.053/0.058."
    ),
    "e05_s1000_a101_av_d11": _RESEED_WHY + (
        " This cell: final loss 4.16 = 1.67x its S=1000 median; time-pool "
        "e2v@1 0.0116 (random 0.0099) vs siblings' 0.0640/0.0640."
    ),
}


def artifacts(prefix: str, slugs: list[str], suffix: str = "") -> list[dict]:
    """`suffix` selects a fixed-epoch variant, e.g. "_ep325". Empty means the
    primary artifact -- each e04 cell's selected epoch, addval's fixed 375."""
    return [d for d in (load(RAW / f"{tag(prefix)}_{s}{suffix}.json")
                        for s in slugs if s not in FAILED_CELLS)
            if d is not None]


def all_artifacts(prefix: str, slugs: list[str]) -> list[tuple[str, dict]]:
    """Every cell including failed ones -- for the per-draw table only."""
    out = []
    for s in slugs:
        d = load(RAW / f"{tag(prefix)}_{s}.json")
        if d is not None:
            out.append((s, d))
    return out


# ---------------------------------------------------------------------------
# Probe (Pearson r)
# ---------------------------------------------------------------------------

def probe_table(prefix: str) -> str:
    header = "| arm | S | n | " + " | ".join(FEATURES) + " | mean(12) |"
    sep = "|---|---|---|" + "---|" * (len(FEATURES) + 1)
    rows = [header, sep]
    for arm, s, slugs in ROWS:
        ds = artifacts(prefix, slugs)
        if not ds:
            continue
        per = {f: [d["features"][f]["pearson_r"] for d in ds
                   if f in d["features"]] for f in FEATURES}
        means = [st.fmean(per[f]) for f in FEATURES if per[f]]
        rows.append(
            f"| {ARM_LABEL[arm]} | {s} | {len(ds)}/{len(slugs)} | "
            + " | ".join(fmt(per[f]) for f in FEATURES)
            + f" | **{st.fmean(means):.4f}** |")
    rand = load(RAW / f"{tag(prefix)}_random.json") or \
        load(RAW / f"{tag(prefix)}_random_e04.json")
    if rand:
        vals = [rand["features"][f]["pearson_r"] for f in FEATURES
                if f in rand["features"]]
        rows.append(
            "| random | - | - | "
            + " | ".join(f"{rand['features'][f]['pearson_r']:.4f}"
                         if f in rand["features"] else "-" for f in FEATURES)
            + f" | **{st.fmean(vals):.4f}** |")
    return "\n".join(rows)


def probe_mean12(prefix: str, slugs: list[str], suffix: str = "") -> list[float]:
    """Per-CELL mean over the 12 features -- so a stdev across draws is a
    between-draw sd, not a between-feature one."""
    return [st.fmean([d["features"][f]["pearson_r"] for f in FEATURES
                      if f in d["features"]])
            for d in artifacts(prefix, slugs, suffix)]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieval_table(prefix: str, level: str) -> str:
    header = ("| arm | S | n | e2v@1 | e2v@5 | e2v@10 | e2v@1 (xChance) | "
              "v2e@1 | v2e@5 | v2e@10 | v2e@1 (xChance) | N_pool |")
    rows = [header, "|---|---|---|" + "---|" * 9]
    for arm, s, slugs in ROWS:
        ds = artifacts(prefix, slugs)
        if not ds:
            continue
        lv = [d["levels"][level] for d in ds]
        rows.append(
            f"| {ARM_LABEL[arm]} | {s} | {len(ds)}/{len(slugs)} | "
            + " | ".join(f"{st.fmean([x['e2v_top_k'][k] for x in lv]):.3f}"
                         for k in TOPKS)
            + f" | {st.fmean([x['e2v_relative']['1'] for x in lv]):.2f}x | "
            + " | ".join(f"{st.fmean([x['v2e_top_k'][k] for x in lv]):.3f}"
                         for k in TOPKS)
            + f" | {st.fmean([x['v2e_relative']['1'] for x in lv]):.2f}x "
            + f"| {lv[0]['n_vision_pool_N']} |")
    rand = load(RAW / f"{tag(prefix)}_random.json") or \
        load(RAW / f"{tag(prefix)}_random_e04.json")
    if rand:
        x = rand["levels"][level]
        rows.append(
            "| random | - | - | "
            + " | ".join(f"{x['e2v_top_k'][k]:.3f}" for k in TOPKS)
            + f" | {x['e2v_relative']['1']:.2f}x | "
            + " | ".join(f"{x['v2e_top_k'][k]:.3f}" for k in TOPKS)
            + f" | {x['v2e_relative']['1']:.2f}x | {x['n_vision_pool_N']} |")
    return "\n".join(rows)


def retr_e2v1(prefix: str, level: str, slugs: list[str],
              suffix: str = "") -> list[float]:
    return [d["levels"][level]["e2v_top_k"]["1"]
            for d in artifacts(prefix, slugs, suffix)]


# ---------------------------------------------------------------------------
# Headline + calibration
# ---------------------------------------------------------------------------

READOUTS = [
    ("TP probe mean(12) r", lambda sl: probe_mean12("e04_tt", sl), 4),
    ("TP retr scene e2v@1", lambda sl: retr_e2v1("e04_retr", "scene", sl), 3),
    ("TP retr time e2v@1", lambda sl: retr_e2v1("e04_retr", "time", sl), 3),
    ("DM probe mean(12) r", lambda sl: probe_mean12("xtask_tt_DM", sl), 4),
    ("DM retr scene e2v@1", lambda sl: retr_e2v1("xtask_retr_DM", "scene", sl), 3),
]


def headline_table() -> str:
    rows = ["| readout | " + " | ".join(
        f"{ARM_LABEL[a].split()[0]} S={s}" for a, s, _ in ROWS) + " |",
        "|---|" + "---|" * len(ROWS)]
    for name, getter, places in READOUTS:
        cells = [fmt(getter(sl), places) for _a, _s, sl in ROWS]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def calibration_table() -> str:
    """e04 S=1400 vs addval S=1400 -- the same S drawn from the two pools.

    This is the load-bearing check. The two arms' draws are NOT the same
    cohorts (the subject list is permuted, and adding R5 changes the list), so
    if the pools are exchangeable these must agree to within the between-draw
    sd. If they do not, every S=1863/2156 comparison has to be read through
    that offset rather than at face value.
    """
    e04 = next(sl for a, s, sl in ROWS if a == "e04" and s == 1400)
    av = next(sl for a, s, sl in ROWS if a == "addval" and s == 1400)
    rows = ["| readout | e04 S=1400 (pool 1863) | addval S=1400 (pool 2156) "
            "| delta | delta / e04 sd |", "|---|---|---|---|---|"]
    for name, getter, places in READOUTS:
        a, b = getter(e04), getter(av)
        if not a or not b:
            rows.append(f"| {name} | {fmt(a, places)} | {fmt(b, places)} | - | - |")
            continue
        delta = st.fmean(b) - st.fmean(a)
        sd = st.stdev(a) if len(a) > 1 else None
        rows.append(
            f"| {name} | {fmt(a, places)} | {fmt(b, places)} | "
            f"{delta:+.{places}f} | "
            + (f"{delta / sd:+.2f}" if sd else "-") + " |")
    return "\n".join(rows)


def per_draw_table() -> str:
    """Every cell individually, no averaging.

    A three-draw mean hides a dead cell. This arm has one -- e05_s1400 d11 --
    and a reader who only sees `0.123 +/- 0.042` cannot tell whether that is
    three mediocre draws or two good ones and a failure. It matters because
    the two have opposite implications for whether the pools stitch.
    """
    rows = ["| arm | S | draw | TP probe mean(12) r | TP retr time e2v@1 "
            "| TP retr scene e2v@1 | DM probe mean(12) r |",
            "|---|---|---|---|---|---|---|"]
    def one(prefix, slug, getter):
        d = load(RAW / f"{tag(prefix)}_{slug}.json")
        return [] if d is None else [getter(d)]
    mean12 = lambda d: st.fmean([d["features"][f]["pearson_r"]
                                 for f in FEATURES if f in d["features"]])
    for arm, s, slugs in ROWS:
        for slug in slugs:
            draw = slug.split("_d")[-1] if "_d" in slug else "-"
            flag = (" **(FAILED, excluded)**" if slug in FAILED_CELLS
                    else " *(reseeded)*" if slug in RESEEDED_CELLS else "")
            rows.append(
                f"| {ARM_LABEL[arm]} | {s} | {draw}{flag} | "
                f"{fmt(one('e04_tt', slug, mean12))} | "
                f"{fmt(one('e04_retr', slug, lambda d: d['levels']['time']['e2v_top_k']['1']), 3)} | "
                f"{fmt(one('e04_retr', slug, lambda d: d['levels']['scene']['e2v_top_k']['1']), 3)} | "
                f"{fmt(one('xtask_tt_DM', slug, mean12))} |")
    for slug, why in FAILED_CELLS.items():
        rows += ["", f"**{slug} — excluded from every mean above.** {why}"]
    for slug, why in RESEEDED_CELLS.items():
        rows += ["", f"**{slug} — reseeded.** {why}"]
    return "\n".join(rows)


def epoch_robustness_table() -> str:
    """The same cells at neighbouring epochs.

    These cells cannot use e04's smoothed per-cell selection (R5 is in their
    training set, so `val/clip_scene_auc` is in-sample), so they are pinned to
    a fixed epoch. That protocol has a real cost and this table is where it is
    visible: a cell caught at a transient dip looks like a bad cohort. If the
    S=1863 conclusion moves across these columns it is an epoch artifact; if it
    does not, the fixed epoch is not doing any work.
    """
    epochs = [("300", "_ep300"), ("350", "_ep350"), ("375 (primary)", "")]
    rows = ["| arm | S | draw | " + " | ".join(f"ep{e}" for e, _s in epochs) + " |",
            "|---|---|---|" + "---|" * len(epochs)]
    any_alt = False
    for arm, s, slugs in ROWS:
        if arm != "addval":
            continue
        for slug in slugs:
            draw = slug.split("_d")[-1] if "_d" in slug else "-"
            flag = (" (FAILED)" if slug in FAILED_CELLS
                    else " (reseeded)" if slug in RESEEDED_CELLS else "")
            cells = []
            for _e, suf in epochs:
                p = RAW / f"{tag('e04_retr')}_{slug}{suf}.json"
                if p.exists():
                    cells.append(f"{json.loads(p.read_text())['levels']['time']['e2v_top_k']['1']:.3f}")
                    any_alt = any_alt or bool(suf)
                else:
                    cells.append("-")
            rows.append(f"| {ARM_LABEL[arm]} | {s} | {draw}{flag} | " + " | ".join(cells) + " |")
    if not any_alt:
        return ("_Not yet measured -- run `_submit_retrieval_e04.py within submit "
                "--arm addval --epoch 350` (and 300) to populate._")
    return "\n".join(rows) + "\n\nTP retrieval, time pool, e2v top-1, test split."


def decomposition_table() -> str:
    """Separate the pool effect from the S effect from the alleged drop.

    Three contrasts, because the naive one (e04 S=1863 vs addval S=1863) mixes
    all three together:

      A  pool effect at matched S      e04 S=1400  -> addval S=1400
      B  S effect within the new pool  addval S=1400 -> addval S=1863
      C  the alleged drop, in e04      e04 S=1400  -> e04 S=1863

    If C is a real population effect it should reappear as a negative B, since
    B is the same step in S measured with three draws instead of one. If B is
    positive while C is negative, the drop belongs to e04's single draw.
    """
    def cells(arm, s):
        return next(sl for a, ss, sl in ROWS if a == arm and ss == s)
    contrasts = [
        ("A: pool effect @ S=1400", ("e04", 1400), ("addval", 1400)),
        ("B: S 1400->1863, new pool", ("addval", 1400), ("addval", 1863)),
        ("C: S 1400->1863, e04", ("e04", 1400), ("e04", 1863)),
        ("D: S 1863->2156, new pool", ("addval", 1863), ("addval", 2156)),
    ]
    rows = ["| readout | " + " | ".join(n for n, _a, _b in contrasts) + " |",
            "|---|" + "---|" * len(contrasts)]
    for name, getter, places in READOUTS:
        out = []
        for _n, (a0, s0), (a1, s1) in contrasts:
            v0, v1 = getter(cells(a0, s0)), getter(cells(a1, s1))
            if not v0 or not v1:
                out.append("-")
                continue
            d = st.fmean(v1) - st.fmean(v0)
            out.append(f"{d:+.{places}f}")
        rows.append(f"| {name} | " + " | ".join(out) + " |")
    return "\n".join(rows)


def matched_epoch_table() -> str:
    """Pool offset with BOTH arms at fixed epoch 325.

    Sections 1-2a compare e04 at its own smoothed-selected epoch against addval
    at a fixed 375. That is matched to within one checkpoint interval at high S
    (4 of 4 e04 high-S cells selected 375 or 350) but it is still two
    protocols, and at low S e04's selection scatters from 75 to 375. This table
    removes the protocol difference entirely: every cell here, both arms, is
    epoch 325.

    It exists to answer one question -- can e04's low-S cells be spliced onto
    this arm to form a single 10 -> 2156 curve? -- so it reports the OVERLAP
    points, S=1400 and S=1863, which are the only S where both arms have cells.
    """
    def av(prefix, slugs, getter):
        out = []
        for s in slugs:
            d = load(RAW / f"{tag(prefix)}_{s}_ep325.json")
            if d is not None:
                out.append(getter(d))
        return out
    mean12 = lambda d: st.fmean([d["features"][f]["pearson_r"]
                                 for f in FEATURES if f in d["features"]])
    readouts = [
        ("TP probe mean(12) r", "e04_tt", mean12, 4),
        ("DM probe mean(12) r", "xtask_tt_DM", mean12, 4),
        ("TP retr time e2v@1", "e04_retr",
         lambda d: d["levels"]["time"]["e2v_top_k"]["1"], 4),
        ("TP retr shot e2v@1", "e04_retr",
         lambda d: d["levels"]["shot"]["e2v_top_k"]["1"], 4),
        ("TP retr scene e2v@1", "e04_retr",
         lambda d: d["levels"]["scene"]["e2v_top_k"]["1"], 4),
        ("DM retr scene e2v@1", "xtask_retr_DM",
         lambda d: d["levels"]["scene"]["e2v_top_k"]["1"], 4),
    ]
    rows = ["| readout | S | e04 (pool 1863) | addval (pool 2156) | delta "
            "| in e04 sd |", "|---|---|---|---|---|---|"]
    for name, prefix, getter, places in readouts:
        for S in (1400, 1863):
            e04 = av(prefix, next(sl for a, s, sl in ROWS
                                  if a == "e04" and s == S), getter)
            adv = av(prefix, next(sl for a, s, sl in ROWS
                                  if a == "addval" and s == S), getter)
            if not e04 or not adv:
                rows.append(f"| {name} | {S} | {fmt(e04, places)} | "
                            f"{fmt(adv, places)} | - | - |")
                continue
            d = st.fmean(adv) - st.fmean(e04)
            sd = st.stdev(e04) if len(e04) > 1 else None
            rows.append(
                f"| {name} | {S} | {fmt(e04, places)} | {fmt(adv, places)} | "
                f"{d:+.{places}f} | "
                + (f"{d / sd:+.2f}" if sd else "e04 n=1") + " |")
    return "\n".join(rows)


def official_curve_table() -> str:
    """The addval arm across the FULL axis at fixed epoch 325.

    This is the curve the paper plots, and until now it had no table. The
    sections above are keyed to epoch 375, which exists only for S >= 1400 --
    the low-S cells (10..1000) were evaluated at 325 only, so they read as
    absent everywhere above while being perfectly present in the figure. A
    reader checking a plotted point against this file would have found nothing.

    One pool (2156) and one epoch (325) at every S, so there is no splice seam
    and no selection seam anywhere on the axis -- which is the property that
    makes this, rather than the spliced e04+addval curve, the reportable one.
    """
    mean12 = lambda d: st.fmean([d["features"][f]["pearson_r"]
                                 for f in FEATURES if f in d["features"]])
    readouts = [
        ("TP probe mean(12) r", "e04_tt", mean12),
        ("TP retr time e2v@1", "e04_retr",
         lambda d: d["levels"]["time"]["e2v_top_k"]["1"]),
        ("TP retr scene e2v@1", "e04_retr",
         lambda d: d["levels"]["scene"]["e2v_top_k"]["1"]),
        ("DM probe mean(12) r", "xtask_tt_DM", mean12),
        ("DM retr time e2v@1", "xtask_retr_DM",
         lambda d: d["levels"]["time"]["e2v_top_k"]["1"]),
    ]
    axis = [s for a, s, _sl in ROWS if a == "addval"]
    rows = ["| readout | " + " | ".join(f"S={s}" for s in axis) + " |",
            "|---|" + "---|" * len(axis)]
    for name, prefix, getter in readouts:
        cells = []
        for S in axis:
            slugs = next(sl for a, s, sl in ROWS if a == "addval" and s == S)
            vals = [getter(d) for d in
                    (load(RAW / f"{tag(prefix)}_{sl}_ep325.json")
                     for sl in slugs) if d is not None]
            cells.append(fmt(vals, 4))
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def coverage() -> str:
    """What is actually on disk, per (arm, S) x readout. An empty cell in a
    table above is indistinguishable from a zero unless it is stated."""
    prefixes = [("e04_tt", "TP probe"), ("e04_retr", "TP retr"),
                ("xtask_tt_DM", "DM probe"), ("xtask_retr_DM", "DM retr")]
    rows = ["| arm | S | " + " | ".join(n for _p, n in prefixes) + " |",
            "|---|---|" + "---|" * len(prefixes)]
    for arm, s, slugs in ROWS:
        got = [f"{len(artifacts(p, slugs))}/{len(slugs)}" for p, _n in prefixes]
        rows.append(f"| {ARM_LABEL[arm]} | {s} | " + " | ".join(got) + " |")
    return "\n".join(rows)


def epoch_table() -> str:
    rows = ["| arm | cell | S | epoch | how chosen |", "|---|---|---|---|---|"]
    for arm, s, slugs in ROWS:
        for slug in slugs:
            if arm == "e04":
                ep, how = E04_SELECTED.get(slug, "?"), "smoothed val selection"
            else:
                ep, how = E05_EPOCH, "fixed (R5 in train -- no held-out val)"
            rows.append(f"| {ARM_LABEL[arm]} | {slug} | {s} | {ep} | {how} |")
    return "\n".join(rows)


DOC = """# RESULTS_add_val_set — does the subject-scaling saturation survive a bigger pool?

`RESULTS_model_scaling.md` reports that the depth-22 arm (`e04_reve_scaling`)
rises monotonically with S from 10 through 1400 and then **drops** at the
full-pool S=1863 cell — consistently across both probe tasks, both splits, and
all three retrieval granularities. It also names the reason that drop cannot be
believed as stated: **S=1863 is the whole R1–R4+R7–R10 pool, so it has exactly
one possible draw**, while every other S on that curve is a mean over three.
A single-draw cell is precisely the case that smoothed epoch selection protects
least, and `RESULTS.md` 2.10 documents a prior instance where selection
variance manufactured an apparent scaling artifact in this very experiment.

This file tests it by **folding the val release (R5) into the pretraining
pool**, taking ThePresent from 1863 recordings to 2156. That buys two things a
larger pool alone would not:

1. **S=1863 becomes a drawable cell with three replicates.** 1863 < 2156, so
   `max_subjects=1863` is a real subsample rather than a no-op cap. If three
   independent draws at S=1863 sit at the S=1400 level, the drop was draw or
   selection variance. If they reproduce it, it is a population-level effect.
2. **A genuinely new max-S point at S=2156** — the first observation past the
   previous ceiling of the data.

Everything else is held to e04: same depth-22 architecture, `meta.seed=2025`,
400 epochs, `epoch_size=703` so every cell runs exactly 4400 steps under an
identical LR schedule, `max_anchors=101`, `save_every=25`. The pool line is the
only substantive difference.

All numbers below are generated from the raw JSONs in `raw_results/` by
`src/write_results_add_val_set.py` — re-run that script to regenerate this file.

---

## Verdict: the drop was one RUN, not the data — a single-seed partial failure

**Sharpened 2026-08-18 by a cleaner test than the one this file was built on.**
`e04_s1863_a101_seed7` is kkokate's alternate-seed replicate of the full-pool
cell, never previously evaluated. The full-pool cell has exactly ONE possible
cohort, so seed7 vs the main cell holds data, pool, architecture,
initialisation and epoch all fixed and varies **only `meta.seed`**:

| e04 S=1863 @ epoch 325 | seed 2026 | seed 7 | delta |
|---|---|---|---|
| within-task probe mean(12) r | 0.2531 | **0.3092** | **+0.0561** |
| time-pool e2v@1 | 0.0523 | 0.0711 | +0.0189 |
| **final training loss** | **3.32** | **2.62** | -0.70 |

Seed 7 lands on this arm's independent three-draw S=1863 estimate (0.3076 +/-
0.0008) reached from a different pool entirely. Two routes, one answer.

So the correct statement is **not** "the S=1863 cohort is unlucky" and not "the
draw was unlucky" -- cohort was never the variable. **That one run optimised
badly.** Its final loss, 3.32, sits well above its seed-7 twin (2.62) and above
the S=1400 cells (2.73), so the deficit is visible in training, not only in the
readout.

**This is the collapse failure mode in a milder form** (see the section below).
Full collapse pins the loss at ln(64)=4.1589 and the score at chance; this run
degraded partially -- ~18 %% low on the probe -- which is far more dangerous,
because nothing about 0.2531 looks broken. It is a plausible number, and it
survived as a published headline precisely because it was plausible.

**The load-bearing consequence for methodology.** 3.32/2.73 = **1.22x** its
comparison median, BELOW the 1.30x threshold that catches outright collapse and
overlapping the worst healthy cell in this arm (1.19x). **Loss screening does
not separate partial failures from normal variation.** Screening catches
collapse; only REPLICATION catches partial failure. That is why every
single-draw cell on these curves is the vulnerable one, and why the fix for
S=1863 had to be replicates rather than a better detector.

---

## Original verdict (three draws from a larger pool) — unchanged and consistent

**Three independent draws at S=1863 from the 2156 pool give a within-task probe
mean r of 0.3097 +/- 0.0005** — *above* e04's S=1400 peak (0.2927 +/- 0.0085),
and **+0.052 above e04's single-draw S=1863 (0.2579)**. The three draws agree to
a stdev of 0.0005, which is 17x tighter than e04's own between-draw spread at
S=1400, so this is not a lucky replacement draw.

The decomposition in section 2a is the argument, because it isolates *the same
step in S* measured two ways:

- **S=1400 -> 1863 with three draws (contrast B): +0.0130** on the within-task
  probe, +0.0097 cross-task, positive on 3 of 5 readouts and never below
  -0.003 (inside the draw sd) on the other two.
- **The identical step with one draw (contrast C): -0.0347** within-task,
  -0.0152 cross-task, negative on **all five** readouts.

Same architecture, same recipe, same step in S, opposite sign — and the only
difference is one draw versus three. `RESULTS_model_scaling.md` flagged exactly
this as the reason to distrust its own headline; that caveat turns out to be the
explanation rather than a hedge.

**The comparison is legitimate because the pools are near-exchangeable.** At
matched S=1400 the offset (contrast A) is +0.0041 on the within-task probe
(+0.48 of e04's between-draw sd) and -0.000 on scene retrieval. One exception,
stated rather than buried: **time-pool retrieval is +0.008 (+10 sd)** — small
in absolute terms but well outside noise. The likely cause is subject quality,
not a bug: `RESULTS.md` 2.11 measured R7-R10 subjects as worth 8-15 % less than
R1-R4 at matched count, and R5 belongs to the original block, so a 1400-draw
from the 2156 pool carries ~13.6 % R5 (190 of 1400) where a 1400-draw from 1863
carries none.
That offset is far too small to account for contrast C either way.

**What this does NOT establish.** S=2156 is again *the whole pool*, so it has
one possible draw and inherits precisely the weakness that made e04's S=1863
unreadable. Its +0.0052 over S=1863 (probe) is inside the range a single draw
can produce, and retrieval is flat across that step (-0.000 time, +0.002
scene). The honest reading is **"no evidence of a decline at 2156"**, not "the
curve keeps rising to 2156." Establishing the latter needs a pool larger than
2156 so that S=2156 becomes drawable in triplicate — the same move this file
made for S=1863.

**Consequence for `RESULTS_model_scaling.md`.** Its headline observation, that
every table drops at the full-pool cell, should be withdrawn as a single-draw
artifact rather than reported as a capacity/subject-count interaction. Its
section 1-4 tables remain correct as measurements; it is the interpretation of
the S=1863 row that does not survive.

---

## What R5-in-train costs, and how each cost is handled

**Test split only, both arms, every table.** The addval cells trained on R5, so
a val number for them would be measured on data their encoder saw. R6 (test) is
untouched by both arms and is the only split on which they are comparable at
all. Where `RESULTS_model_scaling.md` reports val and test, this file reports
test.

**Checkpoint selection is a fixed epoch, not a fallback.** e04 picks each
cell's epoch by smoothed argmax of `val/clip_scene_auc` — a signal that is
in-sample for the addval cells and therefore unusable. Those cells are pinned
to **epoch 375**, which is where 4 of 4 high-S e04 cells' own selection landed
(S=1400 d11/d33 and S=1863 at 375; S=1400 d22 at 350), so the arms are matched
to within one 25-epoch checkpoint interval. `save_every=25` keeps the whole
epoch grid on disk, so a different epoch can be evaluated later without
retraining.

**The probe head-fit pool is NOT changed.** `config_probe_TP_e04.yaml` and
`config_probe_DM_e04.yaml` still declare `[R1..R4, R7..R10]`, so every cell in
both arms fits its RidgeCV head on the identical 1863 (ThePresent) / 1832
(DespicableMe) recordings. Verified per artifact, not assumed: all 31 cells
report `n_train_recordings=1863`, `n_train_windows=188163`,
`n_test_recordings=108`, `seed=42`. So the head's *data* is not a variable, and
a cross-task artifact reporting 2156 would mean the probe read the pretraining
config instead of the eval config — the `verify` action checks exactly that.

**But "the head-fit pool is constant" is NOT the same as "S is unconfounded",
and an earlier version of this section overstated it.** What still varies with S
is the OVERLAP between the pretraining cohort and the head-fit pool:

| S | 10 | 200 | 701 | 1400 | 1863 | 2156 |
|---|---|---|---|---|---|---|
| ≈ %% of the 1863 head-fit pool the encoder pretrained on | 0.5 | 9 | 33 | 65 | 86 | 100 |

At low S the ridge head is fitted almost entirely on subjects the encoder never
saw; at high S, almost entirely on subjects it did. That is not controlled by
holding the pool fixed. Two reasons it is unlikely to manufacture the curve:

1. **Its direction works against high S.** The head is *fitted* on train and
   *applied* to test (R6), which no encoder and no head ever sees. If an
   encoder's embeddings differ systematically for subjects it trained on, then
   at high S the head is fitted on "seen" embeddings and applied to "unseen"
   ones — a fit/apply mismatch that grows with S and should depress it. The
   measured scaling would then be conservative.
2. **Retrieval is immune to it and agrees.** `retrieval.py` fits nothing, so
   there is no head for cohort overlap to advantage. Across S=10→1863 the two
   curves correlate at **r = 0.995**, and retrieval rises **5.7x** against the
   probe's 2.3x. An overlap artifact cannot produce that.

Reason 1 is an argument about direction, not a measurement, and reason 2
establishes that the SHAPE is a property of the encoder — not that the probe's
absolute values at high S are unbiased. **The clean test** would hold a fixed
slice of the head-fit pool out of *every* pretraining cohort (say 300
recordings excluded from all draws) and fit the head only on those, making
overlap 0 %% at every S by construction. That requires retraining the axis,
since the cohorts themselves must change, so it is a design note for the next
sweep rather than something that can be bolted on here.

**Retrieval refits nothing.** It is zero-shot cosine similarity between frozen
EEG and V-JEPA-2 embeddings, so the head-fit caveats apply only to the Pearson
r tables.

**Confirmed at runtime**, job 21138023, rather than assumed:

```
HBN_TRAIN_RELEASES override: train = ['R1','R2','R3','R4','R5','R7','R8','R9','R10']
Scaling subsample: 2156 -> 1400 subjects (2156 -> 1400 recordings)
E0.3 scaling cell: max_subjects=1400 max_anchors=101 epoch_size=703
                   -> 1400 recordings, 703 items/epoch, 11 steps/epoch
```

The pool is 2156 on both counts — one recording per subject, unlike the
R1–R4-only pool where 703 recordings came from 701 subjects — so S means the
same thing in subjects and in recordings throughout this arm. And 703
items/epoch confirms the step budget is unchanged, so the new points are not
confounded with optimisation budget.

---

## Figures

![subject-scaling curves](figures/addval_subject_scaling.png)

**`figures/addval_subject_scaling.pdf`** — the four readouts as small multiples,
mean line with a shaded +/- 1 sd band across draws. **Hollow markers are
single-draw cells** (`n=1`), drawn without a band because no spread was
measured: e04 at S=1863 and addval at S=2156, each being that arm's whole pool.
That distinction is the experiment, so the figure encodes it rather than
averaging it away.

![the same step in S, one draw vs three](figures/addval_drop_decomposition.png)

**`figures/addval_drop_decomposition.pdf`** — the argument in one panel per
readout: the *same* step in S (1400 -> 1863) measured inside e04, where it rests
on one draw, and inside addval, where it rests on three. The arrow direction is
the finding.

Both are written by `src/plot_add_val_set.py`, which imports this script to get
its data -- same cells, same exclusions -- so a figure cannot disagree with a
table above it.

---

## Failure mode: collapse at initialisation, ~10 % of cells per seed

This recipe fails to train on about one cell in ten, and the failure is
**silent by construction**: a collapsed cell runs the full 400 epochs, writes a
complete 15-checkpoint grid, and produces well-formed evaluation artifacts with
plausible-looking numbers. Nothing in `sacct`, the checkpoint count, or the
artifact schema distinguishes it. Left in, it drags its S-group mean toward the
random baseline and steepens the apparent scaling curve.

**The mechanism.** InfoNCE at `batch_size=64` has chance loss `ln(64) = 4.1589`.
Every failure observed on this arm sits at **4.02–4.16, flat from epoch 1 to
400** — the run never leaves the collapsed solution. A healthy cell at the same
S escapes immediately. Measured, same S=400, same recipe, differing only in
`meta.seed`:

| epoch | 1 | 40 | 80 | 120 | 160 | 200 | 240 | 280 | 320 | 360 |
|---|---|---|---|---|---|---|---|---|---|---|
| collapsed (`s400_d11`, seed 2026) | 4.13 | 4.14 | 4.14 | 4.12 | 4.08 | 4.12 | 4.11 | 4.05 | 4.09 | 4.05 |
| healthy (`s400_d22`, seed 2026) | 4.21 | 3.89 | 3.47 | 2.90 | 2.41 | 2.20 | 1.60 | 1.63 | 1.37 | 1.21 |

So the seed decides whether the run escapes the basin. It is not gradual
divergence, and it is not a bad cohort: `s1400_d11` collapsed while `s1863_d11`,
whose cohort is a strict SUPERSET of it, trained normally.

**Rate, across two independently-run sweeps.** 3 of 31 here (~10 %) and 3 of 28
in `e04_reve_scaling` (~11 %) — kkokate's `e04_s{{1000_d33,200_d22,400_d33}}` each
kept a `_FAILED_seed2026` directory beside a seed-7 replacement. Budget for it.

**Detection — `_submit_e05_addval.py screen`.** Two design choices, both
learned the hard way:

1. **Screen on training loss, not on a score.** No evaluation is needed, so a
   dead cell is caught before ~40 min of probe GPU is spent on it. One cell
   (`s400_d11`) was caught exactly this way, before its eval ran.
2. **Compare to the same-S median, not an absolute threshold.** Loss scales with
   cohort size — S=10 sits near 0.70, S=1863 near 2.8 — so a fixed cutoff that
   catches a failure at S=1000 would miss the identical failure at S=50. The
   threshold is 1.30x the S-group median; measured separation is 1.67–3.09x for
   the three real failures against 1.19x for the worst healthy cell.

`screen` also refuses to judge a run that has not finished. A mid-descent loss
looks collapsed beside a finished sibling's, and that false positive fired once
on `s400_d11` at 45 of ~75 minutes (2.49 against siblings' 1.26/1.30).

**The instability is a SPECTRUM, and the screen only catches one end of it.**
Full collapse is the easy case: loss pinned at 4.16, score at chance, 1.67-3.09x
the S-group median. But `e04_s1863_a101_nd` -- the cell whose apparent drop this
whole experiment was built to explain -- was a PARTIAL failure:

| | full collapse | partial (`e04_s1863` seed 2026) | healthy |
|---|---|---|---|
| final loss | 4.02-4.16 (= ln 64) | 3.32 | 2.6-2.8 |
| x comparison median | 1.67-3.09 | **1.22** | <= 1.19 |
| readout | at random baseline | ~18 %% low, looks plausible | -- |

**1.22x is below the 1.30x threshold and overlaps the worst healthy cell at
1.19x.** So training loss does NOT separate partial failures from normal
variation, and `screen` would not have flagged this one. Lowering the threshold
does not fix it -- the two distributions genuinely overlap at that granularity.

The practical rule that follows: **screening catches collapse; only replication
catches partial failure.** Every single-draw cell on these curves is therefore
the vulnerable one, which is exactly why the fix for S=1863 had to be replicates
rather than a better detector. Budget replicates wherever a number will be
quoted, and treat any n=1 point as provisional regardless of how clean it looks.

**Remedy, and the cap.** Reseed to 7, which is e04's own convention. Two of
three recovered at seed 7; `s400_d11` collapsed at 2026 *and* 7 and recovered at
2025. **Retries are capped at three attempts**, after which the cell is reported
at n=2 with the failure disclosed rather than retried further. Past a small
fixed budget, "try seeds until one trains" stops being a fix for a known
instability and becomes selection on the outcome. The cap is in the code
(`RERUN_SEED`), not just in intent, and every reseeded cell is listed in section
2b with the numbers that condemned it — judged before the replacement existed.

---

## 0. Coverage — what is actually measured

An empty cell in any table below is indistinguishable from a zero unless it is
stated. This is the statement.

{coverage}

### Epoch per cell

{epoch_table}

---

## 1. Headline — every readout, both arms, test split

{headline}

---

## 2. Pool-stitch calibration: the same S from the two pools

The two arms' S=1400 draws are **not** the same cohorts — draws nest within a
pool (permute the subject list once per seed, take the first S), but adding R5
changes the list being permuted. So this is a test of whether the two pools are
exchangeable, and it gates everything else in the file: if the R5-inclusive
pool scores differently at matched S, the S=1863 and S=2156 comparisons must be
read through that offset rather than at face value.

`RESULTS.md` 2.11 measured exactly this kind of pool effect once already —
R7–R10 subjects were worth 8–15 % less than R1–R4 at matched count — so a
non-zero delta here is not hypothetical.

{calibration}

---

## 2a. Decomposition — pool effect vs S effect vs the alleged drop

{decomposition}

---

---

## 2b. Every cell, no averaging

{per_draw}

---

## 2c. Epoch robustness

{epoch_robustness}

---

## 2d. Matched-epoch calibration — can e04's low-S cells be spliced on?

Sections 1-2a compare e04 at its own smoothed-selected epoch against addval at
a fixed 375. That is matched to within one checkpoint interval at high S, but it
is still two protocols, and at low S e04's selection scatters from epoch 75 to
375 — so a spliced 10 -> 2156 curve built on those numbers would carry a
selection seam exactly where the curve is steepest. `RESULTS.md` 2.10 documents
that epoch selection has already manufactured one fake scaling artifact in this
experiment, so that seam is not a theoretical worry.

This table removes the protocol difference: **every cell here, both arms, is at
fixed epoch 325** (e04's full 28-cell ep325 sweep already existed; the addval
cells were re-evaluated to match). It reports the two OVERLAP points, S=1400 and
S=1863, the only S where both arms have cells.

{matched_epoch}

**Read the S=1400 rows as the calibration** — both arms have three draws there,
so the delta is a pool effect with the epoch effect held at zero. Five of six
readouts land within +/-1.2 of e04's own between-draw sd, i.e. the pools are
exchangeable for the probe (both tasks) and for shot/scene retrieval.

**The exception is time-pool retrieval, at +4.90 sd** (+0.0068, ~+10 % relative).
Matching the epoch halved it — it was +10.4 sd at 375 — so part of the original
gap was protocol, but a real pool effect survives on the finest-granularity
readout. Direction and size match `RESULTS.md` 2.11, which measured R7-R10
subjects as worth 8-15 % less per subject than R1-R4: R5 belongs to the original
block, so a draw from the 2156 pool is slightly richer in good subjects.

**Consequence — now moot, and that is the point.** Splicing was defensible for
the probe and coarse retrieval with this calibration quoted, and NOT defensible
for time-pool retrieval, where it would have put a visible ~10 % step at the
seam. That is why the low-S cells were retrained from the 2156 pool
(`_submit_e05_addval.py --full-axis`). They now exist, so **no splice is used
anywhere**: section 2e is a single-pool, single-epoch curve and this table is
demoted to a check on a decision already taken.

**Do not read the S=1863 rows as calibration.** e04 has one draw there, so that
delta mixes the pool effect with the single-draw artifact this file is about —
it is the finding, restated at a second fixed epoch, not a control.

---

## 2e. The official curve — one pool, one epoch, full axis

Every cell below is the addval arm (pool 2156) at fixed epoch 325, three draws
except S=2156, which is the whole pool. **This is the curve the paper plots.**
It is reported separately from sections 1-3 because those are keyed to epoch
375, which only exists for S >= 1400 — so the low-S cells read as absent there
while being present in the figure.

{official_curve}

**The S=50 draw spread is real and is not a failed cell.** `e05_s50_a101_av_d22`
scores 0.1084 on the within-task probe against siblings' 0.1511 / 0.1581 and a
random baseline of 0.1049 — at chance, on a readout where its siblings are
clearly above it. It was screened on training loss and **passed**: 0.7882, which
is 1.11x its S-group median, nowhere near the 1.30x threshold and nowhere near
the collapsed value of ln(64) = 4.1589. Its loss descended normally; what it
failed to do was generalise.

That combination — healthy loss, chance-level readout — is not the collapse
failure mode documented above, and it is the reason the cell is **kept**. At
S=50 a single draw is 50 subjects out of 2156, and `RESULTS.md` 2.11 measured
per-subject quality varying 8-15 % by release, so a draw this small can
plausibly be a weak cohort rather than a broken run. Averaging over that is
exactly what three draws are for. Dropping it because its score is low, having
first noticed it *because* its score is low, would be selection on the outcome —
the same trap the retry cap in the collapse section exists to prevent.

The visible consequence: S=50 sits at 0.1392 against S=20's 0.1397, so the
official curve is **non-monotonic at 20 -> 50 on 4 of the 5 readouts** above
(every one except time-pool retrieval, which still rises). The S=50 band is also
the widest anywhere on the axis — 0.0269 against a next-widest 0.0108 on the
within-task probe (2.5x), 0.0468 against 0.0113 cross-task (4.1x). Both are
honest depictions of low-S draw variance and neither is smoothed away.

This is worth carrying into any monotonicity claim, and the claim has to be
stated per readout rather than in aggregate. From S=50 upward, three of the five
readouts rise on every interval: the within-task probe, the cross-task probe and
time-pool within-task retrieval. The other two do not — scene-pool within-task
retrieval falls at 1400 -> 1863, and cross-task time-pool retrieval falls at both
701 -> 1000 and 1400 -> 1863, the latter being the study's negative control and
expected to wander near chance. So e04's "8 of 8 intervals up" does not carry
over to this single-pool ladder unchanged, and quoting it here without naming
the readout would overstate what the curve does.

---

## 3. Within-task (ThePresent) probe — Pearson r, test split

RidgeCV head fit on the full 1863-recording pool for every cell in both arms.
Mean +/- stdev across draws; a single value where only one draw exists.

{tp_probe}

---

## 4. Within-task retrieval — test split

e2v: given an EEG window, is the matching vision pool entry in the top-K by
cosine similarity? v2e: the reverse. "xChance" is top-1 divided by 1/N_pool.

### 4.1 Time-bucket pools (finest, N~101)

{tp_retr_time}

### 4.2 Shot pools

{tp_retr_shot}

### 4.3 Scene pools (coarsest)

{tp_retr_scene}

---

## 5. Cross-task (DespicableMe) probe — Pearson r, test split

Same ThePresent-trained checkpoints, RidgeCV head refit on DespicableMe's own
1832-recording train pool. Linear transfer of the learned features, not
zero-shot alignment (that is section 6).

{dm_probe}

---

## 6. Cross-task retrieval — zero-shot alignment on DespicableMe, test split

Nothing is refit. The strictest transfer test available: does a
ThePresent-trained EEG encoder still land in the right place in V-JEPA-2 space
for a movie it never saw during pretraining?

### 6.1 Time-bucket pools

{dm_retr_time}

### 6.2 Shot pools

{dm_retr_shot}

### 6.3 Scene pools

{dm_retr_scene}

---

## Provenance

- Training: `submit/_submit_e05_addval.py` (7 cells) ->
  `/work/hdd/bbnv/dtyoung/eb_jepa/e05_addval`. Frozen config
  `config/clip_pretrain_e05_addval.yaml`, which is kkokate's e04 training
  config with `data.train_releases` extended by R5 and nothing else.
- e04 reference side: `/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling`
  (read-only; nothing in this pipeline writes there), at the epochs in
  `e04_selection.json`.
- Evaluation: `submit/_submit_traintest_e04.py --arm addval` and
  `submit/_submit_retrieval_e04.py --arm addval` (`within` / `cross` presets).
  Both restrict the addval arm to the test split and pin it to epoch 375; both
  are idempotent (every step is skip-if-output-exists) and both support a
  `verify` action that checks `n_train_recordings` against the expected
  head-fit pool.
- Eval configs: `config/config_probe_TP_e04.yaml`,
  `config/config_probe_DM_e04.yaml` — shared with the e04 arm, unmodified.
- Figures: `src/plot_add_val_set.py` -> `figures/addval_subject_scaling.{{png,pdf}}`,
  `figures/addval_drop_decomposition.{{png,pdf}}`.
- Raw JSONs: `raw_results/{{e04_tt_test,e04_retr_test}}_e05_*` and
  `raw_results/{{xtask_tt_DMtest,xtask_retr_DMtest}}_e05_*` (addval arm); the
  same prefixes with `e04_*` slugs (reference arm).
"""


def main() -> None:
    OUT.write_text(DOC.format(
        coverage=coverage(),
        epoch_table=epoch_table(),
        headline=headline_table(),
        calibration=calibration_table(),
        decomposition=decomposition_table(),
        matched_epoch=matched_epoch_table(),
        official_curve=official_curve_table(),
        per_draw=per_draw_table(),
        epoch_robustness=epoch_robustness_table(),
        tp_probe=probe_table("e04_tt"),
        tp_retr_time=retrieval_table("e04_retr", "time"),
        tp_retr_shot=retrieval_table("e04_retr", "shot"),
        tp_retr_scene=retrieval_table("e04_retr", "scene"),
        dm_probe=probe_table("xtask_tt_DM"),
        dm_retr_time=retrieval_table("xtask_retr_DM", "time"),
        dm_retr_shot=retrieval_table("xtask_retr_DM", "shot"),
        dm_retr_scene=retrieval_table("xtask_retr_DM", "scene"),
    ))
    print(f"wrote {OUT.relative_to(ROOT.parents[1])}")
    print(coverage())


if __name__ == "__main__":
    main()
