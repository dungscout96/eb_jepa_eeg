"""Nested-draw aggregator for the E0.3 / E0.4 subject-scaling sweeps.

``analyse_e03.py`` keys cells by ``(S, A)``, so the three replicate draws at one
S overwrite each other, and its S axis is a hardcoded list. This script keys by
``(S, draw_seed)`` instead, discovers cells by glob, and serves both the ``e03_``
(depth-12) and ``e04_`` (full REVE) prefixes so the two sweeps can be read side
by side. It is what produced RESULTS.md 2.11/2.12, made reproducible.

Three readouts of the same selected checkpoints are aggregated per S:

  1. **Δr²** -- mean r² over the 12 movie features from the within-split CV
     probe on val, minus a random-init null. Raw r² carries a large constant
     that any encoder attains and a constant offset flattens a log-log slope, so
     exponents fitted on raw r² understate both axes (see ``analyse_e03.py``).
  2. **Pearson r** from ``probe_traintest`` -- head fitted on the full train
     pool, evaluated on val and on test. Different protocol, different data for
     the head, so it is an independent check on (1).
  3. **scene-level e→v retrieval top-1/5/10 on TEST.** Never val: the random
     null beats chance at scene level on val while sitting at/below chance on
     test, so val retrieval measures pool structure rather than learning
     (RESULTS.md 2.12c). This script does not read ``*_retr_val_*.json`` at all.

The null is a required argument for every readout, never a default constant.
Δr² measured against a null of different model shape is silently wrong -- it is
a plausible number computed from the wrong subtrahend -- and the E0.3 and E0.4
encoders share ``encoder_dim=512`` while differing in depth, patch size and
init, so the recorded ``encoder_dim`` alone CANNOT tell them apart. The filename
prefix is therefore enforced as well, and the checks below are necessary rather
than sufficient: pairing the right null with the right sweep stays the caller's
responsibility.

Step exponents are ``log(y1/y0) / log(x1/x0)`` between adjacent S, each reported
in units of the pooled between-draw sd so that a step smaller than the cohort
noise is visible as such.

    uv run --group eeg python experiments/snr_scaling/aggregate_nested.py \\
        --cells-glob 'e03_s*_a101_nd*' --prefix=e03 \\
        --null-probe=e03_probe_val_random.json \\
        --null-tt-val=e03_tt_val_random.json \\
        --null-tt-test=e03_tt_test_random.json \\
        --null-retr-test=e03_retr_test_random.json \\
        --ceiling-val=isc_val_ThePresent.json \\
        --ceiling-test=isc_test_ThePresent.json \\
        --extra-step=701:1863 --full-pool-s=1863 \\
        --output=raw_results/e03_nested.json
"""

import argparse
import fnmatch
import glob
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw_results"

# Readout kind -> the filename infix it is stored under. Order is report order.
READOUTS = ("probe_val", "tt_val", "tt_test", "retr_test")
READOUT_LABELS = {
    "probe_val": "CV probe (val)",
    "tt_val": "train→test (val)",
    "tt_test": "train→test (test)",
    "retr_test": "scene retr (test)",
}

RETR_LEVEL = "scene"
RETR_TOPKS = ("1", "5", "10")

# The study token is everything before the first "_s<digits>": "e03", "e04",
# and the arm-carrying prefixes that came later ("e12cb" / "e12cbr" for the
# CBraMod warm / random arms). It is opaque here -- the arm is chosen by the
# caller's --prefix and --cells-glob, never inferred from this token.
SLUG_RE = re.compile(
    r"^(?P<study>e\d[^_]*)_s(?P<s>\d+)_a(?P<a>\d+)"
    r"(?P<tag>(?:_[A-Za-z][A-Za-z0-9]*)*?)(?:_d(?P<draw>\d+))?$"
)

# Ceiling = Spearman-Brown CC_max at K=1 from the combined 5-component CorrCA
# reliability; RESULTS.md 2.1 argues this row rather than a single component.
CEILING_PATH = ("spearman_brown", "corrca_held_out_combined", "ceiling_by_k", "1")


def _resolve(name: str, results_dir: Path) -> Path:
    """Accept a bare filename (looked up beside the cells) or any path."""
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return results_dir / p.name


def _load(path: Path, what: str) -> dict:
    if not path.exists():
        sys.exit(
            f"{what} not found: {path}\n"
            "A shape-matched null is REQUIRED. Δr² against a null probed from a "
            "different encoder shape is not a smaller number, it is a wrong "
            "one; there is no default to fall back to. Probe a random-init "
            "encoder at this sweep's exact geometry and pass it here."
        )
    return json.loads(path.read_text())


def _study_prefix(value: str) -> str:
    """argparse type for --prefix: the study token SLUG_RE would extract.

    Replaces the former fixed ``choices=["e03", "e04"]``: the nulls are still
    matched to the arm by filename prefix, so the value must be a whole study
    token (no underscore) or the prefix check on the nulls becomes meaningless.
    """
    if not re.fullmatch(r"e\d[^_]*", value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a study prefix (expected e.g. e03, e04, e12cb)"
        )
    return value


def parse_slug(slug: str) -> dict | None:
    """``e03_s1863_a101_nd`` and ``e04_s400_a101_nd_d11`` -> S, A, tag, draw.

    The whole-pool cell carries no ``_d`` suffix because only one draw of the
    whole pool exists; ``draw`` is None there rather than a fabricated seed.
    """
    m = SLUG_RE.fullmatch(slug)
    if m is None:
        return None
    return {
        "study": m.group("study"),
        "S": int(m.group("s")),
        "A": int(m.group("a")),
        "tag": m.group("tag").lstrip("_"),
        "draw": int(m.group("draw")) if m.group("draw") else None,
    }


def discover(results_dir: Path, prefix: str, cells_glob: str) -> dict:
    """Map slug -> {readout: path} for every cell matching *cells_glob*.

    Discovery is by glob over the result files, never an enumerated cell list: a
    list drifts from the launcher and a cell that failed to produce a file
    disappears instead of showing up as a missing row.
    """
    found: dict[str, dict[str, Path]] = {}
    for kind in READOUTS:
        pat = re.compile(rf"{re.escape(prefix)}_{kind}_(.+?)(?:_best)?\.json")
        for f in sorted(glob.glob(str(results_dir / f"{prefix}_{kind}_*.json"))):
            m = pat.fullmatch(Path(f).name)
            if m is None:
                continue
            slug = m.group(1)
            if not fnmatch.fnmatch(slug, cells_glob):
                continue
            if parse_slug(slug) is None:
                print(f"warning: unparseable slug {slug!r} in {f}", file=sys.stderr)
                continue
            prev = found.setdefault(slug, {}).get(kind)
            if prev is not None:
                sys.exit(
                    f"two files claim readout {kind!r} for cell {slug!r}:\n"
                    f"  {prev}\n  {f}\n"
                    "Protocols would be mixed. Move or rename one."
                )
            found[slug][kind] = Path(f)
    return found


def mean_r2(d: dict) -> float:
    return st.fmean(v["r2_mean"] for v in d["features"].values())


def mean_pearson_r(d: dict) -> float:
    return st.fmean(v["pearson_r"] for v in d["features"].values())


def scene_topk(d: dict) -> dict:
    lv = d["levels"][RETR_LEVEL]
    return {
        "top_k": {k: lv["e2v_top_k"][k] for k in RETR_TOPKS},
        "chance": {k: lv["e2v_chance"][k] for k in RETR_TOPKS},
        "relative": {k: lv["e2v_relative"][k] for k in RETR_TOPKS},
        "pool_N": lv["n_vision_pool_N"],
    }


SAME_DATA_HINT = (
    "The cells were not evaluated on the same data, so the S curve is not a "
    "curve. Re-run the mismatched cells before aggregating."
)
SAME_PROTOCOL_HINT = (
    "Cells whose slugs carry different tags came from different protocols or "
    "different subject pools (E0.3: '' = fixed budget, 'es' = early stopped, "
    "'ext' = extended pool, 'nd' = nested draws). They are all probed on the "
    "same val recordings, so no shape check can separate them -- averaging "
    "them per S would report a cross-protocol mean as a between-draw mean. "
    "Narrow --cells-glob to one protocol."
)


def _require_same(
    name: str, values: dict, what: str, hint: str = SAME_DATA_HINT
) -> object:
    uniq = set(values.values())
    if len(uniq) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(values.items()))
        sys.exit(f"{what} differ in {name}: {detail}\n{hint}")
    return uniq.pop() if uniq else None


def loglog_slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares exponent b in y ~ x^b. None if any y is non-positive."""
    if len(xs) < 2 or any(y <= 0 for y in ys):
        return None
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = st.fmean(lx), st.fmean(ly)
    den = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den else None


def local_slope(x0: float, y0: float, x1: float, y1: float) -> float | None:
    if y0 <= 0 or y1 <= 0 or x0 <= 0 or x1 <= 0 or x0 == x1:
        return None
    return math.log(y1 / y0) / math.log(x1 / x0)


def pooled_sd(groups: dict) -> float | None:
    """Between-draw sd pooled over the S with >= 2 draws.

    sqrt(Σ(n-1)s² / Σ(n-1)) -- the usual pooled variance, which reduces to the
    RMS of the per-S sample sds when every S has the same number of draws.
    """
    num = den = 0.0
    for g in groups.values():
        vals = g["values"]
        if len(vals) >= 2:
            num += (len(vals) - 1) * st.variance(vals)
            den += len(vals) - 1
    return math.sqrt(num / den) if den else None


def group_by_s(cells: dict, key: str) -> dict:
    """Per-S mean/sd/n and the individual draw values for one scalar readout."""
    groups: dict[int, dict] = {}
    for slug, c in cells.items():
        if c.get(key) is None:
            continue
        g = groups.setdefault(c["S"], {"values": [], "draws": [], "slugs": []})
        g["values"].append(c[key])
        g["draws"].append(c["draw"])
        g["slugs"].append(slug)
    for g in groups.values():
        order = sorted(
            range(len(g["values"])),
            key=lambda i: (g["draws"][i] is None, g["draws"][i]),
        )
        for f in ("values", "draws", "slugs"):
            g[f] = [g[f][i] for i in order]
        g["n"] = len(g["values"])
        g["mean"] = st.fmean(g["values"])
        g["sd"] = st.stdev(g["values"]) if g["n"] >= 2 else None
    return dict(sorted(groups.items()))


def fmt(x: float | None, nd: int = 5) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def cell_mean(groups: dict, s: int, nd: int) -> str:
    """A per-S mean, or a MISSING marker -- never a blank that reads as covered."""
    g = groups.get(s)
    return f"{g['mean']:.{nd}f}" if g else "**MISSING**"


def pct_change(y0: float, y1: float) -> float | None:
    """Percent change, defined only for a strictly positive baseline.

    With y0 <= 0 the ratio flips sign: Δr² going -0.0005 -> +0.046 is a large
    improvement that ``100*(y1-y0)/y0`` renders as -9329 %. A cell at or below
    the null is entirely possible (a from-scratch arm at small S), so this
    returns None rather than a plausible number with the wrong sign.
    """
    return 100.0 * (y1 - y0) / y0 if y0 > 0 else None


def cell_pct(groups: dict, s0: int, s1: int) -> str:
    g0, g1 = groups.get(s0), groups.get(s1)
    if not g0 or not g1:
        return "—"
    p = pct_change(g0["mean"], g1["mean"])
    return "n/a (base ≤ 0)" if p is None else f"{p:+.1f} %"


def fmt_signed(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:+.{nd}f}"


def draws_cell(g: dict, s: int, full_pool_s: int | None) -> str:
    """The individual draw values, labelled by their ``_d`` seed.

    A slug with no ``_d`` seed means only that the launcher did not vary
    ``data.subsample_seed`` for it -- NOT that only one draw of that S exists.
    The whole-pool claim is therefore made only for the S the caller declares
    via ``--full-pool-s``, never inferred from the filename.
    """
    if g["n"] == 1 and g["draws"][0] is None:
        if full_pool_s is not None and s == full_pool_s:
            return "whole pool, only one draw exists"
        return "1 draw; the slug carries no _d seed"
    return ", ".join(
        f"{'d%d' % d if d is not None else 'no _d'} {v:.5f}"
        for d, v in zip(g["draws"], g["values"])
    )


def steps(
    groups: dict, sd: float | None, extra: list[tuple[int, int]]
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Adjacent-S steps, then any explicitly requested wide steps.

    Returns the rows plus the requested steps that could not be computed, so a
    step whose endpoint has no readout shows up as MISSING instead of vanishing
    from the table that the section's headline turns on.
    """
    axis = sorted(groups)
    adjacent = list(zip(axis, axis[1:]))
    pairs, absent = list(adjacent), []
    for p in extra:
        if p in pairs:
            continue
        (pairs if p[0] in groups and p[1] in groups else absent).append(p)
    out = []
    for s0, s1 in pairs:
        y0, y1 = groups[s0]["mean"], groups[s1]["mean"]
        out.append(
            {
                "from": s0,
                "to": s1,
                "delta": y1 - y0,
                "in_sd": (y1 - y0) / sd if sd else None,
                "exponent": local_slope(s0, y0, s1, y1),
                "pct": pct_change(y0, y1),
                "adjacent": (s0, s1) in adjacent,
            }
        )
    return out, absent


def load_ceiling(
    path: Path | None, split: str, task: str, n_rec: int | None
) -> float | None:
    """CC_max at K=1 for *split*, refusing to hand back another split's ceiling."""
    if path is None:
        print(
            f"warning: no --ceiling-{split}; r/ceiling for {split} will be blank. "
            "A ratio against the other split's ceiling is not an approximation, "
            "it is a different quantity (RESULTS.md 2.2).",
            file=sys.stderr,
        )
        return None
    d = _load(path, f"--ceiling-{split}")
    if d.get("split") != split:
        sys.exit(f"{path} is split={d.get('split')!r}, needed {split!r}")
    if d.get("task") != task:
        sys.exit(f"{path} is task={d.get('task')!r}, needed {task!r}")
    if n_rec is not None and d.get("n_recordings") != n_rec:
        sys.exit(
            f"{path} was estimated on {d.get('n_recordings')} recordings but the "
            f"{split} cells were evaluated on {n_rec}. The ceiling is a property "
            "of the cohort it was measured on; these are different cohorts."
        )
    node = d
    for k in CEILING_PATH:
        node = node[k]
    return float(node)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cells-glob",
        default="e03_s*_a101_nd*",
        help="Slug glob, e.g. 'e04_s*_a101_nd*'.",
    )
    ap.add_argument(
        "--prefix",
        default="e03",
        type=_study_prefix,
        help="Study prefix of the result files and of the nulls, e.g. e03, "
        "e04, e12cb, e12cbr. Must be the study token of SLUG_RE.",
    )
    ap.add_argument("--results-dir", default=str(RAW_DIR))
    ap.add_argument("--task", default="ThePresent")
    ap.add_argument(
        "--null-probe",
        required=True,
        help="Random-init CV probe on val, IDENTICAL model shape.",
    )
    ap.add_argument("--null-tt-val", required=True)
    ap.add_argument("--null-tt-test", required=True)
    ap.add_argument("--null-retr-test", required=True)
    ap.add_argument(
        "--ceiling-val",
        default=None,
        help="isc_val_<task>.json, for the val r/ceiling column.",
    )
    ap.add_argument("--ceiling-test", default=None)
    ap.add_argument(
        "--extra-step",
        action="append",
        default=[],
        metavar="S0:S1",
        help="Additional non-adjacent step to report, e.g. 701:1863.",
    )
    ap.add_argument(
        "--full-pool-s",
        type=int,
        default=None,
        help="The S that IS the whole train pool (E0.3/E0.4: 1863). Only that "
        "S may be reported as having no other possible draw; absence of a _d "
        "seed elsewhere means the launcher did not replicate it, not that "
        "replicates do not exist.",
    )
    ap.add_argument("--output", default=None, help="Machine-readable JSON.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    extra = []
    for spec in args.extra_step:
        a, _, b = spec.partition(":")
        if not a.isdigit() or not b.isdigit():
            sys.exit(f"--extra-step must be S0:S1 with integers, got {spec!r}")
        extra.append((int(a), int(b)))

    found = discover(results_dir, args.prefix, args.cells_glob)
    if not found:
        sys.exit(
            f"No {args.prefix}_* result files matched slug glob "
            f"{args.cells_glob!r} under {results_dir}"
        )
    if glob.glob(str(results_dir / f"{args.prefix}_retr_val_*.json")):
        print(
            "note: *_retr_val_* files exist and are deliberately ignored; a "
            "random encoder beats chance on val scene retrieval (RESULTS.md 2.12c).",
            file=sys.stderr,
        )

    for name, path in (
        ("--null-probe", args.null_probe),
        ("--null-tt-val", args.null_tt_val),
        ("--null-tt-test", args.null_tt_test),
        ("--null-retr-test", args.null_retr_test),
    ):
        if not Path(path).name.startswith(f"{args.prefix}_"):
            sys.exit(
                f"{name}={path} does not start with {args.prefix!r}. E0.3 and E0.4 "
                "share encoder_dim=512, so a cross-study null cannot be detected "
                "from the file contents. If this null really was probed at this "
                f"sweep's geometry, name it {args.prefix}_*."
            )
    null_probe = _load(_resolve(args.null_probe, results_dir), "--null-probe")
    null_tt_val = _load(_resolve(args.null_tt_val, results_dir), "--null-tt-val")
    null_tt_test = _load(_resolve(args.null_tt_test, results_dir), "--null-tt-test")
    null_retr = _load(_resolve(args.null_retr_test, results_dir), "--null-retr-test")
    for name, d in (
        ("--null-probe", null_probe),
        ("--null-tt-val", null_tt_val),
        ("--null-tt-test", null_tt_test),
        ("--null-retr-test", null_retr),
    ):
        if not d.get("random_baseline"):
            sys.exit(
                f"{name} has random_baseline={d.get('random_baseline')!r}, needed true"
            )
    null_r2 = mean_r2(null_probe)
    null_retr_scene = scene_topk(null_retr)

    cells: dict[str, dict] = {}
    missing: list[dict] = []
    probe_rec, probe_win, probe_dim = {}, {}, {}
    tt_shape: dict[str, dict] = {"tt_val": {}, "tt_test": {}}
    retr_shape: dict[str, tuple] = {}

    def cell_order(slug: str) -> tuple:
        m = parse_slug(slug)
        return (m["S"], m["draw"] is None, m["draw"] or 0)

    for slug in sorted(found, key=cell_order):
        meta = parse_slug(slug)
        c = dict(meta)
        c["slug"] = slug
        paths = found[slug]
        for kind in READOUTS:
            if kind not in paths:
                missing.append({"slug": slug, "readout": kind})
                print(f"warning: {slug} has no {kind} readout", file=sys.stderr)
        if "probe_val" in paths:
            d = json.loads(paths["probe_val"].read_text())
            c["r2"] = mean_r2(d)
            c["d_r2"] = c["r2"] - null_r2
            probe_rec[slug] = d["n_recordings"]
            probe_win[slug] = d["n_windows_total"]
            probe_dim[slug] = d["encoder_dim"]
            if sorted(d["features"]) != sorted(null_probe["features"]):
                sys.exit(f"{slug} probes a different feature set than --null-probe")
        for kind in ("tt_val", "tt_test"):
            if kind in paths:
                d = json.loads(paths[kind].read_text())
                null_d = null_tt_val if kind == "tt_val" else null_tt_test
                if sorted(d["features"]) != sorted(null_d["features"]):
                    sys.exit(
                        f"{slug} {kind} covers a different feature set than "
                        f"--null-{kind.replace('_', '-')}. A mean r over a "
                        "different set of features is a different number, not "
                        "a noisier one."
                    )
                c[f"{kind}_r"] = mean_pearson_r(d)
                tt_shape[kind][slug] = (
                    d["n_train_recordings"],
                    d["n_test_recordings"],
                    d["n_test_windows"],
                    d["encoder_dim"],
                )
        if "retr_test" in paths:
            d = json.loads(paths["retr_test"].read_text())
            sc = scene_topk(d)
            c["retr"] = sc
            for k in RETR_TOPKS:
                c[f"retr_top{k}"] = sc["top_k"][k]
            retr_shape[slug] = (d["n_recordings"], sc["pool_N"])
        cells[slug] = c

    a_vals = {slug: c["A"] for slug, c in cells.items()}
    _require_same("A (anchors)", a_vals, "cells in the glob")
    _require_same(
        "protocol tag (the slug between _a{A} and the _d draw seed)",
        {slug: c["tag"] or "<none>" for slug, c in cells.items()},
        "cells in the glob",
        SAME_PROTOCOL_HINT,
    )
    n_rec = _require_same("n_recordings", probe_rec, "CV probe cells")
    n_win = _require_same("n_windows_total", probe_win, "CV probe cells")
    enc_dim = _require_same("encoder_dim", probe_dim, "CV probe cells")
    if enc_dim is not None and null_probe["encoder_dim"] != enc_dim:
        sys.exit(
            f"--null-probe encoder_dim={null_probe['encoder_dim']} but the cells "
            f"report {enc_dim}. Δr² against a mismatched-shape null is wrong, not "
            "noisy. Probe a random-init encoder at this sweep's geometry."
        )
    if n_rec is not None and (
        null_probe["n_recordings"],
        null_probe["n_windows_total"],
    ) != (n_rec, n_win):
        sys.exit(
            f"--null-probe was evaluated on {null_probe['n_recordings']}/"
            f"{null_probe['n_windows_total']} but the cells on {n_rec}/{n_win}."
        )
    for kind in ("tt_val", "tt_test"):
        shape = _require_same(
            "(n_train, n_test_rec, n_test_win, encoder_dim)",
            tt_shape[kind],
            f"{kind} cells",
        )
        null_d = null_tt_val if kind == "tt_val" else null_tt_test
        null_shape = (
            null_d["n_train_recordings"],
            null_d["n_test_recordings"],
            null_d["n_test_windows"],
            null_d["encoder_dim"],
        )
        if shape is not None and shape != null_shape:
            sys.exit(
                f"--null-{kind.replace('_', '-')} shape {null_shape} != cells {shape}"
            )
    retr = _require_same("(n_recordings, scene pool N)", retr_shape, "retrieval cells")
    null_retr_key = (null_retr["n_recordings"], null_retr_scene["pool_N"])
    if retr is not None and retr != null_retr_key:
        sys.exit(f"--null-retr-test shape {null_retr_key} != cells {retr}")

    g_d_r2 = group_by_s(cells, "d_r2")
    g_tt_val = group_by_s(cells, "tt_val_r")
    g_tt_test = group_by_s(cells, "tt_test_r")
    g_retr = {k: group_by_s(cells, f"retr_top{k}") for k in RETR_TOPKS}

    sd_pool = pooled_sd(g_d_r2)
    axis = sorted(g_d_r2)
    fit_s = loglog_slope([float(s) for s in axis], [g_d_r2[s]["mean"] for s in axis])
    step_rows, absent_steps = steps(g_d_r2, sd_pool, extra)
    for s0, s1 in absent_steps:
        gone = [s for s in (s0, s1) if s not in g_d_r2]
        print(
            f"warning: --extra-step={s0}:{s1} has no Δr² at S={gone}; "
            "reported as MISSING rather than dropped",
            file=sys.stderr,
        )

    ceil_val = load_ceiling(
        _resolve(args.ceiling_val, results_dir) if args.ceiling_val else None,
        "val",
        args.task,
        n_rec,
    )
    tt_test_rec = (
        next(iter(tt_shape["tt_test"].values()))[1] if tt_shape["tt_test"] else None
    )
    ceil_test = load_ceiling(
        _resolve(args.ceiling_test, results_dir) if args.ceiling_test else None,
        "test",
        args.task,
        tt_test_rec,
    )

    print(
        f"# {args.prefix.upper()} nested-draw aggregation — cells `{args.cells_glob}`\n"
    )
    print(f"- results dir: `{results_dir}`")
    print(f"- task: {args.task}; anchors A = {a_vals[next(iter(a_vals))]}")
    print(
        f"- CV probe null: `{Path(args.null_probe).name}`, mean r² = {null_r2:.5f} "
        f"(encoder_dim {null_probe['encoder_dim']})"
    )
    print(
        f"- every CV probe cell evaluated on {n_rec} val recordings / {n_win} windows"
    )
    print(
        "- `encoder_dim` is the only shape field the probe JSONs record; it does NOT "
        "separate E0.3 from E0.4. The null pairing is enforced by filename prefix and "
        "is ultimately the caller's responsibility.\n"
    )

    print("## Coverage\n")
    head = " | ".join(READOUT_LABELS[k] for k in READOUTS)
    print(f"| slug | S | draw | {head} |")
    print("|---|---:|:-:|" + ":-:|" * len(READOUTS))
    for slug, c in cells.items():
        marks = " | ".join(
            "yes" if k in found[slug] else "**MISSING**" for k in READOUTS
        )
        draw = f"d{c['draw']}" if c["draw"] is not None else "whole"
        print(f"| `{slug}` | {c['S']} | {draw} | {marks} |")
    print()
    if missing:
        print(
            f"{len(missing)} missing readout(s); every one is listed above as MISSING "
            "rather than dropped.\n"
        )

    print("## (1) Δr² above the shape-matched random-init null — CV probe, val\n")
    print("| S | n | mean Δr² | sd | draws |")
    print("|---:|:-:|---:|---:|---|")
    for s in axis:
        g = g_d_r2[s]
        cell = draws_cell(g, s, args.full_pool_s)
        print(f"| {s} | {g['n']} | {g['mean']:.5f} | {fmt(g['sd'])} | {cell} |")
    print()
    n_repl = sum(1 for g in g_d_r2.values() if g["n"] >= 2)
    if sd_pool is None:
        print(
            "Pooled between-draw sd = **—**: no S has ≥2 draws, so every `in sd` "
            "below is blank rather than assumed."
        )
    else:
        print(
            f"Pooled between-draw sd = **{fmt(sd_pool)}** "
            f"(pooled over the {n_repl} S with ≥2 draws)."
        )
    print(
        f"Fitted exponent d log Δr² / d log S = **{fmt_signed(fit_s)}** "
        "(least squares over all S; the local steps below matter more).\n"
    )

    print("## (2) Step exponents on Δr²\n")
    print("| step | Δ | in sd | exponent |")
    print("|---|---:|---:|---:|")
    for r in step_rows:
        label = f"{r['from']} → {r['to']}"
        if not r["adjacent"]:
            label = f"**{label}**"
        print(
            f"| {label} | {r['delta']:+.5f} | {fmt_signed(r['in_sd'], 2)} | "
            f"{fmt_signed(r['exponent'])} |"
        )
    for s0, s1 in absent_steps:
        print(f"| **{s0} → {s1}** | **MISSING** | — | — |")
    print(
        "\n`in sd` is Δ divided by the pooled between-draw sd: a step under 1 sd is "
        "within cohort noise.\n"
    )

    n_tt_feat = len(null_tt_val["features"])
    print(f"## (3) train→test Pearson r, mean over {n_tt_feat} features\n")
    print("| S | n | val *r* | test *r* | val *r*/ceiling | test *r*/ceiling |")
    print("|---:|:-:|---:|---:|---:|---:|")
    rv, rt = mean_pearson_r(null_tt_val), mean_pearson_r(null_tt_test)
    print(f"| random | 1 | {rv:.4f} | {rt:.4f} | — | — |")
    split_n = False
    for s in sorted(set(g_tt_val) | set(g_tt_test)):
        gv, gt = g_tt_val.get(s), g_tt_test.get(s)
        # One n for two readouts would report the better-covered split's draw
        # count beside the other split's single draw, which is exactly the
        # "n=3 mean next to an n=1 number" trap this table exists to avoid.
        nv, nt = (gv["n"] if gv else 0), (gt["n"] if gt else 0)
        n = str(nv) if nv == nt else f"{nv}/{nt}"
        split_n = split_n or nv != nt
        vr = f"{gv['mean']:.4f}" if gv else "**MISSING**"
        tr = f"{gt['mean']:.4f}" if gt else "**MISSING**"
        vc = f"{gv['mean'] / ceil_val:.3f}" if gv and ceil_val else "—"
        tc = f"{gt['mean'] / ceil_test:.3f}" if gt and ceil_test else "—"
        print(f"| {s} | {n} | {vr} | {tr} | {vc} | {tc} |")
    print()
    if split_n:
        print(
            "`n` is written *val/test* where the two splits have different "
            "numbers of draws; the columns are not equally replicated.\n"
        )
    have = [
        f"{sp} ceiling = {c:.5f}"
        for sp, c in (("val", ceil_val), ("test", ceil_test))
        if c
    ]
    if have:
        print(
            "CC_max at K=1 from the combined 5-component CorrCA: "
            + "; ".join(have)
            + ". Each *r* is divided by the ceiling of its OWN split."
        )
    over = (
        [(s, g_tt_test[s]["mean"] / ceil_test) for s in g_tt_test] if ceil_test else []
    )
    over = [(s, v) for s, v in over if v > 1.0]
    if over:
        worst = max(v for _, v in over)
        print(
            f"\n⚠️ {len(over)} test ratio(s) exceed 1.0 (max {worst:.3f}). "
            "A probe cannot exceed a true ceiling, so this ceiling is provably an "
            "underestimate — do not quote a normalised test number from it "
            "(RESULTS.md 2.2, 2.12a)."
        )
    print()

    print(
        f"## (4) scene e→v retrieval, TEST split "
        f"(pool N={null_retr_scene['pool_N']}, chance@1 = "
        f"{null_retr_scene['chance']['1']:.3f})\n"
    )
    print("| S | n | top-1 | top-5 | top-10 |")
    print("|---:|:-:|---:|---:|---:|")
    nk = null_retr_scene["top_k"]
    print(f"| random | 1 | {nk['1']:.4f} | {nk['5']:.4f} | {nk['10']:.4f} |")
    for s in sorted(set().union(*(set(g) for g in g_retr.values()))):
        n = max((g[s]["n"] for g in g_retr.values() if s in g), default=0)
        vals = " | ".join(cell_mean(g_retr[k], s, 4) for k in RETR_TOPKS)
        print(f"| {s} | {n} | {vals} |")
    print()
    print(
        f"Null sits at {null_retr_scene['relative']['1']:.2f}× chance at top-1. "
        "Val retrieval is not reported: the null beats chance there, so the val "
        "numbers measure pool structure as much as learning (RESULTS.md 2.12c).\n"
    )

    print("## (5) Three readouts side by side\n")
    print(
        "| S | Δr² (CV, val) | *r* (train→test, val) | *r* (train→test, **test**) "
        "| scene e→v top-1 | scene e→v top-5 |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    combined = sorted(set(g_d_r2) | set(g_tt_val) | set(g_tt_test) | set(g_retr["1"]))
    shown = [
        ("d_r2", g_d_r2, 5),
        ("tt_val_r", g_tt_val, 4),
        ("tt_test_r", g_tt_test, 4),
        ("retr_top1", g_retr["1"], 4),
        ("retr_top5", g_retr["5"], 4),
    ]
    col_labels = {
        "d_r2": "Δr²",
        "tt_val_r": "r val",
        "tt_test_r": "r test",
        "retr_top1": "top-1",
        "retr_top5": "top-5",
    }
    for s in combined:
        vals = " | ".join(cell_mean(g, s, nd) for _, g, nd in shown)
        print(f"| {s} | {vals} |")
    # This table has one row per S and five differently-covered readouts in it,
    # so without this line a 3-draw mean sits beside a 1-draw number with
    # nothing to say so -- the defect in the RESULTS.md 2.11 combined table.
    col_ns = {}
    for name, g, _ in shown:
        ns = sorted({g[s]["n"] for s in combined if s in g}) or [0]
        col_ns[name] = str(ns[0]) if ns[0] == ns[-1] else f"{ns[0]}–{ns[-1]}"
    pct_pairs = extra or ([(combined[0], combined[-1])] if len(combined) >= 2 else [])
    pct_rows = []
    for s0, s1 in pct_pairs:
        row = {name: cell_pct(g, s0, s1) for name, g, _ in shown}
        pct_rows.append({"from": s0, "to": s1, "pct": row})
        vals = " | ".join(f"**{row[name]}**" for name, _, _ in shown)
        print(f"| **{s0} → {s1}** | {vals} |")
    print()
    if len(set(col_ns.values())) > 1:
        print(
            "**n draws per column:** "
            + "; ".join(f"{col_labels[name]} {n}" for name, n in col_ns.items())
            + ". The columns are NOT equally replicated — a single-draw column "
            "carries no error bar, so a % change read off it is one cohort, "
            "not a cohort mean.\n"
        )
    print(
        "The readouts are not interchangeable and have disagreed before: quote which "
        "one a claim about the S axis refers to (RESULTS.md 2.12)."
    )

    if args.output:
        out = {
            "prefix": args.prefix,
            "cells_glob": args.cells_glob,
            "results_dir": str(results_dir),
            "task": args.task,
            "full_pool_s": args.full_pool_s,
            "anchors_A": a_vals[next(iter(a_vals))],
            "nulls": {
                "probe_val": str(_resolve(args.null_probe, results_dir)),
                "tt_val": str(_resolve(args.null_tt_val, results_dir)),
                "tt_test": str(_resolve(args.null_tt_test, results_dir)),
                "retr_test": str(_resolve(args.null_retr_test, results_dir)),
            },
            "null_values": {
                "probe_val_mean_r2": null_r2,
                "tt_val_r": rv,
                "tt_test_r": rt,
                "retr_test_scene": null_retr_scene,
            },
            "ceilings": {"val": ceil_val, "test": ceil_test},
            "eval_sets": {
                "probe_val_n_recordings": n_rec,
                "probe_val_n_windows": n_win,
                "encoder_dim": enc_dim,
            },
            "pooled_between_draw_sd": sd_pool,
            "fitted_exponent_S_d_r2": fit_s,
            "steps_d_r2": step_rows,
            "pct_change": pct_rows,
            "by_S": {
                str(s): {
                    "d_r2": (
                        {
                            k: g_d_r2[s][k]
                            for k in ("n", "mean", "sd", "values", "draws")
                        }
                        if s in g_d_r2
                        else None
                    ),
                    "tt_val_r": (
                        {
                            k: g_tt_val[s][k]
                            for k in ("n", "mean", "sd", "values", "draws")
                        }
                        if s in g_tt_val
                        else None
                    ),
                    "tt_test_r": (
                        {
                            k: g_tt_test[s][k]
                            for k in ("n", "mean", "sd", "values", "draws")
                        }
                        if s in g_tt_test
                        else None
                    ),
                    "retr_test_scene": {
                        f"top{k}": (
                            {
                                f2: g_retr[k][s][f2]
                                for f2 in ("n", "mean", "sd", "values", "draws")
                            }
                            if s in g_retr[k]
                            else None
                        )
                        for k in RETR_TOPKS
                    },
                }
                for s in combined
            },
            "cells": [
                {k: v for k, v in c.items() if k != "retr"}
                | {"retr_scene": c.get("retr")}
                for c in cells.values()
            ],
            "missing": missing,
            "missing_extra_steps": [{"from": a, "to": b} for a, b in absent_steps],
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
