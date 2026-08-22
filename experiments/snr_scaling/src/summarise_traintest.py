"""Summarise train->eval linear probes (Pearson r) into the tables in RESULTS.

Reads the JSONs written by ``eb_jepa/evaluation/clip_probe/probe_traintest.py``
(submitted via ``submit/_submit_traintest.py``) and prints one row per cell with
the mean Pearson r over the 12 scalar features, on each split it finds.

Two families live side by side in ``raw_results/``:

    e03_tt_{val,test}_<slug>.json        ThePresent   (RESULTS.md 2.12 / 2.13)
    xtask_tt_DM{val,test}_<slug>.json    DespicableMe (RESULTS_cross_task.md 2)

Mean r is unweighted over features, matching how the published tables were
assembled. Features are averaged only if present in *every* cell of the family,
so a cell that silently dropped a feature cannot shift the mean relative to its
neighbours -- it is reported as a warning instead.

Usage:
    uv run --group eeg python experiments/snr_scaling/src/summarise_traintest.py
    uv run --group eeg python experiments/snr_scaling/src/summarise_traintest.py --family xtask
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiments/snr_scaling/src
ROOT = HERE.parent                              # experiments/snr_scaling
RAW = ROOT / "raw_results"

FAMILIES = {
    # name -> (glob, regex with named groups `split` and `slug`)
    "e03": ("e03_tt_*.json", re.compile(r"^e03_tt_(?P<split>val|test)_(?P<slug>.+)$")),
    "xtask": ("xtask_tt_DM*.json", re.compile(r"^xtask_tt_DM(?P<split>val|test)_(?P<slug>.+)$")),
}

# Sort order: random first, then by S, then by draw. Keeps the printed table in
# the same order as the published one regardless of filesystem order.
def _sort_key(slug: str) -> tuple:
    if slug == "random":
        return (0, 0, 0, "")
    m = re.search(r"_s(\d+)_", slug)
    s = int(m.group(1)) if m else 0
    d = re.search(r"_d(\d+)$", slug)
    return (1, s, int(d.group(1)) if d else 0, slug)


def load_family(name: str) -> dict:
    """{split: {slug: {feature: pearson_r}}} for one family."""
    pattern, rx = FAMILIES[name]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted(RAW.glob(pattern)):
        m = rx.match(path.stem)
        if not m:
            continue
        d = json.loads(path.read_text())
        out.setdefault(m["split"], {})[m["slug"]] = {
            f: v["pearson_r"] for f, v in d["features"].items()
        }
    return out


def mean_r(per_feature: dict, features: list[str]) -> float:
    return sum(per_feature[f] for f in features) / len(features)


def report(name: str, match: str = "") -> None:
    fam = load_family(name)
    if match:
        # The xtask_tt_DM* namespace is shared: the depth-12 e03 arm and the
        # depth-22 e04 arm write side by side, so a bare listing interleaves two
        # architectures into one table. Filter by slug substring, e.g. "e03_".
        fam = {sp: {s: f for s, f in cells.items() if match in s or s == "random"}
               for sp, cells in fam.items()}
        fam = {sp: cells for sp, cells in fam.items() if cells}
    if not fam:
        print(f"[{name}] no artifacts in {RAW}")
        return

    splits = [s for s in ("val", "test") if s in fam]
    slugs = sorted({s for sp in splits for s in fam[sp]}, key=_sort_key)

    # Only average over features every cell has, so the mean is comparable.
    shared = set.intersection(*(set(fam[sp][s]) for sp in splits for s in fam[sp]))
    for sp in splits:
        for slug, feats in fam[sp].items():
            missing = shared.symmetric_difference(feats)
            if missing:
                print(f"  ! {name} {sp} {slug}: feature set differs by {sorted(missing)}")
    features = sorted(shared)

    print(f"\n[{name}]  {len(slugs)} cell(s), mean Pearson r over {len(features)} features")
    head = "  ".join(f"{sp+' r':>9}" for sp in splits)
    print(f"{'cell':<28}  {head}")
    print("-" * (30 + 11 * len(splits)))
    baseline = {sp: fam[sp].get("random") for sp in splits}
    for slug in slugs:
        cells = []
        for sp in splits:
            f = fam[sp].get(slug)
            cells.append(f"{mean_r(f, features):9.4f}" if f else f"{'-':>9}")
        print(f"{slug:<28}  " + "  ".join(cells))
    for sp in splits:
        if baseline[sp]:
            print(f"  random baseline ({sp}): {mean_r(baseline[sp], features):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    ap.add_argument("--match", default="",
                    help="Keep only cells whose slug contains this, e.g. 'e03_'. "
                         "The random baseline is always kept.")
    args = ap.parse_args()
    for name in (FAMILIES if args.family == "all" else [args.family]):
        report(name, args.match)


if __name__ == "__main__":
    main()
