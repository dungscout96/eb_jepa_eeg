"""The initialisation comparison with neither arm handicapped by the other's protocol.

Both arms of the published comparison were evaluated at a single fixed epoch. That
epoch sits near the warm-started arm's broad optimum and hundreds of epochs before
the from-scratch arm's, and above S=701 the from-scratch arm's optimum lay outside
its training budget entirely -- it was stopped before it converged. So part of the
reported gap is the protocol rather than the initialisation.

This rebuilds the comparison with every cell at its own probe-selected optimum, and
prints it beside the fixed-epoch version so the difference is legible.

WHICH CHECKPOINT EACH CELL USES is not derivable from its name: the from-scratch arm
draws from two checkpoint trees that hold identically-named cells at identical S,
differing only in training budget. The composite selection file is the authority, and
this script reads slugs from it rather than reconstructing them -- reconstructing
would silently pick the wrong budget for half the ladder.

Usage:
    uv run --group eeg python experiments/snr_scaling/src/analyse_optimum_comparison.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_results"
WARM_SEL = RAW / "addval_selection_probe.json"
SCRATCH_SEL = RAW / "fromscratch_optimum_selection.json"
FIXED = "_ep325"
OPT = "_epprb"
DRAWS = (11, 22, 33)


def s_of(cell: str) -> int:
    return int(cell.split("_")[1][1:])


def mean_r(path: Path) -> float | None:
    if not path.exists():
        return None
    feats = json.loads(path.read_text())["features"]
    return sum(v["pearson_r"] for v in feats.values()) / len(feats)


def e2v1(path: Path, level: str = "time") -> float | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())["levels"][level]["e2v_top_k"]["1"]


def cells_by_s(selection: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for cell in json.loads(selection.read_text()):
        out.setdefault(s_of(cell), []).append(cell)
    return out


def fixed_slug(cell: str) -> str:
    """The fixed-epoch artifact for a cell.

    Only the 400-epoch tree was ever evaluated at the fixed epoch, so a retrained
    cell's fixed-epoch counterpart is its ORIGINAL slug, without the budget marker.
    Comparing a retrained cell against itself at a fixed epoch would otherwise be
    comparing two runs, not two protocols.
    """
    return cell.replace("_ep800", "")


def collect(cells: list[str], prefix: str, suffix: str, reader, slug_fn=lambda c: c):
    vals = [reader(RAW / f"{prefix}_{slug_fn(c)}{suffix}.json") for c in cells]
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else None, len(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readout", default="probe", choices=["probe", "retrieval"])
    ap.add_argument("--task", default="within", choices=["within", "cross"])
    args = ap.parse_args()

    if args.readout == "probe":
        prefix = "e04_tt_test" if args.task == "within" else "xtask_tt_DMtest"
        reader, label = mean_r, "mean(12) Pearson r"
    else:
        prefix = "e04_retr_test" if args.task == "within" else "xtask_retr_DMtest"
        reader, label = e2v1, "time-pool e->v@1"

    warm = cells_by_s(WARM_SEL)
    scratch = cells_by_s(SCRATCH_SEL)

    print(f"INITIALISATION COMPARISON -- {args.task}-task, {label}, test split")
    print("fixed: both arms at epoch 325.  opt: each cell at its own probe optimum,")
    print("the from-scratch arm drawing on 8800 steps where S>=701.\n")
    print(f"{'S':>6} | {'warm':^17} | {'from scratch':^17} | {'gap':^17} | {'':>8}")
    print(f"{'':>6} | {'fixed':>8}{'opt':>9} | {'fixed':>8}{'opt':>9} | "
          f"{'fixed':>8}{'opt':>9} | {'change':>8}")
    rows = []
    for S in sorted(warm):
        wf, _ = collect(warm[S], prefix, FIXED, reader)
        wo, nwo = collect(warm[S], prefix, OPT, reader)
        sf, _ = collect(scratch.get(S, []), prefix, FIXED, reader, fixed_slug)
        so, nso = collect(scratch.get(S, []), prefix, OPT, reader)
        if None in (wf, wo, sf, so):
            missing = [n for n, v in (("warm/opt", wo), ("scratch/opt", so),
                                      ("warm/fixed", wf), ("scratch/fixed", sf))
                       if v is None]
            print(f"{S:>6} |  incomplete: {', '.join(missing)}")
            continue
        gf, go = wf - sf, wo - so
        rows.append((S, gf, go))
        print(f"{S:>6} | {wf:>8.4f}{wo:>9.4f} | {sf:>8.4f}{so:>9.4f} | "
              f"{gf:>8.4f}{go:>9.4f} | {go - gf:>+8.4f}")

    if not rows:
        print("\nNo complete rows yet -- the re-evaluation is still running.")
        return
    hi = [(S, gf, go) for S, gf, go in rows if S >= 701]
    print(f"\nOver the whole ladder: gap {st.mean(g for _, g, _ in rows):.4f} fixed "
          f"-> {st.mean(g for _, _, g in rows):.4f} at each cell's optimum")
    if hi:
        mf, mo = st.mean(g for _, g, _ in hi), st.mean(g for _, _, g in hi)
        print(f"At S>=701, where the budget was binding: {mf:.4f} -> {mo:.4f} "
              f"({100 * (mf - mo) / mf:.0f} % of the reported gap was protocol)")
    print("\nThe comparison survives if the optimum-column gap stays clearly positive;")
    print("what changes is its size, and whether it still grows with S.")


if __name__ == "__main__":
    main()
