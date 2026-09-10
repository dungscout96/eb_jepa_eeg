"""The E1.2 results writer must render every number from the aggregates and
never invent one: a missing arm at some S renders as a dash, not as a zero or a
crash, and the level ratio is warm/random of the MEANS."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "write_results_cbramod",
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "snr_scaling"
    / "src"
    / "write_results_cbramod.py",
)
wr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wr)


def _group(mean, n=3, sd=0.001):
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "values": [mean] * n,
        "draws": [11, 22, 33][:n],
    }


def _agg(level: float, ss=(50, 200, 701, 1863)):
    by_s = {}
    for i, s in enumerate(ss):
        y = level * (1 + i)
        by_s[str(s)] = {
            "d_r2": _group(y),
            "tt_val_r": _group(0.1 + 0.01 * i),
            "tt_test_r": _group(0.05 + 0.01 * i),
            "retr_test_scene": {
                f"top{k}": _group(0.02 * int(k)) for k in ("1", "5", "10")
            },
        }
    steps = [
        {
            "from": a,
            "to": b,
            "delta": 0.01,
            "in_sd": 2.0,
            "exponent": 0.3,
            "pct": 10.0,
            "adjacent": True,
        }
        for a, b in zip(ss, ss[1:])
    ]
    return {
        "null_values": {
            "probe_val_mean_r2": 0.0123,
            "retr_test_scene": {"top1": 0.01, "top5": 0.05, "top10": 0.1},
        },
        "ceilings": {"val": 0.313, "test": 0.172},
        "pooled_between_draw_sd": 0.002,
        "fitted_exponent_S_d_r2": 0.4,
        "steps_d_r2": steps,
        "by_S": by_s,
        "cells": [{} for _ in ss],
        "missing": [],
    }


def test_ratio_and_tables_render():
    warm, rand = _agg(0.02), _agg(0.01)
    md = wr.render(warm, rand, today="2026-09-10")
    assert "| 50 | 0.02000 ± 0.00100 (n=3) | 0.01000 ± 0.00100 (n=3) | 2.00× |" in md
    assert "| 701 → 1863 | 0.300 | 2.0 | 0.300 | 2.0 |" in md
    assert "median 2.00×" in md
    # r / val ceiling uses the VAL ceiling, never the test one
    assert f"| 50 | 0.0500 ± 0.0010 (n=3) | {0.05 / 0.313:.3f} |" in md


def test_missing_arm_cell_renders_dash_not_zero():
    warm, rand = _agg(0.02), _agg(0.01, ss=(50, 200, 701))
    md = wr.render(warm, rand, today="2026-09-10")
    row = [line for line in md.splitlines() if line.startswith("| 1863 |")][0]
    assert "—" in row and "0.00000" not in row


def test_single_draw_cell_has_no_sd():
    warm, rand = _agg(0.02), _agg(0.01)
    warm["by_S"]["1863"]["d_r2"] = _group(0.08, n=1, sd=None)
    md = wr.render(warm, rand, today="2026-09-10")
    assert "0.08000 (n=1)" in md
