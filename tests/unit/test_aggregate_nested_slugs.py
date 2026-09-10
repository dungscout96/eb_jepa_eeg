"""Slug parsing in aggregate_nested.py must accept every arm prefix in use.

The failure is silent: an unparseable slug is skipped with a warning on stderr
and its cell simply vanishes from the table, so an arm prefix the regex does
not know produces an empty (not a wrong) aggregation -- easy to mistake for
"no readouts landed yet".
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_nested",
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "snr_scaling"
    / "aggregate_nested.py",
)
agg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(agg)


@pytest.mark.parametrize(
    "slug, study, s, tag, draw",
    [
        ("e03_s1863_a101_nd", "e03", 1863, "nd", None),
        ("e04_s400_a101_nd_d11", "e04", 400, "nd", 11),
        ("e12cb_s701_a101_nd_d22", "e12cb", 701, "nd", 22),
        ("e12cbr_s1863_a101_nd", "e12cbr", 1863, "nd", None),
        ("e11scad12_s701_a101_av_d33", "e11scad12", 701, "av", 33),
    ],
)
def test_parse_slug(slug, study, s, tag, draw):
    got = agg.parse_slug(slug)
    assert got is not None, slug
    assert (got["study"], got["S"], got["A"], got["tag"], got["draw"]) == (
        study,
        s,
        101,
        tag,
        draw,
    )


@pytest.mark.parametrize("bad", ["s701_a101_nd", "e12cb_701_a101", "e12cb_s701"])
def test_rejects_non_slugs(bad):
    assert agg.parse_slug(bad) is None
