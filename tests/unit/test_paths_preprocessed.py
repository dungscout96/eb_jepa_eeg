"""Precedence tests for resolve_preprocessed_dir.

The env var was silently inert here for the code path that matters (the
training entry point passes this function's result explicitly), so five
extended-cohort cells trained against the wrong root and died on a missing R7
while reporting COMPLETED 0:0.
"""

from pathlib import Path

from eb_jepa.paths import resolve_preprocessed_dir


def test_explicit_config_wins_over_env(monkeypatch):
    monkeypatch.setenv("HBN_PREPROCESS_DIR", "/from/env")
    assert resolve_preprocessed_dir("/from/config") == Path("/from/config")


def test_env_wins_over_autodetect(monkeypatch, tmp_path):
    """The whole bug: an exported HBN_PREPROCESS_DIR must beat a path that
    merely happens to exist on this cluster."""
    existing = tmp_path / "autodetected"
    existing.mkdir()
    monkeypatch.setattr("eb_jepa.paths.PREPROCESSED_DIRS", [existing])
    monkeypatch.setenv("HBN_PREPROCESS_DIR", "/from/env")
    assert resolve_preprocessed_dir(None) == Path("/from/env")


def test_autodetect_used_when_env_absent(monkeypatch, tmp_path):
    existing = tmp_path / "autodetected"
    existing.mkdir()
    monkeypatch.setattr("eb_jepa.paths.PREPROCESSED_DIRS", [existing])
    monkeypatch.delenv("HBN_PREPROCESS_DIR", raising=False)
    assert resolve_preprocessed_dir(None) == existing


def test_none_when_nothing_resolves(monkeypatch):
    monkeypatch.setattr("eb_jepa.paths.PREPROCESSED_DIRS",
                        [Path("/definitely/not/here")])
    monkeypatch.delenv("HBN_PREPROCESS_DIR", raising=False)
    assert resolve_preprocessed_dir(None) is None


def test_empty_env_does_not_shadow_autodetect(monkeypatch, tmp_path):
    """An empty string is not a path; it must fall through, not resolve to ''."""
    existing = tmp_path / "autodetected"
    existing.mkdir()
    monkeypatch.setattr("eb_jepa.paths.PREPROCESSED_DIRS", [existing])
    monkeypatch.setenv("HBN_PREPROCESS_DIR", "")
    assert resolve_preprocessed_dir(None) == existing
