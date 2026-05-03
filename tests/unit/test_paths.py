"""Tests for eb_jepa.paths.resolve_preprocessed_dir."""

from pathlib import Path

import pytest

from eb_jepa.paths import PREPROCESSED_DIRS, resolve_preprocessed_dir


def test_explicit_path_wins(tmp_path):
    """An explicit configured path is returned verbatim, even if it doesn't exist."""
    explicit = tmp_path / "does_not_exist"
    assert resolve_preprocessed_dir(str(explicit)) == explicit


def test_returns_none_when_none_exist(monkeypatch):
    """With no configured path and no auto-detect hits, returns None."""
    monkeypatch.setattr(
        "eb_jepa.paths.PREPROCESSED_DIRS",
        [Path("/definitely/does/not/exist/eb_jepa_test")],
    )
    assert resolve_preprocessed_dir(None) is None


def test_autodetect_picks_first_existing(tmp_path, monkeypatch):
    """Auto-detect returns the first existing path in priority order."""
    later = tmp_path / "later"
    later.mkdir()
    monkeypatch.setattr(
        "eb_jepa.paths.PREPROCESSED_DIRS",
        [Path("/missing/first"), later, tmp_path / "third"],
    )
    assert resolve_preprocessed_dir(None) == later


def test_default_dirs_are_absolute_paths():
    """The packaged default list is a non-empty list of absolute Path objects."""
    assert len(PREPROCESSED_DIRS) > 0
    for p in PREPROCESSED_DIRS:
        assert isinstance(p, Path)
        assert p.is_absolute()
