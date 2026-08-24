"""Tests for E0.3 best-epoch selection.

The failure this guards is silent, not loud: a cell can hold several wandb run
directories -- a crashed attempt leaves a tiny stub beside the real run -- and
glob returns them in arbitrary order. Taking files[0] picks the stub about half
the time. When the stub is empty that errors out; when it is PARTIAL it yields a
plausible but wrong best epoch, and the wrong checkpoint gets probed and
reported as the cell's early-stopped result.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "select_and_probe_e03",
    Path(__file__).resolve().parents[2] / "experiments" / "snr_scaling" / "src"
    / "select_and_probe_e03.py",
)
sel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sel)


def _cell(tmp_path, runs, ckpt_epochs=(25, 50, 75, 100)):
    """runs: {run_dir_name: [values]} -- history per wandb run dir."""
    d = tmp_path / "cell"
    (d / "wandb").mkdir(parents=True)
    for name in runs:
        rd = d / "wandb" / name
        rd.mkdir()
        # Must match the run-*/run-*.wandb glob the selector uses.
        (rd / f"run-{name.split('-')[-1]}.wandb").touch()
    for e in ckpt_epochs:
        (d / f"epoch_{e}.pth.tar").touch()
    return d


def test_picks_the_richest_run_not_the_first_glob_hit(tmp_path, monkeypatch):
    """The real bug: a crashed 8 KB stub beside a 9.5 MB run. Selection must
    follow the data, not filesystem ordering."""
    # A PLATEAU, not a single spike: select_epoch smooths (window 25) before
    # argmax, so a lone spike would be flattened and the expected epoch would
    # depend on window size. The plateau centre survives smoothing.
    series = [0.1] * 50 + [0.9] * 25 + [0.1] * 25
    d = _cell(tmp_path, {"run-A-stub": [], "run-B-real": series})

    def fake_hist(path, key):
        run_dir = Path(path).parent.name
        return [] if "stub" in run_dir else series

    monkeypatch.setattr(sel, "read_history", fake_hist)
    out = sel.select_epoch(d, "val/clip_scene_auc")
    assert 50 <= out["best_epoch"] <= 75, out["best_epoch"]
    assert out["raw_best_value"] == pytest.approx(0.9)


def test_a_partial_run_does_not_win_over_the_complete_one(tmp_path, monkeypatch):
    """The dangerous case: the stub is non-empty, so it would not error -- it
    would just report the wrong epoch."""
    def fake_hist(path, key):
        # Match on the run DIRECTORY, not the full path: pytest puts the test's
        # own name in tmp_path, and this test's name contains "partial", so a
        # substring check against `path` is true for every run.
        run_dir = Path(path).parent.name
        if "partial" in run_dir:
            return [0.5, 0.6, 0.7]          # peaks at epoch 2
        return [0.1] * 190 + [0.95] * 25 + [0.3] * 185   # plateau ~200

    d = _cell(tmp_path, {"run-A-partial": [1], "run-B-full": [1]},
              ckpt_epochs=(50, 100, 150, 200, 250))
    monkeypatch.setattr(sel, "read_history", fake_hist)
    out = sel.select_epoch(d, "k")
    assert 190 <= out["best_epoch"] <= 215, "picked the partial run's peak"
    assert out["selected_epoch"] == 200


def test_reports_error_when_no_run_has_the_metric(tmp_path, monkeypatch):
    d = _cell(tmp_path, {"run-A-x": [], "run-B-y": []})
    monkeypatch.setattr(sel, "read_history", lambda p, k: [])
    out = sel.select_epoch(d, "k")
    assert "no 'k' in history" in out["error"]
    assert "2 run dir(s)" in out["error"]


def test_snaps_to_the_nearest_saved_checkpoint(tmp_path, monkeypatch):
    d = _cell(tmp_path, {"run-A-z": [1]}, ckpt_epochs=(25, 50, 75, 100))
    monkeypatch.setattr(sel, "read_history",
                        lambda p, k: [0.1] * 55 + [0.9] * 20 + [0.2] * 25)
    out = sel.select_epoch(d, "k")
    assert 55 <= out["best_epoch"] <= 75, out["best_epoch"]
    assert out["selected_epoch"] in (50, 75)   # nearest of 25/50/75/100


def test_refuses_when_no_periodic_checkpoints_exist(tmp_path, monkeypatch):
    """Must not silently fall back to latest.pth.tar -- that would reproduce the
    fixed-budget bug the early-stopped protocol exists to fix."""
    d = _cell(tmp_path, {"run-A-z": [1]}, ckpt_epochs=())
    monkeypatch.setattr(sel, "read_history", lambda p, k: [0.1, 0.9, 0.2])
    out = sel.select_epoch(d, "k")
    assert "no periodic checkpoints" in out["error"]
    assert "checkpoint" not in out
