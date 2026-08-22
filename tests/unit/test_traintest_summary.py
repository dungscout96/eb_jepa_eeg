"""Tests for the train->eval probe submitter and its summariser.

Two silent failures are guarded here.

The submitter reconstructs how every published ``e03_tt_*`` Pearson r was
produced -- those runs originally came from throwaway shell scripts that no
longer exist. If a slug rule or a filename tag drifts, the script still runs
cleanly and simply measures *different* checkpoints into *different* files, and
nothing in a dry run would say so. So the `tp` + `low-s` presets are pinned
against the artifacts actually on disk.

The summariser averages Pearson r across features. If one cell is missing a
feature, a naive mean silently compares a 12-feature average against an
11-feature one -- a difference of the same order as the effects being reported.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

_EXP = Path(__file__).resolve().parents[2] / "experiments" / "snr_scaling"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def submit_mod():
    # neurolab is a cluster-side dependency and is not importable in CI; the
    # submitter only needs Job at call time, never at import time.
    if "neurolab.jobs" not in sys.modules:
        pkg, sub = ModuleType("neurolab"), ModuleType("neurolab.jobs")
        sub.Job = object
        pkg.jobs = sub
        sys.modules["neurolab"], sys.modules["neurolab.jobs"] = pkg, sub
    return _load(_EXP / "submit" / "_submit_traintest.py", "_submit_traintest")


@pytest.fixture(scope="module")
def summarise_mod():
    return _load(_EXP / "src" / "summarise_traintest.py", "summarise_traintest")


def _outputs(steps) -> set[str]:
    return set(re.findall(r"raw_results/(\S+\.json)", " ".join(steps)))


def test_tp_and_lows_presets_reproduce_the_artifacts_on_disk(submit_mod):
    """The provenance claim in the docstring, checked rather than asserted."""
    generated = _outputs(submit_mod.build_steps("tp")) | _outputs(
        submit_mod.build_steps("low-s")
    )
    on_disk = {p.name for p in (_EXP / "raw_results").glob("e03_tt_*.json")}
    assert generated == on_disk, (
        f"missing: {sorted(generated - on_disk)}  extra: {sorted(on_disk - generated)}"
    )


def test_full_pool_cell_has_no_draw_suffix(submit_mod):
    """S=1863 is the whole pool, so it was trained once with no draw seed."""
    assert submit_mod._slug(1863) == "e03_s1863_a101_nd"
    assert submit_mod._slug(701) == "e03_s701_a101_nd_d11"


def test_cross_task_presets_use_the_shared_dm_config_on_both_splits(submit_mod):
    for preset in ("cross", "dm-ref"):
        steps = submit_mod.build_steps(preset)
        assert all(submit_mod.DM_CONFIG in s for s in steps)
        outs = _outputs(steps)
        assert sum("DMval" in o for o in outs) == sum("DMtest" in o for o in outs) > 0
        # A cell's own ThePresent config here would silently evaluate the wrong
        # movie while still producing a plausible-looking JSON.
        assert not any("config_probe.yaml" in s for s in steps)


def test_extended_train_pool_is_opted_into(submit_mod):
    """The bug this caught: without it the ridge fits on R1-R4 only.

    The first submission of these runs omitted HBN_TRAIN_RELEASES. All five
    jobs reported COMPLETED 0:0 and wrote 26 well-formed JSONs -- off 741
    DespicableMe recordings instead of 1832, because the loader falls back to
    the R1-R4 default. Nothing but n_train_recordings distinguished them.
    """
    assert submit_mod.ENV["HBN_TRAIN_RELEASES"] == "R1,R2,R3,R4,R7,R8,R9,R10"
    assert submit_mod.ENV["HBN_PREPROCESS_DIR"].endswith("/hbn_preprocessed")


def test_verify_flags_the_wrong_training_pool(submit_mod, tmp_path, capsys):
    """A JSON off the wrong pool is well-formed; only the count betrays it."""
    good = {"n_train_recordings": 1832, "features": {str(i): {} for i in range(12)}}
    bad = {**good, "n_train_recordings": 741}

    steps = submit_mod.build_steps("dm-ref")
    outs = [re.search(r"--output (\S+)", s).group(1).split("/")[-1] for s in steps]
    for name in outs[:-1]:
        (tmp_path / name).write_text(json.dumps(good))
    (tmp_path / outs[-1]).write_text(json.dumps(bad))

    assert submit_mod.verify("dm-ref", tmp_path) == 1
    out = capsys.readouterr().out
    assert "WRONG POOL" in out and "741" in out and "1832" in out

    # And an absent artifact is not silently a pass.
    (tmp_path / outs[0]).unlink()
    assert submit_mod.verify("dm-ref", tmp_path) == 2
    assert "MISSING" in capsys.readouterr().out


def test_verify_expects_the_right_pool_per_task(submit_mod):
    """ThePresent and DespicableMe have different extended-pool sizes.

    Both values are measured from artifacts, not assumed: 1863 from the 17
    e03_tt_* files, 1832 from the 26 xtask_tt_DM* files (and corroborated by 84
    independent depth-22 e04 artifacts). RESULTS_cross_task.md 1 quotes 1841 for
    DespicableMe, which is the requested `max_subjects` cap -- a cap above the
    pool is a silent no-op -- not the realised count.
    """
    assert submit_mod.EXPECTED_TRAIN_RECORDINGS == {
        "ThePresent": 1863, "DespicableMe": 1832,
    }


def test_verify_passes_on_the_artifacts_already_on_disk(submit_mod):
    """The 17 published ThePresent artifacts must satisfy their own checker."""
    assert submit_mod.verify("tp", _EXP / "raw_results") == 0
    assert submit_mod.verify("low-s", _EXP / "raw_results") == 0


def test_every_step_is_skippable_so_a_timed_out_job_can_resume(submit_mod):
    for preset in submit_mod.PRESETS:
        for step in submit_mod.build_steps(preset):
            out = re.search(r"--output (\S+)", step).group(1)
            assert step.startswith(f"([ -f {out} ] &&")


def test_chunking_partitions_without_loss_or_duplication(submit_mod):
    steps = submit_mod.build_steps("cross")
    for n in (1, 3, 4, 7, 100):
        groups = submit_mod.chunk(steps, n)
        flat = [s for g in groups for s in g]
        assert flat == steps
        assert all(groups), "no empty jobs"
        assert max(map(len, groups)) - min(map(len, groups)) <= 1


def _write(raw: Path, name: str, feats: dict) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / name).write_text(json.dumps(
        {"features": {f: {"pearson_r": r} for f, r in feats.items()}}
    ))


def test_summary_averages_only_features_shared_by_every_cell(summarise_mod, tmp_path,
                                                             monkeypatch, capsys):
    """A cell missing a feature must not be averaged over a different set."""
    raw = tmp_path / "raw_results"
    # b is worthless in the shared feature, but excellent in the one only it has.
    _write(raw, "xtask_tt_DMval_a.json", {"x": 0.4, "y": 0.2})
    _write(raw, "xtask_tt_DMval_b.json", {"x": 0.0, "z": 0.9})
    monkeypatch.setattr(summarise_mod, "RAW", raw)

    summarise_mod.report("xtask")
    out = capsys.readouterr().out

    assert "over 1 features" in out          # only x is shared
    assert "0.4000" in out and "0.0000" in out
    assert "0.4500" not in out               # b's own mean over {x, z}
    assert "feature set differs" in out      # and it says so


def test_summary_reads_both_splits_and_sorts_random_first(summarise_mod, tmp_path,
                                                          monkeypatch, capsys):
    raw = tmp_path / "raw_results"
    for split in ("val", "test"):
        _write(raw, f"xtask_tt_DM{split}_random.json", {"x": 0.1})
        _write(raw, f"xtask_tt_DM{split}_e03_s701_a101_nd_d11.json", {"x": 0.3})
        _write(raw, f"xtask_tt_DM{split}_e03_s10_a101_nd_d11.json", {"x": 0.2})
    monkeypatch.setattr(summarise_mod, "RAW", raw)

    summarise_mod.report("xtask")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(("random", "e03"))]

    assert [ln.split()[0] for ln in lines] == [
        "random", "e03_s10_a101_nd_d11", "e03_s701_a101_nd_d11",
    ]
    assert lines[0].split()[1:] == ["0.1000", "0.1000"]   # val and test columns


def test_summary_survives_a_cell_measured_on_only_one_split(summarise_mod, tmp_path,
                                                            monkeypatch, capsys):
    """Half a sweep landing (or a job timing out mid-chunk) must still report."""
    raw = tmp_path / "raw_results"
    _write(raw, "xtask_tt_DMval_e03_s10_a101_nd_d11.json", {"x": 0.2})
    _write(raw, "xtask_tt_DMtest_e03_s10_a101_nd_d11.json", {"x": 0.15})
    _write(raw, "xtask_tt_DMval_e03_s701_a101_nd_d11.json", {"x": 0.3})
    monkeypatch.setattr(summarise_mod, "RAW", raw)

    summarise_mod.report("xtask")
    out = capsys.readouterr().out

    assert "0.3000" in out
    assert re.search(r"e03_s701\S*\s+0\.3000\s+-", out), "missing split shown as '-'"
