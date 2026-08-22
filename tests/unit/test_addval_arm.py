"""Tests for the e05_addval arm -- E0.6, R5 folded into the pretraining pool.

The failure modes guarded here are all silent: each one produces a job that
runs cleanly, writes a well-formed JSON, and measures the wrong thing.

  1. **A val artifact for an addval cell.** Those encoders trained on R5, so a
     val number is measured on data they saw. It is not a weaker result, it is
     a meaningless one -- and it looks exactly like a good result.
  2. **A moved probe head-fit pool.** If an addval cell's eval read its own
     pretraining config instead of the shared one, the ridge head would fit on
     2156 recordings while every e04 cell's fits on 1863, and the arm-to-arm
     delta would no longer be attributable to the encoder.
  3. **Colliding output filenames.** The two arms share the same output
     directory and the same filename tags; only the slug separates them.
  4. **The env var and the config disagreeing on the release list.** The
     config wins (hbn.py::_resolve_releases), so a stale env var would be
     invisible -- but so would a stale config if some path read only the env.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_EXP = Path(__file__).resolve().parents[2] / "experiments" / "snr_scaling"
_ADDVAL_RELEASES = ["R1", "R2", "R3", "R4", "R5", "R7", "R8", "R9", "R10"]
_HEADFIT_RELEASES = ["R1", "R2", "R3", "R4", "R7", "R8", "R9", "R10"]


class _StubJob:
    """Records its kwargs. `Job = object` (the stub the sibling test files use)
    is not enough here -- these tests inspect a built job's command."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _load(path: Path, name: str) -> ModuleType:
    # neurolab is a cluster-side dependency and is not importable in CI; the
    # submitters only need Job/ssh_run at call time, never at import time.
    pkg = sys.modules.setdefault("neurolab", ModuleType("neurolab"))
    sub = sys.modules.setdefault("neurolab.jobs", ModuleType("neurolab.jobs"))
    submit = sys.modules.setdefault(
        "neurolab.jobs.submit", ModuleType("neurolab.jobs.submit"))
    sub.Job = _StubJob
    submit.ssh_run = lambda *a, **k: None
    sub.submit = submit
    pkg.jobs = sub
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def train_mod():
    return _load(_EXP / "submit" / "_submit_e05_addval.py", "_submit_e05_addval")


@pytest.fixture(scope="module")
def tt_mod():
    return _load(_EXP / "submit" / "_submit_traintest_e04.py", "_submit_traintest_e04")


@pytest.fixture(scope="module")
def retr_mod():
    return _load(_EXP / "submit" / "_submit_retrieval_e04.py", "_submit_retrieval_e04")


def _outputs(steps) -> set[str]:
    return set(re.findall(r"raw_results/(\S+\.json)", " ".join(steps)))


# --- 1. no val artifacts ever leave this arm --------------------------------

@pytest.mark.parametrize("arm", ["addval", "fromscratch"])
@pytest.mark.parametrize("preset", ["within", "cross"])
def test_addval_evaluates_test_split_only(tt_mod, retr_mod, preset, arm):
    for mod in (tt_mod, retr_mod):
        root, epochs, suffix, _desc = mod.resolve_arm(arm, None)
        steps = mod.build_steps(preset, epochs, suffix, root, arm)
        assert steps, f"{mod.__name__} {preset} {arm} produced no steps"
        assert not any("val" in o for o in _outputs(steps)), (
            "an addval cell's encoder trained on R5; a val number for it is "
            f"measured on data it saw. Got: {sorted(_outputs(steps))}")
        assert all("test" in o for o in _outputs(steps))


def test_e04_arm_still_reports_both_splits(tt_mod):
    """The --arm plumbing must not narrow the default arm."""
    assert tt_mod.ARM_SPLITS["e04"] is None


# --- 2. the head-fit pool does not move -------------------------------------

@pytest.mark.parametrize("cfg_name", ["config_probe_TP_e04.yaml",
                                      "config_probe_DM_e04.yaml"])
def test_eval_configs_keep_the_e04_head_fit_pool(cfg_name):
    """Shared by both arms, so R5 must NOT appear here even though the addval
    encoders trained on it. Only the encoder's cohort is allowed to differ."""
    cfg = yaml.safe_load((_EXP / "config" / cfg_name).read_text())
    assert cfg["data"]["train_releases"] == _HEADFIT_RELEASES


def test_addval_probe_snapshot_reverts_to_the_head_fit_pool(train_mod):
    """The per-cell config_probe.yaml the training job writes is what the
    retrieval `within` preset reads. It must clear the scaling knobs AND undo
    the R5 opt-in, or a consumer of it would fit on the wrong pool."""
    job = train_mod.build_job(1863, 11, "gpuA40x4", "03:00:00", 400)
    probe_patch = [c for c in job.command.split(" && ") if "config_probe.yaml" in c]
    assert probe_patch, "no config_probe.yaml is written"
    patch = probe_patch[0]
    assert "c.data.train_releases = ['R1','R2','R3','R4','R7','R8','R9','R10']" in patch
    for knob in ("max_subjects", "max_anchors", "epoch_size"):
        assert f"c.data.{knob} = None" in patch


def test_expected_train_recordings_is_not_per_arm(tt_mod):
    """1863/1832 must hold for addval artifacts too -- a 2156 there means the
    probe read the pretraining config instead of the eval config."""
    assert tt_mod.EXPECTED_TRAIN_RECORDINGS == {
        "ThePresent": 1863, "DespicableMe": 1832}


# --- 3. the two arms cannot collide -----------------------------------------

@pytest.mark.parametrize("preset", ["within", "cross"])
def test_arm_outputs_are_disjoint(tt_mod, retr_mod, preset):
    for mod in (tt_mod, retr_mod):
        e04 = _outputs(mod.build_steps(preset, {"e04_s1863_a101_nd": 375}))
        root, epochs, suffix, _ = mod.resolve_arm("addval", None)
        av = _outputs(mod.build_steps(preset, epochs, suffix, root, "addval"))
        assert e04.isdisjoint(av), f"colliding artifacts: {sorted(e04 & av)}"


def test_every_arm_reads_its_own_checkpoint_root(tt_mod, retr_mod, train_mod):
    """Three arms share one output directory and one set of filename tags; only
    the slug and the checkpoint root separate them. A crossed root would
    evaluate warm-started checkpoints under from-scratch slugs, which is
    indistinguishable downstream from "the warm start buys nothing"."""
    roots = {}
    for arm in ("addval", "fromscratch"):
        root, epochs, suffix, _ = tt_mod.resolve_arm(arm, 325)
        steps = tt_mod.build_steps("within", epochs, suffix, root, arm)
        assert all(root in s for s in steps)
        other = (tt_mod.E05FS_CKPT_ROOT if arm == "addval"
                 else tt_mod.E05_CKPT_ROOT)
        assert not any(other in s for s in steps), f"{arm} reads {other}"
        roots[arm] = root
        assert tt_mod.ARM_SPLITS[arm] == ["test"]
    assert roots["addval"] != roots["fromscratch"]
    assert tt_mod.E05FS_CKPT_ROOT == train_mod.FS_CKPT_ROOT
    outs_a = _outputs(tt_mod.build_steps(
        "within", *tt_mod.resolve_arm("addval", 325)[1:3][::-1][::-1][:1],
        tt_mod.resolve_arm("addval", 325)[2], tt_mod.E05_CKPT_ROOT, "addval"))
    assert outs_a


def test_addval_reads_its_own_checkpoint_root(tt_mod, train_mod):
    root, epochs, suffix, _ = tt_mod.resolve_arm("addval", None)
    assert root == train_mod.CKPT_ROOT, (
        "the eval submitter would read e04's checkpoints under e05 slugs")
    steps = tt_mod.build_steps("within", epochs, suffix, root, "addval")
    assert all(train_mod.CKPT_ROOT in s for s in steps)
    assert not any("kkokate" in s for s in steps)


def test_addval_cell_list_matches_what_training_submits(tt_mod, retr_mod, train_mod):
    """The eval submitters must cover exactly the cells training submits --
    including the low-S extension, or the official curve silently loses its
    left half while every table still renders."""
    trained = {train_mod.slug(s, d)
               for s, d in train_mod.CELLS + train_mod.FULL_AXIS_CELLS}
    assert set(tt_mod.E05_CELLS) == trained
    assert set(retr_mod.E05_CELLS) == trained


# --- 4. the pool declaration agrees with itself -----------------------------

def test_training_config_and_env_var_declare_the_same_pool(train_mod):
    """The config wins at runtime, so a stale env var is invisible -- and a
    stale config would be invisible to any path that reads only the env."""
    cfg = yaml.safe_load((_EXP / "config" / "clip_pretrain_e05_addval.yaml").read_text())
    assert cfg["data"]["train_releases"] == _ADDVAL_RELEASES
    assert train_mod.ENV["HBN_TRAIN_RELEASES"] == ",".join(_ADDVAL_RELEASES)


def test_warm_start_is_passed_on_the_cli(train_mod):
    """The regression that invalidated the first e05 sweep.

    e04 records `encoder_init_from: null` in its config.yaml and then overrides
    it on the CLI, so the recipe cannot be reconstructed from the file. An arm
    built from the file alone trains from scratch, scores ~37 % below e04 at
    matched S, flattens the S curve entirely — and every artifact it writes is
    well-formed. Nothing but the invocation distinguishes them.
    """
    job = train_mod.build_job(1863, 11, "gpuA40x4", "03:00:00", 400)
    assert f"--meta.encoder_init_from={train_mod.ENCODER_INIT_FROM}" in job.command
    assert train_mod.ENCODER_INIT_FROM.endswith("reve_base_eet_init.pth.tar")
    assert "--meta.seed=2026" in job.command


def test_from_scratch_arm_omits_the_warm_start_and_cannot_collide(train_mod):
    """The inverse of the test above, and just as load-bearing.

    The whole point of this arm is the ABSENCE of --meta.encoder_init_from. If it
    leaked back in, the arm would silently reproduce the warm-started numbers and
    the comparison would read as "the warm start buys nothing" -- the exact
    opposite of the truth. It must also write somewhere else entirely, or it
    would overwrite the warm-started checkpoints.
    """
    train_mod.FROM_SCRATCH = True
    try:
        job = train_mod.build_job(1400, 11, "gpuA40x4", "03:00:00", 400)
        assert "--meta.encoder_init_from" not in job.command
        assert train_mod.FS_CKPT_ROOT in job.command
        assert train_mod.CKPT_ROOT + "/" not in job.command
        assert train_mod.slug(1400, 11) == "e05fs_s1400_a101_av_d11"
    finally:
        train_mod.FROM_SCRATCH = False
    # ...and the warm-started arm is unaffected by that flag flipping.
    warm = train_mod.build_job(1400, 11, "gpuA40x4", "03:00:00", 400)
    assert f"--meta.encoder_init_from={train_mod.ENCODER_INIT_FROM}" in warm.command
    assert train_mod.slug(1400, 11) == "e05_s1400_a101_av_d11"


def test_config_placeholders_are_not_mistaken_for_the_recipe(train_mod):
    """The frozen config's meta block is deliberately stale (it mirrors e04's,
    which is also stale). Pinned so that a future edit "fixing" it to match the
    CLI does not create a second source of truth that can drift."""
    cfg = yaml.safe_load((_EXP / "config" / "clip_pretrain_e05_addval.yaml").read_text())
    assert cfg["meta"]["encoder_init_from"] is None
    assert cfg["meta"]["seed"] != train_mod.SEED


def test_step_budget_is_held_to_e04(train_mod):
    """epoch_size decouples optimisation budget from pool size. Without it the
    new points would be confounded with the extra data they are measuring."""
    cfg = yaml.safe_load((_EXP / "config" / "clip_pretrain_e05_addval.yaml").read_text())
    assert cfg["data"]["epoch_size"] == train_mod.EPOCH_SIZE == 703
    assert cfg["data"]["max_anchors"] == train_mod.ANCHORS == 101
    # 2026, matching e04's CLI override -- NOT the config's 2025.
    assert train_mod.SEED == 2026 and train_mod.EPOCHS == 400


def test_architecture_matches_the_e04_arm():
    """Only the pool line may differ; a depth or patch-size drift would make
    the arms incomparable while every table still rendered."""
    av = yaml.safe_load((_EXP / "config" / "clip_pretrain_e05_addval.yaml").read_text())
    e04 = yaml.safe_load((_EXP / "config" / "config_probe_TP_e04.yaml").read_text())
    assert av["model"] == e04["model"]
    assert av["loss"] == e04["loss"]


def test_1863_is_a_real_subsample_of_the_addval_pool(train_mod):
    """The whole point: in e04, max_subjects=1863 is a no-op cap on a
    1863-subject pool, so there is one possible draw. Here it must bite."""
    assert train_mod.E04_POOL_RECORDINGS < train_mod.POOL_RECORDINGS
    s_values = {s for s, _d in train_mod.CELLS}
    assert train_mod.E04_POOL_RECORDINGS in s_values
    draws = [d for s, d in train_mod.CELLS if s == train_mod.E04_POOL_RECORDINGS]
    assert len(draws) == 3 and None not in draws
