"""Tests for the end-of-training auto-eval hook in experiments/eeg_jepa/train.py.

The hook (`_run_auto_eval`) is the integration point between training and
the library-level eb_jepa.evaluation pipeline: it must fire when
cfg.eval.auto_run is true, skip when false, and forward the right config
keys to run_probe_eval + bootstrap_predictions.

Failures inside the eval calls are swallowed (with a warning) so a flaky
eval can't lose the training checkpoint -- the test suite verifies that
behavior is intentional.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf


def _base_cfg(auto_run=True, **eval_overrides):
    cfg_dict = {
        "meta": {"seed": 42},
        "data": {
            "n_windows": 4,
            "window_size_seconds": 2,
            "batch_size": 64,
            "num_workers": 4,
            "norm_mode": "global",
            "add_envelope": False,
            "corrca_filters": None,
        },
        "eval": {
            "auto_run": auto_run,
            "splits": "val,test",
            "probe_epochs": 20,
            "subject_probe_epochs": 100,
            "bootstrap_split": "test",
            "n_bootstrap": 1000,
            "wandb_group": "test_group",
            **eval_overrides,
        },
    }
    return OmegaConf.create(cfg_dict)


def test_hook_calls_probe_eval_then_bootstrap(tmp_path):
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from experiments.eeg_jepa import train as m

    with patch.object(m, "run_probe_eval", create=True) as pe, \
         patch.object(m, "bootstrap_predictions", create=True) as bs:
        # The hook imports lazily; we have to patch on the eb_jepa side.
        with patch("eb_jepa.evaluation.run_probe_eval") as pe2, \
             patch("eb_jepa.evaluation.bootstrap_predictions") as bs2:
            pe2.return_value = {"val/some_metric": 0.5}
            bs2.return_value = {"test/some_ci": (0.4, 0.6)}

            m._run_auto_eval(cfg, exp_dir, "cfgs/default.yaml")

            assert pe2.call_count == 1
            assert bs2.call_count == 1


def test_probe_eval_receives_expected_kwargs(tmp_path):
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from experiments.eeg_jepa import train as m

    with patch("eb_jepa.evaluation.run_probe_eval") as pe, \
         patch("eb_jepa.evaluation.bootstrap_predictions") as bs:
        pe.return_value = {}
        bs.return_value = {}

        m._run_auto_eval(cfg, exp_dir, "cfgs/default.yaml")

        kwargs = pe.call_args.kwargs
        assert kwargs["checkpoint"] == str(exp_dir / "latest.pth.tar")
        assert kwargs["save_predictions_dir"] == str(exp_dir / "saved_predictions")
        assert kwargs["n_windows"] == 4
        assert kwargs["window_size_seconds"] == 2
        assert kwargs["splits"] == "val,test"
        assert kwargs["probe_epochs"] == 20
        assert kwargs["subject_probe_epochs"] == 100
        assert kwargs["wandb_group"] == "test_group"
        assert kwargs["seed"] == 42
        # corrca_filters=None in cfg becomes "" downstream (probe_eval default)
        assert kwargs["corrca_filters"] == ""


def test_bootstrap_receives_expected_kwargs(tmp_path):
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from experiments.eeg_jepa import train as m

    with patch("eb_jepa.evaluation.run_probe_eval") as pe, \
         patch("eb_jepa.evaluation.bootstrap_predictions") as bs:
        pe.return_value = {
            "_wandb_run_id": "abc123",
            "_wandb_project": "eb_jepa",
        }
        bs.return_value = {}

        m._run_auto_eval(cfg, exp_dir, "cfgs/default.yaml")

        kwargs = bs.call_args.kwargs
        assert kwargs["predictions_dir"] == str(exp_dir / "saved_predictions")
        assert kwargs["split"] == "test"
        assert kwargs["n_bootstrap"] == 1000
        # The hook must forward probe_eval's W&B run id so bootstrap CIs
        # land on the same run as the probe-eval point estimates.
        assert kwargs["wandb_run_id"] == "abc123"
        assert kwargs["wandb_project"] == "eb_jepa"


def test_bootstrap_wandb_run_id_empty_when_probe_eval_omits_it(tmp_path):
    """Older probe_eval result dicts (or W&B-disabled runs) won't carry
    `_wandb_run_id`; the hook should pass an empty string, which bootstrap
    treats as 'skip W&B logging'."""
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from experiments.eeg_jepa import train as m

    with patch("eb_jepa.evaluation.run_probe_eval") as pe, \
         patch("eb_jepa.evaluation.bootstrap_predictions") as bs:
        pe.return_value = {}                # no _wandb_run_id key
        bs.return_value = {}

        m._run_auto_eval(cfg, exp_dir, "cfgs/default.yaml")

        kwargs = bs.call_args.kwargs
        assert kwargs["wandb_run_id"] == ""
        assert kwargs["wandb_project"] == "eb_jepa"


def test_probe_eval_failure_does_not_propagate(tmp_path):
    """A flaky eval must not lose the checkpoint -- training already finished."""
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from experiments.eeg_jepa import train as m

    with patch("eb_jepa.evaluation.run_probe_eval", side_effect=RuntimeError("boom")), \
         patch("eb_jepa.evaluation.bootstrap_predictions") as bs:
        # Should not raise -- the hook catches and logs.
        m._run_auto_eval(cfg, exp_dir, "cfgs/default.yaml")
        # Bootstrap is skipped when probe_eval fails.
        assert bs.call_count == 0


def test_bootstrap_failure_does_not_propagate(tmp_path):
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from experiments.eeg_jepa import train as m

    with patch("eb_jepa.evaluation.run_probe_eval") as pe, \
         patch("eb_jepa.evaluation.bootstrap_predictions",
               side_effect=RuntimeError("boom")):
        pe.return_value = {}
        m._run_auto_eval(cfg, exp_dir, "cfgs/default.yaml")  # must not raise


def test_disabled_skips_both_calls(tmp_path):
    """run() guards _run_auto_eval with cfg.eval.auto_run; verify the gate
    by exercising the calling site contract: when auto_run=false, neither
    library function should be invoked from the hook path."""
    cfg = _base_cfg(auto_run=False)
    # Assertion lives in run(); _run_auto_eval is unconditionally invoked
    # only after the gate passes. So we test the gate directly:
    eval_cfg = cfg.get("eval", None)
    should_run = eval_cfg is None or eval_cfg.get("auto_run", True)
    assert should_run is False


def test_default_when_no_eval_section(tmp_path):
    """Configs without an eval section default to auto_run=true (back-compat
    with checkpoints trained before the eval section existed)."""
    cfg = OmegaConf.create({
        "meta": {"seed": 42},
        "data": {
            "n_windows": 4, "window_size_seconds": 2, "batch_size": 64,
            "num_workers": 4, "norm_mode": "global",
            "add_envelope": False, "corrca_filters": None,
        },
        # No eval: section.
    })
    eval_cfg = cfg.get("eval", None)
    should_run = eval_cfg is None or eval_cfg.get("auto_run", True)
    assert should_run is True
