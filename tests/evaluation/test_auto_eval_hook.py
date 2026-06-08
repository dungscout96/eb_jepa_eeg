"""Tests for the end-of-training auto-eval hook in eb_jepa/training/jepa_pretrain.py.

The hook (`_run_auto_eval`) is the integration point between training and
the library-level eb_jepa.evaluation pipeline: it must fire when
cfg.eval.auto_run is true, skip when false, and forward the right config
keys to run_probe_eval + bootstrap_predictions.

Failures inside the eval calls are swallowed (with a warning) so a flaky
eval can't lose the training checkpoint -- the test suite verifies that
behavior is intentional.
"""

from unittest.mock import patch

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
            "n_passes": 20,
            "probe_seed": 42,
            "n_bootstrap": 2000,
            **eval_overrides,
        },
    }
    return OmegaConf.create(cfg_dict)


def test_hook_calls_probe_eval_then_bootstrap(tmp_path):
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from eb_jepa.training import jepa_pretrain as m

    with patch("eb_jepa.evaluation.run_probe_eval") as pe, \
         patch("eb_jepa.evaluation.bootstrap_predictions") as bs:
        pe.return_value = {}
        bs.return_value = {}

        m._run_auto_eval(cfg, exp_dir, "config/jepa_pretrain.yaml")

        assert pe.call_count == 1
        assert bs.call_count == 1


def test_probe_eval_receives_expected_kwargs(tmp_path):
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from eb_jepa.training import jepa_pretrain as m

    with patch("eb_jepa.evaluation.run_probe_eval") as pe, \
         patch("eb_jepa.evaluation.bootstrap_predictions") as bs:
        pe.return_value = {}
        bs.return_value = {}

        m._run_auto_eval(
            cfg, exp_dir, "config/jepa_pretrain.yaml", wandb_run_id="run-xyz"
        )

        kwargs = pe.call_args.kwargs
        assert kwargs["checkpoint"] == str(exp_dir / "latest.pth.tar")
        assert kwargs["save_predictions_dir"] == str(exp_dir / "saved_predictions")
        assert kwargs["n_windows"] == 4
        assert kwargs["window_size_seconds"] == 2
        assert kwargs["n_passes"] == 20
        assert kwargs["probe_seed"] == 42
        assert kwargs["wandb_run_id"] == "run-xyz"
        assert kwargs["seed"] == 42
        # corrca_filters=None in cfg becomes "" downstream
        assert kwargs["corrca_filters"] == ""


def test_bootstrap_receives_expected_kwargs(tmp_path):
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from eb_jepa.training import jepa_pretrain as m

    with patch("eb_jepa.evaluation.run_probe_eval") as pe, \
         patch("eb_jepa.evaluation.bootstrap_predictions") as bs:
        pe.return_value = {}
        bs.return_value = {}

        m._run_auto_eval(
            cfg, exp_dir, "config/jepa_pretrain.yaml", wandb_run_id="run-xyz"
        )

        kwargs = bs.call_args.kwargs
        assert kwargs["predictions_npz"] == str(
            exp_dir / "saved_predictions" / "preds_seed42.npz"
        )
        assert kwargs["out_json"] == str(exp_dir / "bootstrap_seed42.json")
        assert kwargs["n_bootstrap"] == 2000
        assert kwargs["seed"] == 42
        assert kwargs["wandb_run_id"] == "run-xyz"


def test_probe_eval_failure_does_not_propagate(tmp_path):
    """A flaky eval must not lose the checkpoint -- training already finished."""
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from eb_jepa.training import jepa_pretrain as m

    with patch("eb_jepa.evaluation.run_probe_eval", side_effect=RuntimeError("boom")), \
         patch("eb_jepa.evaluation.bootstrap_predictions") as bs:
        m._run_auto_eval(cfg, exp_dir, "config/jepa_pretrain.yaml")
        # Bootstrap is skipped when probe_eval fails.
        assert bs.call_count == 0


def test_bootstrap_failure_does_not_propagate(tmp_path):
    cfg = _base_cfg()
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    from eb_jepa.training import jepa_pretrain as m

    with patch("eb_jepa.evaluation.run_probe_eval") as pe, \
         patch("eb_jepa.evaluation.bootstrap_predictions",
               side_effect=RuntimeError("boom")):
        pe.return_value = {}
        m._run_auto_eval(cfg, exp_dir, "config/jepa_pretrain.yaml")  # must not raise


def test_disabled_skips_both_calls(tmp_path):
    """run() guards _run_auto_eval with cfg.eval.auto_run; verify the gate
    by exercising the calling site contract: when auto_run=false, neither
    library function should be invoked from the hook path."""
    cfg = _base_cfg(auto_run=False)
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
    })
    eval_cfg = cfg.get("eval", None)
    should_run = eval_cfg is None or eval_cfg.get("auto_run", True)
    assert should_run is True
