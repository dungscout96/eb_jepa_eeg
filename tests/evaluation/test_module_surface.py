"""Surface tests for eb_jepa.evaluation.

Probe_eval and validation_loop need a real checkpoint and dataset to
exercise end-to-end -- those run on the cluster, not in CI -- so this
file just guards the public Python surface and CLI signatures the rest
of the codebase depends on.
"""

import inspect


def test_top_level_surface_is_stable():
    """The three names experiments/eeg_jepa/main.py and sweep launchers
    import must continue to be exposed."""
    from eb_jepa.evaluation import (  # noqa: F401
        bootstrap_predictions,
        run_probe_eval,
        validation_loop,
    )


def test_probe_eval_run_signature():
    """The fire CLI surface (`run`) is what every sweep launcher calls.
    These keyword names are wire-format and must not regress."""
    from eb_jepa.evaluation.probe_eval import run

    sig = inspect.signature(run)
    required = {
        "checkpoint", "n_windows", "window_size_seconds", "batch_size",
        "num_workers", "probe_epochs", "splits", "save_predictions_dir",
        "wandb_run_id", "wandb_project", "wandb_group", "fname", "seed",
    }
    missing = required - set(sig.parameters)
    assert not missing, f"probe_eval.run missing params: {sorted(missing)}"


def test_bootstrap_run_signature():
    from eb_jepa.evaluation.bootstrap import run

    sig = inspect.signature(run)
    required = {"predictions_dir", "split", "n_bootstrap"}
    missing = required - set(sig.parameters)
    assert not missing, f"bootstrap.run missing params: {sorted(missing)}"


def test_validation_loop_signature():
    from eb_jepa.evaluation import validation_loop

    sig = inspect.signature(validation_loop)
    required = {
        "val_loader", "jepa", "regression_probe", "classification_probe",
        "device", "feature_stats", "feature_median", "feature_names",
    }
    missing = required - set(sig.parameters)
    assert not missing, f"validation_loop missing params: {sorted(missing)}"


def test_module_cli_entrypoints_runnable():
    """Both submodules expose `__main__` blocks so they can be invoked as
    `python -m eb_jepa.evaluation.{probe_eval,bootstrap}`."""
    import eb_jepa.evaluation.bootstrap as bs
    import eb_jepa.evaluation.probe_eval as pe

    # Source must contain the fire dispatch -- guards against accidental
    # removal during refactors.
    assert "fire.Fire(run)" in inspect.getsource(pe)
    assert "fire.Fire(run)" in inspect.getsource(bs)
