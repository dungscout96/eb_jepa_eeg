"""Surface tests for eb_jepa.evaluation.

probe_eval and bootstrap need a real checkpoint and dataset to exercise
end-to-end — those run on the cluster, not in CI — so this file just
guards the public Python surface and CLI signatures the rest of the
codebase depends on.
"""

import inspect


def test_top_level_surface_is_stable():
    """The names jepa_pretrain auto-eval and sweep launchers import must
    continue to be exposed."""
    from eb_jepa.evaluation import (  # noqa: F401
        bootstrap_predictions,
        run_probe_eval,
    )


def test_probe_eval_run_signature():
    """The fire CLI surface (`run`) is what every sweep launcher calls.
    These keyword names are wire-format and must not regress."""
    from eb_jepa.evaluation.probe_eval import run

    sig = inspect.signature(run)
    required = {
        "checkpoint", "n_windows", "window_size_seconds", "batch_size",
        "num_workers", "norm_mode", "corrca_filters", "add_envelope",
        "n_passes", "probe_seed", "save_predictions_dir",
        "wandb_run_id", "wandb_project", "wandb_group", "fname", "seed",
    }
    missing = required - set(sig.parameters)
    assert not missing, f"probe_eval.run missing params: {sorted(missing)}"


def test_bootstrap_run_signature():
    from eb_jepa.evaluation.bootstrap import run

    sig = inspect.signature(run)
    required = {
        "predictions_npz", "out_json", "n_bootstrap", "seed",
        "wandb_run_id", "wandb_project",
    }
    missing = required - set(sig.parameters)
    assert not missing, f"bootstrap.run missing params: {sorted(missing)}"


def test_module_cli_entrypoints_runnable():
    """Both submodules expose `__main__` blocks so they can be invoked as
    `python -m eb_jepa.evaluation.{probe_eval,bootstrap}`."""
    import eb_jepa.evaluation.bootstrap as bs
    import eb_jepa.evaluation.probe_eval as pe

    assert "fire.Fire(run)" in inspect.getsource(pe)
    assert "fire.Fire(run)" in inspect.getsource(bs)
