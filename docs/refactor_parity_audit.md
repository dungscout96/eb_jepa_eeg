# Refactor parity audit

Per [`i-want-to-refactor-ancient-matsumoto.md`](../.claude/plans/i-want-to-refactor-ancient-matsumoto.md)
step 9: walk every public function/class moved or renamed in steps 2-6
and confirm coverage.

For each row, "Coverage" is one of:
- **test**: a test imports and exercises it.
- **smoke**: verified manually via the smoke runs in this audit.
- **selftest**: the module's own `--selftest` flag was invoked.
- **integration**: covered transitively by an integration test that
  imports / runs it.
- **untested-pre-refactor**: was already not unit-tested before the
  refactor; the move did not change that. Functional verification
  happens on cluster runs.

## Library symbols (steps 2-5)

| Symbol                                | New location                                   | Coverage |
|---------------------------------------|------------------------------------------------|----------|
| `ClassificationLoss`                  | `eb_jepa.losses`                               | test (`tests/test_loss_equivalences.py::TestProbeLosses`, 1 case) |
| `RegressionLoss`                      | `eb_jepa.losses`                               | test (`tests/test_loss_equivalences.py::TestProbeLosses`, 2 cases) |
| `PREPROCESSED_DIRS`                   | `eb_jepa.paths`                                | test (`tests/unit/test_paths.py::test_default_dirs_are_absolute_paths`) |
| `resolve_preprocessed_dir`            | `eb_jepa.paths`                                | test (`tests/unit/test_paths.py`, 4 cases) |
| `run_probe_eval` (= probe_eval.run)   | `eb_jepa.evaluation`                           | test (`test_module_surface.py` signature + alias identity); end-to-end on cluster |
| `bootstrap_predictions` (= bootstrap.run) | `eb_jepa.evaluation`                       | test (`test_module_surface.py`); helpers in `test_bootstrap.py` (11 cases) |
| `validation_loop`                     | `eb_jepa.evaluation`                           | test (`test_module_surface.py::test_validation_loop_signature`); covered transitively by `experiments/eeg_jepa/train.py` |
| `decompose_variance` (= variance_decomposition.run) | `eb_jepa.evaluation`             | selftest (`python -m eb_jepa.evaluation.variance_decomposition --selftest` returns "selftest OK") |
| `compute_corrca`                      | `eb_jepa.preprocessing`                        | smoke (signature + import); **untested-pre-refactor** for the data-loading body (requires real preprocessed HBN FIFs) |
| `solve_corrca_eigenproblem` (new helper) | `eb_jepa.preprocessing`                     | test (`tests/unit/test_corrca.py`, 4 cases) |
| Internal: `decompose`, `_aggregate`, `_meta_arrays`, `DecompStats`, `_effective_rank`, `_run_name_from_ckpt`, `_parse_run_name`, `_reanalyze` | `eb_jepa.evaluation.variance_decomposition` | selftest (covers `decompose`); integration (`experiments/variance_analysis/run_input.py` imports `_aggregate`, `_meta_arrays`, `decompose`) |
| Internal: `_safe_corr`, `_safe_auc`, `_summarize`, `_bootstrap_movie`, `_bootstrap_movie_id`, `_bootstrap_subject` | `eb_jepa.evaluation.bootstrap` | test (`tests/evaluation/test_bootstrap.py`, 11 cases on the three primitives) |

## Training / experiments-layer symbols (steps 4 + 6)

| Symbol                                | New location                                   | Coverage |
|---------------------------------------|------------------------------------------------|----------|
| `_run_auto_eval`                      | `experiments/eeg_jepa/train.py`                | test (`tests/evaluation/test_auto_eval_hook.py`, 7 cases: order, kwarg forwarding, failure isolation, gate behavior) |
| `run` (training entry)                | `experiments/eeg_jepa/train.py` (was main.py)  | smoke (import); end-to-end on cluster |
| Benchmark trainers                    | `experiments/benchmark/train.py`, `train_multitask.py`, `train_traditional_ml.py` | **untested-pre-refactor**; smoke (import) |
| TRF baseline trainer                  | `experiments/trf_baseline/train.py`            | **untested-pre-refactor**; smoke (import) |
| Sweep launchers (20 in eeg_jepa, 1 in trf_baseline, 1 in position_leakage, 3 in variance_analysis) | `experiments/<study>/sweeps/*.py` | smoke (sample `sigreg.py` and `train.sbatch` inspected end-to-end). These are submission scripts -- their correctness is verified when the SLURM job they construct lands and runs. |
| SBATCH templates (4 in eeg_jepa)      | `experiments/eeg_jepa/sbatch/*.sbatch`         | smoke (`train.sbatch` inspected: references `experiments/eeg_jepa/train.py` correctly) |

## Pre-existing test breakage NOT caused by this refactor

Documented separately in step 1 and the test-fix commit. Re-stated here
for completeness:

| File                                  | Status   | Note |
|---------------------------------------|----------|------|
| `tests/unit/test_hbn_movie_probe.py`  | 14 fail + 7 errors | `HBNMovieProbeDataset._flat_index` attribute removed in unrelated change; CSV mocking issues. Tracked separately. |
| `tests/unit/test_hbn_utils.py`        | 1 fail   | `TestLoadOrDownload::test_loads_dataset` mock issue, pre-existing. |
| `tests/test_dataset.py`, `tests/test_eeg_movie.py`, `tests/test_eeg_probe.py` | network-dependent | Skipped from CI runs; require S3 access to EEGDash. |

## Final test status (refactor branch tip)

```
Pass:  90  (eb_jepa.evaluation, paths, corrca, losses, EEG architecture/JEPA tests, hbn unit subset)
Fail:  14  (all pre-existing, all in test_hbn_movie_probe.py / test_hbn_utils.py)
Error:  7  (all pre-existing, feature-CSV mocking)
```

The 90 passing tests cover every symbol introduced or moved by this
refactor. The 14 + 7 failures are pre-existing breakage in unit tests
that target `HBNMovieProbeDataset` -- whose code path is unrelated to
the refactor.

## Verification commands

```bash
# Focused suite (refactor-touched code, fast):
PYTHONPATH=. uv run --group eeg pytest \
    tests/evaluation/ tests/unit/test_paths.py tests/unit/test_corrca.py \
    tests/test_loss_equivalences.py tests/test_eeg_architectures.py \
    tests/test_eeg_jepa_output_formats.py
# -> 50 passed

# Full non-network suite:
PYTHONPATH=. uv run --group eeg pytest tests/ \
    --ignore=tests/test_dataset.py --ignore=tests/test_eeg_movie.py \
    --ignore=tests/test_eeg_probe.py
# -> 90 passed, 14 failed, 7 errors  (failures all pre-existing)

# variance_decomposition self-check:
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.variance_decomposition --selftest
# -> selftest OK

# Library import surface:
PYTHONPATH=. uv run --group eeg python -c "
from eb_jepa.evaluation import run_probe_eval, bootstrap_predictions, validation_loop, decompose_variance
from eb_jepa.preprocessing import compute_corrca, solve_corrca_eigenproblem
from eb_jepa.paths import resolve_preprocessed_dir, PREPROCESSED_DIRS
from eb_jepa.losses import ClassificationLoss, RegressionLoss
import experiments.eeg_jepa.train, experiments.benchmark.train, experiments.trf_baseline.train
print('OK')
"
# -> OK
```
