# Experiments

Each subfolder is a self-contained study with its own entry point,
sweep launchers, sbatch files, and (for non-trivial studies) configs.

| Study                  | What it tests                                                                                  | Entry point                                       |
|------------------------|------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`eeg_jepa/`](eeg_jepa/)             | Self-supervised V-JEPA-style masked prediction on HBN movie-watching EEG. The main study.       | `experiments/eeg_jepa/train.py`                  |
| [`trf_baseline/`](trf_baseline/)     | Supervised Braindecode Transformer baseline on the same movie-feature targets.                  | `experiments/trf_baseline/train.py`              |
| [`benchmark/`](benchmark/)           | Supervised baselines (EEGNet, REVE, BIOT, classical ML) for movie-feature decoding.             | `experiments/benchmark/train.py`                 |
| [`position_leakage/`](position_leakage/) | Diagnostic: does the encoder leak time-in-movie? Trivial baselines for upper-bound calibration. | `experiments/position_leakage/sweeps/run_delta.py` |
| [`variance_analysis/`](variance_analysis/) | Per-checkpoint variance decomposition (subject / stimulus / residual) and predictability.      | `experiments/variance_analysis/run_input.py`     |

## Conventions

Every study folder has the same shape:

```
experiments/<study>/
  train.py        (or run.py for analyses)   -- the entry point a sweep submits
  cfgs/                                       -- yaml configs (where applicable)
  sweeps/                                     -- launcher scripts (one per sweep)
  sbatch/                                     -- raw SBATCH templates for direct submission
  README.md
```

## Post-training validation

All training entry points (`eeg_jepa/train.py`, `trf_baseline/train.py`,
`benchmark/train.py`) auto-invoke `eb_jepa.evaluation.run_probe_eval` and
`eb_jepa.evaluation.bootstrap_predictions` at end of training when
`cfg.eval.auto_run=true` (default). Disable for fast smoke runs by
overriding `--eval.auto_run=false`.

## Cluster wrappers

Cluster + data utilities live in [`scripts/`](../scripts/):

- `preprocess_hbn.py` -- raw HBN -> .fif preprocessing (run once)
- `compute_corrca.py` -- thin CLI over `eb_jepa.preprocessing.compute_corrca` (run once before CorrCA-conditioned training)
- `submit_job_{delta,expanse,jamming}.py` -- generic neurolab job submitters
- `pull_wandb.py`, `extract_*_results.py` -- W&B aggregation helpers
