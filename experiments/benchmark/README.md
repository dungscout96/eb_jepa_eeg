# Supervised benchmarks

Supervised baselines (EEGNet, REVE, BIOT, classical ML) for movie-feature
decoding from HBN EEG. Used as comparison points for JEPA probe results.

## Entry points

| Script                    | What it trains                                                          |
|---------------------------|-------------------------------------------------------------------------|
| [`train.py`](train.py)                          | EEGNet / REVE / BIOT (single-task, one model per feature) |
| [`train_multitask.py`](train_multitask.py)      | Multi-task variant (all movie features predicted jointly)  |
| [`train_traditional_ml.py`](train_traditional_ml.py) | Classical ML (LogReg, SVM, RF, GB) on flattened / PCA-reduced EEG |
| [`generate_report.py`](generate_report.py)      | Aggregate saved benchmark JSON into TSV / plots             |

## Run

```bash
PYTHONPATH=. uv run --group eeg python experiments/benchmark/train.py
```

Hyperparameter grid search and CV folds are built into each train
script. See [`sweeps/`](sweeps/) for cluster submission wrappers (none
yet -- launch directly via the train scripts when first set up on a
new cluster).

## Outputs

`outputs/benchmark/<exp_name>/all_results.json` is the input to
`generate_report.py`.
