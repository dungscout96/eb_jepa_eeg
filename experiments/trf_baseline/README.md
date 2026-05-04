# TRF baseline

Supervised Braindecode Transformer (TRF) baseline trained directly on
movie-feature regression + classification. Used as the ceiling for what
a fully-supervised model can extract from the same EEG inputs that
JEPA sees.

## Entry point

```bash
PYTHONPATH=. uv run --group eeg python experiments/trf_baseline/train.py
```

Hyperparameter sweep is built into the script (hidden_dim, n_layers,
lr, batch_size). For one-off jamming submission see
[`sweeps/submit_jamming.py`](sweeps/submit_jamming.py).

## Sanity check

[`sanity.py`](sanity.py) runs collapse / gradient / linear-probe checks
on a TRF checkpoint -- mirrors the SanityCheckHook used during JEPA
training.

## Notes

This study does not use `eb_jepa.evaluation.run_probe_eval` (the model
is already supervised on the same targets a probe would predict). The
TRF metrics serve as a comparison ceiling for JEPA probe results.
