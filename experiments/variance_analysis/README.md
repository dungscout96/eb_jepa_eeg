# Variance analysis

Per-checkpoint variance decomposition (subject / stimulus / residual)
and input-space predictability. The geometric counterpart to probe_eval:
shows where the encoded signal lives in representation space, even when
a linear probe can't lift it.

## Entry points

| Script                      | Purpose                                                      |
|-----------------------------|--------------------------------------------------------------|
| `python -m eb_jepa.evaluation.variance_decomposition`         | Encoder-output variance decomposition. The core analysis -- lives in the library because every study uses it. |
| [`run_input.py`](run_input.py)               | Input-space (pre-encoder) variance + predictability decomposition (RMS, band-power, alpha-phase features) |
| [`diagnose.py`](diagnose.py)                 | Embedding diagnostic plots (t-SNE, PCA, covariance structure) |
| [`sweeps/run_delta.py`](sweeps/run_delta.py) | Per-checkpoint variance decomposition sweep (Delta)           |
| [`sweeps/run_corrca_delta.py`](sweeps/run_corrca_delta.py) | Same, restricted to CorrCA-trained checkpoints              |
| [`sweeps/run_input_delta.py`](sweeps/run_input_delta.py)   | Input-space sweep (4 conditions: raw/per-rec x with/without CorrCA) |

## Quick run

```bash
# Single checkpoint:
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.variance_decomposition \
    --checkpoint=/path/to/latest.pth.tar \
    --n_windows=4 --window_size_seconds=4 --n_clips_per_rec=32 \
    --split=val --output_dir=outputs/variance_decomp

# Aggregate per-checkpoint outputs into a comparison table:
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.variance_decomposition \
    --aggregate_dir=outputs/variance_decomp

# Self-test (no real data):
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.variance_decomposition \
    --selftest
```

## Why this study exists

A high `eta_subject` with low `eta_stimulus` means the encoder is
learning subject fingerprints rather than stimulus features, even when
the stimulus probe corr looks acceptable on val. The variance
decomposition catches this geometrically; probe_eval catches it
behaviorally. They should agree -- when they don't, see
[../../docs/variance_decomposition_report.md](../../docs/variance_decomposition_report.md).
