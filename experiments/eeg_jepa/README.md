# EEG JEPA pretraining

Self-supervised V-JEPA-style masked prediction for EEG during the HBN
movie-watching task.

Entry point: `experiments/eeg_jepa/train.py` (fire CLI).
Default config: `cfgs/default.yaml`.

## Run

```bash
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/train.py
```

CLI overrides use OmegaConf dot syntax, e.g.:

```bash
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/train.py \
    --optim.lr=5e-4 --data.n_windows=4 --data.window_size_seconds=4
```

## Components

- **Encoder (`EEGEncoderTokens`)** — REVE-style transformer over EEG patch
  tokens with 4D Fourier positional encoding (channel xyz + time).
- **Predictor (`MaskedPredictor`)** — masked-token predictor in
  representation space (V-JEPA style).
- **Regularizer** — `VCLoss` (variance/covariance, default) or `SIGRegLoss`.
- **Probes** — `MovieFeatureHead` trained alongside JEPA for online
  regression/classification on movie features (luminance, contrast,
  position-in-movie, narrative-event-score).

## Post-training validation

End-of-training auto-runs library-level probe eval and bootstrap CI on
the saved checkpoint. See `eb_jepa.evaluation.probe_eval` and
`eb_jepa.evaluation.bootstrap`.

> README will be expanded in the per-study refactor pass (see plan
> `i-want-to-refactor-ancient-matsumoto.md`, step 7).
