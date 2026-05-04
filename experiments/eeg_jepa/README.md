# EEG JEPA pretraining

Self-supervised V-JEPA-style masked prediction for EEG during the HBN
movie-watching task. The main study in this repo.

## Entry point

```bash
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/train.py
```

CLI overrides use OmegaConf dot syntax:

```bash
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/train.py \
    --optim.lr=5e-4 --data.n_windows=4 --data.window_size_seconds=4
```

Default config: [`cfgs/default.yaml`](cfgs/default.yaml).

## Architecture

- **Encoder (`EEGEncoderTokens`)** -- REVE-style transformer over EEG
  patch tokens with 4D Fourier positional encoding (channel xyz + time).
- **Predictor (`MaskedPredictor`)** -- masked-token predictor in
  representation space (V-JEPA style).
- **Regularizer** -- `VCLoss` (variance/covariance, default) or `SIGRegLoss`.
- **Online probes (`MovieFeatureHead`)** -- regression + classification
  on movie features (luminance, contrast, position-in-movie,
  narrative-event-score) trained jointly with JEPA for in-loop
  diagnostics.

## Post-training auto-eval

End-of-training calls `eb_jepa.evaluation.run_probe_eval` on the saved
checkpoint, then `eb_jepa.evaluation.bootstrap_predictions` on the
dumped per-clip predictions. Configured via the `eval:` section in
[`cfgs/default.yaml`](cfgs/default.yaml). Disable for fast smoke runs:

```bash
... train.py --eval.auto_run=false
```

## Sweeps

Each launcher in [`sweeps/`](sweeps/) is a self-contained submission
script that builds N command lines and submits them via neurolab.
Group examples:

| Sweep file                           | What it tries                                                  |
|--------------------------------------|----------------------------------------------------------------|
| `phase1.py`, `phase1_resume.py`      | Temporal config sweep (n_windows x window_size x 3 seeds)      |
| `sigreg.py`                          | SIGReg coefficient sweep on the Phase 1 winners                |
| `vicreg.py`, `vicreg_noproj.py`      | VICReg coefficient sweep, with and without projector           |
| `retrain_best.py`, `retrain_best_perrec.py` | Re-train the best Phase 1 configs with new seeds / per-rec norm |
| `sigreg_corrca.py`                   | SIGReg + CorrCA spatial filtering combination                  |
| `trivial_baseline_raw.py`, `trivial_shuffle_and_aligned.py` | Trivial baselines for sanity checks  |
| `probe_eval_*.py`                    | Submit `eb_jepa.evaluation.probe_eval` for a list of checkpoints from a previous sweep |

Direct SBATCH templates are in [`sbatch/`](sbatch/) for one-off runs
that don't need the python launcher (e.g. Phase D diagnostic checkpoints).

## Outputs

Training writes to `outputs/eeg_jepa/<exp_name>/`:

- `latest.pth.tar`, `epoch_<N>.pth.tar` -- checkpoints
- `saved_predictions/<split>.npz` -- per-clip probe predictions (auto-eval)
- W&B run logged to `WANDB_PROJECT` (default `eb_jepa`)
