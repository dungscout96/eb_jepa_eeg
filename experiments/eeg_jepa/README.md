# EEG JEPA pretraining — LEGACY (frozen 2026-05-16)

> **This folder is preserved as a frozen reproducibility snapshot of the
> pre-refactor pipeline.** Do not extend it. The training entry, default
> config, and canonical sbatches have been promoted to
> [`eb_jepa/training/`](../../eb_jepa/training/), and the sweeps have
> been regrouped into self-contained studies under `experiments/`
> (see [`experiments/README.md`](../README.md)).
>
> Why it's still here: every sweep launcher under `sweeps/` invokes
> `experiments/eeg_jepa/train.py` directly, so leaving the directory
> untouched means any pre-refactor sweep run remains reproducible
> against the exact code that produced it.
>
> For new work, use:
> - `python -m eb_jepa.training.jepa_pretrain` instead of `experiments/eeg_jepa/train.py`
> - `experiments/<study>/` (canonical_replication, temporal_sweep,
>   regularizer_study, retrain_best, trivial_baselines, corrca_study)
>   instead of `experiments/eeg_jepa/sweeps/`

---

Self-supervised V-JEPA-style masked prediction for EEG during the HBN
movie-watching task. The main study in this repo (pre-refactor entry).

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
