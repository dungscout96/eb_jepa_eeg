# EEG JEPA

Self-supervised V-JEPA-style masked prediction for EEG, applied to the
Healthy Brain Network (HBN) movie-watching task. The library trains a
masked Joint-Embedding Predictive Architecture on raw EEG, and the
post-training pipeline measures what the encoder actually learned via
linear probes (movie features + subject traits) with bootstrap CIs over
recordings.

This repo started as a fork of [Meta's eb_jepa](https://github.com/facebookresearch/eb_jepa)
(originally a CV / planning library) and has been adapted for EEG. The
upstream image / video / planning code paths have been removed; see the
[refactor plan](.claude/plans/i-want-to-refactor-ancient-matsumoto.md)
for the full transition.

## Layout

```
eb_jepa/                       # the library
  architectures.py             # REVE backbone, EEGEncoderTokens, MaskedPredictor, heads
  jepa.py                      # MaskedJEPA, MaskedJEPAProbe, JEPA
  losses.py                    # VCLoss, SIGRegLoss, ClassificationLoss, RegressionLoss
  masking.py                   # MultiBlockMaskCollator (V-JEPA style 2D channel x patch masks)
  sanity_checks.py             # SanityCheckHook (collapse + linear-probe diagnostics)
  paths.py                     # cluster-aware preprocessed-dir resolver
  datasets/hbn.py              # JEPAMovieDataset, HBNMovieProbeDataset
  preprocessing/corrca.py      # CorrCA spatial filter computation
  evaluation/                  # post-training pipeline
    probe_eval.py              # frozen-encoder probes (movie features + subject traits)
    bootstrap.py               # recording-level bootstrap CIs over saved predictions
    validation_loop.py         # in-loop val metrics during training
    variance_decomposition.py  # subject / stimulus / residual decomposition

experiments/                   # one folder per study
  eeg_jepa/                    # the main JEPA pretraining study
  trf_baseline/                # supervised TRF baseline
  benchmark/                   # EEGNet / REVE / BIOT / classical ML baselines
  position_leakage/            # diagnostic: does the encoder leak time-in-movie?
  variance_analysis/           # per-checkpoint variance & predictability decomposition

scripts/                       # cluster + data utilities (not study-specific)
  preprocess_hbn.py / .sbatch  # raw HBN -> .fif preprocessing (run once)
  compute_corrca.py / .sbatch  # thin CLI over eb_jepa.preprocessing.compute_corrca
  submit_job_{delta,expanse,jamming}.py
  pull_wandb.py, extract_*_results.py

tests/                         # pytest
```

See [`experiments/README.md`](experiments/README.md) for the per-study
index, and each study's own README for details.

## Install

We use [uv](https://docs.astral.sh/uv/guides/projects/) for package management.

```bash
uv sync
# EEG-specific dependencies (braindecode, eegdash, neurolab) live in the `eeg` group:
uv sync --group eeg
```

For development (pytest, black, isort, autoflake):

```bash
uv sync --group dev
```

## Quick start

```bash
# 1. Preprocess HBN once (downloads + windows the data; long-running).
PYTHONPATH=. uv run --group eeg python scripts/preprocess_hbn.py

# 2. Train a JEPA model with default config.
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/train.py

# 3. End-of-training auto-runs probe_eval + bootstrap on the saved
#    checkpoint (gated by cfg.eval.auto_run, default true). To run
#    manually on an existing checkpoint:
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.probe_eval \
    --checkpoint=/path/to/latest.pth.tar
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.bootstrap \
    --predictions_dir=/path/to/saved_predictions --split=test
```

CLI overrides use OmegaConf dot syntax (`--optim.lr=5e-4`,
`--data.n_windows=4`, `--eval.auto_run=false`, etc.).

## Cluster

- **Delta** (NCSA): SLURM, see [`scripts/submit_job_delta.py`](scripts/submit_job_delta.py)
  and study-specific `sbatch/` directories under `experiments/`.
- **Expanse** (SDSC): [`scripts/submit_job_expanse.py`](scripts/submit_job_expanse.py)
- **Jamming** (local workstation): [`scripts/submit_job_jamming.py`](scripts/submit_job_jamming.py),
  no SLURM, direct execution.

Sweep launchers under `experiments/<study>/sweeps/` are thin wrappers
around `neurolab.jobs.Job`; they construct command lines and submit
batches.

## Tests

```bash
uv run --group eeg pytest tests/
```

The fast focused suite (`tests/evaluation/`, `tests/unit/`,
`tests/test_loss_equivalences.py`) runs in seconds and does not require
any cluster fixtures or downloaded data.

## Development

Before contributing, format with:

```bash
autoflake --remove-all-unused-imports -r --in-place .
python -m isort eb_jepa experiments tests
python -m black eb_jepa experiments tests
```

## Citing the upstream library

This work builds on Meta's eb_jepa scaffolding:

```bibtex
@misc{terver2026lightweightlibraryenergybasedjointembedding,
      title={A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures},
      author={Basile Terver and Randall Balestriero and Megi Dervishi and David Fan and Quentin Garrido and Tushar Nagarajan and Koustuv Sinha and Wancong Zhang and Mike Rabbat and Yann LeCun and Amir Bar},
      year={2026},
      eprint={2602.03604},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03604},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE.md).
