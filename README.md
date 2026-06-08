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
  anti_collapse.py             # AntiCollapse: DINO / VICReg / SIGReg strategies
  architectures.py             # REVE backbone, EEGEncoderTokens, MaskedPredictor, heads
  jepa.py                      # MaskedJEPA, MaskedJEPAProbe, JEPA
  losses.py                    # VCLoss, SIGRegLoss, ClassificationLoss, RegressionLoss
  masking.py                   # MultiBlockMaskCollator (V-JEPA style 2D channel x patch masks)
  sanity_checks.py             # SanityCheckHook (collapse + linear-probe diagnostics)
  paths.py                     # cluster-aware preprocessed-dir resolver
  datasets/hbn.py              # JEPAMovieDataset, HBNMovieDataset, HBNDataset
  preprocessing/corrca.py      # CorrCA spatial filter computation
  training/
    jepa_pretrain.py           # canonical pretraining entry point (Fire CLI)
    builder.py                 # shared MaskedJEPA builder
  evaluation/                  # post-training pipeline
    probe_eval.py              # frozen-encoder probes (movie features + subject traits)
    probe_eval_canonical.py    # spec-faithful sklearn-probe variant
    bootstrap.py               # recording-level bootstrap CIs over saved predictions
    validation_loop.py         # in-loop val metrics during training
    variance_decomposition.py  # subject / stimulus / residual decomposition

config/                        # OmegaConf / Hydra configs
  jepa_pretrain.yaml           # JEPA pretraining defaults
  preprocess_hbn.yaml          # HBN preprocessing defaults
  benchmark.yaml               # benchmark experiment defaults

experiments/                   # one folder per study
  eeg_jepa/                    # FROZEN legacy snapshot of the original entry point
  trf_baseline/                # supervised TRF baseline
  benchmark/                   # EEGNet / REVE / BIOT / classical ML baselines
  position_leakage/            # diagnostic: does the encoder leak time-in-movie?
  variance_analysis/           # per-checkpoint variance & predictability decomposition
  regularizer_study/, retrain_best/, temporal_sweep/   # ablation sweeps

scripts/                       # cluster + data utilities (not study-specific)
  preprocess_hbn.py / .sbatch  # raw HBN -> .fif preprocessing (run once)
  compute_corrca.py / .sbatch  # thin CLI over eb_jepa.preprocessing.compute_corrca
  submit_job_{delta,expanse,jamming}.py
  pull_wandb.py, extract_*_results.py

tests/                         # pytest
```

`experiments/eeg_jepa/` is a frozen reproducibility snapshot of the
pre-refactor entry point — do not patch it. New work goes through
`eb_jepa.training.jepa_pretrain`. See
[`experiments/README.md`](experiments/README.md) for the per-study
index.

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
PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.jepa_pretrain

# 3. End-of-training auto-runs probe_eval + bootstrap on the saved
#    checkpoint (gated by cfg.eval.auto_run, default true). To run
#    manually on an existing checkpoint:
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.probe_eval \
    --checkpoint=/path/to/latest.pth.tar
PYTHONPATH=. uv run --group eeg python -m eb_jepa.evaluation.bootstrap \
    --predictions_dir=/path/to/saved_predictions --split=test
```

CLI overrides use OmegaConf dot syntax (`--optim.lr=5e-4`,
`--data.n_windows=4`, `--loss.anti_collapse=sigreg`,
`--eval.auto_run=false`, etc.).

## Data: HBN Movie-Watching EEG

HBN contains EEG recordings of children watching short movies, hosted on
OpenNeuro:

| Release | OpenNeuro ID | Split  |
|---------|--------------|--------|
| R1      | ds005505     | train / val |
| R6      | ds005510     | test   |

> R2–R4 are available but currently disabled in `SPLIT_RELEASES`.

- **EEG system**: EGI GSN-HydroCel-129 (129 channels)
- **Sampling rate**: 100 Hz raw, resampled to 200 Hz during preprocessing
- **Default movie task**: `ThePresent` (3m 23s, 24 fps, 4878 frames)
- **Movie features**: `contrast_rms`, `luminance_mean`, `position_in_movie`,
  `narrative_event_score`

Train and val both use R1 recordings but load disjoint subsets via
EEGDash; test uses R6. Each recording is paired with frame-level movie
features via timestamp alignment.

## Preprocessing

[`scripts/preprocess_hbn.py`](scripts/preprocess_hbn.py) applies a
two-pass pipeline following REVE (Défossez et al. 2023):

1. Drop recordings shorter than `min_duration_s` (default 10 s).
2. Resample to `target_sfreq` (default 200 Hz).
3. Bandpass filter `[l_freq, h_freq]` (default 0.5–99.5 Hz).
4. Convert to float32.
5. Z-score per channel using stats across all kept recordings.
6. Clip to ±`clip_std` standard deviations (default 15).

### Run

```bash
# Single release/task
PYTHONPATH=. uv run --group eeg python scripts/preprocess_hbn.py release=R1 task=ThePresent

# Multiple tasks at once
PYTHONPATH=. uv run --group eeg python scripts/preprocess_hbn.py --multirun \
    task=ThePresent,DespicableMe,RestingState,ContrastChangeDetection
```

Config: [`config/preprocess_hbn.yaml`](config/preprocess_hbn.yaml). Output
goes to `$HBN_PREPROCESS_DIR` (or `./data/hbn_preprocessed`):

```
hbn_preprocessed/R1/ThePresent/
├── intermediate/         # after pass 1 (resampled, filtered)
├── preprocessed/         # after pass 2 (z-scored, clipped)
├── normalization_stats.npz
└── summary.json
```

Each numbered subdir holds one recording's `.fif` plus a
`description.json` with subject metadata.

## Datasets

All dataset classes live in [`eb_jepa/datasets/hbn.py`](eb_jepa/datasets/hbn.py)
and use lazy loading: only file paths and sample indices are kept in
memory; EEG data is read from FIF on demand.

- **`HBNDataset`** — self-supervised crops, returns `[n_windows, C, T]`.
- **`HBNMovieDataset`** — supervised, pairs each window with movie
  features. Filters recordings on annotation quality.
- **`JEPAMovieDataset`** — extends `HBNMovieDataset` with temporal
  striding, per-channel EEG z-normalization, and frame-feature tensors.
  Provides `get_eeg_norm_stats()` (used for val/test), `get_chs_info()`
  (REVE channel positions), and `compute_feature_stats()` /
  `compute_feature_median()` for probe loss normalization.
- **`HBNMovieProbeDataset`** — flat-indexed `HBNMovieDataset` for
  per-window evaluation; each `__getitem__` returns a single
  `(window, features)` pair.

## Model

The masked path composes three pluggable parts:

```
EEG [B, T, C, W]
   │
   ▼
EEGEncoderTokens       ──► context tokens (masked)
                       ──► target tokens (depends on anti-collapse strategy)
MaskedPredictor        ──► predicts target representations from context + positions
PredictionLoss (MSE / smooth_l1) + AntiCollapse auxiliary loss
```

### Anti-collapse strategies

| Strategy | Class | Targets | Auxiliary loss | Combine |
|----------|-------|---------|----------------|---------|
| `dino`   | `DINOAntiCollapse` | EMA copy of encoder (frozen) | — | n/a |
| `vicreg` | `VICRegAntiCollapse` | stop-grad on online encoder | VCLoss on context tokens | additive |
| `sigreg` | `SIGRegAntiCollapse` | online encoder (grad flows) | Epps-Pulley test on pooled embeddings | convex |
| `none`   | `AntiCollapse` | stop-grad | — | (collapse expected; for ablations) |

DINO = V-JEPA; SIGReg = LeJEPA (arXiv:2511.08544); VICReg is the standard
variance/covariance recipe.

### Components

| Component | Class | Description |
|-----------|-------|-------------|
| Encoder   | `EEGEncoderTokens` | REVE backbone exposing per-token representations on a `[C, T, P]` grid. |
| Predictor | `MaskedPredictor`  | Transformer that predicts representations at masked positions from context + positional embeddings. |
| Masking   | `MultiBlockMaskCollator` | Samples short / long mask blocks on the `[C, P]` grid (V-JEPA style). |
| Anti-collapse | `AntiCollapse` subclass | DINO / VICReg / SIGReg (see above). |
| Probes    | `MaskedJEPAProbe` + `MovieFeatureHead` | Regression (z-MSE on continuous features) and classification (BCE on median-thresholded features) on the frozen encoder. |

### Defaults (`config/jepa_pretrain.yaml`)

| Section | Key | Default | Notes |
|---------|-----|---------|-------|
| `meta`  | `seed`           | 2025  | |
| `meta`  | `device`         | auto  | |
| `data`  | `batch_size`     | 64    | recordings per batch |
| `data`  | `n_windows`      | 4     | windows per sample |
| `data`  | `window_size_seconds` | 2 | |
| `data`  | `temporal_stride`| 1     | stride between sampled windows |
| `data`  | `preprocessed`   | true  | |
| `data`  | `feature_names`  | contrast_rms, luminance_mean, position_in_movie, narrative_event_score | |
| `model` | `encoder_embed_dim`  | 64 | |
| `model` | `encoder_depth`      | 4  | |
| `model` | `encoder_heads`      | 4  | |
| `model` | `encoder_head_dim`   | 16 | |
| `model` | `predictor_depth`    | 2  | shallower than encoder |
| `model` | `predictor_embed_dim`| null | null = same as encoder dim |
| `model` | `patch_size` / `patch_overlap` | 50 / 20 | |
| `model` | `ema_momentum` / `ema_momentum_end` | 0.996 / 1.0 | DINO only |
| `loss`  | `anti_collapse`  | vicreg | `dino` / `vicreg` / `sigreg` / `none` |
| `loss`  | `pred_loss_type` | mse    | `mse` or `smooth_l1` |
| `loss`  | `vicreg.std_coeff` / `vicreg.cov_coeff` | 1.0 / 1.0 | VICReg only |
| `loss`  | `vicreg.use_projector` | true | wraps VICReg with a Projector MLP |
| `loss`  | `sigreg.coeff`   | 0.05  | λ in `(1-λ)·pred + λ·sigreg` |
| `loss`  | `sigreg.num_slices` | 1024 | random projection directions |
| `optim` | `epochs`         | 100   | |
| `optim` | `lr`             | 3e-4  | |
| `optim` | `warmup_epochs`  | 5     | linear warmup before cosine decay |
| `logging` | `log_wandb`    | true  | project `eb_jepa` |

### Training loop

Each epoch iterates over recordings. For each batch:

1. **JEPA step** — encode context tokens, get target representations
   via the anti-collapse strategy, predict masked tokens, combine
   prediction loss with the auxiliary anti-collapse loss, backprop.
   DINO additionally EMA-updates the target encoder on a cosine
   momentum schedule (`ema_momentum → ema_momentum_end`).
2. **Online probes** — train `MovieFeatureHead` regression and
   classification heads on frozen encoder representations.

Validation runs every `logging.log_every` epochs (default every epoch).
Checkpoints save every `logging.save_every` epochs (default 10), and
`latest.pth.tar` + `best.pth.tar` (by `val/reg_loss`) are always kept.

### Post-training auto-eval

When `cfg.eval.auto_run=true` (default), training finishes by invoking
`probe_eval` + `bootstrap_predictions` on the saved checkpoint so each
run produces a self-contained metrics set. Disable for fast smoke runs:

```bash
PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.jepa_pretrain \
    --eval.auto_run=false
```

## Other experiments

```bash
# Binary classification benchmark (Hydra)
PYTHONPATH=. uv run --group eeg python experiments/benchmark/train.py

# Regression / tertile variants
PYTHONPATH=. uv run --group eeg python experiments/benchmark/train.py benchmark.task_mode=regression
PYTHONPATH=. uv run --group eeg python experiments/benchmark/train.py benchmark.task_mode=tertile
```

`experiments/benchmark/` trains supervised models (EEGNet, REVE, BIOT,
classical ML) directly on movie-feature prediction. See each study's
README for sweep / submission details.

## Cluster

- **Delta** (NCSA): SLURM, see [`scripts/submit_job_delta.py`](scripts/submit_job_delta.py)
  and study-specific `sbatch/` directories under `experiments/`.
- **Expanse** (SDSC): [`scripts/submit_job_expanse.py`](scripts/submit_job_expanse.py)
- **Jamming** (local workstation): [`scripts/submit_job_jamming.py`](scripts/submit_job_jamming.py),
  no SLURM, direct execution.

Sweep launchers under `experiments/<study>/sweeps/` are thin wrappers
around `neurolab.jobs.Job`; they build command lines and submit batches.

## Environment variables

| Variable             | Description                              | Default |
|----------------------|------------------------------------------|---------|
| `HBN_PREPROCESS_DIR` | Preprocessed data directory              | `./data/hbn_preprocessed` |
| `HBN_CACHE_DIR`      | Raw data cache (EEGDash)                 | `~/.cache/eb_jepa/datasets/eegdash_cache` |
| `EBJEPA_CKPTS`       | Checkpoint directory                     | `./checkpoints` |

## Gotchas

- Raw EEG is in volts (~1e-5 scale). Per-channel z-normalization is
  essential — without it the encoder produces constant representations.
- Always reuse `train_set.get_eeg_norm_stats()` when constructing
  val / test sets; mismatched normalization quietly hurts probes.
- REVE needs `chs_info` with spatial positions; use the dataset's
  `get_chs_info()`.
- `n_times` must match `window_size_seconds * sfreq` (default `400 = 2 · 200`).
- Old checkpoints saved before the anti-collapse refactor cannot be
  loaded — `check_old_checkpoint_format` raises a clear error rather
  than silently mis-loading. Either retrain or pin a pre-refactor
  commit for evaluation.

## Tests

```bash
uv run --group eeg pytest tests/
```

The fast focused suite (`tests/evaluation/`, `tests/unit/`,
`tests/test_loss_equivalences.py`, `tests/test_jepa_refactor.py`) runs in
seconds and does not require any cluster fixtures or downloaded data.

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
