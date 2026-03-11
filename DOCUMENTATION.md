# EB-JEPA EEG: Project Documentation

Self-supervised EEG representation learning using JEPA (Joint Embedding Predictive Architecture) on HBN (Healthy Brain Network) movie-watching data.

---

## Project Structure

```
eb_jepa_eeg/
├── eb_jepa/                    # Core library
│   ├── jepa.py                 # JEPA model classes
│   ├── architectures.py        # Encoders, predictors, heads
│   ├── losses.py               # Loss functions (VCLoss, SquareLoss, etc.)
│   ├── training_utils.py       # Device setup, checkpointing, W&B
│   ├── logging.py              # Logger configuration
│   ├── nn_utils.py             # NN utilities (TemporalBatchMixin)
│   ├── scheduling.py           # LR schedulers
│   └── datasets/
│       └── hbn.py              # HBN EEG dataset classes
├── experiments/
│   ├── eeg_jepa/               # Main EEG JEPA experiment
│   │   ├── main.py             # Training script
│   │   ├── eval.py             # Validation loop
│   │   └── cfgs/default.yaml   # Training config
│   ├── run_benchmark.py        # Supervised baseline benchmarks
│   └── run_benchmark_multitask.py
├── scripts/
│   └── preprocess_hbn.py       # Data preprocessing pipeline
├── config/
│   ├── default.yaml            # Global config (benchmark, data params)
│   └── preprocess_hbn.yaml     # Preprocessing config
├── examples/                   # Original EB-JEPA examples (image, video, AC)
├── tests/                      # Test suite
└── data/
    ├── movies/                 # Movie stimulus files
    └── output/The_Present/     # Extracted movie features (CSV)
```

---

## Data: HBN Movie-Watching EEG

### Overview

The Healthy Brain Network (HBN) dataset contains EEG recordings of children watching short movies. Each release is hosted on OpenNeuro:

| Release | OpenNeuro ID | Split  |
|---------|-------------|--------|
| R1      | ds005505    | train / val |
| R6      | ds005510    | test   |

> R2–R4 are available but currently commented out in `SPLIT_RELEASES`.

### Recording Setup

- **EEG system**: EGI GSN-HydroCel-129 (129 channels)
- **Sampling rate**: 100 Hz (raw) or 200 Hz (preprocessed)
- **Default movie task**: `ThePresent` (3m 23s, 24 fps, 4878 frames)
- **Movie features**: `contrast_rms`, `luminance_mean`, `entropy`, `scene_natural_score`

### Data Splits

Train and val both use R1 recordings but load different subsets via EEGDash. Test uses R6. Each recording is paired with frame-level movie features via timestamp alignment.

---

## Preprocessing

### Pipeline

The preprocessing script (`scripts/preprocess_hbn.py`) applies a two-pass pipeline following REVE (Defossez et al. 2023):

**Pass 1** (intermediate output):
1. Remove recordings shorter than 10 seconds
2. Resample to 200 Hz
3. Apply 0.5–99.5 Hz band-pass filter
4. Convert to float32

**Pass 2** (final output):
5. Z-score normalization (per-channel, stats computed across all kept recordings)
6. Clip values exceeding +/-15 standard deviations

### Running Preprocessing

```bash
# Single release/task
PYTHONPATH=. uv run scripts/preprocess_hbn.py release=R1 task=ThePresent

# Multiple tasks
PYTHONPATH=. uv run scripts/preprocess_hbn.py --multirun \
    task=ThePresent,DespicableMe,RestingState,contrastChangeDetection
```

### Configuration

Config file: `config/preprocess_hbn.yaml`

| Parameter      | Default | Description                               |
|----------------|---------|-------------------------------------------|
| `release`      | R1      | HBN release to process                    |
| `task`         | ThePresent | EEG task name                          |
| `target_sfreq` | 200.0  | Target sampling frequency (Hz)            |
| `l_freq`       | 0.5    | Band-pass low cutoff (Hz)                 |
| `h_freq`       | 99.5   | Band-pass high cutoff (Hz)                |
| `min_duration_s` | 10.0 | Minimum recording duration to keep        |
| `clip_std`     | 15.0   | Clip threshold (standard deviations)      |
| `output_dir`   | `$HBN_PREPROCESS_DIR` or `./data/hbn_preprocessed` | Output path |

### Output Structure

```
hbn_preprocessed/
└── R1/
    └── ThePresent/
        ├── intermediate/       # After pass 1 (resampled, filtered)
        │   ├── 0/0-raw.fif
        │   ├── 1/1-raw.fif
        │   └── ...
        ├── preprocessed/       # After pass 2 (z-scored, clipped)
        │   ├── 0/0-raw.fif
        │   ├── 1/1-raw.fif
        │   └── ...
        ├── normalization_stats.npz   # Per-channel mean/std
        └── summary.json              # Processing summary
```

Each numbered subdirectory contains one recording's FIF file and a `description.json` with subject metadata.

---

## Dataset Classes

All dataset classes are in `eb_jepa/datasets/hbn.py`. They use lazy loading: only file paths and sample indices are stored in memory; EEG data is read from FIF files on demand in `__getitem__`.

### HBNDataset

Self-supervised dataset for general JEPA pretraining. Each item is a random crop of `n_windows` contiguous windows from one recording.

```python
HBNDataset(split, n_windows=16, window_size_seconds=2, task="ThePresent",
           preprocessed=False, preprocessed_dir=None)
# Returns: Tensor[n_windows, n_channels, n_times]
```

### HBNMovieDataset

Supervised dataset pairing EEG windows with movie features. Filters recordings by annotation quality (requires `video_start`/`video_stop` events, sufficient duration).

```python
HBNMovieDataset(split, window_size_seconds=2, task="ThePresent",
                cfg=cfg, preprocessed=False, preprocessed_dir=None)
# Returns: (Tensor[n_windows, n_channels, n_times], pd.Series of feature dicts)
```

### JEPAMovieDataset

Extends `HBNMovieDataset` for JEPA training. Adds temporal striding, per-channel EEG normalization, and feature tensor extraction.

```python
JEPAMovieDataset(split, n_windows=16, window_size_seconds=2,
                 task="ThePresent", feature_names=None,
                 eeg_norm_stats=None, temporal_stride=1,
                 cfg=cfg, preprocessed=False, preprocessed_dir=None)
# Returns: (Tensor[n_windows, n_channels, n_times], Tensor[n_windows, n_features])
```

Key methods:
- `get_eeg_norm_stats()` — returns `{"mean": ..., "std": ...}` for val/test normalization
- `get_chs_info()` — returns MNE channel info with positions (required by REVE encoder)
- `compute_feature_stats()` / `compute_feature_median()` — for loss normalization

### HBNMovieProbeDataset

Flat-indexed version of `HBNMovieDataset` for per-window evaluation. Each `__getitem__` returns a single `(window, features)` pair.

---

## Model Architecture

### JEPA

The JEPA model (`eb_jepa/jepa.py`) learns representations by predicting future latent states from past latent states, without reconstructing raw inputs.

```
EEG windows ──> Encoder ──> representations ──> Predictor ──> predicted future repr.
                                              │
                              Target Encoder ──> target future repr.
                              (EMA of Encoder)
```

**Components** (configured in `experiments/eeg_jepa/cfgs/default.yaml`):

| Component | Class | Description |
|-----------|-------|-------------|
| Encoder | `EEGEncoder` | Wraps braindecode REVE model. Maps `[B, C, T]` EEG to `[B, D]` representations. |
| Target encoder | `EEGEncoder` | Exponential moving average copy of encoder. |
| Predictor | `StateOnlyPredictor(MLPEEGPredictor)` | Concatenates consecutive state pairs, predicts next state. |
| Projector | `Projector` | MLP projector for loss computation. |
| Regularizer | `VCLoss` | Variance-Covariance loss preventing representation collapse. |
| Prediction loss | `SquareLossSeq` | L2 loss between predicted and target representations. |

### REVE Encoder Parameters

```yaml
encoder_embed_dim: 64   # Embedding dimension
encoder_depth: 4        # Transformer blocks
encoder_heads: 4        # Attention heads
encoder_head_dim: 16    # Per-head dimension
# ~755K parameters (small config for limited data)
```

The REVE encoder requires channel position information via `chs_info` (GSN-HydroCel-129 montage).

### Evaluation Probes

Two `JEPAProbe` modules evaluate representation quality during training:

1. **Regression probe**: `MovieFeatureHead` + MSE loss (z-normalized targets)
2. **Classification probe**: `MovieFeatureHead` + BCE loss (median-thresholded binary targets)

Probes train their own heads on frozen encoder representations.

---

## Running Experiments

### EEG JEPA Training

```bash
# Default config (using uv)
PYTHONPATH=. uv run experiments/eeg_jepa/main.py

# Or with explicit python
PYTHONPATH=. python experiments/eeg_jepa/main.py

# Override hyperparameters
PYTHONPATH=. uv run experiments/eeg_jepa/main.py \
    --data.batch_size=32 --optim.lr=1e-4
```

> `PYTHONPATH=.` is required so that `eb_jepa` and `experiments` packages are importable.

### Training Configuration

Config file: `experiments/eeg_jepa/cfgs/default.yaml`

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `meta`  | `seed`    | 2025    | Random seed |
| `meta`  | `device`  | auto    | Device selection |
| `data`  | `batch_size` | 64   | Batch size (number of recordings per batch) |
| `data`  | `n_windows` | 16    | Windows sampled per recording |
| `data`  | `window_size_seconds` | 2 | Window duration |
| `data`  | `temporal_stride` | 4 | Stride between windows (4 = 8s apart) |
| `data`  | `preprocessed` | true | Use preprocessed data |
| `data`  | `preprocessed_dir` | null | Path to preprocessed data |
| `model` | `dstc`    | 64      | Representation dimension |
| `model` | `hpre`    | 128     | Predictor hidden dimension |
| `model` | `hdec`    | 64      | Probe head hidden dimension |
| `model` | `steps`   | 4       | Prediction steps |
| `loss`  | `std_coeff` | 1.0   | Variance loss weight |
| `loss`  | `cov_coeff` | 1.0   | Covariance loss weight |
| `optim` | `epochs`  | 100     | Training epochs |
| `optim` | `lr`      | 3e-4    | Learning rate |
| `logging` | `log_wandb` | true | Enable Weights & Biases logging |

### Training Loop

Each epoch iterates over recordings. For each batch:

1. **JEPA step** (self-supervised): unroll encoder + predictor for `steps` timesteps, compute prediction loss + VC regularization
2. **Probe step** (supervised, frozen encoder): train regression and classification heads on movie features

Validation runs every `log_every` epochs. Checkpoints saved every `save_every` epochs.

### Supervised Benchmarks

```bash
# Binary classification benchmark (default)
PYTHONPATH=. uv run experiments/run_benchmark.py

# Regression benchmark
PYTHONPATH=. uv run experiments/run_benchmark.py benchmark.task_mode=regression

# Tertile classification
PYTHONPATH=. uv run experiments/run_benchmark.py benchmark.task_mode=tertile
```

Benchmarks train supervised models (EEGNet, REVE, BIOT) directly on movie feature prediction. Metrics: R², Pearson correlation, accuracy, balanced accuracy, ROC-AUC.

---

## Important Notes

- **EEG normalization**: Raw data is in volts (~1e-5 scale). Per-channel z-normalization is essential — without it, the encoder outputs constant representations.
- **Val/test normalization**: Always use the training set's normalization stats via `eeg_norm_stats=train_set.get_eeg_norm_stats()`.
- **n_times**: Must match `window_size_seconds * sfreq`. Default: 200 (2s * 100Hz).
- **Channel positions**: REVE requires `chs_info` with spatial positions. Use `get_chs_info()` from the dataset.
- **W&B**: Logs to project `eb_jepa`. Set `logging.log_wandb: false` to disable.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HBN_PREPROCESS_DIR` | Preprocessed data directory | `./data/hbn_preprocessed` |
| `HBN_CACHE_DIR` | Raw data cache (EEGDash) | `~/.cache/eb_jepa/datasets/eegdash_cache` |
| `EBJEPA_CKPTS` | Checkpoint directory | `./checkpoints` |

---

## Tests

```bash
# Run all tests
uv run pytest tests/

# Key test files
tests/test_eeg_architectures.py      # Encoder/decoder shape tests
tests/test_eeg_movie.py              # Movie dataset tests
tests/test_eeg_probe.py              # JEPAProbe tests
tests/test_eeg_jepa_output_formats.py # JEPA output validation
```
