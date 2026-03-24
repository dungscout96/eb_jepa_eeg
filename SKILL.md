# EEG JEPA Hyperparameter Sweep

Autonomous hyperparameter sweep for masked causal predictive coding (V-JEPA) on HBN EEG data.

## Setup

To set up a new sweep, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `sweep/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b sweep/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `experiments/eeg_jepa/cfgs/default.yaml` — all configurable hyperparameters.
   - `scripts/submit_job_jamming.py` — job submission entry point.
   - `experiments/eeg_jepa/main.py` — training loop.
4. **Verify SSH connectivity**: Run `ssh 100.113.196.11 hostname` to ensure the connection to jamming is live.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good with the user.

Once you get confirmation, kick off the experimentation.

## How experiments work

Each experiment runs on the jamming workstation (direct SSH execution, no SLURM). Jobs are submitted via `scripts/submit_job_jamming.py` which uses the `neurolab` library to:
- SSH into jamming
- cd into the repo, checkout the branch, pull
- Launch the training command with `nohup` in the background
- Return the remote PID

**Entry point**: `scripts/submit_job_jamming.py`

**Primary modification**: The `command` string in `submit_job_jamming.py` to pass config overrides. The training script (`experiments/eeg_jepa/main.py`) accepts `**overrides` via `fire.Fire()` in dot notation:

```python
command=(
    "PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py"
    " --optim.lr=0.001"
    " --data.batch_size=32"
    " --masking.min_context_fraction=0.2"
),
```

**You may also modify any other file** (source code, configs, training loop, etc.) as long as all changes are on the sweep branch. The branch isolates your work from main. If a hyperparameter change requires code changes (e.g. adding a new scheduler, changing the optimizer, restructuring the training loop), go ahead and make those changes.

## Hyperparameter search space

Scan over everything. The full set of tunable parameters (with defaults from `default.yaml`):

### Data / task structure
| Parameter | Config key | Default | Suggested range |
|---|---|---|---|
| Batch size | `data.batch_size` | 64 | 16, 32, 64, 128 |
| Windows per sample | `data.n_windows` | 4 | 2, 4, 8 |
| Window duration (s) | `data.window_size_seconds` | 2 | 1, 2, 4 |
| Temporal stride | `data.temporal_stride` | 1 | 1, 2, 4 |

### Model architecture
| Parameter | Config key | Default | Suggested range |
|---|---|---|---|
| Encoder embed dim | `model.encoder_embed_dim` | 64 | 32, 64, 128 |
| Encoder depth | `model.encoder_depth` | 4 | 2, 4, 6, 8 |
| Encoder heads | `model.encoder_heads` | 4 | 2, 4, 8 |
| Encoder head dim | `model.encoder_head_dim` | 16 | 16, 32 |
| Patch size (samples) | `model.patch_size` | 50 | 25, 50, 100 |
| Patch overlap | `model.patch_overlap` | 20 | 0, 10, 20, 30 |
| MLP dim ratio | `model.mlp_dim_ratio` | 2.66 | 2.0, 2.66, 4.0 |
| Predictor depth | `model.predictor_depth` | 2 | 1, 2, 4 |
| EMA momentum start | `model.ema_momentum` | 0.996 | 0.99, 0.996, 0.999 |

### Masking (controls the pretraining task)
| Parameter | Config key | Default | Suggested range |
|---|---|---|---|
| Short masks count | `masking.n_pred_masks_short` | 2 | 1, 2, 4 |
| Long masks count | `masking.n_pred_masks_long` | 2 | 0, 1, 2, 4 |
| Short channel scale | `masking.short_channel_scale` | [0.08, 0.15] | vary lo/hi |
| Short patch scale | `masking.short_patch_scale` | [0.3, 0.6] | vary lo/hi |
| Long channel scale | `masking.long_channel_scale` | [0.15, 0.35] | vary lo/hi |
| Long patch scale | `masking.long_patch_scale` | [0.5, 1.0] | vary lo/hi |
| Min context fraction | `masking.min_context_fraction` | 0.15 | 0.05, 0.10, 0.15, 0.25 |

### Loss
| Parameter | Config key | Default | Suggested range |
|---|---|---|---|
| VC cov coefficient | `loss.cov_coeff` | 1.0 | 0.1, 1.0, 10.0 |
| VC std coefficient | `loss.std_coeff` | 1.0 | 0.1, 1.0, 10.0 |

### Optimization
| Parameter | Config key | Default | Suggested range |
|---|---|---|---|
| Learning rate | `optim.lr` | 3e-4 | 1e-4, 3e-4, 1e-3, 3e-3 |
| Epochs | `optim.epochs` | 100 | 30 (short sweep), 100 (full) |

## Metrics

Results are tracked via W&B (`eb_jepa` project) and remote log files. The key metrics to extract:

**Primary — pretraining health** (from sanity checks, W&B prefix `sanity/`):
- `train_step/pred_loss` — masked prediction loss (primary pretraining signal, should decrease)
- `sanity/embed_std` — embedding standard deviation (should not collapse to 0)
- `sanity/cosim_mean` — mean cosine similarity (high → representation collapse)
- `sanity/grad_norm` — gradient norm (should be stable, not exploding)
- `sanity/linear_probe_acc` — linear probe accuracy on subject metadata (should increase if encoder learns useful representations)

**Secondary** (downstream probe losses, lower is better):
- `val/reg_loss` — regression probe loss on validation set
- `val/cls_loss` — classification probe loss on validation set

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

Header row and columns:

```
commit	pid	pred_loss	probe_acc	cosim	embed_std	val_reg	val_cls	status	description
```

1. `commit` — git commit hash (short, 7 chars)
2. `pid` — remote PID returned by job.submit()
3. `pred_loss` — final train pred_loss (use 0.000 for crashes)
4. `probe_acc` — final sanity/linear_probe_acc (use 0.000 for crashes)
5. `cosim` — final sanity/cosim_mean (use 0.000 for crashes)
6. `embed_std` — final sanity/embed_std (use 0.000 for crashes)
7. `val_reg` — final val/reg_loss (use 0.000 for crashes)
8. `val_cls` — final val/cls_loss (use 0.000 for crashes)
9. `status` — `keep`, `discard`, or `crash`
10. `description` — short text of what this experiment tried

Example:

```
commit	pid	pred_loss	probe_acc	cosim	embed_std	val_reg	val_cls	status	description
a1b2c3d	12345	0.034	0.62	0.45	1.23	0.892	0.651	keep	baseline (defaults)
b2c3d4e	12350	0.029	0.68	0.38	1.45	0.875	0.642	keep	lr=1e-3
c3d4e5f	12360	0.000	0.00	0.00	0.00	0.000	0.000	crash	embed_dim=256 (OOM)
```

## Checking results

Since jobs run remotely, use these methods to check results:

1. **Check if job is still running**:
   ```python
   from neurolab.jobs.monitor import monitor_jobs
   monitor_jobs(["<PID>"], cluster="jamming")
   ```
   Or via SSH: `ssh 100.113.196.11 "ps -p <PID> -o pid,stat,etime,comm"`

2. **Read remote logs**:
   ```bash
   ssh 100.113.196.11 "tail -n 50 /home/dung/Documents/eb_jepa_eeg/logs/eeg_jepa_sanity_checks.out"
   ```

3. **Check for crashes**:
   ```bash
   ssh 100.113.196.11 "tail -n 50 /home/dung/Documents/eb_jepa_eeg/logs/eeg_jepa_sanity_checks.err"
   ```

4. **Check W&B** for detailed metrics (user can provide the W&B run URL).

## The experiment loop

The experiment runs on a dedicated branch (e.g. `sweep/mar23`).

LOOP:

1. **Plan**: Look at results so far, decide which hyperparameter(s) to vary next. Start with the most impactful parameters first (lr, masking ratios, model capacity). Change **one thing at a time** unless doing a deliberate interaction test.
2. **Configure**: Edit the `command` string in `scripts/submit_job_jamming.py` with the chosen overrides. If needed, modify any other source files (training loop, architectures, configs, etc.) — all changes are on the sweep branch.
3. **Commit**: `git add <changed files> && git commit -m "sweep: <description>"`
4. **Push**: `git push` (so jamming pulls the latest).
5. **Submit**: `uv run python scripts/submit_job_jamming.py` (with DRY_RUN=False).
6. **Wait**: The job runs on jamming (~5-30 min depending on epochs). Check status periodically.
7. **Collect results**: Once the job finishes, read the remote logs or W&B to extract metrics.
8. **Record**: Append results to `results.tsv` (do NOT commit this file — keep it untracked).
9. **Decide**:
   - If improved: **keep** the commit, advance the branch.
   - If equal or worse: **discard**, `git reset --hard HEAD~1` to revert.
10. **Repeat**: Go to step 1.

### Strategy guidelines

- **First run**: Always establish the baseline with default config.
- **Short runs first**: Use `optim.epochs=30` for initial sweeps to iterate faster, then do full 100-epoch runs for promising configs.
- **One variable at a time**: Vary a single parameter per experiment to isolate effects. Only combine changes when testing known-good individual improvements together.
- **Prioritize**: Start with learning rate and masking parameters (they control the core pretraining task), then model capacity, then loss coefficients.
- **Watch for collapse**: If `sanity/cosim_mean` approaches 1.0 or `sanity/embed_std` drops near 0, the representations are collapsing — increase VC regularization or reduce learning rate.
- **Crashes**: If a run OOMs or errors, log it as `crash`, revert, and try a less aggressive setting.

### Timeout

Each short sweep (~30 epochs) should take ~5-15 minutes. Each full run (~100 epochs) should take ~15-45 minutes. If a job has been running for more than 2x the expected time, check for hangs.

### When to pause

Unlike a local experiment loop, remote jobs have non-trivial wait times. After submitting a job, **inform the user** that a job is running and what you're testing. Ask if they want you to:
- Wait and check back (if they're staying around)
- Stop here and they'll check results later

Do NOT silently loop forever — the user needs to be in the loop for remote job monitoring.
