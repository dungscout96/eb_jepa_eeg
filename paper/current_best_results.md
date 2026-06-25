# Canonical baseline evaluation pipeline (Protocol B)

Locks the protocol that produces the JEPA + CorrCA narrative reference (`narr = 0.156 ± 0.023`) and ships a 10-method × 12-metric apples-to-apples comparison under that exact protocol.

After cleanup, the PR scope is exclusively the canonical training+eval pipeline used to generate `docs/canonical_baselines_2026-05-04.md`.

---

## Training methodology — Phase D, nw=2, ws=4

### Encoder

| Component | Value |
|---|---|
| Class | `EEGEncoderTokens` (`eb_jepa/architectures.py`) |
| `embed_dim` | 64 |
| `depth` | 2 transformer blocks |
| `heads` | 4 |
| `head_dim` | 16 |
| `patch_size` | 50 samples (250 ms @ 200 Hz) |
| `patch_overlap` | 20 samples |
| `mlp_dim_ratio` | 2.66 (GEGLU FFN) |
| Positional encoding | 4D Fourier on `(channel.x, channel.y, channel.z, time_index)` |
| Norm | RMSNorm in attention / FFN; LayerNorm in pos MLP |
| Nonlinearity | GEGLU + softmax-attention with residual streams |
| Output token grid | `[B, C×T×P, D] = [B, 5×2×26, 64]` per clip |

The same module is used during SSL pretraining (one update path per gradient step) and at canonical-eval time (frozen, no gradient).

### SSL objective — JEPA + VC regularizer

`MaskedJEPA` (`eb_jepa/jepa.py`) maps an 8-second 5-channel CorrCA-projected EEG clip through:

1. Patches → linear projection to 64-d → 4D Fourier pos encoding.
2. Mask roughly half the patches via `MultiBlockMaskCollator` on the `[C, P]` grid (mask is replicated across windows so a masked position is masked at every timepoint within the clip).
3. **Context encoder** (trained) processes only the visible patches → `[B, n_visible, 64]`.
4. **Target encoder** (EMA copy of context, no gradient) processes all patches without mask → `[B, n_all, 64]`.
5. **Masked predictor** (trained) projects context tokens through a 64→24 bottleneck, runs 24-d transformer over masked-position queries, projects back 24→64 → `[B, n_masked, 64]`.
6. Loss = `smooth_l1(predicted, target.detach())` + `VCLoss(context_tokens)` where `VCLoss = std_coeff * HingeStdLoss + cov_coeff * CovarianceLoss`.
7. Target encoder updated by EMA from context encoder weights at each step.

Predictor + target + VC projector are training-time only — discarded at eval.

### Training configuration (`scripts/train_phaseD.sbatch` + `experiments/eeg_jepa/main.py`)

| Setting | Value |
|---|---|
| Optimizer | AdamW, `lr=5e-4`, cosine decay to `1e-6` |
| Warmup | 5 epochs |
| Epochs | 100 |
| Batch size | 64 |
| `n_windows` × `window_size_seconds` | 2 × 4s (8s clip total) |
| `data.norm_mode` | `per_recording` (each clip z-normalized using its OWN mean/std) |
| `data.corrca_filters` | `corrca_filters.npz` — 129 → 5 component spatial filter, **fit on train-only recordings** by `scripts/compute_corrca.py` |
| `data.tasks` | `ThePresent` |
| `model.encoder_depth` | 2 |
| `model.predictor_embed_dim` | 24 (bottleneck) |
| `loss.std_coeff` / `cov_coeff` | 0.25 / 0.25 |
| `loss.pred_loss_type` | `smooth_l1` |
| `optim.early_stopping_patience` | 0 (full 100 epochs, no early stop) |
| Seeds | {42, 123, 456, 789, 2025} (5 encoder seeds) |
| Checkpoints saved | `latest.pth.tar` and `best_by_online_probe.pth.tar` per seed |

### Train / val / test data splits (`eb_jepa/datasets/hbn.py`)

The HBN dataset is split by **physical OpenNeuro release**, guaranteeing subject-level disjointness by construction:

| Split | Releases | OpenNeuro | Subjects |
|---|---|---|---|
| Train | R1, R2, R3, R4 | ds005505 / ds005506 / ds005507 / ds005508 | ~700 |
| Val | R5 | ds005509 | 108 |
| Test | R6 | ds005510 | 108 |

CorrCA filters are fit ONLY from the train releases (`SPLIT_RELEASES["train"]`). `eeg_norm_stats` is computed train-only and passed to val/test. Under `per_recording` norm mode the saved stats are loaded but never actually used (each clip uses its own mean/std).

---

## Eval methodology — Protocol B (canonical)

### Pipeline overview

`scripts/probe_unified_tier.sbatch` → `scripts/unified_probe_eval.py` → produces per-clip predictions NPZ → `scripts/bootstrap_unified.py` (or `bootstrap_tier_native.py`) → per-seed L2 JSON → `scripts/aggregate_and_print.py` → 5-seed L3 JSON + t-test vs chance → `docs/canonical_baselines_2026-05-04.md`.

Same pipeline for every method in the comparison; only the per-clip feature extraction step varies.

### Per-clip feature extraction (`--feature_source` flag)

A single clip enters as `eeg [n_windows=2, n_chans=5_CorrCA, T_samp=800] = 8000 numbers`. Each `--feature_source` produces a 1-D feature vector for the Ridge probe:

| `--feature_source` | What it does | Per-clip D |
|---|---|---|
| `jepa` | frozen `context_encoder` forward → kc-pool: tokens `[B, C, T, P, D]` → `mean(P)` → `mean(T)` → `[B, C×D]` | **320** = 5 × 64 |
| `random_init` | same arch as `jepa` but random weights, no checkpoint load | 320 |
| `raw_corrca` | CorrCA-5, box-mean-pool 800→100, `mean(over windows)`, flatten | 500 (legacy; not in headline table) |
| `raw_corrca_64` | CorrCA-5, box-mean-pool 800→64, `mean(over windows)`, flatten — matched-D linear ceiling | **320** |
| `raw_corrca_pca` | CorrCA-5, box-mean-pool 800→200; per-channel PCA(64) fit train-only; concat across 5 chans | **320** |
| `corrca_stats` | per-channel 7 summary stats (mean, std, log-pow in δ/θ/α/β/γ); 5 × 7 | **35** |
| `corrca_stats_chan1` | first CorrCA channel only × 7 stats | **7** |
| `corrca_stats_pooled` | corrca_stats but stats averaged across chans then tiled (rank-7) | 35 |
| `raw_stats` | 129 raw chans × 7 stats; **CorrCA filter not applied** | **903** |
| `raw_stats_pooled` | raw_stats but stats averaged across chans then tiled (rank-7) | 903 |
| `psd_band` | Welch 5-band log-power × 129 raw chans, `mean(windows)`, no CorrCA | 645 |

Every source feeds the **same downstream pipeline**. The only thing that changes per row in the comparison table is the function `eeg_clip → 1-D feature vector`.

### n_passes data augmentation + train iteration order

For each split (train/val/test), each recording goes through the dataset's `__getitem__` 20 times, each time returning a different random clip start (`torch.randint` on the global RNG). This produces:

- Train Ridge fit: `[700 × 20 = 14000 samples, D] @ Ridge → predict`
- Val Ridge eval: `[108 × 20 = 2160 samples, D]` (used only for protocol audit, never reported)
- Test Ridge eval: `[108 × 20 = 2160 samples, D]`

**Train iteration order** (`train_order=True` in `_extract`):

```python
rng = torch.Generator().manual_seed(probe_seed)
for p in range(n_passes):
    for rec_idx in torch.randperm(n_rec, generator=rng).tolist():
        eeg, feats, _ = dataset[rec_idx]
        ...
```

Outer pass × inner shuffled-recording — matches the locked protocol so different runs at the same seed produce the same training matrix.

Val/test extraction iterates `for rec in range(n_rec): for _ in range(n_passes)` (sequential rec × passes) so the saved NPZ has predictions in `(n_rec, n_passes)` major order for direct recording-level bootstrap reshape.

### Probe seed convention

| Method category | `--seed` policy | What `σ_5seed` captures |
|---|---|---|
| JEPA (`pB_phaseD_*`) | **probe_seed = 42 fixed** across all 5 encoder seeds | Pure encoder-training variance; probe sees identical clip set across encoders |
| Trivial baselines (`pB_t1_*`) | **probe_seed = enc_seed** paired (1:1) | For deterministic feature paths (corrca_stats, raw_corrca_*, raw_stats), `σ` = probe-side n_passes RNG variance only. For random_init, `σ` = combined random-weight init + probe RNG variance |

This asymmetric protocol gives every row a non-zero `σ` while keeping JEPA's variance interpretable as "what does another SSL training run produce?".

### Probe heads (canonical, deterministic)

| Family | Head | Solver | Reg |
|---|---|---|---|
| Stim regression (lum, cont, pos, narr, age) | `sklearn.Ridge(α=1.0)` | closed-form | L2 |
| Stim classification (median-split AUC + bal-acc) | `sklearn.LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=1000)` | LBFGS | L2 |
| Sex (binary) | `sklearn.LogisticRegression(C=1, lbfgs)` | LBFGS | L2 |
| Movie ID (20-bin top-1, top-5) | `sklearn.LogisticRegression(C=1, multi_class='multinomial', solver='lbfgs', max_iter=2000)` | LBFGS | L2 |

All deterministic given (encoder weights, probe_seed, train/val/test data). One Ridge fit per stim feature, one LogReg per cls feature, one multinomial LogReg for movie_id. No SGD, no minibatching.

### Train-only quantities (no leakage at any step)

- `mu`, `sd` for feature standardization → `Xtr_flat.mean(0)` / `Xtr_flat.std(0)` only.
- `ym`, `ys` for label normalization → `ytr.mean()` / `ytr.std()` only.
- `med = np.nanmedian(ytr)` for median-split classification → train-only.
- Movie_id bin edges → `np.linspace(pos_tr_rec.min(), pos_tr_rec.max(), 21)` train-only.
- PCA basis for `raw_corrca_pca` → `PCA(64).fit(tr_flat)` per channel, train-only.
- CorrCA spatial filter → `compute_corrca.py` over `SPLIT_RELEASES["train"]` only.

Audit log: `git grep -n 'fit\|.mean\|.std\|.median' scripts/unified_probe_eval.py` traces every train-only fit.

### Per-clip Pearson r on flat (2160,) — the L1 metric

```python
ytr = Ytr_flat_clips[:, fi]                 # train-clip labels
probe = Ridge(α=1).fit(Xtr_flat_clips, (ytr - ym) / ys)
pred  = probe.predict(Xt_flat_clips) * ys + ym
r     = pearsonr(pred, yt_flat).statistic   # ← test r on flat 2160
```

Pearson r is computed over all 2160 test rows directly; no per-recording aggregation at the metric stage.

### Recording-level bootstrap → L2 → L3

`scripts/bootstrap_unified.py` reads test_*-prefixed keys ONLY from the saved NPZ. Per-clip arrays of shape `(2160,)` are reshaped to `(108_rec, 20_passes)`. For each of B = 2000 iterations:

```python
idx = rng.integers(0, 108, size=108)        # resample recordings with replacement
pred_sub = pred_grp[idx].reshape(-1)        # flatten back to subset×20
metric_b = pearsonr(pred_sub, tgt_sub).statistic
```

- **L1** = single raw Pearson r over the full 2160 test rows (per encoder seed)
- **L2** = `mean ± 95% CI` over 2000 bootstrap resamples (per encoder seed)
- **L3** = `mean ± 1σ` of L2 across the 5 encoder seeds (the headline number)

`scripts/aggregate_and_print.py` then runs a 1-sample two-sided t-test of the 5 per-seed L2 means against the per-metric chance value (0 for r, 0.5 for AUC, 0.05/0.25 for movie_id top-1/5) and emits ✓ / ns flags matching the tracking-table format.

---

## Headline finding (full table in `docs/canonical_baselines_2026-05-04.md`)

At matched 320-d Ridge capacity on the same 8-second CorrCA-projected EEG input:

| Method (D=320) | reg_lum | reg_cont | reg_pos | reg_narr | age_corr | sex_auc |
|---|---|---|---|---|---|---|
| **raw_corrca_pca** (no encoder) | **0.292±0.016** | 0.217±0.018 | **0.261±0.007** | **0.235±0.033** | 0.090±0.064 | 0.549±0.040 |
| **raw_corrca_64** (no encoder) | **0.292±0.018** | 0.217±0.015 | **0.262±0.007** | **0.232±0.031** | 0.128±0.085 | 0.565±0.041 |
| pB_phaseD_issue10best (JEPA SSL) | 0.226±0.018 | **0.223±0.014** | 0.226±0.016 | 0.155±0.023 | 0.390±0.014 ⚠ | 0.727±0.019 ⚠ |
| pB_t1_random_init | 0.218±0.020 | 0.194±0.022 | 0.199±0.023 | 0.129±0.049 | 0.437±0.052 ⚠ | 0.705±0.028 ⚠ |

Reading: at the same Ridge capacity (320-d) on the same CorrCA-5 input, a **fixed linear projection of the EEG outperforms the JEPA SSL encoder on every stim regression**. JEPA loses 0.077 narr to a fixed linear basis on its own input, and inflates subject-identity leakage 3× (sex_auc 0.73 vs 0.55) in exchange. The matched-D linear ceiling is the unambiguous form of the paper's claim "stim signal lives in the CorrCA spatial filter".

JEPA reproduces the doc reference `narr = 0.156 ± 0.023` exactly: per-seed L1 = {0.137, 0.160, 0.148, 0.142, 0.194} → 5-seed mean 0.1559, matching `docs/narrative_research_2026-05-04.md` §6.4 within 0.0001 per seed.

---

## File scope after cleanup

```
scripts/                         13 files (canonical training+eval+bootstrap)
  aggregate_and_print.py
  bootstrap_tier_native.py
  bootstrap_unified.py
  compute_corrca.py / .sbatch
  preprocess_hbn.py / .sbatch
  print_l3_table.py
  probe_unified_tier.sbatch
  submit_job_delta.py
  submit_phaseD_nw2ws4_5seed.sh
  train_phaseD.sbatch
  unified_probe_eval.py

experiments/                      4 files
  eeg_jepa/main.py / eval.py / cfgs/default.yaml / README.md

docs/                             6 files
  canonical_baselines_2026-05-04.md   ← final report
  evaluation_guide.md                 ← locked protocol spec
  CODE_OF_CONDUCT.md / CONTRIBUTING.md
  archi-schema-eb-jepa.png / teaser.png

eb_jepa/                         11 files (only EEG-relevant modules)
  architectures.py / jepa.py / losses.py / masking.py / sanity_checks.py
  schedulers.py / training_utils.py / nn_utils.py / logging.py
  datasets/hbn.py / utils.py
```

120 legacy files removed (toy domains, abandoned experimental cells, sweep scripts, older diagnostic docs). The remaining tree is exactly what's needed to reproduce the canonical training and eval.

## Test plan

- [x] Per-seed JEPA narrative L1 reproduces the doc reference {0.137, 0.160, 0.148, 0.142, 0.194} → 0.156 ± 0.023 ✓
- [x] All 50 canonical jobs (10 methods × 5 seeds) completed end-to-end on Delta under per-clip flat protocol
- [x] Train/val/test leakage audit through code (CorrCA train-only, norm stats train-only, PCA fit train-only, standardization train-only, Ridge/LogReg fit train-only, bootstrap reads test-only)
- [x] Black + isort applied to all 5 canonical scripts
- [x] Pre-existing test failures verified independent of this PR (braindecode library version mismatch + HBN data fetch in local env)

🤖 Generated with [Claude Code](https://claude.com/claude-code)