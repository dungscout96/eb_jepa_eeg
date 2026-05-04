# Evaluation Guide

This is the **canonical** evaluation protocol for any encoder run in this repo. Every run reported in any document, table, or memo MUST follow this protocol unless explicitly noted otherwise. Single canonical probe per metric family. **Bootstrap is mandatory; seed-level significance testing is on request only.**

The driver that implements this protocol is **`scripts/unified_probe_eval.py`**. Use it. Don't reinvent.

## 1. Metrics that MUST be tracked

For every encoder evaluated against the HBN movie probe set, the following 18 headline metrics must appear in the per-recording predictions JSON and be reported (or summarized) in any results table.

### 1.1 Stimulus features — regression + classification (4 features × 4 metrics = 16)

Four movie-derived features, each evaluated as both regression (Pearson r, R²) and binary classification (AUC, balanced accuracy; binarized at train-set median):

| Feature | Regression | Classification | Chance |
|---|---|---|---|
| `luminance_mean` | r, R² | AUC, bal_acc | 0.0 / 0.5 |
| `contrast_rms` | r, R² | AUC, bal_acc | 0.0 / 0.5 |
| `position_in_movie` | r, R² | AUC, bal_acc | 0.0 / 0.5 |
| `narrative_event_score` | r, R² | AUC, bal_acc | 0.0 / 0.5 |

JSON keys: `test/reg_<feature>_corr`, `test/reg_<feature>_r2`, `test/cls_<feature>_auc`, `test/cls_<feature>_bal_acc`. Same with `val/` prefix for val split.

### 1.2 Subject-side metrics (4 numbers)

| Metric | Type | Chance | JSON key |
|---|---|---|---|
| `subject/age_reg/corr` | regression r on continuous age | 0.0 | `test/subject/age_reg/corr` |
| `subject/age_reg/r2` | R² on continuous age | 0.0 | `test/subject/age_reg/r2` |
| `subject/sex/auc` | binary classification AUC | 0.5 | `test/subject/sex/auc` |
| `subject/sex/bal_acc` | binary classification bal_acc | 0.5 | `test/subject/sex/bal_acc` |

Subject metrics are an **anti-target** for stimulus-driven SSL — a method that improves stim probes by leaking subject identity is failing the assignment, not winning at it.

### 1.3 Movie-ID retrieval (2 numbers)

Discretize `position_in_movie` into 20 equal-width bins (per train-set min/max); train a 20-way classifier on encoder embeddings.

| Metric | Chance | JSON key |
|---|---|---|
| `movie_id/top1` | 0.05 | `test/movie_id/top1` |
| `movie_id/top5` | 0.25 | `test/movie_id/top5` |

This is the **upper-bound check on stimulus-locked information** — any encoder that fails to exceed CorrCA-5's `0.136 / 0.448` (raw_corrca tier1, see `docs/tier1_results.md`) is adding noise on top of what's linearly recoverable.

## 2. Canonical probe heads — one per metric family

**kc-pool features are the default and only supported pooling**. Mean-pool is dropped (loses cross-channel structure that narrative needs).

Encoder forward: `eeg [n_windows, C, T] → encoder.encode_tokens → tokens [B, C×T×P, D] → reshape [B, C, T, P, D] → mean over P → permute & reshape [B, T, C×D] → mean over T → [B, C×D]`. Default `C=5, D=64 → 320-d per clip`. Average over `n_passes=20` random clip draws per recording.

Then per metric family, the **single canonical probe head**:

| Metric family | Probe head | Solver | Reg | Why |
|---|---|---|---|---|
| **Stim regression** (4 features × {r, R²}) | `sklearn.Ridge(α=1.0)` | closed-form | L2 | Closed-form, deterministic, regularized; what PR #15 used; outperforms gradient-trained linear at 320-d feature size |
| **Stim classification** (4 features × {AUC, bal_acc}; binarized at train median) | `sklearn.LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=1000)` | LBFGS | L2 | Closed-form-ish; deterministic; matches Ridge regularization scale; existing `trivial_ridge_baseline.py` already used this |
| **Subject age** (regression r, R²) | `sklearn.Ridge(α=1.0)` | closed-form | L2 | Continuous target; same family as stim regression |
| **Subject sex** (AUC, bal_acc) | `sklearn.LogisticRegression(C=1, lbfgs)` | LBFGS | L2 | Binary target; same family as stim cls |
| **Movie ID** (20-class top-1, top-5) | `sklearn.LogisticRegression(C=1, multi_class='multinomial', solver='lbfgs', max_iter=2000)` | LBFGS | L2 | Multi-class closed-form-ish; matches raw_corrca tier1 protocol from `tier1_results.md` |

**All 5 probe heads are deterministic** given (encoder weights, n_passes seed, train/val/test splits). No SGD, no random init, no learning-rate hyperparam to tune. **One run, one number, fully reproducible.**

**Train/eval discipline**:
- Fit each probe on the **train-split kc-pool features** (averaged over n_passes).
- Standardize features with **train-only** mean/std before fitting.
- Eval on val and test splits (both keys included in JSON).
- Per-recording predictions are saved to NPZ for bootstrap.

## 3. Bootstrap (mandatory)

Every reported metric must come with a recording-level bootstrap interval. We use **B = 2000** resamples by default.

**Why mandatory:** raw test-set metrics are dominated by recording-level variance — a single hard or easy recording can swing the headline number by ±0.05. The bootstrap collapses that variance into a per-seed point estimate that's stable across re-evaluations.

### 3.1 Bootstrap protocol

For each (encoder seed, probe seed) pair:

1. Run `unified_probe_eval.py` with `--save_predictions_dir=...` → per-recording `(prediction, target)` saved to disk.
2. Compute bootstrap point estimates by resampling **at the recording level** (not the clip level — clips within a recording are highly correlated):
   - For B=2000 iterations, resample recordings with replacement.
   - Compute the metric on the resampled set.
   - Record the bootstrap mean (and 95% CI from the 2.5th / 97.5th percentiles if needed).

Driver script: `scripts/bootstrap_trivial_perseed.py` (existing utility); operates on the NPZ produced by `unified_probe_eval.py`.

### 3.2 What to report

In any results table, prefer:

**Bootstrap mean ± 1σ across encoder seeds** (5-seed mean + std):
- `0.207 ± 0.018`

Never report a raw single-evaluation metric without bootstrap context.

## 4. Seed-level significance testing (on request)

When the question is "is method A significantly different from method B", do a paired test over matched encoder seeds:

- **Paired t-test** on the per-seed bootstrap means, A vs B.
- Report Δ, t-statistic, two-sided p-value.
- Only run this when explicitly asked.

## 5. Standard reporting template

Every results section in any memo / doc / paper must conform to this skeleton:

```
Method: <name>
Encoder seeds: <list>
Probe seed: <int>
Pretrain config: nw_ws=<X_Y>, grad_steps=<N>, ...
Eval driver: scripts/unified_probe_eval.py
Bootstrap: B=2000, recording-level resampling

| Metric                       | Test (5-seed mean ± 1σ) |
|------------------------------|-------------------------|
| reg_luminance_mean_corr      | ...                     |
| reg_contrast_rms_corr        | ...                     |
| reg_position_in_movie_corr   | ...                     |
| reg_narrative_event_score_corr| ...                    |
| cls_luminance_mean_auc       | ...                     |
| cls_contrast_rms_auc         | ...                     |
| cls_position_in_movie_auc    | ...                     |
| cls_narrative_event_score_auc| ...                     |
| subject/age_reg/corr         | ...                     |
| subject/sex/auc              | ...                     |
| movie_id/top1                | ...                     |
| movie_id/top5                | ...                     |
```

Optional: include cls bal_acc and reg r² alongside.

When a paired comparison is reported, add:

```
Paired t-test (5 enc seeds, matched probe seed):
| Metric | Δ (A − B) | t | p (two-sided) |
|---|---|---|---|
...
```

## 6. Worked example: Phase D nw2_ws4 (5-seed kc+Ridge, 2026-05-04)

```
Method: Phase D nw2_ws4
Encoder seeds: [42, 123, 456, 789, 2025]
Probe seed: 42
Pretrain config: nw=2, ws=4, ~1.1k grad steps
Eval driver: scripts/unified_probe_eval.py
Bootstrap: B=2000, recording-level

| Metric                          | Test (5-seed mean ± 1σ) |
|---------------------------------|-------------------------|
| reg_luminance_mean_corr         | 0.225 ± 0.018           |
| reg_contrast_rms_corr           | 0.220 ± 0.017           |
| reg_position_in_movie_corr      | 0.222 ± 0.016           |
| reg_narrative_event_score_corr  | 0.156 ± 0.023           |
```

## 7. Anti-patterns (don't do these)

- ❌ Report a single test-set Pearson r without specifying bootstrap, n_passes, or seed protocol.
- ❌ Report online-probe correlations (`val/reg_*_corr` from training-time validation) as headline numbers — these come from a noisy SGD probe trained at every checkpoint, not the canonical Ridge probe.
- ❌ Use mean-pool instead of kc-pool. Mean-pool collapses cross-channel structure that narrative depends on.
- ❌ Use `probe_eval.py` with `probe_type=linear` or `mlp` for headline numbers. Those are diagnostic / exploratory probes, not canonical. Use `unified_probe_eval.py`.
- ❌ Compare across runs that used different `(n_windows, window_size_seconds)`, `(norm_mode)`, or `(corrca_filters)` — these change the input feature space and invalidate cross-run comparison. Match the eval config to the encoder's training config.
- ❌ Cherry-pick the best epoch by online probe and report only that. Track `best_by_online_probe.pth.tar` AND `latest.pth.tar`, eval both, and report which one was used. They should give similar Ridge numbers; if they don't, that's diagnostic information.
- ❌ Drop subject-side metrics from the reporting. They're the anti-target — a method that improves stim probes by leaking subject identity is failing the assignment, not winning at it.
- ❌ Add new probe heads to the canonical set without updating this guide. If a new probe variant is needed, propose it as a *diagnostic* probe (separate output) — not as a replacement for Ridge / LogReg above.

## 8. Driver invocation

The single canonical eval invocation:

```bash
PYTHONPATH=. uv run --group eeg python scripts/unified_probe_eval.py \
    --checkpoint=/path/to/<flavor>.pth.tar \
    --n_windows=<NW> --window_size_seconds=<WS> \
    --norm_mode=per_recording --corrca_filters=corrca_filters.npz \
    --n_passes=20 --seed=<probe_seed> \
    --output_json=<results_path>.json \
    --save_predictions_dir=<predictions_dir>
```

Output:
- `<output_json>` — single JSON with all 18 headline metrics + protocol metadata.
- `<save_predictions_dir>/test_seed<seed>.npz` — per-recording predictions for all 5 probe families, suitable for bootstrap and paired-t.

For SLURM, an sbatch wrapper around this driver is `scripts/probe_unified.sbatch` (to be created when needed). Until then, embed the call in any tier-specific sbatch.

## 9. Versioning

This document is **the source of truth** for evaluation. When changes are needed:
1. Update `unified_probe_eval.py` and re-run any open evaluations.
2. Update this doc with the new protocol.
3. Note the change date in §10 below.
4. Re-eval any historical numbers being compared in published artifacts.

## 10. Changelog

- **2026-05-03** — Initial doc. Mandatory bootstrap + paired-t. Multiple probe types per family.
- **2026-05-04** — **Canonical protocol locked**: one probe head per family (Ridge for reg, LogReg LBFGS for cls/multinomial), kc-pool only (mean-pool dropped), `unified_probe_eval.py` is the canonical driver. All 18 headline metrics in one JSON. Reverted multi-probe sweep (was diagnostic clutter).
