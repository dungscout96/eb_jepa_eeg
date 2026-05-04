# Evaluation Guide

This is the standard evaluation protocol for any encoder run in this repo. Every run reported in any document, table, or memo should follow this protocol unless explicitly noted otherwise. **Bootstrap is mandatory; seed-level significance testing is on request only.**

## 1. Metrics that MUST be tracked

For every encoder evaluated against the HBN movie probe set, the following must all appear in the per-recording predictions JSON and be reported (or summarized) in any results table.

### 1.1 Stimulus features — regression + classification

Four movie-derived features, each evaluated as both regression and classification:

| Feature | Regression metric | Classification metric | Chance |
|---|---|---|---|
| `luminance_mean` | Pearson r, R² | AUC, balanced accuracy, accuracy | 0.0, 0.5 |
| `contrast_rms` | Pearson r, R² | AUC, balanced accuracy, accuracy | 0.0, 0.5 |
| `position_in_movie` | Pearson r, R² | AUC, balanced accuracy, accuracy | 0.0, 0.5 |
| `narrative_event_score` | Pearson r, R² | AUC, balanced accuracy, accuracy | 0.0, 0.5 |

JSON keys produced by `experiments/eeg_jepa/probe_eval.py` and `scripts/trivial_ridge_baseline.py`:
- `test/reg_<feature>_corr`, `test/reg_<feature>_r2`
- `test/cls_<feature>_acc`, `test/cls_<feature>_auc`, `test/cls_<feature>_bal_acc`

### 1.2 Subject-side metrics (all four)

These probe whether the encoder leaks subject identity / fingerprints (an anti-target for stimulus-driven SSL).

| Metric | Type | Chance |
|---|---|---|
| `subject/age_reg/corr` | regression r on continuous age | 0.0 |
| `subject/age_reg/r2` | R² on continuous age | 0.0 |
| `subject/sex/auc` | binary classification AUC | 0.5 |
| `subject/sex/bal_acc` | binary classification balanced accuracy | 0.5 |

JSON keys: `test/subject/age_reg/corr`, `test/subject/age_reg/r2`, `test/subject/sex/auc`, `test/subject/sex/bal_acc`.

### 1.3 Movie-ID retrieval

Discretize `position_in_movie` into 20 equal-width bins; train a 20-way classifier on encoder embeddings.

| Metric | Chance |
|---|---|
| `movie_id_top1` | 0.05 (= 1/20) |
| `movie_id_top5` | 0.25 (= 5/20) |

This is the *upper-bound check on stimulus-locked information* — any encoder that fails to exceed CorrCA-5's `0.136 / 0.448` (raw_corrca tier1, see `docs/tier1_results.md`) is adding noise on top of what's linearly recoverable.

JSON keys: `test/movie_id/top1`, `test/movie_id/top5`.

## 2. Bootstrap (mandatory)

Every reported metric must come with a recording-level bootstrap interval. We use **B = 2000** resamples by default.

**Why mandatory:** raw test-set Pearson r values are dominated by recording-level variance — a single hard or easy recording can swing the headline number by ±0.05. The bootstrap collapses that variance into a per-seed point estimate that's stable across re-evaluations.

### 2.1 Bootstrap protocol

For each (encoder seed, probe seed) pair:

1. Run probe eval → save per-clip `(prediction, target, recording_id)` predictions to disk (`save_predictions_dir` flag).
2. Compute bootstrap point estimates by resampling **at the recording level** (not the clip level — clips within a recording are highly correlated):
   - For B=2000 iterations, resample recordings with replacement.
   - Within each resampled recording, use *all* clips.
   - Compute the metric on the resampled set.
   - Record the bootstrap mean and 95% CI from the 2.5th / 97.5th percentiles.

This is implemented in:
- `scripts/bootstrap_probe_eval.py` (general bootstrap utilities, used internally)
- `scripts/bootstrap_trivial_perseed.py` (driver script for trivial baselines + per-seed t-test against chance)

For encoder runs (Phase D, Cell B, Lever 1, etc.), the same bootstrap utilities apply — point them at the predictions directory produced by `probe_eval.py` or `trivial_ridge_baseline.py`.

### 2.2 What to report

In any results table, prefer one of:

**Bootstrap mean ± 95% CI half-width per seed**, then average across seeds:
- `0.207 ± 0.018` (5-seed mean of per-seed bootstrap means; spread is the 5-seed std)
- This is the format for headline tables.

**Bootstrap mean per seed, listed**:
- Useful when the per-seed spread is the point of interest.

Never report the raw test-set Pearson r without bootstrap context — recording-level variance makes that number unreliable to within ±0.05 even in 5-seed runs.

## 3. Seed-level significance testing (on request)

When the question is "is method A significantly different from method B", do a paired test over matched encoder seeds (5 seeds × 5 probe seeds = 25 pairs typically; or 5 encoder seeds × 1 probe seed = 5 pairs):

- **Paired t-test** on the per-seed bootstrap means, A vs B.
- Report Δ, t-statistic, two-sided p-value.
- Only run this when explicitly asked. Do not bake it into routine reporting.

Implementation: `scripts/bootstrap_trivial_perseed.py` already exposes the t-test machinery (`t_test_against_chance` and the per-seed pipeline); paired t-tests across two methods are a thin wrapper.

## 4. Standard reporting template

Every results section in any memo / doc / paper must conform to this skeleton:

```
Method: <name>
Encoder seeds: <list>
Probe seed(s): <list>
Bootstrap: B=2000, recording-level resampling
Protocol: kc-pool + Ridge(α=1.0) + n_passes=20  [or whatever the actual protocol is]

| Probe | r (5-seed mean ± 1σ) | AUC (5-seed mean ± 1σ) | Bootstrap CI half-width |
|---|---|---|---|
| luminance | ... | ... | ... |
| contrast | ... | ... | ... |
| position | ... | ... | ... |
| narrative | ... | ... | ... |

Subject metrics: age_corr=..., sex_auc=..., age_auc=..., age_bal_acc=...
Movie-ID: top1=..., top5=...
```

When a paired comparison is being reported (e.g., "Cell B vs Phase D"), add:

```
Paired t-test (5 enc seeds, matched probe seed):
| Probe | Δ (A − B) | t | p (two-sided) |
|---|---|---|---|
...
```

## 5. Worked example: Cell B vs Phase D, 2026-05-03

Cell B 5-seed × kc + Ridge:

```
| Probe | r (5-seed mean) | Phase D r (PR #15) | Δ |
|---|---|---|---|
| luminance | 0.207 | 0.208 | −0.001 |
| contrast | 0.166 | 0.159 | +0.007 |
| position | 0.158 | 0.144 | +0.014 |
| narrative | 0.090 | 0.090 | +0.000 |
```

These are 5-seed test means from `scripts/trivial_ridge_baseline.py` with `--keep_channels=True` `--n_passes=20`. Per-recording predictions are saved at `/projects/bbnv/kkokate/eb_jepa_eeg/predictions/lever1/cellB_*_seed42/` for the bootstrap to consume. Bootstrap pass + paired-t against Phase D's saved predictions is the next step before this enters the memo as a confirmed result.

## 6. Anti-patterns (don't do these)

- ❌ Report a single test-set Pearson r without specifying bootstrap, n_passes, or seed protocol.
- ❌ Report online-probe correlations (`val/reg_*_corr` from training-time validation) as headline numbers — these come from a noisy SGD probe, not Ridge, and are not comparable to PR #15 numbers.
- ❌ Use `probe_eval.py` default flags (`probe_type=linear`, `keep_channels=False`) for headline comparisons. Use `scripts/trivial_ridge_baseline.py` with `--keep_channels=True` instead.
- ❌ Compare across runs that used different `(n_windows, window_size_seconds)`, `(norm_mode)`, or `(corrca_filters)` — these change the input feature space and invalidate cross-run comparison. Match the eval config to the encoder's training config.
- ❌ Cherry-pick the best epoch by online probe and report only that. The settled policy (memo §2) is to track `best_by_online_probe.pth.tar` AND `latest.pth.tar`, eval both, and report which one was used. The two should give similar Ridge numbers; if they don't, that's diagnostic information.
- ❌ Drop subject-side metrics from the reporting. They're the anti-target — a method that improves stim probes by leaking subject identity is failing the assignment, not winning at it.
