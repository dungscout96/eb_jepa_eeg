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

### 3.1 The three measurement layers (L1 / L2 / L3)

Every metric exists at three distinct layers. Confusing them is the most common source of "why does X say 0.156 and Y say 0.114" arguments. Always state which layer a number is at.

| Layer | Definition | Captures | Where it's stored |
|---|---|---|---|
| **L1 — raw test r** | Single Ridge fit on train clips → `pearsonr(pred, target)` over all test predictions, one number per (encoder seed, probe seed) | Nothing about uncertainty — single point estimate | `results/canonical/<method>/<enc_seed>/metrics.json` |
| **L2 — per-seed bootstrap** | For one (enc seed, probe seed): B=2000 recording-level resamples → mean + 95% CI of resampled Pearson r | **Test-set sampling** variance (would the number change with a different draw of 108 test recordings?) | `results/canonical/<method>/<enc_seed>/bootstrap.json` |
| **L3 — across-seed L2** | Train K=5 encoder seeds → take per-seed L2 bootstrap mean → aggregate as `mean ± 1σ` over the K per-seed bootstrap means | Both test-set sampling and **encoder-training** variance — the headline number | `results/canonical/<method>/aggregate.json` |

**Why L2 might ≠ L1**: Pearson r is non-linear under resampling, so bootstrap-mean r is not guaranteed to equal raw r. When L2 differs materially from L1, that itself is diagnostic information (skewed per-recording residuals); state both layers.

**Which one to report**: **always L3** (if K=5 seeds are available). Single-seed L2 is a fallback when retraining isn't feasible. Bare L1 is diagnostic only.

**Why 5-seed × bootstrap is canonical and not just one or the other**:
- Encoder-training variance σ on narrative is ~0.020; per-seed bootstrap CI half-width is ~0.015. Encoder seed is the **dominant** noise source.
- Reporting only L2 (single-seed bootstrap) hides encoder variance → understates uncertainty.
- Reporting only "5-seed mean of L1" wastes the bootstrap step → harder to detect when test-set noise dominates.
- L3 captures both sources cleanly and is what gets reported in papers.

### 3.2 Bootstrap protocol

For each (encoder seed, probe seed) pair:

1. Run `unified_probe_eval.py` with `--save_predictions_dir=...` → per-recording `(prediction, target)` saved to NPZ → produces L1.
2. Run `scripts/bootstrap_trivial_perseed.py` on the NPZ:
   - Resample recordings with replacement (not clips — clips within a recording are highly correlated).
   - For B=2000 iterations, compute the metric on the resampled set.
   - Record bootstrap mean and 95% CI (2.5th / 97.5th percentiles) → **L2**.
3. Aggregate L2 across encoder seeds → **L3**: `mean ± 1σ` over K per-seed bootstrap means.

### 3.3 What to report

Every results table must state all three layers (or explicitly note which is omitted):

```
| Metric                       | L1 (5-seed raw mean) | L3 (5-seed mean of L2 ± 1σ) |
|------------------------------|----------------------|-----------------------------|
| reg_narrative_event_score_corr | 0.114 ± 0.020      | 0.156 ± 0.023               |
```

L2 (per-seed bootstrap mean + CI) goes in supplementary tables for full transparency.

**Headline number = L3.** Never report a bare L1 in a comparison without flagging it.

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
Reporting: L3 = 5-seed mean of per-seed bootstrap means ± 1σ

| Metric                       | L1 (raw 5-seed mean) | L3 (5-seed mean of L2 ± 1σ) |
|------------------------------|----------------------|-----------------------------|
| reg_luminance_mean_corr      | ...                  | ...                         |
| reg_contrast_rms_corr        | ...                  | ...                         |
| reg_position_in_movie_corr   | ...                  | ...                         |
| reg_narrative_event_score_corr| ...                 | ...                         |
| cls_luminance_mean_auc       | ...                  | ...                         |
| cls_contrast_rms_auc         | ...                  | ...                         |
| cls_position_in_movie_auc    | ...                  | ...                         |
| cls_narrative_event_score_auc| ...                  | ...                         |
| subject/age_reg/corr         | ...                  | ...                         |
| subject/sex/auc              | ...                  | ...                         |
| movie_id/top1                | ...                  | ...                         |
| movie_id/top5                | ...                  | ...                         |
```

Optional supplement: per-seed L2 bootstrap mean + 95% CI table.

When K < 5 (e.g. only 1 seed available), state that explicitly and report L2 ± 95% CI in place of L3 — never hide the layer change.

When a paired comparison is reported, add:

```
Paired t-test (5 enc seeds, matched probe seed):
| Metric | Δ (A − B) | t | p (two-sided) |
|---|---|---|---|
...
```

## 6. Worked example: Phase D nw2_ws4 (canonical Protocol B, locked 2026-05-04)

**Protocol locked**: `unified_probe_eval.py` with `train_order=True` (commit c805692), n_passes=20 train-flatten Ridge α=1, kc-pool features, B=2000 recording-level bootstrap, **probe_seed=42 fixed across all encoder seeds**, 5 encoder seeds {42, 123, 456, 789, 2025}.

All artifact paths use the **`pB_` prefix** for grep-traceability.

**Reproducibility check (issue8 latest, per-seed narrative — exact match against doc):**

| seed | doc | reproduced |
|---|---|---|
| 42 | 0.137 | 0.1367 ✓ |
| 123 | 0.160 | 0.1601 ✓ |
| 456 | 0.148 | 0.1477 ✓ |
| 789 | 0.142 | 0.1410 ✓ |
| 2025 | 0.194 | 0.1938 ✓ |
| **mean ± 1σ** | **0.156 ± 0.023** | **0.1559 ± 0.023** ✓ |

**Full canonical results (issue8 vs issue10) — equivalent within 1σ on every metric:**

| Metric | pB_phaseD_issue8 (L3) | pB_phaseD_issue10best (L3) | pB_phaseD_issue10latest (L3) |
|---|---|---|---|
| reg_luminance_mean_corr | 0.2245 ± 0.018 | 0.2259 ± 0.018 | 0.2255 ± 0.018 |
| reg_contrast_rms_corr | 0.2195 ± 0.017 | 0.2231 ± 0.014 | 0.2235 ± 0.014 |
| reg_position_in_movie_corr | 0.2219 ± 0.016 | 0.2255 ± 0.016 | 0.2256 ± 0.016 |
| **reg_narrative_event_score_corr** | **0.1562 ± 0.023** | 0.1553 ± 0.023 | 0.1554 ± 0.024 |
| cls_luminance_mean_auc | 0.5791 ± 0.004 | 0.5783 ± 0.003 | 0.5785 ± 0.003 |
| cls_contrast_rms_auc | 0.5717 ± 0.011 | 0.5722 ± 0.009 | 0.5724 ± 0.010 |
| cls_position_in_movie_auc | 0.6092 ± 0.007 | 0.6105 ± 0.007 | 0.6105 ± 0.007 |
| cls_narrative_event_score_auc | 0.5467 ± 0.012 | 0.5467 ± 0.012 | 0.5465 ± 0.012 |
| subject/age_reg/corr | 0.3837 ± 0.014 | 0.3895 ± 0.014 | 0.3902 ± 0.014 |
| subject/sex/auc | 0.7313 ± 0.017 | 0.7273 ± 0.019 | 0.7269 ± 0.020 |
| movie_id/top1 (chance 0.05) | 0.094 ± 0.024 | 0.092 ± 0.017 | 0.094 ± 0.020 |
| movie_id/top5 (chance 0.25) | 0.452 ± 0.024 | 0.459 ± 0.032 | 0.459 ± 0.032 |

L1 (raw test r per seed) and L3 (5-seed mean of L2) agree to within 0.001 on every metric.

Source artifacts:
- Per-seed predictions NPZ: `predictions/canonical/pB_phaseD_<fam>/seed<S>/test_seed42.npz`
- Per-seed metrics: `results/canonical/pB_phaseD_<fam>/seed<S>/metrics.json`
- Per-seed bootstrap: `results/bootstrap/pB_phaseD_<fam>_<seed>.json`
- L3 aggregate: `results/bootstrap/pB_phaseD_<fam>_L3.json`

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
- **2026-05-04 (late)** — **L1 / L2 / L3 framework added** (§3.1) and **iteration-order issue surfaced**. The doc previously cited narr 0.156 ± 0.023 (5-seed Phase D), reproducible only under `train_order=True` train extraction (the `c805692` patched version). The pre-patch sequential extraction yields narr 0.116 ± 0.022 on the same checkpoints. Both are valid Ridge fits — they differ because the train-extraction order changes which random clip-starts are drawn for each (rec, pass) cell. **Protocol locking pending.** Worked example in §6 cleared until a single train-extraction order is adopted as canonical.
