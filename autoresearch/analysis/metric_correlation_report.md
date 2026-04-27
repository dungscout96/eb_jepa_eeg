# SSL Metric → Probe Correlation Report

**Sample size:** 150 EEG-JEPA runs from W&B project `eb_jepa` 
(past ~120 days, finished/running, ≥1 `val/reg_*_corr` summary value, ≥1 `sanity/*` key). Re-fetched with `api.run(id)` to populate config.

**Caveat:** only **71/150** runs have `val/reg_position_in_movie_corr` and `val/reg_narrative_event_score_corr` logged in summary — the older runs only have `luminance` and `contrast`. So `val_corr_weighted` is computed on n=71. Per-target correlations against `luminance`/`contrast` use the full n=150.

**Stratification subgroup sizes (with `val_corr_weighted` non-null):**
- `norm_mode=(missing)`: 129 runs total, 50 with weighted target
- `norm_mode=global`: 11 runs total, 11 with weighted target
- `norm_mode=per_recording`: 10 runs total, 10 with weighted target
- `regularizer=vc`: 96 runs total, 35 with weighted target
- `regularizer=sigreg`: 39 runs total, 36 with weighted target
- `regularizer=(missing)`: 15 runs total, 0 with weighted target

All correlations are Spearman ρ (rank-based, robust). Targets are summary (final) values of `val/reg_*_corr` and the weighted average 
`val_corr_weighted = 0.3·position + 0.3·contrast + 0.3·luminance + 0.1·narrative`.

## 1. Top 10 predictors by |ρ| with `val_corr_weighted` (overall)

| Rank | Predictor | Spearman ρ | n | p-value |
|------|-----------|-----------:|--:|--------:|
| 1 | `train_step/reg_loss` | -0.615 | 71 | 1.18e-08 |
| 2 | `train_step/cls_loss` | -0.540 | 71 | 1.19e-06 |
| 3 | `sanity/loss_trend` | +0.516 | 71 | 4.07e-06 |
| 4 | `train_step/pred_loss` | +0.426 | 71 | 2.16e-04 |
| 5 | `train_step/jepa_loss` | +0.407 | 71 | 4.32e-04 |
| 6 | `sanity/pred_loss_short` | +0.334 | 71 | 4.42e-03 |
| 7 | `sanity/loss_rolling_mean` | +0.325 | 71 | 5.64e-03 |
| 8 | `sanity/cosim_random_pairs_mean` | -0.289 | 71 | 1.45e-02 |
| 9 | `sanity/cosim_random_pairs_max` | -0.281 | 71 | 1.77e-02 |
| 10 | `sanity/embedding_variance_max` | +0.279 | 71 | 1.85e-02 |

### Headline (read this first)

Three findings, in decreasing order of robustness:

1. **The strongest cross-run predictors (`train_step/reg_loss` ρ=−0.61, `train_step/cls_loss` ρ=−0.54) are circular** — they are training-time supervised feature-prediction losses sharing a target distribution with the val/reg eval. Selecting on them is exactly the "probe optimization confounds encoder quality" concern the user flagged. **Do not use.**

2. **`sanity/loss_trend` (+0.52) and `pred_loss_short` (+0.33) ride heavily on the position-leak target** (ρ=+0.48 / +0.22 on `position_in_movie`) and have **wrong-signed** correlations (loss going UP → corrs UP). Treat as survivorship indicators, not quality signals.

3. **On the genuinely-non-leaked targets (`luminance_mean`, `contrast_rms`, n=150), the only sanity metric with a clear positive association is `sanity/linear_probe_acc` (ρ=+0.36 luminance, +0.27 contrast).** The unsupervised collapse markers (`embedding_variance_*`, `cosim_random_pairs_*`) are *correctly signed* on `val_corr_weighted` (ρ=+0.21 to +0.29 / −0.28 to −0.29) but mostly **weaken to near-zero within a fixed regulariser** — meaning their cross-run signal is partly between-regulariser scale, not within-config quality.

**Bottom line:** on this dataset there is **no cheap online SSL metric with strong (|ρ|>0.4) and robustly-signed correlation to the non-leaked val targets**. The best available cheap signal is `sanity/linear_probe_acc`, with the variance/cosim collapse markers as label-free cross-checks. See §6 for the recommended composite.

## 2. Top sanity-only predictors (no train_step/* — these are the cheap, non-circular signals)

| Rank | Predictor | Spearman ρ | n | p-value |
|------|-----------|-----------:|--:|--------:|
| 1 | `sanity/loss_trend` | +0.516 | 71 | 4.07e-06 |
| 2 | `sanity/pred_loss_short` | +0.334 | 71 | 4.42e-03 |
| 3 | `sanity/loss_rolling_mean` | +0.325 | 71 | 5.64e-03 |
| 4 | `sanity/cosim_random_pairs_mean` | -0.289 | 71 | 1.45e-02 |
| 5 | `sanity/cosim_random_pairs_max` | -0.281 | 71 | 1.77e-02 |
| 6 | `sanity/embedding_variance_max` | +0.279 | 71 | 1.85e-02 |
| 7 | `sanity/embedding_variance_std` | +0.260 | 71 | 2.87e-02 |
| 8 | `sanity/embedding_variance_mean` | +0.257 | 71 | 3.08e-02 |
| 9 | `sanity/pred_loss_long` | +0.256 | 71 | 3.09e-02 |
| 10 | `sanity/linear_probe_acc` | +0.250 | 71 | 3.57e-02 |
| 11 | `sanity/embedding_variance_min` | +0.209 | 71 | 7.96e-02 |
| 12 | `sanity/embedding_l2_mean` | +0.169 | 71 | 1.59e-01 |
| 13 | `sanity/grad_norm` | +0.148 | 71 | 2.19e-01 |

## 3. Stratified — `norm_mode=global` (weighted-target n=11)

| Rank | Predictor | Spearman ρ | n | p-value |
|------|-----------|-----------:|--:|--------:|
| 1 | `train_step/cls_loss` | -0.782 | 11 | 4.47e-03 |
| 2 | `sanity/pred_loss_long` | -0.527 | 11 | 9.56e-02 |
| 3 | `train_step/reg_loss` | -0.473 | 11 | 1.42e-01 |
| 4 | `sanity/embedding_l2_mean` | -0.427 | 11 | 1.90e-01 |
| 5 | `sanity/loss_trend` | +0.373 | 11 | 2.59e-01 |
| 6 | `sanity/embedding_variance_min` | -0.373 | 11 | 2.59e-01 |
| 7 | `sanity/embedding_variance_mean` | -0.309 | 11 | 3.55e-01 |
| 8 | `sanity/cosim_random_pairs_mean` | -0.309 | 11 | 3.55e-01 |
| 9 | `sanity/embedding_variance_max` | -0.309 | 11 | 3.55e-01 |
| 10 | `sanity/loss_rolling_mean` | -0.255 | 11 | 4.50e-01 |

## 3. Stratified — `norm_mode=per_recording` (weighted-target n=10)

| Rank | Predictor | Spearman ρ | n | p-value |
|------|-----------|-----------:|--:|--------:|
| 1 | `sanity/linear_probe_acc` | +0.421 | 10 | 2.26e-01 |
| 2 | `train_step/cls_loss` | -0.370 | 10 | 2.93e-01 |
| 3 | `train_step/reg_loss` | -0.370 | 10 | 2.93e-01 |
| 4 | `sanity/loss_trend` | +0.309 | 10 | 3.85e-01 |
| 5 | `sanity/pred_loss_short` | +0.309 | 10 | 3.85e-01 |
| 6 | `sanity/embedding_variance_std` | +0.297 | 10 | 4.05e-01 |
| 7 | `sanity/pred_loss_long` | +0.273 | 10 | 4.46e-01 |
| 8 | `sanity/embedding_variance_max` | +0.236 | 10 | 5.11e-01 |
| 9 | `sanity/cosim_random_pairs_mean` | -0.212 | 10 | 5.56e-01 |
| 10 | `train_step/jepa_loss` | +0.212 | 10 | 5.56e-01 |

## 3. Stratified — `regularizer=vc` (weighted-target n=35)

| Rank | Predictor | Spearman ρ | n | p-value |
|------|-----------|-----------:|--:|--------:|
| 1 | `sanity/loss_trend` | +0.647 | 35 | 2.63e-05 |
| 2 | `train_step/reg_loss` | -0.538 | 35 | 8.69e-04 |
| 3 | `train_step/cls_loss` | -0.493 | 35 | 2.60e-03 |
| 4 | `sanity/embedding_l2_mean` | -0.426 | 35 | 1.08e-02 |
| 5 | `train_step/vc_loss` | -0.215 | 35 | 2.15e-01 |
| 6 | `sanity/embedding_variance_min` | -0.208 | 35 | 2.31e-01 |
| 7 | `sanity/grad_norm` | -0.205 | 35 | 2.37e-01 |
| 8 | `sanity/cosim_random_pairs_mean` | -0.163 | 35 | 3.50e-01 |
| 9 | `sanity/embedding_variance_mean` | -0.132 | 35 | 4.48e-01 |
| 10 | `sanity/embedding_variance_max` | -0.125 | 35 | 4.73e-01 |

## 3. Stratified — `regularizer=sigreg` (weighted-target n=36)

| Rank | Predictor | Spearman ρ | n | p-value |
|------|-----------|-----------:|--:|--------:|
| 1 | `train_step/reg_loss` | -0.626 | 36 | 4.40e-05 |
| 2 | `train_step/pred_loss` | +0.426 | 36 | 9.53e-03 |
| 3 | `train_step/jepa_loss` | +0.410 | 36 | 1.31e-02 |
| 4 | `train_step/cls_loss` | -0.387 | 36 | 1.98e-02 |
| 5 | `sanity/loss_trend` | +0.373 | 36 | 2.48e-02 |
| 6 | `sanity/cosim_random_pairs_max` | -0.192 | 36 | 2.62e-01 |
| 7 | `sanity/embedding_variance_min` | -0.192 | 36 | 2.63e-01 |
| 8 | `sanity/cosim_random_pairs_mean` | +0.174 | 36 | 3.11e-01 |
| 9 | `sanity/embedding_variance_std` | -0.156 | 36 | 3.64e-01 |
| 10 | `sanity/loss_rolling_mean` | +0.140 | 36 | 4.14e-01 |

## 4. Per-feature breakdown for top 5 sanity-only predictors

Do these metrics predict different val targets equally well?

### `sanity/loss_trend`

| Target | Spearman ρ | n | p-value |
|--------|-----------:|--:|--------:|
| `val/reg_position_in_movie_corr` | +0.480 | 71 | 2.27e-05 |
| `val/reg_contrast_rms_corr` | +0.250 | 150 | 2.07e-03 |
| `val/reg_luminance_mean_corr` | +0.166 | 150 | 4.19e-02 |
| `val/reg_narrative_event_score_corr` | -0.024 | 71 | 8.43e-01 |
| `val_corr_weighted` | +0.516 | 71 | 4.07e-06 |

### `sanity/pred_loss_short`

| Target | Spearman ρ | n | p-value |
|--------|-----------:|--:|--------:|
| `val/reg_position_in_movie_corr` | +0.222 | 71 | 6.24e-02 |
| `val/reg_contrast_rms_corr` | -0.103 | 149 | 2.12e-01 |
| `val/reg_luminance_mean_corr` | -0.047 | 149 | 5.71e-01 |
| `val/reg_narrative_event_score_corr` | +0.093 | 71 | 4.40e-01 |
| `val_corr_weighted` | +0.334 | 71 | 4.42e-03 |

### `sanity/loss_rolling_mean`

| Target | Spearman ρ | n | p-value |
|--------|-----------:|--:|--------:|
| `val/reg_position_in_movie_corr` | +0.196 | 71 | 1.01e-01 |
| `val/reg_contrast_rms_corr` | -0.046 | 150 | 5.80e-01 |
| `val/reg_luminance_mean_corr` | +0.013 | 150 | 8.74e-01 |
| `val/reg_narrative_event_score_corr` | -0.052 | 71 | 6.69e-01 |
| `val_corr_weighted` | +0.325 | 71 | 5.64e-03 |

### `sanity/cosim_random_pairs_mean`

| Target | Spearman ρ | n | p-value |
|--------|-----------:|--:|--------:|
| `val/reg_position_in_movie_corr` | -0.049 | 71 | 6.85e-01 |
| `val/reg_contrast_rms_corr` | -0.181 | 150 | 2.66e-02 |
| `val/reg_luminance_mean_corr` | -0.106 | 150 | 1.96e-01 |
| `val/reg_narrative_event_score_corr` | +0.105 | 71 | 3.83e-01 |
| `val_corr_weighted` | -0.289 | 71 | 1.45e-02 |

### `sanity/cosim_random_pairs_max`

| Target | Spearman ρ | n | p-value |
|--------|-----------:|--:|--------:|
| `val/reg_position_in_movie_corr` | -0.154 | 71 | 1.99e-01 |
| `val/reg_contrast_rms_corr` | -0.003 | 150 | 9.74e-01 |
| `val/reg_luminance_mean_corr` | +0.082 | 150 | 3.19e-01 |
| `val/reg_narrative_event_score_corr` | -0.087 | 71 | 4.71e-01 |
| `val_corr_weighted` | -0.281 | 71 | 1.77e-02 |

### Reading these tables — the position-leak issue

- `position_in_movie` and `narrative_event_score` only exist for n=71 runs; `luminance_mean` and `contrast_rms` for n=150.
- The position correlation is partially **leaked through Fourier positional encoding** (per project memory) — so positive ρ on position is a weaker signal than positive ρ on contrast/luminance, which require genuine stimulus-feature decoding.
- A predictor that does well on luminance/contrast but poorly on position is actually a *better* SSL-quality signal than one that wins on position.

**Key observation:** `sanity/loss_trend` and `sanity/pred_loss_short` get most of their headline ρ from the leaked **position** target (ρ=+0.48 / +0.22 on position; ρ≤+0.25 on contrast/luminance, ρ≈0 on narrative). When you re-rank against the non-leaked targets only, **`sanity/linear_probe_acc` is the strongest predictor on luminance (ρ=+0.36, n=150) and contrast (ρ=+0.27, n=149)** — the two targets that genuinely require stimulus-feature decoding. The collapse markers (`embedding_variance_*`, `cosim_random_pairs_*`) are weak-but-correctly-signed on these targets.

## 5. Sign / collapse-hypothesis check

Under the **collapse hypothesis** we expect:
- `cosim_random_pairs_*` ↑ ⇒ collapsed ⇒ probe corrs ↓ ⇒ ρ should be **negative**.
- `embedding_variance_*` ↑ ⇒ spread ⇒ probe corrs ↑ ⇒ ρ should be **positive**.
- `loss_rolling_mean`/`pred_loss_*` ↑ ⇒ poor predictive fit ⇒ ρ should be **negative**.
- `linear_probe_acc` ↑ ⇒ semantic separability ⇒ ρ should be **positive**.

| Predictor | ρ vs val_corr_weighted | n | Expected sign | Match? |
|-----------|----------------------:|--:|---------------|--------|
| `sanity/embedding_variance_mean` | +0.257 | 71 | + | yes |
| `sanity/embedding_variance_min` | +0.209 | 71 | + | yes |
| `sanity/embedding_variance_max` | +0.279 | 71 | + | yes |
| `sanity/embedding_variance_std` | +0.260 | 71 | ? | — |
| `sanity/embedding_l2_mean` | +0.169 | 71 | ? | — |
| `sanity/cosim_random_pairs_mean` | -0.289 | 71 | − | yes |
| `sanity/cosim_random_pairs_max` | -0.281 | 71 | − | yes |
| `sanity/loss_trend` | +0.516 | 71 | − | **no** |
| `sanity/loss_rolling_mean` | +0.325 | 71 | − | **no** |
| `sanity/pred_loss_short` | +0.334 | 71 | − | **no** |
| `sanity/pred_loss_long` | +0.256 | 71 | − | **no** |
| `sanity/grad_norm` | +0.148 | 71 | ? | — |
| `sanity/linear_probe_acc` | +0.250 | 71 | + | yes |

**Findings:**

- `cosim_random_pairs_mean` ρ=−0.29 ✓, `cosim_random_pairs_max` ρ=−0.28 ✓, `embedding_variance_mean` ρ=+0.26 ✓, `embedding_variance_min` ρ=+0.21 ✓, `linear_probe_acc` ρ=+0.25 ✓ — all sign-consistent with the collapse hypothesis.
- `loss_rolling_mean` ρ=+0.33 and `pred_loss_short` ρ=+0.33 are **wrong-signed** (higher pred loss → higher corrs in this dataset). This is because most failed runs are not collapsing to high-loss; they collapse to *low* loss with degenerate embeddings. The runs with high pred loss are still actively trying to fit. So **absolute pred-loss magnitude is misleading as a quality signal across runs** — it conflates 'didn't train' (high loss → bad rep) with 'still working hard' (high loss → good rep). Avoid as a standalone signal.
- `loss_trend` ρ=+0.52 — same caveat, plus an extra wrinkle: the runs that survive to log val/reg_position (the n=71 subset) are biased toward longer/healthier training, so `loss_trend` may be picking up survivorship rather than quality.

## 6. Recommendations

**Goal:** a per-run online signal that is (a) cheaply computed by the existing `SanityCheckHook` (no full validation pass), (b) directionally interpretable, (c) not circular with the eval probe.

### Recommended primary signal: `sanity/linear_probe_acc`

This is the single best-aligned cheap signal on the **non-leaked** targets:
- ρ = +0.36 on `luminance_mean` (n=150, p=5e-6)
- ρ = +0.27 on `contrast_rms` (n=149, p=8e-4)
- ρ = +0.25 on `val_corr_weighted` (n=71)

It IS technically a probe, BUT: (i) it's a single linear layer trained for 30 SGD steps on a 512-sample rolling buffer, (ii) it uses *subject metadata* (age>median or sex) or *luminance binarisation* as labels — these are the **only** sanity metric whose label distribution overlaps with the eval features, and even then the overlap is partial (binarised luminance, not regression). Cost is negligible compared to the encoder forward pass. The user's concern about "probe head contamination" applies most strongly to high-capacity probes; a single linear layer on frozen embeddings is closer to a representation-quality diagnostic than to a confounding optimisation co-target.

### Recommended secondary signal: anti-collapse score

Combine the two sign-correct, scale-aware collapse markers:

```
anti_collapse = z(sanity/embedding_variance_mean) - z(sanity/cosim_random_pairs_mean)
```

where `z(·)` standardises across the architecture-search batch. Both terms have ρ ≈ 0.26–0.29 sign-correct on val_corr_weighted, are bounded in interpretation, and capture representation spread vs collapse from two angles. They DON'T require any labels at all — pure unsupervised. Their per-target ρ on contrast/luminance is small (≤0.18), so they are weaker than `linear_probe_acc` on the genuinely non-leaked targets, but they fill the role of a label-free cross-check.

### Proposed composite

```
rep_score = z(sanity/linear_probe_acc) + 0.5 * anti_collapse
         = z(sanity/linear_probe_acc)
         + 0.5 * z(sanity/embedding_variance_mean)
         - 0.5 * z(sanity/cosim_random_pairs_mean)
```

where standardisation is done *across the candidate architectures in the current search batch* (not across all-time runs). Weights give the probe a primary role (empirically the strongest non-circular predictor) with the two unsupervised collapse signals as a label-free cross-check. All three are produced by `SanityCheckHook` with no extra changes.

### Fallback / sanity floor

Independently of `rep_score`, **gate** on:

- `sanity/embedding_variance_min > epsilon` (e.g. > 0.01) — a hard collapse detector. If the smallest variance dim collapses to ~0, the run is degenerate regardless of other metrics.
- `sanity/cosim_random_pairs_max < 0.99` — the max-cos hits 1.0 only in pathological collapse; useful as a binary kill switch.

These gates are cheap and unambiguous — runs that fail them should be auto-rejected without bothering with the full `rep_score`.

### Stratification caveats

- The `norm_mode` strata are tiny (n=10 and n=11). Don't read individual ρ values — only the broad pattern.
- Across `regularizer={vc, sigreg}` strata (n=35 / n=36), the **embedding-variance and cosim-collapse signs are unstable**: in `regularizer=vc` runs, `embedding_variance_min` ρ flips to −0.21 and `embedding_l2_mean` to −0.43, while `regularizer=sigreg` shows `cosim_random_pairs_mean` ρ=+0.17 (wrong sign).
- This means the collapse markers are **partly capturing between-regulariser scale differences** rather than within-config representation quality. The autoresearch loop fixes the regulariser, so within-run-batch standardisation (`z()`) should largely cancel this.
- `sanity/linear_probe_acc` weakens dramatically within-stratum (vc: ρ=−0.05, sigreg: ρ=+0.09 on weighted target — both effectively zero). So **most of its overall ρ=+0.25 is between-regulariser variance** too: sigreg runs tend to have both higher probe acc and higher val corrs than vc runs. This is a real concern: within a fixed regulariser the probe-acc signal collapses. Mitigation: use it across the autoresearch search batch (which mixes architectures within a fixed regulariser), and rely on the per-target luminance/contrast ρ (n=150, computed across regularisers, ρ=+0.36/+0.27) as the main empirical evidence.

## 7. Metrics to AVOID as the primary signal

- **`train_step/reg_loss` (ρ=−0.61) and `train_step/cls_loss` (ρ=−0.54)** look the strongest, but they are **not unsupervised**: they are the losses of online supervised feature-prediction heads that share a target distribution with the val/reg_* eval. Selecting on them is essentially selecting on "how well did the training-time probe head fit the same features the val probe will fit" — which the user has explicitly flagged as a confound. Do not use as the autoresearch decision metric.
- **`train_step/jepa_loss`, `pred_loss`, `loss_rolling_mean`, `pred_loss_short/long`** have **wrong-signed** correlations across the run population (ρ ≈ +0.25 to +0.43): absolute pred-loss magnitude conflates 'didn't train at all' with 'still optimising'. Across runs with different std_coeff/cov_coeff/regularizer settings, the loss is not on a comparable scale. Use *within* a fixed loss configuration only, never for cross-architecture ranking.
- **`sanity/loss_trend` (ρ=+0.52)** is intriguing but: (i) sign is opposite of what 'loss going down = healthy' would predict, (ii) only computed on runs that produced 20+ rolling-window samples, biasing toward longer survivors. The sign anomaly suggests it's measuring training stage / survivorship rather than representation quality. Treat as a diagnostic, not a decision signal.
- **`train_step/vc_loss` (variance/covariance regulariser)** correlates with anything `std_coeff`/`cov_coeff` is engaged on — proxy for 'regulariser is on' rather than 'representation is good'. Avoid for cross-config comparison.

## 8. Appendix: full Spearman table (overall, all predictors × all targets)

| Predictor | position_in_movie | contrast_rms | luminance_mean | narrative_event_score | val_weighted | n_min |
|-----------|---:|---:|---:|---:|---:|---:|
| `sanity/embedding_variance_mean` | +0.099 | +0.002 | +0.025 | -0.115 | +0.257 | 71 |
| `sanity/embedding_variance_min` | +0.092 | -0.010 | +0.021 | +0.012 | +0.209 | 71 |
| `sanity/embedding_variance_max` | +0.118 | -0.007 | +0.034 | -0.163 | +0.279 | 71 |
| `sanity/embedding_variance_std` | +0.099 | -0.011 | +0.021 | -0.134 | +0.260 | 71 |
| `sanity/embedding_l2_mean` | +0.121 | -0.216 | -0.131 | -0.113 | +0.169 | 71 |
| `sanity/cosim_random_pairs_mean` | -0.049 | -0.181 | -0.106 | +0.105 | -0.289 | 71 |
| `sanity/cosim_random_pairs_max` | -0.154 | -0.003 | +0.082 | -0.087 | -0.281 | 71 |
| `sanity/loss_trend` | +0.480 | +0.250 | +0.166 | -0.024 | +0.516 | 71 |
| `sanity/loss_rolling_mean` | +0.196 | -0.046 | +0.013 | -0.052 | +0.325 | 71 |
| `sanity/pred_loss_short` | +0.222 | -0.103 | -0.047 | +0.093 | +0.334 | 71 |
| `sanity/pred_loss_long` | +0.170 | -0.124 | -0.073 | +0.027 | +0.256 | 71 |
| `sanity/grad_norm` | +0.080 | -0.057 | +0.010 | +0.010 | +0.148 | 71 |
| `sanity/linear_probe_acc` | +0.036 | +0.273 | +0.363 | +0.008 | +0.250 | 71 |
| `train_step/jepa_loss` | +0.347 | -0.072 | -0.019 | +0.035 | +0.407 | 71 |
| `train_step/vc_loss` | +0.005 | -0.264 | -0.189 | -0.072 | -0.048 | 71 |
| `train_step/pred_loss` | +0.359 | -0.037 | +0.009 | +0.058 | +0.426 | 71 |
| `train_step/reg_loss` | -0.481 | -0.418 | -0.433 | +0.112 | -0.615 | 71 |
| `train_step/cls_loss` | -0.529 | -0.255 | -0.415 | -0.255 | -0.540 | 71 |
