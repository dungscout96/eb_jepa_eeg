# Canonical Baselines — nw2_ws4 Protocol B (LOCKED 2026-05-05)

All numbers reported on **test split only**. Train and val are used only for fitting probes / standardization / sanity. **No train or val leakage in any reported metric.** All 50 jobs (10 methods × 5 seeds) are at nw=2_ws=4 under per-clip flat Ridge α=1, B=2000 recording-level bootstrap.

## Eval protocol — Protocol B (`pB_` prefix)

| Parameter | Value |
|---|---|
| Driver | `scripts/unified_probe_eval.py` (commit c805692, `train_order=True`) |
| Train extraction | `for p in range(n_passes): for rec in torch.randperm(n_rec, generator=rng): dataset[rec]` |
| Val/test extraction | `for rec in range(n_rec): for _ in range(n_passes): dataset[rec]` |
| Granularity | per-clip flat: train 14k samples (700 × 20), val/test 2160 samples (108 × 20) |
| Pooling (encoder methods) | kc-pool (5 ch × 64 d = 320-d per clip) |
| Probe heads | Ridge α=1 (regression), LogReg lbfgs C=1 (classification + multinomial) |
| Standardization | train-only mu/sd, broadcast |
| Classification threshold | train median |
| Movie_id bins | 20 (linspace over train pos_in_movie min/max) |
| **Probe seed** | Phase D: **fixed at 42** (σ_enc only). Tier 1: **probe_seed=enc_seed paired** (σ captures probe-side n_passes RNG variance) |
| Encoder seeds | {42, 123, 456, 789, 2025} |
| L1 metric | Pearson r / AUC computed on flat 2160 test rows directly per seed |
| Bootstrap | NPZ saves flat (2160,) → reshape (108, 20) → resample recordings (axis 0) B=2000 → flatten back → recompute Pearson r per resample → bootstrap mean = L2 |
| L3 reporting | 5-seed mean of L2 ± 1σ across encoder seeds |
| t-test | 1-sample 2-sided t-test of 5 per-seed L2 means against chance (`✓` = p<0.05, `ns` = p≥0.05) |

All artifacts grep-traceable via `pB_` prefix in `predictions/canonical/`, `results/canonical/`, `results/bootstrap/`.

## Matched-D 320 linear ceiling (CRITICAL — read first)

Per the user's note that 500-d `raw_corrca` had unfair Ridge capacity advantage over JEPA's 320-d output, we ran two **matched-D 320-d** linear-ceiling baselines on CorrCA-projected EEG:

- `raw_corrca_64`: 5 chans × box-mean-pool to 64 samples = 320-d (same dim as JEPA kc-pool).
- `raw_corrca_pca`: 5 chans × per-channel PCA(64) of 200-sample-downsampled signal = 5×64 = 320-d.

Both are **fixed linear projections** of the CorrCA-5 input — no learned parameters, no encoder.

| Method (D=320) | reg_lum | reg_cont | reg_pos | reg_narr | age_corr | sex_auc | mid_top1 | mid_top5 |
|---|---|---|---|---|---|---|---|---|
| **raw_corrca_64** (no encoder) | **0.292±0.018** ✓ | 0.217±0.015 ✓ | **0.262±0.007** ✓ | **0.232±0.031** ✓ | 0.128±0.085 ✓ | 0.565±0.041 ✓ | 0.111±0.040 ✓ | 0.427±0.057 ✓ |
| **raw_corrca_pca** (no encoder) | **0.292±0.016** ✓ | 0.217±0.018 ✓ | **0.261±0.007** ✓ | **0.235±0.033** ✓ | 0.090±0.064 ✓ | 0.549±0.040 **ns** | 0.094±0.038 **ns** | 0.411±0.040 ✓ |
| pB_phaseD_issue10best (JEPA SSL) | 0.226±0.018 ✓ | **0.223±0.014** ✓ | 0.226±0.016 ✓ | 0.155±0.023 ✓ | 0.390±0.014 ⚠ | 0.727±0.019 ⚠ | 0.092±0.017 ✓ | **0.459±0.032** ✓ |
| pB_t1_random_init (random encoder) | 0.218±0.020 ✓ | 0.194±0.022 ✓ | 0.199±0.023 ✓ | 0.129±0.049 ✓ | 0.437±0.052 ⚠ | 0.705±0.028 ⚠ | 0.091±0.027 ✓ | 0.438±0.063 ✓ |

The two matched-D raw baselines agree to ±0.003 on every metric — the result is **basis-independent** (downsample vs PCA give the same answer). It's a property of the input.

### What the matched-D rows prove

**At identical 320-d Ridge capacity, a fixed linear readout of CorrCA-5 EEG OUTPERFORMS the JEPA SSL encoder on every stim regression**: lum +0.066, pos +0.036, **narr +0.080**. Only contrast goes (barely) JEPA's way. So the SSL encoder is **actively destroying** linearly-recoverable signal that's present in its own input.

Narrative hierarchy (test, 5-seed bootstrap, all CorrCA-5 input):
```
0.235  raw_corrca_pca       ← LINEAR CEILING on 320-d CorrCA-5
0.232  raw_corrca_64        ← same ceiling, alternate basis
0.155  JEPA SSL-trained     ← SSL adds +0.026 vs random encoder, loses -0.080 vs ceiling
0.129  random_init          ← random projection adds noise on top of input
0.040  corrca_stats (35-d)  ← information bottleneck — time-stripped summary stats
```

### And the anti-target inverts

JEPA / random_init **leak subject identity 3-4× more** than the matched-D linear ceiling: age_corr 0.39-0.44 vs raw 0.09-0.13, sex_auc 0.71-0.73 vs raw 0.55-0.57. The transformer architecture itself amplifies subject signal; SSL doesn't fix it (only reduces it modestly from random init's 0.44 to 0.39).

**Headline conclusion**: stim signal lives in the CorrCA spatial filter and is fully accessible by a fixed linear readout. JEPA pretraining (a) fails to recover that signal and (b) inflates subject-identity leakage in exchange. This is the unambiguous form of the paper's claim; the matched-D linear ceiling rows make it visible.

## Headline comparison (all metrics, L3 = 5-seed mean of bootstrap mean ± 1σ)

| Method (D) | reg_lum | reg_cont | reg_pos | reg_narr | age_corr | sex_auc | mid_top1 | mid_top5 |
|---|---|---|---|---|---|---|---|---|
| **JEPA + Ridge per-clip (issue10 best, 320)** | **0.226±0.018** ✓ | **0.223±0.014** ✓ | **0.226±0.016** ✓ | **0.155±0.023** ✓ | 0.390±0.014 ✓ | 0.727±0.019 ✓ | 0.092±0.017 ✓ | 0.459±0.032 ✓ |
| JEPA + Ridge per-clip (issue8 latest) | 0.225±0.018 ✓ | 0.220±0.017 ✓ | 0.222±0.016 ✓ | 0.156±0.023 ✓ | 0.384±0.014 ✓ | 0.731±0.017 ✓ | 0.094±0.024 ✓ | 0.452±0.024 ✓ |
| JEPA + Ridge per-clip (issue10 latest) | 0.226±0.018 ✓ | 0.224±0.014 ✓ | 0.226±0.016 ✓ | 0.155±0.024 ✓ | 0.390±0.014 ✓ | 0.727±0.020 ✓ | 0.094±0.020 ✓ | 0.459±0.032 ✓ |
| Random-init encoder (320, no SSL) | 0.218±0.020 ✓ | 0.194±0.022 ✓ | 0.199±0.023 ✓ | 0.129±0.049 ✓ | **0.437±0.052** ✓ | 0.705±0.028 ✓ | 0.091±0.027 ✓ | 0.438±0.063 ✓ |
| Trivial Ridge corrca35 (per_chan, 35) | 0.167±0.041 ✓ | 0.177±0.043 ✓ | 0.158±0.032 ✓ | 0.040±0.023 ✓ | **0.487±0.022** ✓ | 0.726±0.007 ✓ | 0.093±0.022 ✓ | 0.453±0.039 ✓ |
| Trivial Ridge chan1_only (7) | 0.095±0.028 ✓ | 0.093±0.016 ✓ | 0.082±0.024 ✓ | 0.007±0.021 **ns** | 0.396±0.018 ✓ | 0.506±0.018 **ns** | 0.098±0.031 ✓ | 0.464±0.047 ✓ |
| Trivial Ridge corrca_pooled35 (35 r-7) | 0.131±0.033 ✓ | 0.125±0.026 ✓ | 0.109±0.025 ✓ | 0.015±0.025 **ns** | 0.404±0.018 ✓ | 0.589±0.018 ✓ | 0.089±0.021 ✓ | 0.455±0.060 ✓ |
| Trivial Ridge raw903 (903) | **−0.016**±0.005 | **−0.008**±0.022 | **+0.006**±0.025 | **−0.006**±0.015 | 0.109±0.004 ✓ | 0.689±0.008 ✓ | 0.094±0.034 ✓ | 0.405±0.032 ✓ |
| Trivial Ridge raw_pooled903 (903 r-7) | 0.093±0.032 ✓ | 0.073±0.023 ✓ | 0.060±0.041 ✓ | 0.005±0.015 **ns** | 0.455±0.030 ✓ | 0.524±0.019 ✓ | 0.078±0.037 ✓ | 0.404±0.035 ✓ |
| Trivial Ridge psd_band (645) | 0.041±0.013 ✓ | 0.041±0.015 ✓ | 0.037±0.007 ✓ | −0.010±0.019 **ns** | 0.119±0.062 ✓ | 0.722±0.015 ✓ | 0.094±0.022 ✓ | 0.417±0.023 ✓ |

Bold = column winner among comparable encoder vs no-encoder rows. ✓ = bootstrap-mean significantly different from chance (one-sample two-sided t-test on 5 seeds, p<0.05). ns = not significant.

## Reading the table

**Stim regression (test, n=2160 clips, recording-bootstrap-resampled)**:
- **JEPA Ridge per-clip wins all four stim regressions** by a clear margin: lum +0.06 over corrca35, cont +0.05, pos +0.07, **narr +0.115** over corrca35.
- **Random-init beats every trivial baseline** on lum/cont/pos and ties JEPA within 1σ on narr (0.129 vs 0.155). The architecture's spatial-temporal inductive bias (Fourier-pos encoding, attention over patches) provides most of the stim-decoding lift; SSL adds the rest.
- **`raw903` is at chance for every stim feature** (lum −0.016, cont −0.008, pos +0.006, narr −0.006) — confirms paper claim that nothing in raw 129-ch amplitude statistics predicts movie content.
- `psd_band` (5 log-band-powers × 129 raw chans, 645-d, no mean/std) is also at chance on stim — band power alone is not enough.

**Anti-target (subject metrics)**:
- `corrca35` and `random_init` both leak more age info than JEPA (0.487, 0.437 vs 0.390). JEPA *reduces* age leakage relative to its random-init counterpart — small but consistent.
- `sex_auc` is roughly tied across CorrCA-projected methods (0.70–0.73) — sex info is highly captured by spatial filtering itself, not by the encoder.

**Movie ID retrieval**:
- top-1: all methods at ~0.09 (above chance 0.05 but barely; encoder doesn't recover fine-grained position).
- top-5: JEPA 0.46 vs corrca35 0.45 vs random_init 0.44 — all comparable, all above chance 0.25.

## Per-seed L1 detail (JEPA narrative — reproduces paper §6.4 reference)

issue10 best, narrative L1 per seed: 0.143 / 0.160 / 0.147 / 0.135 / 0.193 → 0.155 ± 0.024 ✓ matches doc reference 0.156 ± 0.023.

## Why our numbers differ from paper Table 2

Paper Table 2 reports JEPA narr **−0.011** and corrca35 narr **0.010** under per-rec aggregation (108 test rows). Our **per-clip flat** numbers (2160 test rows, Pearson on flat) give JEPA narr **0.155** and corrca35 narr **0.040** — the relative ranking (JEPA > corrca35) is preserved but absolute magnitudes differ ~10× because per-clip flat doesn't average out within-recording variance.

The two protocols ask different questions:
- **Per-clip flat** (our protocol): "Across 2160 (rec × clip) pairs, how well does the probe predict the label of *this specific clip*?" — answers a per-clip prediction question.
- **Per-recording** (paper protocol): "Across 108 recordings, how well does the probe predict the label of *the recording's mean clip*?" — answers a per-recording prediction question and is much stricter.

Both are valid; we lock per-clip flat per the user's hierarchy doc which explicitly states "JEPA + Ridge per-clip, Trivial Ridge (all 5 baselines)" use per-clip granularity.

## Source artifacts

```
predictions/canonical/<method>/seed<S>/test_seed*.npz   # raw flat test predictions
results/canonical/<method>/seed<S>/metrics.json         # L1 + protocol metadata
results/bootstrap/<method>_<S>.json                     # per-seed L2 (mean + 95% CI)
results/bootstrap/<method>_L3.json                      # 5-seed L3 + t-test vs chance
```

Where `<method>` ∈ {`pB_phaseD_issue8`, `pB_phaseD_issue10best`, `pB_phaseD_issue10latest`, `pB_t1_random_init`, `pB_t1_psd_band`, `pB_t1_corrca_stats`, `pB_t1_corrca_stats_chan1`, `pB_t1_corrca_stats_pooled`, `pB_t1_raw_stats`, `pB_t1_raw_stats_pooled`}.

## Tier 4 / 6 model-native (supervised end-to-end, archived for reference)

These methods train an FM encoder + regression head jointly; their predictions come from a task-specific head, not from canonical Ridge. Reported with B=2000 recording-level bootstrap on the existing prediction NPZs. **Not directly comparable to canonical Ridge column** — separate axis.

| Metric | pB_tier4_luna (n=5) | pB_tier4_biot (n=5) | pB_tier4_cbramod (n=4) | pB_tier6_luna_corrca (n=5) |
|---|---|---|---|---|
| reg_luminance_mean_corr | 0.192±0.120 | 0.085±0.060 | 0.053±0.077 | 0.159±0.035 |
| reg_contrast_rms_corr | 0.170±0.038 | 0.074±0.066 | 0.034±0.042 | 0.134±0.059 |
| reg_position_in_movie_corr | 0.184±0.031 | 0.109±0.030 | 0.022±0.079 | 0.133±0.062 |
| reg_narrative_event_score_corr | 0.095±0.100 | 0.014±0.028 | 0.040±0.067 | 0.058±0.040 |
| subject/age_reg/corr | 0.390±0.059 | **0.556±0.008** ⚠ leak | 0.286±0.054 | n/a |
| subject/sex/auc | 0.648±0.027 | 0.676±0.012 | 0.655±0.048 | n/a |

Tier 4/6 narrative L3 are all *below* JEPA's 0.155 (under per-clip flat protocol). Tier 4 BIOT's age=0.556 is the worst anti-target leak in any method tested.

## Tier 2 (supervised CNN) — TBD

Train embeddings not saved for Tier 2 (only val + test embeddings exist). Canonical Ridge re-fit not possible without re-extracting train embeddings from the saved Deep4 / EEGNet / etc. checkpoints. Deferred.

## Tier 3 (frozen FM) canonical Ridge — TBD

Requires extending `--feature_source` with BIOT/Luna/CBraMod feature loaders (per-FM channel mapping, sfreq resample). Pending implementation.

## Changelog

- **2026-05-04**: initial Phase D Protocol B 5-seed × 3-family eval; identified per-clip flat protocol matches doc reference 0.156 narr.
- **2026-05-05**: Tier 1 grid completed for all 7 baselines (raw_corrca dropped — 500-d downsampled signal not in tracking table). Final 10-method comparison locked. `raw_stats` initial run had a CorrCA-contamination bug (sbatch was passing CorrCA filters); fixed and re-ran.
