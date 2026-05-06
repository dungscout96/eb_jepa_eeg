# Subject-feature (age, sex) aggregation — 2026-05-06

Consolidated 5-seed L3 (mean ± 1σ) for the two subject-level anti-target
metrics across **all** methods present at canonical nw=2 ws=4. Both ideally
**low** for a stimulus-focused encoder (subject-identity leakage is the
failure mode movie-stim training is trying to avoid). Sources:

- **Block A** — pre-bootstrapped L3 jsons in `results/bootstrap/*_L3.json`
  (Phase D + Ridge, trivial baselines, linear ceilings, random-encoder ablations).
- **Block B** — 5-seed mean ± std computed from per-seed unified JSONs in
  `results/unified/pB_t{2,3}_*_canonical_<seed>_seed<seed>.json` (Tier 2
  Ridge-on-backbone, Tier 3 frozen + Ridge).
- **Block C** — Tier 4 (full FT) recomputed directly from saved NPZs at
  `tier4/predictions/<model>_seed<S>/test_seed<S>.npz`.

## Master table — test split, R6, 108 recordings × 20 passes = 2160 clips

| Category | Method | age_corr ↓ | sex_auc ↓ |
|---|---|---|---|
| **Phase D + Ridge (Cine-JEPA, ours)** | issue10best (5 seeds) | 0.387 ± 0.017 | 0.726 ± 0.019 |
| Random-encoder ablations | random_init (no SSL) | 0.437 ± 0.053 | 0.703 ± 0.029 |
| | random no-transformer | 0.043 ± 0.050 | 0.503 ± 0.054 |
| | random no-attention | 0.437 ± 0.078 | 0.699 ± 0.029 |
| | random no-pos | 0.318 ± 0.135 | 0.665 ± 0.045 |
| Trivial baselines | corrca-stats (35-d) | 0.488 ± 0.022 | 0.724 ± 0.007 |
| | corrca chan1 (7-d) | 0.397 ± 0.018 | 0.504 ± 0.018 |
| | corrca pooled (35-d, rank-7) | 0.405 ± 0.018 | 0.588 ± 0.018 |
| | raw-stats (903-d) | 0.107 ± 0.000 | 0.688 ± 0.008 |
| | raw pooled (903-d, rank-7) | 0.454 ± 0.031 | 0.524 ± 0.019 |
| | psd-band (645-d) | 0.100 ± 0.060 | 0.720 ± 0.014 |
| Linear ceilings (matched-D 320) | raw_corrca_64 | **0.130 ± 0.087** | **0.564 ± 0.042** |
| | raw_corrca_pca | **0.090 ± 0.064** | **0.548 ± 0.040** |
| **Tier 2** (supervised + Ridge backbone) | Deep4 | 0.380 ± 0.077 | 0.644 ± 0.026 |
| **Tier 2 native end-to-end** (Deep4/Shallow/EEGNet/EEGNeX) | n/a (native heads = stim only) | n/a | n/a |
| **Tier 3** (frozen FM + Ridge) | BIOT | 0.474 ± 0.025 | 0.642 ± 0.021 |
| | Luna | 0.458 ± 0.038 | 0.687 ± 0.031 |
| | BENDR | **−0.008 ± 0.068** | **0.480 ± 0.113** |
| | LaBraM | 0.525 ± 0.023 | **0.758 ± 0.037** ⚠ |
| | CBraMod | _(in flight)_ | _(in flight)_ |
| | EEGPT | _(in flight)_ | _(in flight)_ |
| | REVE | _(in flight)_ | _(in flight)_ |
| **Tier 4** (full FT, native head) | BIOT | **0.558 ± 0.007** ⚠ | 0.676 ± 0.012 |
| | Luna | 0.392 ± 0.059 | 0.647 ± 0.026 |
| | CBraMod (retrain nw=2 ws=4) | 0.269 ± 0.058 | 0.663 ± 0.032 |
| | BENDR | **0.038 ± 0.037 ns** | **0.520 ± 0.055 ns** |
| | LaBraM | 0.300 ± 0.115 | 0.690 ± 0.028 |
| | EEGPT | _(in flight)_ | _(in flight)_ |
| | REVE | _(in flight)_ | _(in flight)_ |
| | Luna + CorrCA-5 | n/a (subject-info attach failed) | n/a |
| **mTRF** (Crosse 2016, any input) | n/a (no subject head) | n/a | n/a |

Chance: age 0 (Pearson r), sex 0.5 (binary AUC). Bold = column extreme. ⚠ = highest leak. **ns** = not significant against chance.

## Striking patterns

- **Tier 4 BIOT (full FT) is the worst anti-target leak** by a clear margin: age 0.558 ± 0.007 — predicts subject age from a movie-stim-trained model better than any other configuration we tested. Sex AUC 0.676 is also high.
- **LaBraM Tier 3 is the highest sex_auc** at 0.758, even higher than LaBraM Tier 4 (0.690), Luna T3 (0.687), and Cine-JEPA (0.726). LaBraM's frozen representation strongly encodes subject identity.
- **Linear ceilings (`raw_corrca_64`, `raw_corrca_pca`) are the cleanest among learned-projection methods**: age 0.09–0.13, sex 0.55–0.57. A fixed 320-d linear projection of the EEG does NOT inflate subject leak. Inflation comes from learned encoders (JEPA, random-init transformer, Tier 3/4 FMs).
- **BENDR is uniquely subject-clean both ways**: Tier 3 age −0.008 / sex 0.480 (ns vs chance) and Tier 4 age 0.038 ns / sex 0.520 ns. BENDR's pretrained features simply don't transfer to HBN — they fail to lock onto either stim OR subject identity.
- **Random no-transformer ablation** (no-attn, no-pos, no-FFN — token-only baseline) drops age to 0.043 and sex to 0.503 — confirms the transformer machinery is what amplifies subject signal.
- **Cine-JEPA reduces age leak vs random-init encoder** (0.387 vs 0.437; sex roughly tied). The SSL objective lowers age by ~0.05 vs untrained-arch baseline.
- **Trivial corrca35 leaks more age than JEPA** (0.488 vs 0.387). The CorrCA spatial filter alone, fed to canonical Ridge, recovers more age signal than the JEPA encoder does on the same input — JEPA pretraining *reduces* age leakage relative to a trivial probe of its input.

## Caveats

1. Tier 2 / Tier 3 Block-B numbers are **5-seed mean ± std of L1 values**, not 5-seed mean ± std of bootstrap means. Block A (Phase D, trivial, linear ceilings) uses the canonical L3 protocol (mean ± std of B=2000 bootstrap means). The two are very close in central tendency but Block-B σ may be slightly larger.
2. Tier 6 Luna+CorrCA bootstrap was run only on stim regression metrics — its NPZ does not contain subject heads (the subject-info-attach branch failed for the CorrCA path).
3. Tier 4 native heads predict only stim + age + sex (no movie_id classifier) by design.
4. Tier 2 native end-to-end (Deep4/Shallow/EEGNet/EEGNeX) was run with NO subject heads; only the Tier 2 Deep4 + Ridge-on-backbone path has subject metrics.
5. mTRF is by design stim-only (a backward decoder).

## Source artifacts

```
results/bootstrap/<method>_L3.json                                  # Block A
results/unified/pB_t{2,3}_<model>_canonical_<seed>_seed<seed>.json  # Block B
tier4/predictions/<model>_seed<S>/test_seed<S>.npz                  # Block C
```
