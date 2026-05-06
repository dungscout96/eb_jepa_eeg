# Subject-level + movie-id metrics aggregation — 2026-05-06

Consolidated 5-seed L3 (mean ± 1σ) for the four "anti-target" / retrieval metrics
across all baseline methods present in `results/`. Sources:

- **Block A** — pre-bootstrapped L3 jsons in `results/bootstrap/*_L3.json`.
- **Block B** — 5-seed mean ± std computed from per-seed unified JSONs in
  `results/unified/pB_t{2,3}_*_canonical_<seed>_seed<seed>.json`
  (no bootstrap CI was computed for Tier 2/3 subject metrics — these are
  per-seed L1 means, std taken across the 5 seeds).
- **Block C** — Tier 4/6 native heads only predict stim + age + sex
  (no movie_id), so movie_id columns are n/a.

## Headline (test split, R6, 108 recordings × 20 passes = 2160 clips)

| Method | age_corr ↓ | sex_auc ↓ | mvid_t1 ↑ | mvid_t5 ↑ |
|---|---|---|---|---|
| **Cine-JEPA + Ridge (ours)**          | 0.387 ± 0.017 | 0.726 ± 0.019 | 0.091 ± 0.017 | 0.457 ± 0.031 |
| _Random-init encoder_                 | 0.437 ± 0.053 | 0.703 ± 0.029 | 0.091 ± 0.027 | 0.437 ± 0.063 |
| _Random no-transformer_               | 0.043 ± 0.050 | 0.503 ± 0.054 | 0.080 ± 0.015 | 0.430 ± 0.054 |
| _Random no-attention_                 | 0.437 ± 0.078 | 0.699 ± 0.029 | 0.093 ± 0.031 | 0.446 ± 0.048 |
| _Random no-pos_                       | 0.318 ± 0.135 | 0.665 ± 0.045 | 0.106 ± 0.039 | 0.456 ± 0.050 |
| Trivial corrca-stats (35-d)           | 0.488 ± 0.022 | 0.724 ± 0.007 | 0.093 ± 0.022 | 0.454 ± 0.038 |
| Trivial corrca chan1 (7-d)            | 0.397 ± 0.018 | 0.504 ± 0.018 | 0.098 ± 0.031 | 0.465 ± 0.047 |
| Trivial corrca pooled (35-d, rank-7)  | 0.405 ± 0.018 | 0.588 ± 0.018 | 0.089 ± 0.021 | 0.456 ± 0.061 |
| Trivial raw-stats (903-d)             | 0.107 ± 0.000 | 0.688 ± 0.008 | 0.094 ± 0.034 | 0.406 ± 0.032 |
| Trivial raw pooled (903-d, rank-7)    | 0.454 ± 0.031 | 0.524 ± 0.019 | 0.078 ± 0.037 | 0.404 ± 0.034 |
| Trivial psd-band (645-d)              | 0.100 ± 0.060 | 0.720 ± 0.014 | 0.094 ± 0.022 | 0.417 ± 0.024 |
| Linear ceiling raw_corrca_64 (320-d)  | 0.130 ± 0.087 | 0.564 ± 0.042 | 0.111 ± 0.040 | 0.428 ± 0.057 |
| Linear ceiling raw_corrca_pca (320-d) | 0.090 ± 0.064 | 0.548 ± 0.040 | 0.094 ± 0.037 | 0.411 ± 0.040 |
| Tier 3 BIOT (frozen + Ridge)          | 0.474 ± 0.025 | 0.642 ± 0.021 | 0.098 ± 0.027 | 0.450 ± 0.062 |
| Tier 3 Luna (frozen + Ridge)          | 0.458 ± 0.038 | 0.687 ± 0.031 | 0.117 ± 0.041 | 0.483 ± 0.059 |
| Tier 2 Deep4 (sup + Ridge backbone)   | 0.380 ± 0.077 | 0.644 ± 0.026 | 0.109 ± 0.028 | 0.456 ± 0.078 |
| Tier 4 BIOT (full finetune)           | **0.558 ± 0.007** ⚠ | 0.676 ± 0.012 | n/a | n/a |
| Tier 4 Luna (full finetune)           | 0.392 ± 0.059 | 0.647 ± 0.026 | n/a | n/a |
| Tier 4 CBraMod (full finetune, n=4)   | 0.284 ± 0.054 | 0.656 ± 0.048 | n/a | n/a |
| Tier 6 Luna + CorrCA (full finetune)  | n/a (not bootstrapped) | n/a | n/a | n/a |
| mTRF (Crosse 2016, any input)         | n/a (no subj/movie head) | n/a | n/a | n/a |

Chance: age 0 (Pearson r), sex 0.5 (binary AUC), mvid_t1 0.05 (1/20), mvid_t5 0.25 (5/20).

## Reading the table

**Anti-target axis (age, sex — lower is better, since neither metric is the actual training target):**

- **Tier 4 BIOT (full finetune) is the worst**: age 0.558 ± 0.007 — predicts subject age from a movie-stim-trained model better than any other configuration. This is the largest subject-identity leak we measured.
- **Trivial corrca35 leaks more age than JEPA** (0.488 vs 0.387). The CorrCA spatial filter alone, fed to canonical Ridge, recovers more age signal than the JEPA encoder does on the same input — i.e., JEPA pretraining *reduces* age leakage relative to the trivial probe of its input.
- **Random-init encoder leaks more than JEPA** (age 0.437 vs 0.387; sex 0.703 vs 0.726 — sex is roughly tied). The Cine-JEPA SSL objective lowers age by 0.05 vs untrained-arch baseline.
- **Linear ceilings (`raw_corrca_64`, `raw_corrca_pca`)** are the cleanest on the anti-target — age 0.09–0.13, sex 0.55–0.57. A fixed 320-d linear projection of the EEG does NOT inflate subject leak; the inflation comes from learned encoders (JEPA, random-init transformer, Tier 3/4 FMs).
- **Random no-transformer (no-attn no-pos no-feedforward)** drops age to 0.043 and sex to 0.503 — confirms transformer machinery is what amplifies subject signal.

**Movie-ID retrieval (top-1, top-5):**

- All methods are above chance (top-1 chance 0.05, top-5 chance 0.25) but the spread is small: top-1 ≈ 0.08–0.12, top-5 ≈ 0.40–0.48. Movie-ID retrieval is roughly comparable across the entire ladder — the encoder doesn't recover sharp per-clip movie identity.
- Tier 3 Luna leads (top-1 0.117, top-5 0.483), Cine-JEPA sits at 0.091 / 0.457, and the linear ceilings are within 1σ of JEPA. This metric does not discriminate strongly across methods.

## Caveats

1. The Tier 2 / Tier 3 numbers in Block B are **5-seed mean ± std of L1 values**, NOT 5-seed mean ± std of bootstrap-mean. The Phase D / trivial / linear-ceiling rows in Block A use the canonical L3 protocol (mean ± std of bootstrap means; B = 2000 recording resamples per seed). The bootstrap-mean and seed-mean are very close, but the σ for Block B may be slightly larger than the strict L3 σ would be.
2. Tier 6 Luna+CorrCA bootstrap was run only on stim regression metrics — it does not have age/sex columns in its NPZ.
3. Tier 4/6 full-finetune heads predict only stim + age + sex (no movie_id classifier) by design.

## Source artifacts

```
results/bootstrap/<method>_L3.json                      # Block A, full L3
results/unified/<method>_<seed>_seed<seed>.json         # Block B, per-seed L1
results/bootstrap/pB_tier{4,6}_*_L3.json                # Block C, native heads
```
