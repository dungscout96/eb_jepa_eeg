# Subject + stim-classification metrics — 2026-05-06

Consolidated 5-seed L3 (mean ± 1σ) for **subject** anti-target metrics
(age_corr, age_R², sex_auc, sex_bal_acc) and **stim-feature classification**
AUCs (median-split per stim feature) across all methods at canonical
nw=2 ws=4. Both subject metrics ideally **low**; stim cls AUCs high (above
0.5 chance) means the encoder discriminates stim feature class. Sources:

- **Block A** — pre-bootstrapped L3 jsons in `results/bootstrap/*_L3.json`
  (Phase D + Ridge, trivial baselines, linear ceilings, random-encoder ablations).
  Note: age_R² was not saved by the bootstrap script and shows "—".
- **Block B** — 5-seed mean ± std from per-seed unified JSONs in
  `results/unified/pB_t{2,3}_*.json` (Tier 2 Ridge-on-backbone, Tier 3 frozen + Ridge).
- **Block C** — Tier 4 (full FT) recomputed directly from saved NPZs in
  `tier4/predictions/<model>_seed<S>/test_seed<S>.npz`.

## Table 1 — Subject anti-target metrics

Both columns ideally low (subject identity should NOT be recovered from a stimulus-trained encoder).

| Method | age_corr ↓ | age_R² ↓ | sex_auc ↓ | sex_bal_acc ↓ |
|---|---|---|---|---|
| **Cine-JEPA + Ridge (ours)** | +0.387 ± 0.017 | — | 0.726 ± 0.019 | 0.666 ± 0.031 |
| Random-init (no SSL) | +0.437 ± 0.053 | — | 0.703 ± 0.029 | 0.644 ± 0.034 |
| Random no-transformer | +0.043 ± 0.050 | — | 0.503 ± 0.054 | 0.493 ± 0.034 |
| Random no-attention | +0.437 ± 0.078 | — | 0.699 ± 0.029 | 0.651 ± 0.033 |
| Random no-pos | +0.318 ± 0.135 | — | 0.665 ± 0.045 | 0.617 ± 0.013 |
| Trivial corrca-stats (35-d) | +0.488 ± 0.022 | — | 0.724 ± 0.007 | 0.689 ± 0.031 |
| Trivial corrca chan1 (7-d) | +0.397 ± 0.018 | — | 0.504 ± 0.018 | 0.543 ± 0.022 |
| Trivial corrca pooled (35, rank-7) | +0.405 ± 0.018 | — | 0.588 ± 0.018 | 0.536 ± 0.016 |
| Trivial raw-stats (903-d) | +0.107 ± 0.000 | — | 0.688 ± 0.008 | 0.671 ± 0.018 |
| Trivial raw pooled (903, rank-7) | +0.454 ± 0.031 | — | 0.524 ± 0.019 | 0.511 ± 0.009 |
| Trivial psd-band (645-d) | +0.100 ± 0.060 | — | 0.720 ± 0.014 | 0.695 ± 0.032 |
| Linear ceiling raw_corrca_64 | +0.130 ± 0.087 | — | 0.564 ± 0.042 | 0.518 ± 0.045 |
| Linear ceiling raw_corrca_pca | +0.090 ± 0.064 | — | 0.548 ± 0.040 | 0.527 ± 0.038 |
| Tier 2 Deep4 + Ridge backbone | +0.380 ± 0.077 | +0.041 ± 0.188 | 0.644 ± 0.026 | 0.585 ± 0.036 |
| Tier 3 BIOT (frozen + Ridge) | +0.474 ± 0.025 | +0.122 ± 0.052 | 0.642 ± 0.021 | 0.586 ± 0.012 |
| Tier 3 Luna (frozen + Ridge) | +0.458 ± 0.038 | +0.100 ± 0.070 | 0.687 ± 0.031 | 0.669 ± 0.015 |
| Tier 3 BENDR (frozen + Ridge) | **−0.008 ± 0.068** | **−0.027 ± 0.007** | **0.480 ± 0.113** | 0.500 ± 0.000 |
| Tier 3 LaBraM (frozen + Ridge) | +0.525 ± 0.023 | +0.182 ± 0.044 | **0.758 ± 0.037** ⚠ | 0.697 ± 0.048 |
| Tier 3 CBraMod / EEGPT / REVE | _(in flight)_ | _(in flight)_ | _(in flight)_ | _(in flight)_ |
| Tier 4 BIOT (full FT) | **+0.558 ± 0.007** ⚠ | **+0.302 ± 0.007** ⚠ | 0.676 ± 0.012 | 0.597 ± 0.012 |
| Tier 4 Luna (full FT) | +0.392 ± 0.059 | +0.090 ± 0.036 | 0.647 ± 0.026 | 0.502 ± 0.005 |
| Tier 4 CBraMod (retrain nw=2 ws=4) | +0.269 ± 0.058 | +0.062 ± 0.035 | 0.663 ± 0.032 | 0.534 ± 0.021 |
| Tier 4 BENDR (full FT) | **+0.038 ± 0.037 ns** | **−0.032 ± 0.015** | **0.520 ± 0.055 ns** | 0.537 ± 0.039 |
| Tier 4 LaBraM (full FT) | +0.300 ± 0.115 | +0.080 ± 0.069 | 0.690 ± 0.028 | 0.548 ± 0.029 |
| Tier 4 EEGPT / REVE | _(in flight)_ | _(in flight)_ | _(in flight)_ | _(in flight)_ |
| Tier 6 Luna+CorrCA-5 (full FT) | n/a | n/a | n/a | n/a |
| mTRF (any input) | n/a | n/a | n/a | n/a |
| Tier 2 native end-to-end (4 archs) | n/a (no subject head) | n/a | n/a | n/a |

Chance: age_corr/R² = 0; sex_auc / bal_acc = 0.5. ⚠ = highest leak; **bold** = column extreme; **ns** = not significant vs chance.

### Subject-axis takeaways

- **Tier 4 BIOT has the worst leak**: age_corr 0.558, age_R² 0.302 (explains 30% of test-subject age variance).
- **LaBraM Tier 3 has the highest sex AUC (0.758)** — stronger than its Tier 4 (0.690) and stronger than Luna T3 (0.687) or Cine-JEPA (0.726).
- **BENDR is uniquely subject-clean both ways**: Tier 3 age −0.008 / sex 0.480 (basically chance) and Tier 4 age 0.038 ns / sex 0.520 ns. Combined with weak stim, BENDR's TUEG-pretrain doesn't transfer to HBN at all.
- **Linear ceilings** (raw_corrca_64/pca) are the cleanest among learned-projection methods: age 0.09–0.13, sex 0.55–0.57. The fixed 320-d projection doesn't inflate subject leak; learned encoders do.
- **Cine-JEPA reduces age leak vs random-init** (0.387 vs 0.437); sex roughly tied. SSL lowers age leakage by ~0.05.

## Table 2 — Stim-feature classification AUC (median-split, per feature)

For each stim feature the target is binarized at the median; AUC > 0.5 = above chance discrimination.

| Method | cls_lum AUC | cls_cont AUC | cls_pos AUC | cls_narr AUC |
|---|---|---|---|---|
| **Cine-JEPA + Ridge (ours)** | 0.578 ± 0.003 | 0.572 ± 0.009 | **0.611 ± 0.007** | 0.547 ± 0.012 |
| Random-init (no SSL) | 0.570 ± 0.009 | 0.572 ± 0.016 | 0.598 ± 0.010 | 0.545 ± 0.010 |
| Random no-transformer | 0.516 ± 0.013 | 0.527 ± 0.005 | 0.542 ± 0.015 | 0.525 ± 0.014 |
| Random no-attention | 0.575 ± 0.013 | 0.578 ± 0.016 | 0.606 ± 0.010 | 0.541 ± 0.010 |
| Random no-pos | 0.557 ± 0.013 | 0.552 ± 0.017 | 0.585 ± 0.004 | 0.532 ± 0.008 |
| Trivial corrca-stats (35-d) | 0.552 ± 0.013 | 0.560 ± 0.019 | 0.581 ± 0.017 | 0.533 ± 0.008 |
| Trivial corrca chan1 (7-d) | 0.530 ± 0.012 | 0.529 ± 0.009 | 0.537 ± 0.007 | 0.522 ± 0.015 |
| Trivial corrca pooled (35, rank-7) | 0.538 ± 0.011 | 0.540 ± 0.013 | 0.555 ± 0.013 | 0.529 ± 0.014 |
| Trivial raw-stats (903-d) | 0.505 ± 0.011 | 0.515 ± 0.010 | 0.528 ± 0.010 | 0.502 ± 0.008 |
| Trivial raw pooled (903, rank-7) | 0.504 ± 0.012 | 0.505 ± 0.013 | 0.534 ± 0.019 | 0.531 ± 0.026 |
| Trivial psd-band (645-d) | 0.508 ± 0.013 | 0.522 ± 0.015 | 0.527 ± 0.013 | 0.508 ± 0.012 |
| Linear ceiling raw_corrca_64 | **0.643 ± 0.012** | **0.618 ± 0.016** | 0.652 ± 0.006 | **0.634 ± 0.014** |
| Linear ceiling raw_corrca_pca | **0.647 ± 0.010** | 0.612 ± 0.019 | **0.657 ± 0.006** | **0.636 ± 0.012** |
| Tier 2 Deep4 + Ridge backbone | 0.593 ± 0.007 | 0.575 ± 0.017 | 0.587 ± 0.012 | 0.580 ± 0.012 |
| Tier 3 BIOT | 0.535 ± 0.013 | 0.522 ± 0.014 | 0.547 ± 0.019 | 0.505 ± 0.020 |
| Tier 3 Luna | 0.555 ± 0.010 | 0.562 ± 0.015 | 0.585 ± 0.009 | 0.521 ± 0.013 |
| Tier 3 BENDR | 0.508 ± 0.011 | 0.505 ± 0.014 | 0.515 ± 0.011 | 0.499 ± 0.017 |
| Tier 3 LaBraM | 0.532 ± 0.012 | 0.542 ± 0.014 | 0.544 ± 0.014 | 0.528 ± 0.012 |
| Tier 4 BIOT | 0.529 ± 0.033 | 0.545 ± 0.042 | 0.545 ± 0.016 | 0.516 ± 0.027 |
| Tier 4 Luna | 0.547 ± 0.047 | 0.563 ± 0.030 | 0.590 ± 0.019 | 0.534 ± 0.044 |
| Tier 4 CBraMod | 0.593 ± 0.030 | 0.522 ± 0.007 | 0.572 ± 0.026 | 0.536 ± 0.050 |
| Tier 4 BENDR | 0.531 ± 0.062 | 0.550 ± 0.059 | 0.562 ± 0.027 | 0.497 ± 0.058 |
| Tier 4 LaBraM | 0.529 ± 0.031 | 0.545 ± 0.007 | 0.531 ± 0.020 | 0.543 ± 0.045 |

### Stim-classification takeaways

- **Linear ceilings (raw_corrca_64/pca) win on cls AUC for 3 of 4 features** (lum / cont / narr) — same pattern as the regression Pearson r in Table 1. A fixed 320-d linear projection of the EEG already discriminates median-split stim classes better than any learned encoder.
- **Cine-JEPA wins on position AUC** (0.611, narrowly beating raw_corrca_pca 0.657 — actually pca leads). Re-checking: pca 0.657 > Cine-JEPA 0.611. Cine-JEPA does NOT win any cls AUC column among the non-supervised rows; supervised end-to-end (Tier 2 native) numbers go in a separate table.
- **Tier 3 BENDR cls AUCs are essentially at chance** (0.499–0.515). Same pattern as its subject metrics — its features just don't carry HBN-relevant signal.
- **Tier 4 numbers have wider σ** than Tier 3 / Block A — full fine-tuning is unstable across seeds for these small-data tasks.

## Caveats

1. Block A's age_R² is unavailable because the canonical bootstrap script saved only `age_reg_corr` per seed. Recomputable from the per-seed prediction NPZs (~5 min × 13 methods); not done in this pass.
2. Block B (Tier 2 + Ridge backbone, Tier 3 frozen+Ridge) numbers are 5-seed mean±std of L1 values, not bootstrap means. Block C (Tier 4 NPZ recompute) is also L1-based here.
3. Tier 4 native heads predict only stim regression + age + sex (no movie_id classifier).
4. Tier 2 native end-to-end (Deep4/Shallow/EEGNet/EEGNeX) has no subject heads; the only Tier 2 row with subject metrics is Deep4 + Ridge-on-backbone.
5. mTRF and Tier 6 Luna+CorrCA-5 NPZs do not contain subject heads.

## Source artifacts

```
results/bootstrap/<method>_L3.json                                   # Block A
results/unified/pB_t{2,3}_<model>_canonical_<seed>_seed<seed>.json   # Block B
tier4/predictions/<model>_seed<S>/test_seed<S>.npz                   # Block C
```
