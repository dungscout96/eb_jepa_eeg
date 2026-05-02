# v10 lane #1 bootstrap analysis — nw4ws2_ridge_keepch_bn

Per-encoder bootstrap (B=2000 resamples of recordings),
5 probe seeds averaged before bootstrap. Then 1-sample t-test on
the 5-encoder bootstrap means against chance.

Split: **test** | tag prefix: `nw4ws2_ridge` | suffix: ``

| Metric | Chance | Mean ± σ_enc | p (t-test) | n_enc | Sig |
|---|---:|---:|---:|---:|:-:|
| `age_cls_auc` | 0.5 | +0.7288 ± 0.0264 | 4.2e-05 | 5 | ✓ |
| `age_cls_bal_acc` | 0.0 | +0.6569 ± 0.0181 | 1.4e-07 | 5 | ✓ |
| `age_reg_corr` | 0.0 | +0.4992 ± 0.0520 | 2.8e-05 | 5 | ✓ |
| `age_reg_mae` | 0.0 | +2.7775 ± 0.0502 | 2.6e-08 | 5 | ✓ |
| `cls_contrast_rms_auc` | 0.5 | +0.5278 ± 0.0240 | 0.061 | 5 |   |
| `cls_contrast_rms_bal_acc` | 0.0 | +0.5193 ± 0.0243 | 1.1e-06 | 5 | ✓ |
| `cls_luminance_mean_auc` | 0.5 | +0.5076 ± 0.0153 | 0.33 | 5 |   |
| `cls_luminance_mean_bal_acc` | 0.0 | +0.4925 ± 0.0161 | 2.7e-07 | 5 | ✓ |
| `cls_narrative_event_score_auc` | 0.5 | +0.5041 ± 0.0095 | 0.39 | 5 |   |
| `cls_narrative_event_score_bal_acc` | 0.0 | +0.4799 ± 0.0183 | 5.1e-07 | 5 | ✓ |
| `cls_position_in_movie_auc` | 0.5 | +0.5011 ± 0.0211 | 0.91 | 5 |   |
| `cls_position_in_movie_bal_acc` | 0.0 | +0.4980 ± 0.0110 | 5.7e-08 | 5 | ✓ |
| `reg_contrast_rms_corr` | 0.0 | +0.0300 ± 0.0344 | 0.12 | 5 |   |
| `reg_luminance_mean_corr` | 0.0 | +0.0883 ± 0.0338 | 0.0043 | 5 | ✓ |
| `reg_narrative_event_score_corr` | 0.0 | +0.0548 ± 0.0280 | 0.012 | 5 | ✓ |
| `reg_position_in_movie_corr` | 0.0 | +0.0137 ± 0.0262 | 0.31 | 5 |   |
| `sex_auc` | 0.5 | +0.7138 ± 0.0049 | 6.7e-08 | 5 | ✓ |
| `sex_bal_acc` | 0.0 | +0.6682 ± 0.0151 | 6.3e-08 | 5 | ✓ |