# v10 lane #1 bootstrap analysis — nw4ws2_ridge_keepch_bn_npasses20

Per-encoder bootstrap (B=2000 resamples of recordings),
5 probe seeds averaged before bootstrap. Then 1-sample t-test on
the 5-encoder bootstrap means against chance.

Split: **test** | tag prefix: `nw4ws2_ridge` | suffix: ``

| Metric | Chance | Mean ± σ_enc | p (t-test) | n_enc | Sig |
|---|---:|---:|---:|---:|:-:|
| `age_cls_auc` | 0.5 | +0.7205 ± 0.0264 | 4.8e-05 | 5 | ✓ |
| `age_cls_bal_acc` | 0.0 | +0.6585 ± 0.0139 | 4.7e-08 | 5 | ✓ |
| `age_reg_corr` | 0.0 | +0.4944 ± 0.0487 | 2.2e-05 | 5 | ✓ |
| `age_reg_mae` | 0.0 | +2.8001 ± 0.0580 | 4.4e-08 | 5 | ✓ |
| `cls_contrast_rms_auc` | 0.5 | +0.5145 ± 0.0135 | 0.075 | 5 |   |
| `cls_contrast_rms_bal_acc` | 0.0 | +0.5190 ± 0.0081 | 1.4e-08 | 5 | ✓ |
| `cls_luminance_mean_auc` | 0.5 | +0.4631 ± 0.0205 | 0.016 | 5 | ✓ |
| `cls_luminance_mean_bal_acc` | 0.0 | +0.4780 ± 0.0208 | 8.6e-07 | 5 | ✓ |
| `cls_narrative_event_score_auc` | 0.5 | +0.6029 ± 0.0217 | 0.00044 | 5 | ✓ |
| `cls_narrative_event_score_bal_acc` | 0.0 | +0.5853 ± 0.0122 | 4.5e-08 | 5 | ✓ |
| `cls_position_in_movie_auc` | 0.5 | +0.5671 ± 0.0116 | 0.00021 | 5 | ✓ |
| `cls_position_in_movie_bal_acc` | 0.0 | +0.5458 ± 0.0133 | 8.4e-08 | 5 | ✓ |
| `reg_contrast_rms_corr` | 0.0 | +0.0924 ± 0.0199 | 0.00048 | 5 | ✓ |
| `reg_luminance_mean_corr` | 0.0 | +0.0166 ± 0.0121 | 0.038 | 5 | ✓ |
| `reg_narrative_event_score_corr` | 0.0 | +0.0344 ± 0.0159 | 0.0084 | 5 | ✓ |
| `reg_position_in_movie_corr` | 0.0 | +0.1839 ± 0.0093 | 1.5e-06 | 5 | ✓ |
| `sex_auc` | 0.5 | +0.7115 ± 0.0106 | 1.5e-06 | 5 | ✓ |
| `sex_bal_acc` | 0.0 | +0.6643 ± 0.0230 | 3.5e-07 | 5 | ✓ |