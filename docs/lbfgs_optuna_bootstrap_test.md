# v10 lane #1 bootstrap analysis — nw4ws2_lbfgs_optuna_keepch_bn

Per-encoder bootstrap (B=2000 resamples of recordings),
5 probe seeds averaged before bootstrap. Then 1-sample t-test on
the 5-encoder bootstrap means against chance.

Split: **test** | tag prefix: `nw4ws2_lbfgs` | suffix: ``

| Metric | Chance | Mean ± σ_enc | p (t-test) | n_enc | Sig |
|---|---:|---:|---:|---:|:-:|
| `age_cls_auc` | 0.5 | +0.7250 ± 0.0213 | 1.9e-05 | 5 | ✓ |
| `age_cls_bal_acc` | 0.0 | +0.6626 ± 0.0160 | 8.2e-08 | 5 | ✓ |
| `age_reg_corr` | 0.0 | +0.5104 ± 0.0533 | 2.8e-05 | 5 | ✓ |
| `age_reg_mae` | 0.0 | +2.7561 ± 0.0666 | 8.2e-08 | 5 | ✓ |
| `cls_contrast_rms_auc` | 0.5 | +0.4938 ± 0.0211 | 0.55 | 5 |   |
| `cls_contrast_rms_bal_acc` | 0.0 | +0.4939 ± 0.0173 | 3.6e-07 | 5 | ✓ |
| `cls_luminance_mean_auc` | 0.5 | +0.5103 ± 0.0234 | 0.38 | 5 |   |
| `cls_luminance_mean_bal_acc` | 0.0 | +0.5133 ± 0.0108 | 4.7e-08 | 5 | ✓ |
| `cls_narrative_event_score_auc` | 0.5 | +0.4932 ± 0.0199 | 0.49 | 5 |   |
| `cls_narrative_event_score_bal_acc` | 0.0 | +0.4879 ± 0.0333 | 5.2e-06 | 5 | ✓ |
| `cls_position_in_movie_auc` | 0.5 | +0.4831 ± 0.0246 | 0.2 | 5 |   |
| `cls_position_in_movie_bal_acc` | 0.0 | +0.4923 ± 0.0157 | 2.5e-07 | 5 | ✓ |
| `reg_contrast_rms_corr` | 0.0 | +0.0057 ± 0.0986 | 0.9 | 5 |   |
| `reg_luminance_mean_corr` | 0.0 | -0.0117 ± 0.0504 | 0.63 | 5 |   |
| `reg_narrative_event_score_corr` | 0.0 | +0.0167 ± 0.0356 | 0.35 | 5 |   |
| `reg_position_in_movie_corr` | 0.0 | -0.0457 ± 0.0333 | 0.037 | 5 | ✓ |
| `sex_auc` | 0.5 | +0.7165 ± 0.0088 | 6.4e-07 | 5 | ✓ |
| `sex_bal_acc` | 0.0 | +0.6671 ± 0.0270 | 6.4e-07 | 5 | ✓ |