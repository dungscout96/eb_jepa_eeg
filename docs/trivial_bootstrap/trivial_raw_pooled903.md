# Trivial baseline View 2 + View 3 — trivial_raw_pooled903

Probe seeds: [7, 13, 42, 1234, 2025]
Split: **test** | B=2000 | n_seeds=5

| Metric | Chance | View 2 mean ± σ | V2 p | View 3 mean ± σ | V3 p |
|---|---:|---:|---:|---:|---:|
| `age_cls_auc` | 0.5 | — | — | +0.6957 ± 0.0032 | 1.7e-08 sig |
| `age_cls_bal_acc` | 0.0 | — | — | +0.6486 ± 0.0103 | 1.5e-08 sig |
| `age_reg_corr` | 0.0 | +0.3615 ± 0.0184 | 1.6e-06 sig | +0.3609 ± 0.0202 | 2.3e-06 sig |
| `age_reg_mae` | 0.0 | — | — | +2.9151 ± 0.0293 | 2.5e-09 sig |
| `cls_contrast_rms_auc` | 0.5 | +0.5317 ± 0.0401 | 0.15 ns | +0.5318 ± 0.0401 | 0.15 ns |
| `cls_contrast_rms_bal_acc` | 0.0 | — | — | +0.5142 ± 0.0232 | 9.9e-07 sig |
| `cls_luminance_mean_auc` | 0.5 | +0.5023 ± 0.0501 | 0.92 ns | +0.5025 ± 0.0502 | 0.92 ns |
| `cls_luminance_mean_bal_acc` | 0.0 | — | — | +0.5007 ± 0.0067 | 7.7e-09 sig |
| `cls_narrative_event_score_auc` | 0.5 | +0.5015 ± 0.0268 | 0.91 ns | +0.5012 ± 0.0266 | 0.92 ns |
| `cls_narrative_event_score_bal_acc` | 0.0 | — | — | +0.5103 ± 0.0103 | 4e-08 sig |
| `cls_position_in_movie_auc` | 0.5 | +0.5301 ± 0.0477 | 0.23 ns | +0.5307 ± 0.0471 | 0.22 ns |
| `cls_position_in_movie_bal_acc` | 0.0 | — | — | +0.5015 ± 0.0371 | 7.1e-06 sig |
| `reg_contrast_rms_corr` | 0.0 | -0.0076 ± 0.1041 | 0.88 ns | -0.0066 ± 0.1005 | 0.89 ns |
| `reg_luminance_mean_corr` | 0.0 | +0.0448 ± 0.0770 | 0.26 ns | +0.0468 ± 0.0706 | 0.21 ns |
| `reg_narrative_event_score_corr` | 0.0 | +0.0059 ± 0.0481 | 0.8 ns | +0.0067 ± 0.0455 | 0.76 ns |
| `reg_position_in_movie_corr` | 0.0 | -0.0252 ± 0.0961 | 0.59 ns | -0.0229 ± 0.0908 | 0.6 ns |
| `sex_auc` | 0.5 | +0.4879 ± 0.0150 | 0.14 ns | +0.4875 ± 0.0150 | 0.14 ns |
| `sex_bal_acc` | 0.0 | — | — | +0.5000 ± 0.0000 | nan ns |