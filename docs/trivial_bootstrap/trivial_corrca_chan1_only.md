# Trivial baseline View 2 + View 3 — trivial_corrca_chan1_only

Probe seeds: [7, 13, 42, 1234, 2025]
Split: **test** | B=2000 | n_seeds=5

| Metric | Chance | View 2 mean ± σ | V2 p | View 3 mean ± σ | V3 p |
|---|---:|---:|---:|---:|---:|
| `age_cls_auc` | 0.5 | — | — | +0.5220 ± 0.1175 | 0.7 ns |
| `age_cls_bal_acc` | 0.0 | — | — | +0.5043 ± 0.0618 | 5.3e-05 sig |
| `age_reg_corr` | 0.0 | +0.0823 ± 0.1923 | 0.39 ns | +0.0798 ± 0.1911 | 0.4 ns |
| `age_reg_mae` | 0.0 | — | — | +3.7703 ± 0.8599 | 0.00061 sig |
| `cls_contrast_rms_auc` | 0.5 | +0.5043 ± 0.0312 | 0.77 ns | +0.5043 ± 0.0309 | 0.77 ns |
| `cls_contrast_rms_bal_acc` | 0.0 | — | — | +0.5005 ± 0.0229 | 1e-06 sig |
| `cls_luminance_mean_auc` | 0.5 | +0.5050 ± 0.0209 | 0.62 ns | +0.5057 ± 0.0208 | 0.57 ns |
| `cls_luminance_mean_bal_acc` | 0.0 | — | — | +0.4994 ± 0.0251 | 1.5e-06 sig |
| `cls_narrative_event_score_auc` | 0.5 | +0.5080 ± 0.0294 | 0.58 ns | +0.5075 ± 0.0297 | 0.6 ns |
| `cls_narrative_event_score_bal_acc` | 0.0 | — | — | +0.5097 ± 0.0151 | 1.9e-07 sig |
| `cls_position_in_movie_auc` | 0.5 | +0.5247 ± 0.0263 | 0.1 ns | +0.5250 ± 0.0260 | 0.097 ns |
| `cls_position_in_movie_bal_acc` | 0.0 | — | — | +0.5089 ± 0.0213 | 7.4e-07 sig |
| `reg_contrast_rms_corr` | 0.0 | -0.0100 ± 0.0356 | 0.56 ns | -0.0092 ± 0.0357 | 0.59 ns |
| `reg_luminance_mean_corr` | 0.0 | +0.0492 ± 0.0652 | 0.17 ns | +0.0497 ± 0.0633 | 0.15 ns |
| `reg_narrative_event_score_corr` | 0.0 | +0.0070 ± 0.0504 | 0.77 ns | +0.0080 ± 0.0487 | 0.73 ns |
| `reg_position_in_movie_corr` | 0.0 | +0.0225 ± 0.1012 | 0.64 ns | +0.0216 ± 0.1003 | 0.66 ns |
| `sex_auc` | 0.5 | +0.5224 ± 0.0619 | 0.46 ns | +0.5224 ± 0.0624 | 0.47 ns |
| `sex_bal_acc` | 0.0 | — | — | +0.5236 ± 0.0383 | 6.9e-06 sig |