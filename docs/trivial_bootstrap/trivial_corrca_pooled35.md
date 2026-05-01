# Trivial baseline View 2 + View 3 — trivial_corrca_pooled35

Probe seeds: [7, 13, 42, 1234, 2025]
Split: **test** | B=2000 | n_seeds=5

| Metric | Chance | View 2 mean ± σ | V2 p | View 3 mean ± σ | V3 p |
|---|---:|---:|---:|---:|---:|
| `age_cls_auc` | 0.5 | — | — | +0.6586 ± 0.0183 | 4.2e-05 sig |
| `age_cls_bal_acc` | 0.0 | — | — | +0.6098 ± 0.0271 | 9.4e-07 sig |
| `age_reg_corr` | 0.0 | +0.1586 ± 0.3002 | 0.3 ns | +0.1600 ± 0.3006 | 0.3 ns |
| `age_reg_mae` | 0.0 | — | — | +3.0896 ± 0.1657 | 2e-06 sig |
| `cls_contrast_rms_auc` | 0.5 | +0.5096 ± 0.0298 | 0.51 ns | +0.5094 ± 0.0295 | 0.51 ns |
| `cls_contrast_rms_bal_acc` | 0.0 | — | — | +0.5027 ± 0.0169 | 3.1e-07 sig |
| `cls_luminance_mean_auc` | 0.5 | +0.5069 ± 0.0163 | 0.4 ns | +0.5073 ± 0.0158 | 0.36 ns |
| `cls_luminance_mean_bal_acc` | 0.0 | — | — | +0.4957 ± 0.0105 | 4.8e-08 sig |
| `cls_narrative_event_score_auc` | 0.5 | +0.5269 ± 0.0307 | 0.12 ns | +0.5262 ± 0.0310 | 0.13 ns |
| `cls_narrative_event_score_bal_acc` | 0.0 | — | — | +0.5047 ± 0.0074 | 1.1e-08 sig |
| `cls_position_in_movie_auc` | 0.5 | +0.5194 ± 0.0517 | 0.45 ns | +0.5199 ± 0.0508 | 0.43 ns |
| `cls_position_in_movie_bal_acc` | 0.0 | — | — | +0.5009 ± 0.0333 | 4.7e-06 sig |
| `reg_contrast_rms_corr` | 0.0 | +0.0017 ± 0.0472 | 0.94 ns | +0.0019 ± 0.0479 | 0.93 ns |
| `reg_luminance_mean_corr` | 0.0 | +0.0335 ± 0.0676 | 0.33 ns | +0.0338 ± 0.0663 | 0.32 ns |
| `reg_narrative_event_score_corr` | 0.0 | +0.0026 ± 0.0555 | 0.92 ns | +0.0042 ± 0.0542 | 0.87 ns |
| `reg_position_in_movie_corr` | 0.0 | +0.0377 ± 0.1097 | 0.49 ns | +0.0361 ± 0.1090 | 0.5 ns |
| `sex_auc` | 0.5 | +0.5209 ± 0.0772 | 0.58 ns | +0.5215 ± 0.0778 | 0.57 ns |
| `sex_bal_acc` | 0.0 | — | — | +0.5133 ± 0.0187 | 4.2e-07 sig |