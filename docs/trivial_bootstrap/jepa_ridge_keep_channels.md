# Trivial baseline View 2 + View 3 — jepa_ridge_keep_channels

Probe seeds: [42, 123, 456, 789, 2025]
Split: **test** | B=2000 | n_seeds=5

| Metric | Chance | View 2 mean ± σ | V2 p | View 3 mean ± σ | V3 p |
|---|---:|---:|---:|---:|---:|
| `cls_contrast_rms_auc` | 0.5 | — | — | +0.5000 ± 0.0000 | nan ns |
| `cls_contrast_rms_bal_acc` | 0.0 | — | — | +0.5000 ± 0.0000 | nan ns |
| `cls_luminance_mean_auc` | 0.5 | — | — | +0.5000 ± 0.0000 | nan ns |
| `cls_luminance_mean_bal_acc` | 0.0 | — | — | +0.5000 ± 0.0000 | nan ns |
| `cls_narrative_event_score_auc` | 0.5 | — | — | +0.5000 ± 0.0000 | nan ns |
| `cls_narrative_event_score_bal_acc` | 0.0 | — | — | +0.5000 ± 0.0000 | nan ns |
| `cls_position_in_movie_auc` | 0.5 | — | — | +0.5000 ± 0.0000 | nan ns |
| `cls_position_in_movie_bal_acc` | 0.0 | — | — | +0.5000 ± 0.0000 | nan ns |
| `reg_contrast_rms_corr` | 0.0 | +0.1585 ± 0.0093 | 2.8e-06 sig | +0.1588 ± 0.0094 | 3e-06 sig |
| `reg_luminance_mean_corr` | 0.0 | +0.2076 ± 0.0072 | 3.4e-07 sig | +0.2077 ± 0.0074 | 3.8e-07 sig |
| `reg_narrative_event_score_corr` | 0.0 | +0.0900 ± 0.0113 | 5.9e-05 sig | +0.0903 ± 0.0112 | 5.5e-05 sig |
| `reg_position_in_movie_corr` | 0.0 | +0.1435 ± 0.0167 | 4.4e-05 sig | +0.1436 ± 0.0165 | 4.1e-05 sig |