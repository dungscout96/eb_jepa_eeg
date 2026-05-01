# Trivial baseline View 2 + View 3 — trivial_ridge_corrca35

Probe seeds: [7, 13, 42, 1234, 2025]
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
| `reg_contrast_rms_corr` | 0.0 | +0.1383 ± 0.0344 | 0.00085 sig | +0.1389 ± 0.0345 | 0.00085 sig |
| `reg_luminance_mean_corr` | 0.0 | +0.1408 ± 0.0274 | 0.00033 sig | +0.1417 ± 0.0269 | 0.0003 sig |
| `reg_narrative_event_score_corr` | 0.0 | +0.0436 ± 0.0187 | 0.0064 sig | +0.0439 ± 0.0187 | 0.0063 sig |
| `reg_position_in_movie_corr` | 0.0 | +0.1237 ± 0.0256 | 0.00042 sig | +0.1249 ± 0.0262 | 0.00044 sig |