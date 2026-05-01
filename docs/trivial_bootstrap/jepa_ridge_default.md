# Trivial baseline View 2 + View 3 — jepa_ridge_default

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
| `reg_contrast_rms_corr` | 0.0 | +0.1736 ± 0.0134 | 8.5e-06 sig | +0.1740 ± 0.0135 | 8.5e-06 sig |
| `reg_luminance_mean_corr` | 0.0 | +0.1982 ± 0.0125 | 3.8e-06 sig | +0.1987 ± 0.0129 | 4.2e-06 sig |
| `reg_narrative_event_score_corr` | 0.0 | +0.0331 ± 0.0317 | 0.08 ns | +0.0333 ± 0.0318 | 0.08 ns |
| `reg_position_in_movie_corr` | 0.0 | +0.1524 ± 0.0147 | 2.1e-05 sig | +0.1528 ± 0.0149 | 2.1e-05 sig |