# Trivial baseline View 2 + View 3 — trivial_ridge_raw903

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
| `reg_contrast_rms_corr` | 0.0 | +0.0028 ± 0.0215 | 0.78 ns | +0.0093 ± 0.0179 | 0.31 ns |
| `reg_luminance_mean_corr` | 0.0 | -0.0013 ± 0.0254 | 0.91 ns | +0.0033 ± 0.0249 | 0.78 ns |
| `reg_narrative_event_score_corr` | 0.0 | +0.0179 ± 0.0155 | 0.062 ns | +0.0181 ± 0.0127 | 0.033 sig |
| `reg_position_in_movie_corr` | 0.0 | +0.0026 ± 0.0186 | 0.77 ns | +0.0082 ± 0.0155 | 0.3 ns |