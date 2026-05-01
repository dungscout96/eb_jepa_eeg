# Trivial baseline View 2 + View 3 — trivial_ridge_raw_pooled903

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
| `reg_contrast_rms_corr` | 0.0 | +0.0370 ± 0.0225 | 0.021 sig | +0.0371 ± 0.0225 | 0.021 sig |
| `reg_luminance_mean_corr` | 0.0 | +0.0583 ± 0.0244 | 0.0059 sig | +0.0588 ± 0.0250 | 0.0062 sig |
| `reg_narrative_event_score_corr` | 0.0 | +0.0212 ± 0.0209 | 0.085 ns | +0.0216 ± 0.0205 | 0.078 ns |
| `reg_position_in_movie_corr` | 0.0 | +0.0087 ± 0.0254 | 0.48 ns | +0.0091 ± 0.0254 | 0.47 ns |