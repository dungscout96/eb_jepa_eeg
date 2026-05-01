# Trivial baseline View 2 + View 3 — trivial_ridge_corrca_pooled35

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
| `reg_contrast_rms_corr` | 0.0 | +0.0897 ± 0.0230 | 0.00096 sig | +0.0902 ± 0.0232 | 0.00096 sig |
| `reg_luminance_mean_corr` | 0.0 | +0.0891 ± 0.0180 | 0.00038 sig | +0.0898 ± 0.0182 | 0.00038 sig |
| `reg_narrative_event_score_corr` | 0.0 | +0.0213 ± 0.0127 | 0.02 sig | +0.0213 ± 0.0127 | 0.02 sig |
| `reg_position_in_movie_corr` | 0.0 | +0.0715 ± 0.0071 | 2.4e-05 sig | +0.0721 ± 0.0078 | 3.2e-05 sig |