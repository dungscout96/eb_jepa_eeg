# Trivial baseline View 2 + View 3 — luna_corrca

Probe seeds: [42, 123, 456, 789, 2025]
Split: **test** | B=2000 | n_seeds=5

| Metric | Chance | View 2 mean ± σ | V2 p | View 3 mean ± σ | V3 p |
|---|---:|---:|---:|---:|---:|
| `cls_contrast_rms_auc` | 0.5 | +0.5247 ± 0.0240 | 0.083 ns | +0.5780 ± 0.0256 | 0.0024 sig |
| `cls_contrast_rms_bal_acc` | 0.0 | — | — | +0.5263 ± 0.0198 | 4.8e-07 sig |
| `cls_luminance_mean_auc` | 0.5 | +0.5117 ± 0.0439 | 0.58 ns | +0.5507 ± 0.0332 | 0.027 sig |
| `cls_luminance_mean_bal_acc` | 0.0 | — | — | +0.5251 ± 0.0225 | 8e-07 sig |
| `cls_narrative_event_score_auc` | 0.5 | +0.5090 ± 0.0180 | 0.33 ns | +0.4989 ± 0.0241 | 0.93 ns |
| `cls_narrative_event_score_bal_acc` | 0.0 | — | — | +0.4943 ± 0.0079 | 1.5e-08 sig |
| `cls_position_in_movie_auc` | 0.5 | +0.5245 ± 0.0274 | 0.12 ns | +0.5613 ± 0.0199 | 0.0023 sig |
| `cls_position_in_movie_bal_acc` | 0.0 | — | — | +0.5335 ± 0.0305 | 2.6e-06 sig |
| `reg_contrast_rms_corr` | 0.0 | +0.0841 ± 0.0660 | 0.046 sig | +0.1577 ± 0.0695 | 0.0071 sig |
| `reg_luminance_mean_corr` | 0.0 | +0.0648 ± 0.0431 | 0.028 sig | +0.1629 ± 0.0746 | 0.0081 sig |
| `reg_narrative_event_score_corr` | 0.0 | -0.0010 ± 0.0477 | 0.97 ns | -0.0086 ± 0.0549 | 0.75 ns |
| `reg_position_in_movie_corr` | 0.0 | +0.0866 ± 0.0593 | 0.031 sig | +0.1649 ± 0.0454 | 0.0012 sig |