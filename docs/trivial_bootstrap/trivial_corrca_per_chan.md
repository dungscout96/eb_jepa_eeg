# Trivial baseline View 2 + View 3 — trivial_corrca_per_chan

Probe seeds: [7, 13, 42, 1234, 2025]
Split: **test** | B=2000 | n_seeds=5

| Metric | Chance | View 2 mean ± σ | V2 p | View 3 mean ± σ | V3 p |
|---|---:|---:|---:|---:|---:|
| `age_cls_auc` | 0.5 | — | — | +0.6446 ± 0.0118 | 1.1e-05 sig |
| `age_cls_bal_acc` | 0.0 | — | — | +0.5905 ± 0.0286 | 1.3e-06 sig |
| `age_reg_corr` | 0.0 | +0.1317 ± 0.2839 | 0.36 ns | +0.1339 ± 0.2841 | 0.35 ns |
| `age_reg_mae` | 0.0 | — | — | +3.0797 ± 0.1571 | 1.6e-06 sig |
| `cls_contrast_rms_auc` | 0.5 | +0.5209 ± 0.0382 | 0.29 ns | +0.5209 ± 0.0382 | 0.29 ns |
| `cls_contrast_rms_bal_acc` | 0.0 | — | — | +0.5127 ± 0.0223 | 8.5e-07 sig |
| `cls_luminance_mean_auc` | 0.5 | +0.5095 ± 0.0284 | 0.49 ns | +0.5103 ± 0.0285 | 0.47 ns |
| `cls_luminance_mean_bal_acc` | 0.0 | — | — | +0.4995 ± 0.0169 | 3.2e-07 sig |
| `cls_narrative_event_score_auc` | 0.5 | +0.5224 ± 0.0190 | 0.057 ns | +0.5218 ± 0.0191 | 0.064 ns |
| `cls_narrative_event_score_bal_acc` | 0.0 | — | — | +0.5155 ± 0.0143 | 1.4e-07 sig |
| `cls_position_in_movie_auc` | 0.5 | +0.5306 ± 0.0497 | 0.24 ns | +0.5310 ± 0.0490 | 0.23 ns |
| `cls_position_in_movie_bal_acc` | 0.0 | — | — | +0.5156 ± 0.0415 | 1e-05 sig |
| `reg_contrast_rms_corr` | 0.0 | +0.0349 ± 0.0487 | 0.18 ns | +0.0348 ± 0.0487 | 0.18 ns |
| `reg_luminance_mean_corr` | 0.0 | +0.0598 ± 0.0665 | 0.11 ns | +0.0603 ± 0.0660 | 0.11 ns |
| `reg_narrative_event_score_corr` | 0.0 | -0.0139 ± 0.0454 | 0.53 ns | -0.0130 ± 0.0444 | 0.55 ns |
| `reg_position_in_movie_corr` | 0.0 | +0.0520 ± 0.1003 | 0.31 ns | +0.0506 ± 0.0992 | 0.32 ns |
| `sex_auc` | 0.5 | +0.5594 ± 0.0999 | 0.25 ns | +0.5602 ± 0.1005 | 0.25 ns |
| `sex_bal_acc` | 0.0 | — | — | +0.5138 ± 0.0205 | 6.1e-07 sig |