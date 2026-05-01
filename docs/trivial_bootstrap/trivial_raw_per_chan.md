# Trivial baseline View 2 + View 3 — trivial_raw_per_chan

Probe seeds: [7, 13, 42, 1234, 2025]
Split: **test** | B=2000 | n_seeds=5

| Metric | Chance | View 2 mean ± σ | V2 p | View 3 mean ± σ | V3 p |
|---|---:|---:|---:|---:|---:|
| `age_cls_auc` | 0.5 | — | — | +0.6949 ± 0.0095 | 1.4e-06 sig |
| `age_cls_bal_acc` | 0.0 | — | — | +0.6459 ± 0.0176 | 1.3e-07 sig |
| `age_reg_corr` | 0.0 | +0.3144 ± 0.0860 | 0.0012 sig | +0.3158 ± 0.0828 | 0.001 sig |
| `age_reg_mae` | 0.0 | — | — | +3.0356 ± 0.1842 | 3.2e-06 sig |
| `cls_contrast_rms_auc` | 0.5 | +0.5021 ± 0.0273 | 0.87 ns | +0.5021 ± 0.0272 | 0.87 ns |
| `cls_contrast_rms_bal_acc` | 0.0 | — | — | +0.4899 ± 0.0223 | 1e-06 sig |
| `cls_luminance_mean_auc` | 0.5 | +0.4980 ± 0.0092 | 0.65 ns | +0.4982 ± 0.0103 | 0.71 ns |
| `cls_luminance_mean_bal_acc` | 0.0 | — | — | +0.4989 ± 0.0104 | 4.5e-08 sig |
| `cls_narrative_event_score_auc` | 0.5 | +0.5158 ± 0.0190 | 0.14 ns | +0.5154 ± 0.0192 | 0.15 ns |
| `cls_narrative_event_score_bal_acc` | 0.0 | — | — | +0.5213 ± 0.0200 | 5.2e-07 sig |
| `cls_position_in_movie_auc` | 0.5 | +0.5131 ± 0.0672 | 0.68 ns | +0.5137 ± 0.0662 | 0.67 ns |
| `cls_position_in_movie_bal_acc` | 0.0 | — | — | +0.4883 ± 0.0360 | 7e-06 sig |
| `reg_contrast_rms_corr` | 0.0 | -0.0132 ± 0.0602 | 0.65 ns | -0.0123 ± 0.0556 | 0.65 ns |
| `reg_luminance_mean_corr` | 0.0 | +0.0015 ± 0.0420 | 0.94 ns | +0.0031 ± 0.0411 | 0.87 ns |
| `reg_narrative_event_score_corr` | 0.0 | +0.0420 ± 0.0611 | 0.2 ns | +0.0410 ± 0.0592 | 0.2 ns |
| `reg_position_in_movie_corr` | 0.0 | -0.0087 ± 0.0863 | 0.83 ns | -0.0103 ± 0.0853 | 0.8 ns |
| `sex_auc` | 0.5 | +0.5890 ± 0.0169 | 0.0003 sig | +0.5883 ± 0.0167 | 0.00029 sig |
| `sex_bal_acc` | 0.0 | — | — | +0.5362 ± 0.0101 | 3.1e-08 sig |