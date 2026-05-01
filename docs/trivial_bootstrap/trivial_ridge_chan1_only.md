# Trivial baseline View 2 + View 3 — trivial_ridge_chan1_only

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
| `reg_contrast_rms_corr` | 0.0 | +0.0759 ± 0.0180 | 0.00071 sig | +0.0757 ± 0.0183 | 0.00075 sig |
| `reg_luminance_mean_corr` | 0.0 | +0.0772 ± 0.0314 | 0.0053 sig | +0.0774 ± 0.0319 | 0.0056 sig |
| `reg_narrative_event_score_corr` | 0.0 | +0.0208 ± 0.0121 | 0.019 sig | +0.0208 ± 0.0121 | 0.018 sig |
| `reg_position_in_movie_corr` | 0.0 | +0.0637 ± 0.0111 | 0.00021 sig | +0.0641 ± 0.0117 | 0.00025 sig |