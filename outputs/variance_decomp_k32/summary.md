# Variance Decomposition Summary

| run | regularizer | coeff | config | seed | S | K | D | η²_subj | η²_stim | stim/within | eff_rank(C_subj) | eff_rank(C_within) | angle@k=5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eeg_jepa_bs32_lr0.0005_sigreg0.1_nw4_ws4s_seed2025 | sigreg | 0.1 | nw4_ws4 | 2025 | 293 | 32 | 64 | 0.371 | 0.0041 | 0.0065 | 4.42 | 5.64 | 25.1 |
| eeg_jepa_bs32_lr0.0005_std0.1_cov0.1_nw4_ws4s_seed2025 | vicreg | None | nw4_ws4 | 2025 | 293 | 32 | 64 | 0.278 | 0.0078 | 0.0108 | 1.90 | 3.32 | 19.3 |
| eeg_jepa_bs32_lr0.0005_std1.0_cov1.0_nw4_ws4s_seed2025 | vicreg | None | nw4_ws4 | 2025 | 293 | 32 | 64 | 0.194 | 0.0060 | 0.0074 | 2.30 | 3.09 | 11.4 |
| eeg_jepa_bs64_lr0.0005_sigreg0.1_nw2_ws1s_seed2025 | sigreg | 0.1 | nw2_ws1 | 2025 | 293 | 32 | 64 | 0.348 | 0.0047 | 0.0073 | 2.58 | 5.25 | 16.9 |
| eeg_jepa_bs64_lr0.0005_sigreg1.0_nw1_ws1s_seed2025 | sigreg | 1.0 | nw1_ws1 | 2025 | 293 | 32 | 64 | 0.157 | 0.0054 | 0.0064 | 4.67 | 6.72 | 32.1 |
| eeg_jepa_bs64_lr0.0005_std1.0_cov1.0_nw2_ws1s_seed2025 | vicreg | None | nw2_ws1 | 2025 | 293 | 32 | 64 | 0.061 | 0.0068 | 0.0072 | 3.36 | 3.17 | 10.5 |
