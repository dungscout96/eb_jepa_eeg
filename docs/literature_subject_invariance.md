# Literature: subject-invariant stimulus-driven SSL for EEG

Background for the Exp 9+ line of work. Compiled 2026-04-23.

## Direct observations from our data

- Raw EEG per-rec z-normed: η²_subj = 0.000, η²_stim = 0.026. Stimulus signal lives in delta (1-4 Hz).
- CorrCA component 1 achieves η²_stim = 0.074 — best 1-dim linear stimulus representation.
- Our Exp 6 encoder: PC1 = 70% variance, pure subject (sex AUC 0.76). Re-introduces subject identity not present in input.
- Auxiliary CLIP alignment + V-JEPA 2 target swap + cross-subject permutation have all failed.

## 1. Subject-invariant SSL for EEG

Dominant approach: gradient-reversal subject adversary on a CNN trunk. Mostly validated on classification, not naturalistic stimulus decoding.

- **Ozdenizci 2020** (doi:10.1109/ACCESS.2020.2971600). Adversarial inference via gradient reversal. Shows vanilla CNNs leak 31-63% subject-ID accuracy.
- **Han 2021** (arXiv:1910.07747). Non-adversarial alternative using MI estimation — factorizes latent into class-relevant / subject-relevant.
- **Cai 2025** (arXiv:2501.08693, SDN-Net). MINE-style MI minimization on speech envelope reconstruction from EEG. Most directly analogous to our naturalistic setup.

## 2. Group-mean / consensus-based targets

No published JEPA/MAE uses CorrCA or ISC-projection as target. Closest:

- **Shen 2024 — CL-SSTER** (arXiv:2402.14213, NeuroImage). Cross-subject time-aligned InfoNCE: positives are different subjects at same movie timestamp. Achieves highest reported ISC on naturalistic EEG. This is essentially our "Idea 2" done correctly with InfoNCE instead of JEPA prediction.
- **Richard 2020 — FastSRM** (arXiv:1909.12537). Subject-specific orthogonal projections into a shared time-locked subspace. fMRI consensus analogue — nonlinear generalization of CorrCA.
- **Guo 2021 — Hybrid Hyperalignment** (PubMed:33762217). Joint response + connectivity alignment.

## 3. JEPA variants for neural data

- **Dong 2024 — Brain-JEPA** (arXiv:2409.19407, NeurIPS Spotlight). fMRI JEPA; shows masking geometry is the main lever for what features get learned.
- **Guetschel 2024 — S-JEPA** (arXiv:2403.11772). First EEG-JEPA. Spatial filtering matters more than mask size — matches our observation that subject features leak through channel-wise statistics.
- **LeJEPA EEG 2026** (arXiv:2603.16281). Uses explicit variance/covariance regularization (like our VCLoss). Would penalize a single dominant PC — which is exactly our PC1 = 70% failure mode.
- **EEG2Rep — Foumani 2024** (arXiv:2402.17772, KDD). Robustness to amplitude variability but no PCA/ISC leakage eval.

## 4. Information-theoretic subject removal

- **Cai 2025** — MINE for MI minimization on continuous stimulus decoding; works.
- **Han 2021** — factorized class-relevant / subject-relevant branches.
- **IB SSVEP** (doi:10.3389/fnhum.2021.675091). Information-bottleneck compression improves cross-subject SSVEP.

## 5. Counter-examples

- **VLA-JEPA** (arXiv:2602.10098). On noisy human videos the EMA target encoder "turns into a delta-frame encoder of nuisance motion rather than meaningful transition dynamics". Same failure we observe.
- **Apple ML Research — Implicit Bias of JEPA (2024)**. Proves JEPA's implicit noise suppression fails when noise is **low-rank and high-variance** — which is exactly subject identity. Theoretical grounding for why V-JEPA on EEG recovers subject PC1.

## Additional solutions proposed

1. **SRM-style target encoder (group-consensus JEPA).** Replace EMA target with a subject-specific orthogonal projection W_s into a shared time-locked subspace, fit jointly with the encoder. Generalizes CorrCA to nonlinear features inside JEPA.

2. **CL-SSTER-in-JEPA.** Add cross-subject time-aligned InfoNCE on the predictor output (not the encoder output). Forces the predictor to carry shared signal, preserving encoder capacity.

3. **MI-bottleneck on predictor output vs subject ID (MINE/CLUB).** Unlike adversarial heads, MI regularizers operate downstream of the encoder so cannot destroy latent structure, only penalize subject-predictive axes. Cai 2025 shows this works for continuous stimulus decoding.

## What this means for us

- Exp 9 (MAE with CorrCA target) is a **simpler precursor** of option 1. If it works, scale up to proper SRM.
- Option 2 is a corrected version of Exp 7 (auxiliary CLIP alignment) — the key fix is applying InfoNCE to **predictor** output at **matched movie time across subjects**, not to encoder output against CLIP.
- Option 3 is a safer alternative to Exp 7c's subject-adversarial head — apply MI penalty downstream.
- The Apple ML result explains why even large JEPA models fail here: subject noise is low-rank + high-variance, which defeats JEPA's implicit bias.
