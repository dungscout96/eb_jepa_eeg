# EEG JEPA Experiments Log

## Experiment 1: Narrow Predictor Bottleneck (Baseline Fix)

**Date:** 2026-04-13
**Delta Job:** 17584039 | **W&B:** https://wandb.ai/braindecode/eb_jepa/runs/0j78jkkd
**Checkpoint:** `checkpoints/eeg_jepa/dev_2026-04-13_16-13/eeg_jepa_bs64_lr0.0005_std0.25_cov0.25_nw4_ws2s_seed2025/`

### Config
depth=2, embed_dim=64, **predictor_dim=24** (new), lr=5e-4, VCLoss(0.25,0.25), smooth_l1, 100 epochs, nw=4, ws=2s, bs=64

### Key Change
Reduced predictor internal dimension from 64 → 24 (ratio 0.375, matching V-JEPA). Creates information bottleneck forcing encoder to learn richer representations.

### Results

| Metric | Value | Baseline (buggy Apr 5) |
|--------|-------|----------------------|
| pred_loss (epoch 50) | 0.065 | 1764 (bug) |
| **Cosine similarity** | **0.25** | 0.97 (collapsed) |
| **Participation ratio** | **19.1 / 64** | 3.4 / 64 |
| val/reg_loss (best, ep50) | 0.846 | 2595 |
| val/reg_loss (final, ep99) | 1.12 | — |

### Probe Eval (epoch 50)

| Probe | Val | Test |
|-------|-----|------|
| Age AUC | **0.640** | 0.543 |
| Sex AUC | 0.555 | 0.490 |
| Luminance AUC | 0.537 | 0.523 |
| Contrast corr | 0.048 | 0.056 |
| Movie ID top-1 | 3.8% | 5.6% (chance=5%) |

### Conclusions
- **Collapse solved** — cosim 0.97→0.25, PR 3.4→19.1
- **Bug fixed** — pred_loss in correct range (~0.1, not ~1500)
- **Subject traits detectable** — age AUC 0.64 (val), matches jamming best
- **Stimulus probes still near chance** — movie features ~0.50 bal_acc
- **val/reg_loss U-curve** — best at epoch 50, degraded by epoch 99 → need early stopping
- **Val-test gap** — age AUC 0.64→0.54, overfitting to val distribution

---

## Core Problem Identified

> ### ⚠️ CORRECTION (2026-07-31) — the numbers below were estimates, now measured
>
> The figures in this section were literature estimates written in April 2026,
> before anything here was measured on HBN. They have since been measured
> directly — see
> [`experiments/snr_scaling/RESULTS.md`](experiments/snr_scaling/RESULTS.md).
> **The qualitative claims survive; two of the magnitudes do not.**
>
> | claim below | measured (R5 val, 293 subjects, 2 s windows) | verdict |
> |---|---|---|
> | δ/θ ISC = 0.10–0.28 | **0.040** mean over channels, **0.062** best channel | **2.5–4.5× too high** |
> | α ISC < 0.05 | **0.014** mean, **0.029** best channel | holds |
> | stimulus SNR −24 dB (0.4 % of variance) | **−20 dB (0.96 %)** per channel; **−9.6 dB (9.8 %)** multivariate | see below |
> | fingerprint ≈ 96 % of variance | ~99 % per channel; **~90 %** multivariate | readout-dependent |
>
> **The most important correction is the last two rows.** The −24 dB / 0.4 %
> figure is a *per-channel* quantity. What bounds a multivariate probe — a
> ridge on a 512-d embedding, which is what we actually train — is the
> cross-validated CorrCA reliability of **0.098**, i.e. **−9.6 dB**, an order
> of magnitude more signal than the per-channel number implies. Using the
> per-channel figure to argue "single-trial decoding is at its theoretical
> limit" understates the available signal ~10×, and the measured ceiling
> (probe *r* = 0.313, with the best checkpoint at 48 % of it) shows single-trial
> decoding is **not** at ceiling.
>
> The δ/θ ≫ α ordering — the actual mechanism this section argues for — is
> confirmed: δ/θ exceeds α by 2.8× at the channel mean. Only the absolute
> values were wrong.
>
> The cross-subject aggregation arithmetic below is sound and, with measured
> values, more favourable than stated: at per-channel `rho1 = 0.0096`,
> K = 660 gives **+8.1 dB**; at multivariate `rho1 = 0.098`, **+18.6 dB**.
>
> **When citing this section, cite the measured values, not the ones below.**

### Why the encoder learns subject identity, not stimulus content

**Signal decomposition:**
```
EEG(s, c, t) = stimulus_response(c, τ) + subject_fingerprint(c) + noise(c, t)
                     ~3 μV                    ~30 μV                ~50 μV
```

- Stimulus SNR per single trial: **-24 dB (0.4% of variance)**
  — *superseded: −20 dB (0.96 %) per channel, −9.6 dB (9.8 %) multivariate*
- Subject fingerprint explains **~96% of variance**
  — *superseded: ~99 % per channel, ~90 % multivariate*
- The masked prediction objective is rationally solved by learning subject patterns

**Why spatial masking is trivial for EEG:**
- Adjacent channels have correlation r > 0.9 (volume conduction)
- Predicting masked channels = spatial interpolation, no stimulus content needed

**Where stimulus signal lives:**
- Delta/Theta (1-8 Hz): ISC = 0.10-0.28 (narrative, scene boundaries)
  — *superseded: 0.040 mean over channels, 0.062 best channel*
- Alpha (8-12 Hz): ISC < 0.05 (subject-specific, dominant power)
  — *confirmed: 0.014 mean, 0.029 best channel*
- The encoder is dominated by alpha because that's where the variance is

**Theoretical limit of single-trial stimulus decoding:**
- Movie ID 20-way: P(correct) ≈ 5.4% (we observe 5.6%) — at ceiling
  — *not re-tested; but the "at ceiling" framing is wrong for the continuous
    movie-feature probe, which reaches only 48 % of its measured ceiling*
- Need cross-subject aggregation (√660 subjects ≈ 25.7× SNR boost → +4 dB) to detect stimulus
  — *arithmetic sound; with measured values, +8.1 dB per channel / +18.6 dB
    multivariate*

### Three proposed solutions (in priority order)
1. **Cross-subject contrastive loss** — pull same-time-different-subject pairs together
2. **Per-recording normalization + low-frequency input** — remove subject fingerprint from input (Experiment 2)
3. **Adversarial subject removal + stimulus gradients** — GRL + supervised stimulus loss

---

## Experiment 2: Per-Recording Normalization + Low-Frequency Envelope

**Date:** 2026-04-13
**Goal:** Remove subject fingerprint from input to force encoder toward stimulus content

### Changes
1. **Per-recording z-normalization** — normalize each recording independently (removes subject-specific amplitude/offset)
2. **1-8 Hz envelope channels** — extract Hilbert envelope in delta/theta band where ISC is highest, concatenate as additional input channels

### Hypothesis
- Per-recording norm removes ~96% subject variance (amplitude scale from impedance/skull)
- Low-frequency envelope isolates the band with highest stimulus content (ISC 0.10-0.28)
- Prediction loss may increase (subject shortcut removed), but stimulus probes should improve

### Expected Outcome
- Subject-trait probes (age/sex) should DROP (by design — fingerprint removed)
- Movie feature probes should IMPROVE (encoder forced toward stimulus dynamics)
- pred_loss may increase (harder task without subject shortcut)
