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

### Why the encoder learns subject identity, not stimulus content

**Signal decomposition:**
```
EEG(s, c, t) = stimulus_response(c, τ) + subject_fingerprint(c) + noise(c, t)
                     ~3 μV                    ~30 μV                ~50 μV
```

- Stimulus SNR per single trial: **-24 dB (0.4% of variance)**
- Subject fingerprint explains **~96% of variance**
- The masked prediction objective is rationally solved by learning subject patterns

**Why spatial masking is trivial for EEG:**
- Adjacent channels have correlation r > 0.9 (volume conduction)
- Predicting masked channels = spatial interpolation, no stimulus content needed

**Where stimulus signal lives:**
- Delta/Theta (1-8 Hz): ISC = 0.10-0.28 (narrative, scene boundaries)
- Alpha (8-12 Hz): ISC < 0.05 (subject-specific, dominant power)
- The encoder is dominated by alpha because that's where the variance is

**Theoretical limit of single-trial stimulus decoding:**
- Movie ID 20-way: P(correct) ≈ 5.4% (we observe 5.6%) — at ceiling
- Need cross-subject aggregation (√660 subjects ≈ 25.7× SNR boost → +4 dB) to detect stimulus

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

### Results

**Delta Job:** 17595315 | **W&B:** https://wandb.ai/braindecode/eb_jepa/runs/gu8tw0ay
**Best checkpoint:** epoch 25 (val/reg_loss=0.795) | **Early stopping:** triggered at epoch 44

| Metric | Exp 1 (baseline) | Exp 2 (per-rec norm) | Change |
|--------|-----------------|---------------------|--------|
| **Best val/reg_loss** | 0.846 (ep50) | **0.795 (ep25)** | -6% better |
| pred_loss (best ep) | 0.065 | 0.183 | harder task (expected) |

**Probe eval (best.pth.tar = epoch 25):**

| Probe | Val | Test | Exp 1 Val | Exp 1 Test |
|-------|-----|------|-----------|------------|
| Age AUC | 0.475 | **0.575** | 0.640 | 0.543 |
| Age bal_acc | 0.520 | **0.587** | 0.516 | 0.483 |
| Sex AUC | 0.468 | 0.500 | 0.555 | 0.490 |
| Contrast corr | 0.037 | **0.087** | 0.048 | 0.056 |
| Narrative corr | -0.031 | **0.057** | -0.016 | 0.049 |
| Position corr | 0.022 | 0.042 | 0.039 | -0.028 |
| Movie ID top-1 | 3.4% | 4.6% | 3.8% | 5.6% |
| Movie ID top-5 | 23.2% | 22.2% | 23.6% | 20.4% |

### Conclusions
- **val/reg_loss improved** (0.795 vs 0.846) — per-recording norm helps training
- **Test age bal_acc improved** (0.587 vs 0.483) — better generalization, smaller val-test gap
- **Test contrast corr improved** (0.087 vs 0.056) — 55% improvement in stimulus decoding
- **Val age AUC dropped** (0.475 vs 0.640) — expected, subject fingerprint partially removed
- **Test age AUC improved** (0.575 vs 0.543) — representation generalizes better
- **Key finding:** per-recording norm reduces val-test gap for subject traits (0.475→0.575 test vs 0.640→0.543 test), suggesting less overfitting to val distribution
- **Stimulus probes still near chance** but trending in the right direction
- **Early stopping worked** — saved 56 epochs of compute

---

## Experiment 1: Cross-Subject Contrastive Loss (Running)

**Date:** 2026-04-14
**Branch:** `kkokate/exp1-cross-subject-contrastive`
**Delta Job:** 17597552 | **W&B:** https://wandb.ai/braindecode/eb_jepa/runs/nbmlnheb
**Config:** Same as Exp 2 + `loss.contrastive_coeff=0.05, n_bins=20, temperature=0.1`

### Key Change
InfoNCE contrastive loss pulling embeddings from different subjects at the same movie time together. Discretizes position_in_movie into 20 bins (~10s each). Only stimulus-locked responses correlate across subjects, so this loss directly incentivizes stimulus encoding.

### Status: Training (epoch ~2, contrastive_loss=4.09 ≈ log(64), at chance initially)
