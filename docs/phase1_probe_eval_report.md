# Phase 1 Probe Evaluation Report

**Date:** 2026-04-10
**Sweep:** 11 temporal configs x 3 seeds = 33 experiments, all trained to 100 epochs
**Fixed hyperparams:** depth=2, embed_dim=64, lr=5e-4, VCLoss(0.25,0.25), smooth_l1, bs=64 (bs=32 for nw4_ws4, nw8_ws2)

---

## 1. Movie-Feature Probes (per-clip, stimulus-driven)

Frozen encoder embeddings evaluated with fresh linear probes (20 epochs) on val set.
Each clip is one training sample — labels are movie features at that timestamp.

### Classification (balanced accuracy, chance = 0.50)

| Config | total ctx | contrast | luminance | position | narrative | **avg** |
|--------|-----------|----------|-----------|----------|-----------|---------|
| nw1_ws1 | 1s | 0.505 | 0.488 | 0.491 | 0.492 | 0.494 |
| nw1_ws2 | 2s | 0.496 | 0.504 | 0.502 | 0.499 | 0.500 |
| nw1_ws4 | 4s | 0.489 | 0.477 | 0.519 | 0.496 | 0.495 |
| nw2_ws1 | 2s | 0.495 | **0.513** | **0.509** | **0.522** | **0.510** |
| nw2_ws2 | 4s | 0.503 | 0.500 | 0.506 | 0.477 | 0.497 |
| nw2_ws4 | 8s | **0.515** | 0.485 | 0.507 | 0.494 | 0.500 |
| nw4_ws1 | 4s | 0.502 | 0.506 | 0.486 | 0.492 | 0.497 |
| nw4_ws2 | 8s | 0.511 | **0.514** | 0.505 | 0.494 | 0.506 |
| **nw4_ws4** | **16s** | **0.516** | 0.506 | **0.511** | **0.503** | **0.509** |
| nw8_ws1 | 8s | 0.512 | 0.504 | 0.493 | 0.493 | 0.500 |
| nw8_ws2 | 16s | 0.507 | 0.502 | 0.494 | 0.493 | 0.499 |

### Regression (correlation, chance = 0.0)

| Config | total ctx | contrast | luminance | position | narrative |
|--------|-----------|----------|-----------|----------|-----------|
| **nw2_ws1** | 2s | 0.049 | -0.002 | **0.094** | **0.049** |
| **nw2_ws4** | 8s | **0.070** | -0.030 | 0.039 | -0.004 |
| nw4_ws2 | 8s | 0.056 | 0.032 | 0.013 | 0.006 |
| nw4_ws4 | 16s | 0.033 | 0.039 | 0.015 | -0.008 |
| nw8_ws2 | 16s | 0.050 | 0.007 | 0.019 | -0.005 |

**Verdict:** All near chance. Best: **nw2_ws1** and **nw4_ws4**. Multi-window configs marginally outperform single-window, but signal is weak everywhere.

---

## 2. Subject-Trait Probes (per-recording, pooled embeddings)

Frozen encoder embeds multiple clips per recording, mean-pools to one vector per subject,
then trains a linear probe for subject-level classification/regression.

### Age > median classification (32 clips/rec)

| Config | total ctx | bal_acc | AUC |
|--------|-----------|---------|-----|
| **nw1_ws1** | **1s** | **0.524** | **0.590** |
| nw2_ws1 | 2s | 0.525 | 0.569 |
| nw2_ws4 | 8s | 0.526 | 0.559 |
| **nw4_ws4** | **16s** | **0.525** | **0.582** |
| nw2_ws2 | 4s | 0.509 | 0.575 |
| nw4_ws1 | 4s | 0.511 | 0.572 |
| nw4_ws2 | 8s | 0.508 | 0.569 |
| nw8_ws1 | 8s | 0.514 | 0.566 |
| nw1_ws2 | 2s | 0.512 | 0.555 |
| nw1_ws4 | 4s | 0.508 | 0.555 |
| nw8_ws2 | 16s | 0.503 | 0.548 |

### Sex classification and age regression (4 clips/rec)

| Config | total ctx | sex_ba | sex_auc | age_mae | age_corr | age_r2 |
|--------|-----------|--------|---------|---------|----------|--------|
| nw1_ws1 | 1s | 0.496 | 0.507 | 2.72 | 0.031 | -0.019 |
| nw1_ws2 | 2s | 0.498 | 0.495 | 2.74 | -0.019 | -0.038 |
| nw1_ws4 | 4s | 0.501 | 0.474 | 2.73 | -0.008 | -0.028 |
| nw2_ws1 | 2s | 0.499 | 0.472 | 2.74 | 0.015 | -0.029 |
| nw2_ws2 | 4s | 0.498 | 0.471 | 2.75 | -0.003 | -0.033 |
| nw2_ws4 | 8s | 0.498 | 0.466 | 2.72 | -0.026 | -0.035 |
| nw4_ws1 | 4s | 0.502 | 0.457 | 2.73 | 0.005 | -0.030 |
| nw4_ws2 | 8s | 0.494 | 0.476 | 2.75 | -0.065 | -0.062 |
| **nw4_ws4** | **16s** | 0.496 | **0.526** | **2.71** | -0.039 | -0.028 |
| nw8_ws1 | 8s | 0.502 | 0.466 | 2.72 | -0.035 | -0.038 |
| nw8_ws2 | 16s | 0.496 | 0.497 | 2.72 | -0.090 | -0.046 |

**Verdict:** Strongest subject signal across all probes. Age binary AUC 0.55-0.59 is meaningfully above chance. **nw1_ws1** (simplest) and **nw4_ws4** (most context) are consistently best. Sex and age regression near chance with only 4 clips -- likely needs more clips per recording.

---

## 3. Overall Ranking

| Rank | Config | total ctx | Strength | Notes |
|------|--------|-----------|----------|-------|
| 1 | **nw4_ws4** | 16s | Best all-around | Top movie-feature cls, top subject AUC, but 8x slower (bs=32) |
| 2 | **nw2_ws1** | 2s | Best movie-feature | Strongest regression correlations, good subject bal_acc |
| 3 | **nw1_ws1** | 1s | Best subject-trait | Highest age AUC (0.59), simplest/fastest config |
| 4 | nw4_ws2 | 8s | Balanced | Good movie cls, decent subject signal |
| 5 | nw2_ws4 | 8s | Mixed | Best contrast correlation, but inconsistent |

---

## 4. Key Takeaways

- **Embeddings carry more subject-level signal than stimulus-level signal.** Subject AUC ~0.59 vs movie-feature classification ~0.50 (chance). The encoder learns to represent *who* is watching more than *what* they're watching.

- **More temporal context helps for movie features** (nw4_ws4 > nw1_ws1), but **doesn't help for subject traits** (nw1_ws1 ties nw4_ws4). Subject identity is in the neural "fingerprint" even from 1s of EEG.

- **Multi-window masking (nw>=2) beats single-window (nw=1)** for stimulus features, supporting the temporal masking objective -- but the overall signal is too weak to be conclusive yet.

- **nw8 configs don't improve over nw4** -- diminishing returns or contention from sequence length.

---

## 5. Methodology Notes

- All checkpoints are from epoch 99 (end of training, no early stopping)
- Movie-feature probes: 20-epoch linear probe on frozen encoder, evaluated per-clip
- Subject-trait probes: per-recording mean-pooled embeddings, 100-epoch linear probe
- Age binary: binarized as `age > median(age)` across training split
- Sex: M=1, F=0 binary classification
- Age regression: linear probe with MSE loss, standardized targets
- Features: contrast_rms, luminance_mean, position_in_movie, narrative_event_score
- Movie identity probe (20-bin temporal classification): results pending
