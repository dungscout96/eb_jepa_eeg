# EEG JEPA Experiments Log

## Core Problem
```
EEG = stimulus_response (~3μV) + subject_fingerprint (~30μV) + noise (~50μV)
```
Stimulus SNR = -24 dB (0.4% of variance). Spatial masking trivial (r>0.9 between channels). Encoder learns subject identity, not stimulus content.

---

## Exp 1: Narrow Predictor (Baseline)
**Job:** 17584039 | **W&B:** [0j78jkkd](https://wandb.ai/braindecode/eb_jepa/runs/0j78jkkd)
**Change:** predictor_dim 64→24 (V-JEPA bottleneck ratio 0.375)
**Result:** Collapse solved (cosim 0.97→0.25, PR 3.4→19.1). Age AUC 0.64 val / 0.54 test. Stimulus probes at chance. Val-test gap.

## Exp 2: Per-Recording Norm + Envelope
**Job:** 17595315 | **W&B:** [gu8tw0ay](https://wandb.ai/braindecode/eb_jepa/runs/gu8tw0ay) | Early stopped ep 44
**Change:** Per-recording z-norm (removes subject amplitude) + 1-8Hz envelope channels
**Result:** Best test generalization. Age bal_acc 0.587 test (+21% vs Exp 1). Contrast corr 0.087 (+55%). Val-test gap reversed.

## Exp 1v1: Contrastive Loss — DISCARDED
**Job:** 17597552 | Bugs: .detach() on stability shift, double forward pass, VCLoss mismatch
**Result:** Contrastive loss stuck at log(64)=4.16 for all epochs. Zero effect.

## Exp 3: Fixed Soft Contrastive + Adversarial
**Job:** 17598100 | **W&B:** [w6hcnusv](https://wandb.ai/braindecode/eb_jepa/runs/w6hcnusv) | Early stopped ep 43
**Change:** SoftCLT contrastive + gradient reversal subject discriminator (bugs fixed)
**Result:** Sex AUC 0.549 (first above chance). Luminance AUC 0.545, Position AUC 0.544. Modest improvements, no breakthrough.

---

## Cross-Experiment Comparison (Test Set)

| Metric | Chance | Exp 1 | Exp 2 | Exp 3 | Best |
|--------|--------|-------|-------|-------|------|
| Age bal_acc | 0.50 | 0.483 | **0.587** | 0.505 | Exp 2 |
| Sex AUC | 0.50 | 0.490 | 0.500 | **0.549** | Exp 3 |
| Contrast corr | 0.00 | 0.056 | **0.087** | 0.076 | Exp 2 |
| Luminance AUC | 0.50 | 0.523 | 0.514 | **0.545** | Exp 3 |
| Position AUC | 0.50 | — | 0.491 | **0.544** | Exp 3 |

**All stimulus metrics near theoretical single-trial ceiling (-24 dB).**

---

## Remaining Problems → Next Experiments

### P1: Spatial masking is trivial (r>0.9 between channels)
→ **Exp 4: Temporal-dominant masking** — mask entire time windows across all channels. Forces temporal prediction. (EEG2Rep, Laya)

### P2: Only using 20% of data (1,100 of 5,900 recordings)
→ **Exp 5: Multi-task pretraining + resting-state baseline subtraction** (planned)

### P3: Single-trial SNR at -24 dB ceiling
→ **Exp 6: CorrCA preprocessing** — spatial filters maximizing ISC. 129→5 channels. Expected +15-20 dB. (Parra et al.)

---

## Exp 4: Temporal-Dominant Masking (Running)
**Job:** 17605606 | **Branch:** kkokate/exp1-cross-subject-contrastive
**Change:** Mask 2/4 time windows across ALL channels + 30% spatial in visible windows
**Hypothesis:** Eliminates spatial interpolation shortcut. Temporal autocorrelation at 2s lag is r<0.3 (vs r>0.9 spatial).

## Exp 6: CorrCA Preprocessing (Running)
**CorrCA Job:** 17605607 (computing filters) | **Training:** after CorrCA completes
**Change:** Offline CorrCA eigenvalue solve → 129 channels projected to 5 stimulus-driven components
**Hypothesis:** +15-20 dB SNR boost from projecting out non-stimulus dimensions.

---

## Missing Baselines
1. Random encoder (untrained) + same probes
2. Permutation test (shuffled labels)
3. Handcrafted features (band power) + same probes
4. Multiple seeds (3-5 runs)
