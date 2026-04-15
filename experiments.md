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

| Metric | Chance | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 6 | Best |
|--------|--------|-------|-------|-------|-------|-------|------|
| Age bal_acc | 0.50 | 0.483 | 0.587 | 0.505 | 0.573 | **0.597** | Exp 6 |
| Age AUC | 0.50 | — | — | — | — | **0.655** | Exp 6 |
| Sex AUC | 0.50 | 0.490 | 0.500 | 0.549 | — | **0.611** | Exp 6 |
| Contrast corr | 0.00 | 0.056 | **0.087** | 0.076 | 0.066 | 0.061 | Exp 2 |
| Luminance corr | 0.00 | — | — | — | — | **0.126** | Exp 6 |
| Luminance AUC | 0.50 | 0.523 | 0.514 | 0.545 | — | **0.529** | Exp 3 |
| Position corr | 0.00 | — | — | — | — | **0.191** | Exp 6 |
| Position AUC | 0.50 | — | 0.491 | 0.544 | — | **0.572** | Exp 6 |
| Narrative AUC | 0.50 | — | — | — | — | **0.553** | Exp 6 |

**Exp 6 (CorrCA) is best overall — strongest position/luminance correlations and subject probes.**

---

## Remaining Problems → Next Experiments

### P1: Spatial masking is trivial (r>0.9 between channels)
→ **Exp 4: Temporal-dominant masking** — mask entire time windows across all channels. Forces temporal prediction. (EEG2Rep, Laya)

### P2: Only using 20% of data (1,100 of 5,900 recordings)
→ **Exp 5: Multi-task pretraining + resting-state baseline subtraction** (planned)

### P3: Single-trial SNR at -24 dB ceiling
→ **Exp 6: CorrCA preprocessing** — spatial filters maximizing ISC. 129→5 channels. Expected +15-20 dB. (Parra et al.)

---

## Exp 4: Temporal-Dominant Masking
**Job:** 17605606 | Early stopped ep ~44
**Change:** Mask 2/4 time windows across ALL channels + 30% spatial in visible windows
**Result:** No improvement over Exp 2. Age 0.573, Contrast 0.066, Narrative 0.062. Temporal masking didn't help — problem is input SNR, not masking strategy.

## Exp 6: CorrCA Preprocessing — BEST
**CorrCA Job:** 17614210 | **Training Job:** 17614379 | **Eval Job:** 17614786 | Early stopped ep 44
**Change:** Offline CorrCA eigenvalue solve (R_b w = λ R_w w) → 129ch projected to 5 ISC-maximizing components + per-recording norm
**CorrCA ISC values:** 0.019, 0.012, 0.006, 0.004, 0.002 (701 subjects)
**Result:** Best overall. Position corr **0.191** (new best), Luminance corr **0.126** (new best), Age bal_acc **0.597**, Sex AUC **0.611**, Narrative AUC **0.553**. All stimulus probes above chance. Contrast corr 0.061 (slightly below Exp 2's 0.087 — may need more components).

---

## Next Steps
1. Try CorrCA with more components (10, 20) — contrast corr dropped, may need broader spatial coverage
2. Missing baselines: random encoder, permutation test, band power features, multiple seeds
3. Exp 5: Multi-task pretraining + resting-state baseline subtraction (planned)
