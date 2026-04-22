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

| Metric | Chance | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 6 | Exp 7 | Exp 7b | Best |
|--------|--------|-------|-------|-------|-------|-------|-------|--------|------|
| Age bal_acc | 0.50 | 0.483 | 0.587 | 0.505 | 0.573 | 0.597 | 0.598 | **0.635** | Exp 7b |
| Age AUC | 0.50 | — | — | — | — | 0.655 | 0.678 | **0.708** | Exp 7b |
| Sex AUC | 0.50 | 0.490 | 0.500 | 0.549 | — | 0.611 | 0.604 | **0.661** | Exp 7b |
| Contrast corr | 0.00 | 0.056 | **0.087** | 0.076 | 0.066 | 0.061 | 0.040 | 0.067 | Exp 2 |
| Contrast AUC | 0.50 | — | — | — | — | 0.497 | 0.506 | **0.518** | Exp 7b |
| Luminance corr | 0.00 | — | — | — | — | 0.126 | 0.115 | **0.175** | Exp 7b |
| Luminance AUC | 0.50 | 0.523 | 0.514 | 0.545 | — | 0.529 | 0.537 | **0.563** | Exp 7b |
| Position corr | 0.00 | — | — | — | — | 0.191 | 0.176 | **0.251** | Exp 7b |
| Position AUC | 0.50 | — | 0.491 | 0.544 | — | 0.572 | 0.572 | **0.599** | Exp 7b |
| Narrative corr | 0.00 | — | — | — | — | 0.009 | -0.010 | 0.008 | Exp 6 |
| Narrative AUC | 0.50 | — | — | — | — | **0.553** | 0.559 | 0.537 | Exp 6 |

**Exp 7b (CorrCA + CLIP per-window InfoNCE) is the new best on 8/11 metrics** — largest gains on position (+31% corr, +2.7pp AUC) and luminance (+39% corr, +3.4pp AUC) vs Exp 6. Narrative regresses slightly; sits near zero in Exp 6 so likely within noise.

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

## Exp 6 Multi-Seed Validation (5 seeds: 2025, 42, 123, 456, 789)

| Metric | Mean ± Std | Chance | Sig |
|--------|-----------|--------|-----|
| Position corr | **0.176 ± 0.048** | 0.0 | *** |
| Luminance corr | **0.168 ± 0.059** | 0.0 | *** |
| Contrast corr | **0.115 ± 0.054** | 0.0 | *** |
| Position AUC | **0.580 ± 0.025** | 0.5 | *** |
| Age bal_acc | **0.637 ± 0.024** | 0.5 | *** |
| Sex AUC | **0.618 ± 0.007** | 0.5 | *** |

---

## Exp 6 Optimization Sweep (Complete)

All runs use pure JEPA loss (smooth_l1 pred + VCLoss). Only preprocessing and architecture differ.

| Run | Pos corr | Lum corr | Con corr | Age bal_acc | Sex AUC |
|-----|----------|----------|----------|-------------|---------|
| **Baseline** (5ch) | 0.176 | **0.168** | **0.115** | 0.637 | 0.618 |
| **A: 10 comp** | **0.215** | 0.153 | 0.070 | 0.585 | 0.602 |
| **B: depth=4, pred=48** | -0.004 | 0.104 | 0.047 | 0.655 | 0.430 |
| **C: 4bands x 3** (12ch) | 0.208 | 0.161 | 0.066 | 0.606 | 0.617 |
| **D: 1-8Hz+OAS** (5ch) | 0.200 | 0.161 | **0.089** | 0.587 | **0.639** |

**Note:** Baseline is 5-seed mean; A-D are single seed. High variance expected.

### Key Findings
- **Run A (10 comp):** Best position corr (+22%). More components help temporal localization but dilute contrast/luminance.
- **Run B (deeper+wider): FAILED.** Wider predictor (48 vs 24) removed the information bottleneck → collapsed stimulus representations.
- **Run C (band-specific):** Solid overall, 12ch fixes masking, but no clear win. Alpha/beta ISC near noise floor adds little.
- **Run D (bandpass+OAS):** Best contrast recovery (0.089, matching Exp 2) and best sex AUC. Bandpass focuses on delta/theta where ISC is strongest.
- **Narrow predictor bottleneck (24/64=37.5%) is critical.** Run B proves that relaxing it destroys stimulus encoding.
- **Baseline with 5-seed averaging is still competitive.** Single-seed variance is high (~±0.05).

---

## Iterative Hyperparameter Search (10 iterations, single-seed)

Base: Exp 6 baseline. All use pure JEPA loss (smooth_l1 + VCLoss). Only one variable changed per iteration.

| Iter | Change | Pos corr | Lum corr | Con corr | Nar AUC | Age bal | Sex AUC |
|------|--------|----------|----------|----------|---------|---------|---------|
| **Base** | 5-seed mean | **0.176** | **0.168** | **0.115** | 0.527 | **0.637** | **0.618** |
| 1 | Temporal masking | 0.209 | 0.142 | 0.057 | **0.555** | 0.598 | 0.609 |
| 2 | 8 windows | 0.119 | 0.129 | 0.084 | 0.536 | 0.580 | 0.615 |
| 3 | Overlap 20→5 | 0.191 | 0.122 | 0.057 | 0.555 | 0.604 | 0.610 |
| 4 | VCLoss std=10 cov=1 | 0.061 | 0.061 | -0.008 | 0.523 | 0.617 | 0.622 |
| 5 | EMA 0.99→0.999 | 0.195 | 0.136 | 0.065 | 0.553 | 0.607 | 0.609 |
| 6 | batch_size=256 | 0.128 | 0.063 | 0.031 | 0.543 | 0.643 | 0.617 |
| 7 | Amp augmentation | 0.060 | 0.036 | 0.049 | 0.535 | 0.541 | 0.617 |
| 8 | 80ep no early stop | 0.189 | 0.136 | 0.062 | 0.550 | 0.597 | 0.615 |
| 9 | lr=1e-4 | 0.125 | 0.083 | 0.018 | 0.535 | 0.624 | 0.621 |
| **10** | **10comp + EMA 0.99** | **0.213** | 0.151 | 0.067 | **0.579** | 0.589 | 0.605 |

### Conclusions
1. **Baseline is remarkably robust.** No single-variable change consistently beat it across all metrics.
2. **Input SNR is the bottleneck**, not model/training hyperparameters. Changes that perturb training dynamics (stronger VCLoss, larger batch, augmentation) destroy the fragile -24 dB stimulus signal.
3. **Position vs luminance/contrast are different timescales.** Position (slow drift) is easy to improve; luminance/contrast (fast local features) are very sensitive.
4. **Iter 10 (10 comp + slow EMA)** is best single-seed challenger: position 0.213, narrative AUC 0.579.
5. **Multi-seed validation is essential.** Single-seed variance (~±0.05) exceeds most improvement margins.

---

## Exp 7: CLIP-Style Cross-Modal Alignment (sample-pool, coeff 0.5)
**Branch:** `kkokate/exp7-clip-align` | **Training Job:** 17798390 | **Eval Job:** 17798706 | Early stopped ep ~48 (12m42s)
**Setup:** Exp 6 base + auxiliary symmetric InfoNCE between EEG→MLP projection (64→256→512) and frozen CLIP ViT-L/14 frame embeddings (4877×512, time-aligned to each window). JEPA loss unchanged.
  - coeff 0.5, learnable temperature init 0.1, sample-pool: [B, D] vs [B, D_clip] → 64 negatives
**Training:** val/reg_loss best 0.8130. clip_align_acc stuck at 1–5% vs 1.6% chance (3× ratio).
**Result (test, seed 2025):** Roughly neutral vs Exp 6. Position corr 0.176 (Δ-0.015), luminance corr 0.115 (Δ-0.011), contrast corr 0.040 (Δ-0.021), narrative corr -0.010 (Δ-0.019). AUCs marginally up on 3/4 (position 0.572 flat, narrative 0.559 +0.006, luminance 0.537 +0.008, contrast 0.506 +0.009). **Verdict:** auxiliary signal too weak to break through — 64 negatives and coeff 0.5 gave barely-above-chance alignment, acting as noise rather than useful gradient.

## Exp 7b: Stronger CLIP Alignment (per-window, coeff 1.5) — NEW BEST
**Branch:** `kkokate/exp7-clip-align` | **Training Job:** 17800015 | **Eval Job:** 17800131 | Early stopped ep ~35 (15m11s)
**Setup:** Identical to Exp 7 except:
  - **per_window=true** — pool encoder tokens to [B, T, D] and flatten to [B·T, D] for InfoNCE → **256 negatives** instead of 64
  - **coeff 1.5** (3× stronger) — aux weight ≈ JEPA pred_loss magnitude
**Training:** val/reg_loss best **0.8043** (best across all experiments). clip_align_acc ~2.5% vs 0.4% chance (**6× ratio**, 2× better than Exp 7 in relative terms).

**Stimulus probe results (test, seed 2025):**

| Feature | reg_corr | Δ vs Exp 6 | cls_AUC | Δ vs Exp 6 |
|---------|---------:|-----------:|--------:|-----------:|
| position_in_movie | **0.251** | +0.060 (+31%) | **0.599** | +0.027 |
| luminance_mean | **0.175** | +0.049 (+39%) | **0.563** | +0.034 |
| contrast_rms | 0.067 | +0.006 | **0.518** | +0.021 |
| narrative_event | 0.008 | −0.001 | 0.537 | −0.016 |

**Subject-trait probes (test, seed 2025):** encoder became *more* subject-informative (side effect of aux loss enriching the representation broadly).

| Trait | Metric | Exp 6 | Exp 7b | Δ |
|-------|--------|------:|-------:|---:|
| age | reg_corr | 0.303 | **0.471** | +0.168 |
| age | cls_AUC | 0.655 | **0.708** | +0.053 |
| age | cls_bal_acc | 0.597 | **0.635** | +0.038 |
| sex | cls_AUC | 0.611 | **0.661** | +0.050 |

**Verdict:** Stronger alignment (per-window 256 negatives + 3× coeff) is a clear win — 6/8 stimulus metrics improve with largest gains on position and luminance. JEPA loss is untouched (`total = jepa_loss + 1.5·clip_align_loss`); only additions are the projection head + InfoNCE term. Trade-off: subject identity info also increases (age corr 0.30→0.47), suggesting the CLIP targets correlate with age-correlated visual attention patterns — not necessarily bad for downstream decoding, but worth flagging for a "subject-invariant foundation model" framing.

---

## Exp 7b Multi-Seed Validation (5 seeds: 2025, 42, 123, 456, 789)

**Training jobs:** 17800015 (s2025), 17800724 (s42), 17800725 (s123), 17800728 (s456), 17800735 (s789) — staggered 30s to avoid git race, all COMPLETED.
**Eval jobs:** 17800131 (s2025), 17800975 (s42), 17801039 (s123, resubmit), 17800984 (s456), 17800993 (s789).

### Stimulus probes (mean ± population std, test set)

| Metric | Exp 6 (5s) | Exp 7b (5s) | Δ | Δ / Exp 6 σ | Verdict |
|---|---:|---:|---:|---:|---|
| position corr | 0.176 ± 0.048 | **0.236 ± 0.013** | +0.060 | **+1.25σ** | ✅ WIN |
| luminance corr | 0.168 ± 0.059 | 0.174 ± 0.011 | +0.006 | +0.10σ | ≈ neutral |
| contrast corr | **0.115 ± 0.053** | 0.065 ± 0.004 | −0.050 | −0.94σ | ❌ regress |
| narrative corr | −0.003 ± 0.042 | −0.022 ± 0.029 | −0.019 | −0.45σ | ≈ zero both |
| position AUC | 0.580 ± 0.025 | **0.602 ± 0.014** | +0.022 | +0.88σ | ≈ win |
| luminance AUC | **0.567 ± 0.021** | 0.550 ± 0.013 | −0.017 | −0.82σ | ≈ regress |
| contrast AUC | **0.553 ± 0.032** | 0.507 ± 0.012 | −0.046 | **−1.44σ** | ❌ regress |
| narrative AUC | 0.528 ± 0.025 | 0.534 ± 0.003 | +0.006 | +0.24σ | ≈ neutral |

### Subject-trait probes (mean ± population std, test set)

| Metric | Exp 6 (5s) | Exp 7b (5s) | Δ | Δ / Exp 6 σ | Note |
|---|---:|---:|---:|---:|---|
| age reg_corr | 0.325 ± 0.030 | **0.421 ± 0.064** | +0.096 | **+3.2σ** | Exp 7b encodes more age info |
| age cls_AUC | 0.667 ± 0.013 | **0.678 ± 0.023** | +0.011 | +0.85σ | ranking up |
| age cls_bal_acc | **0.638 ± 0.024** | 0.609 ± 0.019 | −0.029 | −1.21σ | threshold calibration drops |
| sex cls_AUC | 0.618 ± 0.007 | **0.649 ± 0.020** | +0.031 | **+4.4σ** | clear ranking win |
| sex cls_bal_acc | **0.603 ± 0.015** | 0.532 ± 0.046 | −0.071 | −4.7σ | probe collapse, not feature loss |

### Key observations

- **Position is the only clear win.** Both corr (+1.25σ) and AUC (+0.88σ) improve. Position-corr std shrinks 4× (0.048 → 0.013), so the gain is highly reproducible.
- **Contrast regresses on both metrics.** corr −0.94σ and AUC −1.44σ. CLIP is a single-frame semantic target; it under-weights fast local luminance-contrast structure.
- **Luminance and narrative are essentially unchanged.** Earlier single-seed (2025) comparisons suggested luminance wins of +0.049 corr / +0.034 AUC, but those were artifacts of Exp 6 seed 2025 being a low-AUC outlier (lum_auc 0.529 vs 5-seed mean 0.567). The 5-seed comparison is within noise.
- **Strict 3-of-4 go rule fails** — only position clears +1σ, and contrast regresses >1σ on AUC.
- **Subject-trait AUCs rise by 1–4σ while bal_acc drops**: on sex, 3/5 seeds give bal_acc = exactly 0.500 (probe collapses to constant class) despite AUC 0.649 — threshold-calibration artifact, not feature loss. AUC is the informative metric here.
- **Net effect: Exp 7b is *not* uniformly richer** — it trades fast local feature sensitivity (contrast) for slow temporal drift (position), while slightly increasing subject-trait encoding. Good for paper framings that emphasize temporal/narrative decoding; potentially worse for low-level feature reconstruction.

### Next steps (priority order)

1. **Swap CLIP → V-JEPA 2** target (Proposal A): video-native, should rescue narrative and contrast. Embeddings already on Delta.
2. **SigLIP loss** (Proposal B): published lift over InfoNCE at B<4096.
3. **Hard-negative mining via temporal proximity** (Proposal C): specifically targets narrative/contrast weakness.

---

## Missing Baselines
1. Random encoder (untrained) + same probes
2. Permutation test (shuffled labels)
3. Handcrafted features (band power) + same probes
4. Multiple seeds for training (not just eval)
