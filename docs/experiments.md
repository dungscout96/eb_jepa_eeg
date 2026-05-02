# EEG-JEPA Experiment Log

Concise tracking of experiments on the V-JEPA-style masked EEG encoder for HBN movie-watching data. Older per-run notes from early exploratory sweeps live in `observations.md` (separate numbering).

Goal: learn a stimulus-driven EEG encoder. Metric: linear probe on movie features (contrast, luminance, position, narrative) + subject traits (age, sex) from a frozen encoder.

## Summary table

| Exp | Change from baseline | Seeds | Verdict |
|---|---|---|---|
| 1 | Narrow predictor bottleneck (pred_dim 64→24) | 1 | Fixed collapse. Kept. |
| 2 | Per-recording z-norm + optional delta envelope | 1 | Removes subject amplitude fingerprint. Kept. |
| 6 | CorrCA-5 spatial filter + per-rec norm + Exp 1 + VCLoss(0.25/0.25) + smooth_L1 | 5 | **Current baseline.** |
| 7 | + CLIP-ViT-L/14 per-frame alignment (InfoNCE, B=64 negatives, coeff=0.5) | 1 | Neutral. |
| 7b | + CLIP alignment, per-window (256 negatives), coeff=1.5 | 5 | Mixed: pos ↑ (+1.25σ), cont ↓ (−1.44σ). |
| 7c | + V-JEPA 2 video target swap (1408-d @ 2Hz) | 1 | Regression. |
| 7d | + SigLIP pairwise-sigmoid alignment | 1 | Neutral / slight regression. |
| 8 | Cross-subject paired JEPA, p=0.5 (target sees another subject's EEG) | 1 | Broken — all probes at chance. |
| 8b | Same, p=0.15 | 1 | Still broken. |
| 8 diag | TimeAlignedBatchSampler on, cross-subject permutation off (p=0) | 1 | Matches Exp 6 → sampler isn't the bug; permutation is. |
| 9 | MAE with raw CorrCA patch target (drop EMA target encoder) | 1 | Regression: pos −2.6σ vs Exp 6. |

## Baseline: Exp 6 (5-seed mean ± population std)

Test metrics, seeds {42, 123, 456, 789, 2025}:

| Probe | Value |
|---|---:|
| position corr | 0.176 ± 0.048 |
| luminance corr | 0.168 ± 0.059 |
| contrast corr | 0.115 ± 0.053 |
| narrative corr | −0.003 ± 0.042 |
| position AUC | 0.580 ± 0.025 |
| luminance AUC | 0.567 ± 0.021 |
| contrast AUC | 0.553 ± 0.032 |
| narrative AUC | 0.528 ± 0.025 |
| age reg_corr | 0.325 ± 0.030 |
| sex AUC | 0.618 ± 0.007 |

Per-seed σ ≈ 0.04–0.06 on stimulus corrs. Detection threshold for a sweep with 3 seeds per cell: Δ ≳ 0.07.

---

## Per-experiment detail (concise)

### Exp 1 — Predictor bottleneck (2026-04-13)
- pred_dim 64 → 24. Fixed representational collapse (cosim 0.97 → 0.25, participation ratio 3.4 → 19.1).
- Subject probes detectable (age AUC 0.64 val); stimulus probes still at chance.

### Exp 2 — Per-recording normalization (2026-04-13)
- Z-norm per recording removes subject amplitude/offset. Optional 1-8 Hz envelope channel.
- Rationale: η²_subj drops to 0 on input after per-rec z-norm (validated later in raw-data analysis).

### Exp 6 — CorrCA spatial filter baseline (merged PR #9, 2026-04-20)
- Stack: CorrCA-5 input (129 → 5 ISC-maximizing channels) + per-rec norm + Exp 1 bottleneck + VCLoss(0.25, 0.25) + smooth_L1.
- 5-seed numbers above.
- Still exhibits subject dominance in embeddings (see Diagnostics below) — but no longer collapsed; all stimulus probes significantly above chance.

### Exp 7 — CLIP InfoNCE alignment (2026-04-21, single seed)
- Add auxiliary EEG→CLIP InfoNCE (sample-pool negatives, B=64) with coeff=0.5.
- Result: no meaningful change vs Exp 6.

### Exp 7b — Stronger CLIP alignment (PR #10, 5-seed)
- Per-window pooling → 256 negatives, coeff=1.5.
- Position: **0.236 ± 0.013** (+1.25σ, σ 4× tighter). Contrast: **−1.44σ regression**. Luminance/narrative within noise. Age ranking ↑.
- Net: trades fast local feature sensitivity for slow temporal drift + subject-trait ranking.

### Exp 7c — V-JEPA 2 target swap (single seed)
- Replace CLIP target with V-JEPA 2 video embedding (1408-d @ 2Hz), InfoNCE.
- Regressed vs Exp 6 across probes.

### Exp 7d — SigLIP pairwise sigmoid (single seed)
- Replace InfoNCE with SigLIP sigmoid loss.
- Neutral / slight regression.

### Exp 8 — Cross-subject paired JEPA (2026-04-22)
- TimeAlignedBatchSampler + target encoder sees a batch-permuted EEG (different subject at same movie time).
- **Broken at both p=0.5 and p=0.15**: all stimulus probe corrs at chance (0.03–0.07).

### Exp 8 diagnostic — sampler-only (2026-04-23)
- p=0.0, TimeAlignedBatchSampler on (no cross-subject permutation).
- Matches or exceeds Exp 6 on every stimulus probe → **sampler isn't the bug**. The cross-subject target permutation is.

### Exp 9 — MAE with raw CorrCA patch target (2026-04-23, single seed)
- Drop EMA target encoder. Target = raw CorrCA patch values at masked positions. Linear decoder head projects predictor output (D=64) → patch_size=50.
- Result: pos 0.051 (vs 0.176 ± 0.048 baseline, −2.6σ). Multi-sigma regression on every stimulus probe. Sanity cosim_mean = 0.95 (near-collapse).
- Root cause: raw CorrCA time-series per single subject is ~93% within-subject noise. MAE on this target learns subject-specific reconstruction, not stimulus. A stimulus-locked target needs **group-mean across subjects** at matched time — not any single subject.

---

## Diagnostic analyses

### Embedding-space dissection of Exp 6 (2026-04-23)
- PC1 explains **70.6%** of embedding variance, between-rec fraction 0.73, sex AUC 0.76, zero stimulus signal → PC1 is nearly pure subject identity.
- PC2: 10% var, mixed (some stimulus: pos R² 0.055, lum R² 0.080).
- PCs 3–63: small, stimulus distributed across them (cumulative top-20 pos R² = 0.062).
- Effective dim ≈ 10 — 64-dim embedding is low-rank.
- Conclusion: the encoder *re-introduces* subject identity that was zero at the per-rec z-normed input. Projecting out PC1 won't help because even the full embedding sits below the linear CorrCA stimulus ceiling.
- Script: `scripts/embedding_dissection.py`.

### Raw-EEG data ceiling (2026-04-23)
- Mean ISC on raw 129-ch per-rec z-normed EEG: **0.0157** broadband; top-5 channels 0.034. Stimulus lives in **delta (1-4 Hz)**, mean ISC 0.011; alpha/beta/gamma at noise.
- CorrCA train eigenvalues (top-5 ISC components): **[0.019, 0.012, 0.006, 0.004, 0.002]**. On val: [0.058, 0.044, 0.020, 0.013, 0.010].
- Variance decomposition on raw: η²_subj = 0.000 (zeroed by z-norm), η²_stim = 0.026.
- Best 1-dim linear stimulus representation (CorrCA comp 1): **η²_stim = 0.074**.
- Exp 6 encoder distributes stimulus across top-20 PCs to reach pos R² 0.062 — below CorrCA comp 1 alone, despite richer architecture.
- Script: `scripts/raw_data_analysis.py`.

### Variance decomposition on earlier SIGReg/VICReg checkpoints (pre-Exp 6 exploration)
- See `docs/variance_decomposition_report.md`. K=32 clips/rec. η²_subj dominates (0.06–0.37 depending on config); η²_stim near null (0.004–0.008).

---

## Current open question

Exp 6 was not hyperparameter-tuned — coefficients (VCLoss 0.25/0.25) and predictor bottleneck (24) were chosen ad-hoc. Next: multi-seed grid sweep over VCLoss std=cov coeffs × predictor bottleneck dim, keeping everything else at Exp 6 defaults, to find the true Exp 6 ceiling before comparing against further architectural changes.

## Literature review

Subject-invariance in self-supervised EEG, group-consensus targets (SRM / CorrCA / hyperalignment), JEPA variants for neural data, MI-bottleneck methods. See `docs/literature_subject_invariance.md`.

Relevant theoretical result: Apple ML 2024 proves JEPA's implicit noise-suppression bias **fails when noise is low-rank and high-variance** — exactly subject identity. Matches our PC1=70% failure.
