# Lever 1 — Bootstrap & Paired Significance Analysis (2026-05-02)

Anchor: 5 Lever 1 enc seeds (s42/s123/s456/s789/s2025) × kc + Ridge probe with
n_passes=20 random-clip averaging. Best checkpoint per seed selected by
mean online val-stim-corr peak (epoch 60 / latest / 40 / 90 / latest for the 5
seeds respectively — see `docs/lever1_loss_analysis.md`).

Phase-D production baseline: same 5 enc seeds, same Ridge protocol, latest
checkpoint, taken from
`/projects/bbnv/kkokate/eb_jepa_eeg/tier1/jepa_ridge_keep_channels_enc{seed}.json`.

## View 3 — recording-level bootstrap (B=2000), 5-seed t-test against chance

| Metric | Lever 1 best-ckpt V3 mean ± σ | p (vs chance) |
|---|---:|---:|
| reg_position_in_movie_corr     | +0.1835 ± 0.0166 | **1.6e-5** ✓ |
| reg_luminance_mean_corr        | +0.2244 ± 0.0075 | **3.0e-7** ✓ |
| reg_contrast_rms_corr          | +0.1781 ± 0.0227 | **6.2e-5** ✓ |
| reg_narrative_event_score_corr | +0.1065 ± 0.0092 | **1.3e-5** ✓ |
| cls_position_in_movie_auc      | +0.5949 ± 0.0121 | 6.2e-5 ✓ |
| cls_narrative_event_score_auc  | +0.5645 ± 0.0138 | 4.8e-4 ✓ |
| cls_luminance_mean_auc         | +0.5585 ± 0.0110 | 2.9e-4 ✓ |
| cls_contrast_rms_auc           | +0.5567 ± 0.0069 | 5.2e-5 ✓ |

All 8 stim metrics significantly above chance.

## Paired comparison vs Phase-D production

Same protocol on both sides (5 enc seeds × n_passes=20 × kc + Ridge).
Paired t-test computes the per-seed difference Lever 1 − Phase D, then
1-sample t-test against zero on the 5 differences.

| Metric | Phase-D mean | Lever 1 mean | Δ (mean) | per-seed Δ | t | p |
|---|---:|---:|---:|---|---:|---:|
| narrative | +0.0900 | +0.1066 | +0.0166 | [-0.000, +0.020, +0.040, +0.002, +0.021] | 2.28 | 0.084 |
| **position** | +0.1435 | +0.1830 | **+0.0395** | [+0.026, +0.037, +0.075, +0.032, +0.028] | 4.38 | **0.012** ✓ |
| **luminance** | +0.2076 | +0.2246 | **+0.0170** | [+0.015, +0.011, +0.027, +0.016, +0.016] | 6.24 | **0.003** ✓ |
| contrast | +0.1586 | +0.1777 | +0.0192 | [+0.004, +0.056, +0.006, +0.039, -0.010] | 1.56 | 0.193 |

**Position** and **luminance** are significantly improved at p < 0.05.
Narrative and contrast trend in the same direction but don't clear
significance at n=5.

Effect sizes vs the +0.213 cross-subject narrative ceiling:
- Phase D narrative was 42% of the ceiling (+0.090 / +0.213).
- Lever 1 best-ckpt narrative is 50% of the ceiling (+0.107 / +0.213).
- Position lift (+0.039) is meaningful in absolute terms — pos peak was
  +0.144, now +0.183, a 27% relative increase.

## Important caveat — InfoNCE was at chance during training

Per `docs/lever1_loss_analysis.md`, `stim_nce_loss` stayed at log(B=64) ≈ 4.10
across all 5 seeds for all 100 epochs. The auxiliary objective was not
learned. **The stim-probe lift cannot be attributed to InfoNCE working.**
Two non-loss-design candidates explain it:

1. **Best-checkpoint selection.** 3 of 5 seeds used pre-latest checkpoints
   (ep60, ep40, ep90). Combined with the Phase-1 finding that val stim
   peaks before epoch 100 and declines, this alone could explain part of
   the lift.
2. **Implicit regularization.** Carrying an extra loss term in the optimizer
   adds gradient-direction noise; even when the term itself is at chance,
   the noise can act as effective regularization that improves
   generalization. The pos and lum lifts are most consistent with a "less
   overfit to identity-stable features" interpretation.

A clean test of (1) vs (2): probe the original Phase-D 5 enc seeds at their
best-by-val-stim epoch with kc + Ridge, n_passes=20. If Phase-D best-ckpt
matches Lever 1 best-ckpt within paired noise, we're done — Lever 1 v1's
improvement is purely checkpoint selection. If a gap remains, the implicit
regularization is real (and can be replicated more cleanly with explicit
weight decay or dropout).

## What this means for Phase 2

Even with v1 of Lever 1 not training the InfoNCE, the kc + Ridge production
probe lifted significantly on pos and lum. **Lever 1 v2 (structured batch
sampler + kc-pool InfoNCE + λ=2.0)** should test whether actually training
the auxiliary loss adds further lift on top of the v1 result, particularly
on narrative where v1 only trended (+0.017, p=0.084).

## Files

- `predictions/lever1/lever1_nw4ws2_s*_seed*` — per-recording prediction
  npz files (one per enc seed × {best, latest} ckpt).
- `predictions/lever1_best/lever1_best_seed*` — symlinks pointing at the
  best-ckpt-per-seed prediction dirs (used by the bootstrap script).
- `results/lever1/lever1_nw4ws2_s*.json` — flat metric JSONs.
- `results/lever1_best/lever1_best_seed*.json` — symlinks to the best-ckpt
  JSONs.
- `docs/lever1_loss_analysis.md` — pre-training loss curves + InfoNCE
  flat-at-chance evidence.
- `scripts/bootstrap_trivial_perseed.py` — bootstrap driver
  (cherry-picked from `kkokate/trivial-stats-baselines`).
