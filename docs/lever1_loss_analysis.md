# Lever 1 — Loss Curve Analysis (2026-05-02)

Source: `logs/lever1_18011312..18011320.out` (5 enc seeds × 100 epochs, λ_stim=0.5,
τ=0.1, position bucket=4 s). Compared against the Phase-D baseline
`logs/issue8D_17927885..17927889.out` on the same 5 enc seeds, same nw4_ws2
config, no auxiliary loss.

## TL;DR

**The cross-subject InfoNCE auxiliary loss is doing nothing during pretraining.**
Across all 5 enc seeds, `stim_nce_loss` is essentially constant at **4.096 = log(B=64)**
from epoch 0 to epoch 99 — the chance value when embeddings carry no
cross-subject stim-aligned structure. The encoder isn't learning the InfoNCE
objective.

The kc + Ridge probe-eval headline (Lever 1 narrative +0.105 vs Phase-D +0.090)
is **not attributable to the new loss working**; it has to come from probe-seed
luck or some other side effect.

## Per-seed evidence

### Pre-training loss decomposition

| Seed | pred[0] → pred[99] | stim_nce[0] → stim_nce[99] |
|---:|---:|---:|
| s42   | 1.353 → 1.299 | 4.096 → 4.095 |
| s123  | 1.345 → 1.612 | 4.099 → 4.095 |
| s456  | 1.369 → 1.171 | 4.096 → 4.095 |
| s789  | 1.161 → 1.154 | 4.096 → 4.095 |
| s2025 | 1.104 → 1.379 | 4.096 → 4.095 |

Stim-NCE values were derived from the loss decomposition
`total = pred + vc + 0.5 × stim_nce`. The Δ over 100 epochs is < 0.005 — well
within noise from end-of-epoch batch variance. log(64) = 4.158, so this is
within ~0.06 of pure chance.

### Val stim corrs — peak values

| Metric | Phase-D peak (mean of 5) | Lever 1 peak (mean of 5) | Δ |
|---|---:|---:|---:|
| val/reg_position_in_movie_corr   | +0.260 | **+0.193** | **−0.067** |
| val/reg_luminance_mean_corr      | +0.218 | **+0.176** | **−0.042** |
| val/reg_narrative_event_score_corr | +0.092 | +0.094 | ~0 |
| val/reg_contrast_rms_corr (mean) | +0.130 | (similar) | ~0 |

Lever 1's online val stim corrs are *lower* than baseline on position and luminance
during training. Narrative is unchanged. **The auxiliary loss is not helping val
stim corrs during training**, contradicting what we'd expect if it were doing
its job. This is consistent with `stim_nce_loss` being flat at chance.

### Peak epochs

| Seed | Phase-D pos peak (ep) | Lever 1 pos peak (ep) |
|---:|---:|---:|
| s42  | 72 | 72 |
| s123 | 51 | 66 |
| s456 | 32 | 32 |
| s789 | 91 | 91 |
| s2025| 94 | 92 |

Peak-then-decline pattern persists in Lever 1 — the loss-design pathology from
Phase 1 is unchanged.

## Why is the InfoNCE loss flat at chance?

Three candidate root causes, ordered by likelihood:

### 1. Random batches don't carry enough same-stim cross-subject pairs *(most likely)*

The current implementation uses standard random batches of 64. With ~5 movies
× ~100 stim-time buckets each = ~500 buckets, the expected number of
same-stim-different-subject pairs per anchor is ≈ 64 / 500 ≈ 0.13. The loss
spec excludes anchors with no positive in batch, so most batches contribute
zero gradient. The InfoNCE on top of an already-noisy random sample of 5 movies
has no consistent signal to learn from.

**Fix:** structured batch sampler that places K clips per stim-bucket from K
distinct recordings into each batch. Even K=4 across 16 buckets per batch
gives 3 guaranteed positives per anchor and full dense gradient. Was discussed
in the design document but skipped in v1 for simplicity.

### 2. Mean-pool over all C·T·P tokens dilutes channel-specific stim signal

The InfoNCE is computed on `tokens.mean(dim=1)` — a global mean over all
C×T×P tokens. Phase 1 finding 4 showed stim signal is concentrated in a
specific encoder channel (ch1 for pos/lum/contrast, ch4 for narrative). The
global mean dilutes these. Cross-subject InfoNCE may not have enough discriminative
signal at the global-pool level.

**Fix:** apply InfoNCE on the kc-pooled embedding `[B, C·D]` instead of mean-pooled
`[B, D]`, mirroring the production probe protocol.

### 3. λ=0.5 is too small relative to pred + vc gradients

The total loss is dominated by pred (~1.2) + vc (~0.01); stim contributes
~2.05 (= 0.5 × 4.1). But the *gradients* matter, and a flat loss = zero
gradient regardless of magnitude. So this can't be the only cause, but combined
with #1 (sparse positives) it ensures any weak gradient signal gets washed out.

**Fix:** λ=2.0 or 5.0 *paired with* the structured batch sampler.

## What the kc+Ridge result actually shows

| Metric (kc + Ridge, test) | Phase-D (5 enc × 5 probe = 25) | Lever 1 (5 enc × 1 probe = 5) | Comparable? |
|---|---:|---:|---|
| narrative | +0.0900 ± 0.011 | +0.105 ± 0.009 | **Not directly** — Phase D averages 5 probe seeds; Lever 1 doesn't |
| position  | +0.1435 ± 0.017 | +0.182 ± 0.015 | Same caveat |
| luminance | +0.2076 ± 0.007 | +0.222 ± 0.006 | Same caveat |
| contrast  | +0.1585 ± 0.009 | +0.176 ± 0.022 | Same caveat |

Given (a) flat InfoNCE during training, (b) lower val stim corrs during training,
and (c) the probe-protocol asymmetry, the apparent +0.015 narrative lift is
almost certainly probe-seed noise + sample-size mismatch, not InfoNCE
helping. The 3 best-checkpoint Ridge probes queued (jobs `18011786..88`)
won't change the diagnosis on the loss-side question — InfoNCE isn't training.

## Recommended Phase-2 revision

**v1 of Lever 1 (random batch, global pool, λ=0.5) is null.** Three concrete
follow-ups, ordered by cost:

1. **(cheap)** Switch InfoNCE pool from global mean to kc-style channel-concat;
   keep random batches; λ=2.0. ~5 GPU-h.
2. **(medium)** Add `StimAlignedBatchSampler` that ensures K=4 clips per stim
   bucket × 16 buckets per batch; keep λ=0.5; global pool.  ~5 GPU-h + small
   sampler implementation.
3. **(combined)** #1 + #2 together. The right design from the SSL research
   memo's recommendation, but the proper test of whether Lever 1 *can* lift
   stim signal vs whether the implementation is the bug.

A null result from any one of these would be more interpretable than v1's
"loss didn't move" — at least we'd know the encoder *can* be pushed by an
InfoNCE objective and it just wasn't enough to lift stim probes.

Cross-references:
- `docs/phase1_loss_analysis.md` — Phase D loss baseline (peak-then-decline pattern).
- `docs/ssl_research_2026-05-02.md` §1.2 — InfoNCE recommendation.
- `docs/phase2_plan.md` — original Phase 2 grid.
