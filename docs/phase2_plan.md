# Phase 2 Plan — Synthesis (2026-05-02)

This memo combines three threads:
- `phase1_findings.md` — encoder-tap diagnostics on `phaseD_nw4ws2_baseline_s42`
- `phase1_loss_analysis.md` — pre-training loss + val-stim trajectories on all 5 Phase-D enc seeds
- `ssl_research_2026-05-02.md` — 16-paper literature scan on SSL stim/identity disentanglement

The three threads point at the same root cause and the same intervention.

## Convergent diagnosis

| Thread | Observation |
|---|---|
| Phase 1 finding 1 | Block 1 of the 2-block encoder is dead weight for stim (block0 ≈ final on every regression). |
| Phase 1 finding 5 | Encoder learns subject traits (age +0.50, sex +0.71) but not new stim info beyond patch_embed. |
| Loss analysis | Train pred_loss never decreases over 100 epochs (bounces 1.18 → 1.39); val stim peaks at epoch 30–80 then declines. |
| V-JEPA 2.1 (arXiv:2603.14482) | Names this exact failure mode: "shallow encoders concentrate semantic features in early layers and use the rest for prediction-specific bookkeeping." |
| Cheng 2020 (arXiv:2007.04871) | Subject identity is a free shortcut for any biosignal SSL loss; needs explicit removal. |
| EEG SSL field (11 papers) | No paper has addressed stim-vs-subject trade-off via loss design. We're in novel territory. |

**Root cause:** the masked-prediction loss alone doesn't distinguish stim-encoding solutions from identity-encoding solutions. Identity is a strictly easier path to low loss (within-recording, position-invariant, statistically stable). The encoder finds it, then drifts toward it as training proceeds. Stim signal acquired incidentally early in training gets unlearned.

**Two complementary fixes are required:**
1. *Remove the identity shortcut* (negative gradient against identity).
2. *Add a positive gradient toward stim* (the encoder must benefit from encoding stim).

Either fix alone may be insufficient: removing identity without rewarding stim leaves the encoder with no objective; rewarding stim without removing identity leaves a strong competing attractor.

## Phase 2 grid — ranked by expected information per cell

Each cell = full pretraining (5 enc seeds × 100 epochs) + production probe-eval
(kc + MLP, 5 probe seeds, save best checkpoint by val stim corr).

### Tier A — must run (low cost, decisive)

| # | Intervention | Lever | Refs |
|---|---|---|---|
| **A1** | **Cross-subject stim-aligned InfoNCE auxiliary loss** at coefficient λ=0.5, τ=0.1. Builds positive pairs from same stimulus-time across subjects; negatives from different stim-times. | + grad toward stim | CPC arXiv:1807.03748; Cheng arXiv:2007.04871 |
| A2 | Subject-adversarial GRL head, λ_adv=0.1. 2-layer MLP predicts subject id; encoder gets reversed gradient. | − grad against identity | Cheng arXiv:2007.04871 |
| A3 | Per-clip normalization (vs current per-recording z-norm). Encoder loses access to slow recording-level statistics. | − identity shortcut at data level | loss memo + finding 5 |
| A4 | Save best checkpoint by val stim corr — no retrain, just eval-time. | rescue val-corr peak | loss memo |

A1 is the single highest-information cell. It's the only intervention from the literature that creates a *positive* gradient toward stim rather than just removing the identity shortcut.

A4 is free: it just rescue-probes existing checkpoints. **12 verification jobs already queued** (epoch 40/60/80/latest × 3 probe seeds × kc) to quantify the lift.

### Tier B — recommended after Tier A (mechanistic refinements)

| # | Intervention | Lever | Refs |
|---|---|---|---|
| B1 | DSS (deep self-supervision): apply masked-prediction loss at every encoder block, with a small fusion MLP. Pair with **depth 2 → 4**. | exposes deeper layers to direct loss | V-JEPA 2.1 arXiv:2603.14482 |
| B2 | Replace VCLoss with SIGReg / LeJEPA isotropic-Gaussian regularizer (already in `eb_jepa/losses.py`). | better isotropy, no schedule, single hyperparam | LeJEPA arXiv:2511.08544 |
| B3 | EEG2Rep SSP masking (semantic-similarity-driven mask placement). | stops the loss from being trivially solved | EEG2Rep arXiv:2402.17772 |

DSS specifically tests our Phase 1 finding 1. If "block 1 dead weight" is a loss-design artifact (loss only reaches the top), DSS should make block 1 carry signal. If it stays dead even under DSS, the encoder genuinely doesn't need that depth.

### Tier C — combined-fix stress test

| # | Intervention |
|---|---|
| C1 | A1 + A2 + B1 (cross-subject InfoNCE + GRL + DSS, depth=4). |

Run only after we have A-tier numbers. If the individual fixes each lift narrative by say ≤ 0.03, C1 confirms whether they compose to close the +0.090 → +0.213 gap.

### Dropped from prior Phase 2 plan

- Multi-scale patches: lower priority now. Phase 1 finding showed patch_embed already carries the stim signal that survives the encoder; the issue is the encoder *unlearning* it, not the input projection missing it.
- Per-channel attention inside the encoder: deferred. Channel specialization (Phase 1 finding 4) is preserved by the encoder; not a known leak.
- Predictor bottleneck width sweeps: ruled out by Phase 1 finding 3.
- Teacher EMA momentum: ruled out by Phase 1 finding 2.

## Compute budget

| Tier | Cells | Pretraining cost | Probe-eval cost | Total |
|---|---|---|---|---|
| A | 4 (A4 is eval-only) | 3 × 5 enc seeds × ~7h = 105 GPU-h | 3 × 5 × 5 = 75 jobs × 0.5h = 37 GPU-h | ~140 GPU-h |
| B | 3 | 3 × 5 × 7h = 105 GPU-h | 75 × 0.5h = 37 GPU-h | ~140 GPU-h |
| C | 1 | 5 × 7h = 35 GPU-h | 25 × 0.5h = 12 GPU-h | ~47 GPU-h |

**Total Phase 2: ~330 GPU-h** (~3 days on the partition's typical capacity).

## Sequencing

1. **This week** — A4 rescue probes (12 jobs queued), confirm checkpoint-selection lift.
2. **Week 1 of Phase 2** — A1 implementation + 5-seed pretrain + probe-eval.
3. **Week 1 in parallel** — A2 implementation (separate branch).
4. **Week 2** — A3 (per-clip norm sweep), B1 (DSS+depth=4), B2 (SIGReg swap).
5. **Week 3** — C1 combined run if individual cells show signal.

## Open questions to settle empirically

- Does A1 (cross-subject InfoNCE) lift narrative beyond the per-subject ceiling of +0.063 toward the cross-subject ceiling of +0.213? This is the crucial test — the per-subject ceiling is information-theoretic, the cross-subject ceiling needs the encoder to use cross-subject context.
- Does A2 (GRL) *hurt* age/sex probes? It should — that's the point. We should report age/sex degradation as a negative control: if traits don't drop, the GRL isn't doing its job.
- Does A4 alone close the val-corr peak-then-decline gap? If yes, much of the published Phase D number is checkpoint-selection error.

## What we are NOT doing yet

- Adding more pretraining data (HBN release 12, etc.): the loss analysis says the bottleneck is loss design, not data scale.
- Scaling encoder past depth 4: the literature consensus for our 35-hour data budget is depth 4–8 with DSS; depth scaling without DSS makes the block-1-dead-weight problem worse.
- Switching to LaBraM/CBraMod-style 6–12-layer encoders: these are 6000+ hour pretraining models; for our budget the right answer is a small encoder with the right loss, not a big encoder with a misaligned loss.

## References

Full citations and 16-paper bibliography in `docs/ssl_research_2026-05-02.md`.
