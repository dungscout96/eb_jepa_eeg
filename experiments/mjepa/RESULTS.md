# MJEPA-adapted — results

Runs executed 2026-07-30/31 on Delta `gpuA40x4`. Pre-registration in
[NOTES.md](NOTES.md); implementation in `eb_jepa/mjepa.py`.

Protocol identical to every prior result in this repo: 12-feature
`SCALAR_FEATURES_DEFAULT`, 701 train / 293 val / 108 test recordings,
`clip_probe/probe.py` (val, 5-fold r²) and `probe_traintest.py`
(train→eval Pearson r, `--bootstrap 2000 --seed 42`).

## Headline

| checkpoint | val r² | val Δr² | **test r** | val r |
|---|---:|---:|---:|---:|
| random REVE-shape | 0.0155 | — | 0.0516 | 0.1228 |
| **REVE init, untrained** | 0.0256 | +0.0101 | **0.0938** | — |
| MJEPA λ=0 warm, 300 ep | 0.0442 | +0.0287 | 0.1319 | 0.2035 |
| **MJEPA λ=0 warm, 1000 ep** | **0.0593** | **+0.0437** | **0.1551** | **0.2241** |
| MJEPA λ=0 warm, 3000 ep | 0.0181 | +0.0026 | 0.0801 | 0.1209 |
| MJEPA λ=0 scratch, 400 ep | 0.0169 | +0.0014 | 0.0739 | — |
| MJEPA λ=1 warm, 300 ep | 0.0236 | +0.0081 | 0.0801 | — |
| MJEPA λ=1 scratch, 400 ep | 0.0139 | −0.0016 | 0.0474 | — |
| soft-target CLIP (best from-scratch) | — | — | 0.1517 | — |
| **scene_clip warm-start (record)** | — | +0.041…0.043 | **0.2383** | 0.3206 |

K=1 probe-r ceiling from ρ₁(CorrCA, val) = 0.045 is **0.213**.

## Pre-registered predictions

| # | prediction | verdict |
|---|---|---|
| 1 | λ=0 beats 0.1517, best from-scratch contrastive | **matched, not beaten** — 0.1551 vs 0.1517; margin 0.0034 against CIs an order of magnitude wider |
| 2 | λ=0 + REVE warm-start beats 0.2383 | **falsified** — 0.1551; CI overlap on 5/12 features; holds across 300/1000/3000 ep |
| 3 | λ>0 does not help | **confirmed, strongly** — λ=1 costs ~40% of signal at matched budget; λ=1 scratch lands *below* the random floor |

Prediction 3 is the one that contradicts MJEPA's own headline ablation
(shared encoder without cross-modal degrades; adding it lifts both). It does
not transfer to a frozen-teacher setting, where the "other modality" is
precomputed data rather than a co-trained stream.

## The central result

Contrastive extracts **2.4× more** from the same initialization:

| | lift over REVE init (test r) |
|---|---:|
| MJEPA λ=0, 1000 ep | **+0.061** |
| scene_clip contrastive, 300 ep | **+0.145** |

The hypothesis in NOTES.md was that ThePresent's ~101 distinct anchors cap
contrastive objectives, and that L1 regression to a continuous 1408-d target
escapes that cap. **This is refuted.** Removing the class structure entirely
does not help; it costs 45% of the signal at the best MJEPA operating point.
Whatever limits the contrastive line here, anchor count is not it.

## Schedule is the dominant hyperparameter — and it is not monotone

| schedule | drift from REVE | final `pr` | val Δr² | test r |
|---|---:|---:|---:|---:|
| 300 ep | 0.053 | 2.73 | +0.0287 | 0.1319 |
| **1000 ep** | **0.093** | **6.31** | **+0.0437** | **0.1551** |
| 3000 ep | 0.142 | 2.78 | +0.0026 | 0.0801 |

`drift` = ‖θ − θ_REVE‖ / ‖θ_REVE‖ over the 139 matched encoder tensors.

300 ep underfits; 3000 ep drifts far enough to **destroy** the pretrained
structure, landing *below* the untrained REVE init (0.0801 vs 0.0938). The
optimum is near 1000 ep, at ~0.09 drift.

What matters is **completing the cosine decay, not accumulating steps.** At
matched epoch 999 the 1000-ep run (fully annealed) is at `ev`=0.315 /
`pr`=6.31, while the 3000-ep run (still ~75% of peak LR) is at `ev`=0.718 —
pinned to the predict-the-mean floor — with `pr`=1.67. Annealing is what
consolidates the representation; `pr` sits at 1.2–1.9 for the first 1500
epochs of the 3000-ep run.

## Training metrics do not predict probe quality

**This invalidates the diagnostic protocol proposed in the original NOTES.md.**

| run | `ev` (floor 0.711) | `ev_gap` | `pr` | val Δr² |
|---|---:|---:|---:|---:|
| 1000 ep | 0.315 | +0.706 | 6.31 | **+0.0437** |
| 3000 ep | 0.362 | +0.588 | 2.78 | **+0.0026** |

The 3000-ep run has *excellent* training diagnostics — `ev` far below the
floor, a large positive `ev_gap` — and a representation at the random-noise
floor. All of `ev` / `ev_gap` / `pr` are **train-split** quantities
(`train/ev_loss` etc. in wandb); driving them in the right direction is
compatible with destroying downstream performance.

`pr` is not a usable proxy either: 300 ep (`pr`=2.73) and 3000 ep (`pr`=2.78)
have nearly identical participation ratio and differ 11× in Δr². Separately,
the epoch-63 checkpoint at `pr`≈1.3 already probed at Δr² +0.0202 — variance
concentrated into few directions that were nonetheless movie-relevant. `pr`
measures variance spread, not task content.

**Only the probe is informative. Run arms to completion and probe them.** At
~10 s/epoch a 1000-epoch arm is ~3 h on one A40; there is nothing to save by
gating early.

## Collapse and the warm-start

λ=0 **from scratch collapses**: `pr`=1.02, `ev_gap`=+0.0002, probing at 0.0739
against a 0.0516 floor. The REVE warm-start is load-bearing for this
objective, not an enhancement.

SIGReg (λ=1 arms) does prevent the collapse — `pr`=18.9 (warm) and 10.5
(scratch) — but those arms have 4–500× smaller `ev_gap` and worse probes.
**Holding rank high and learning the cross-modal map are in tension here**,
which is the opposite of the free-lunch framing in the plan's anti-collapse
section. The λ=0 warm-start run instead recovers on its own: `pr` falls
18.2 → 1.24 by epoch 48, then rebuilds to 6.31.

## Per-feature structure

Gains from extending 300 → 1000 ep concentrate proportionally in the
low-variance "tail" features, while the dominant ones saturate early
(val Pearson r):

| feature | 300 ep | 1000 ep | Δ | CI separated |
|---|---:|---:|---:|:--|
| motion_energy | 0.1631 | 0.2177 | +0.055 | yes |
| depth_mean | 0.1371 | 0.1748 | +0.038 | yes |
| scene_natural_score | 0.1569 | 0.1867 | +0.030 | yes |
| luminance_mean | 0.2946 | 0.2914 | −0.003 | no |
| position_in_movie | 0.2820 | 0.2758 | −0.006 | no |

In absolute r² the largest gains are the *dominant* features (entropy +0.037,
saturation +0.030) simply because they have more headroom; in relative terms
the tail leads (face_area_frac +63%, narrative_event_score +61%,
motion_energy +56% vs luminance +28%, position +19%). L1 regression is
**variance-ordered**: it fits dominant directions first and reaches the tail
only with substantially more compute.

## Caveats

- **Single seed per arm.** The 300/1000/3000 schedule effect is large and
  monotone in drift, but not seed-controlled. Collapse-recovery timing already
  showed run-to-run variability at matched LR.
- **1000 ep is an operating point found by a 3-point sweep**, not a tuned
  optimum. 500 and 1500 were not run.
- Val and test Pearson r are **not comparable** — the random floor is 0.1228
  on val vs 0.0516 on test (293 vs 108 recordings). Compare within split only.
- λ=0.5 arms were not run: the plan gated them on the endpoints separating,
  and λ=1 lands far below λ=0, so interpolation cannot cross it.

## Reproduce

```bash
# training (arm order 1 -> 2 -> 3 -> 4)
python experiments/mjepa/_submit.py l0-warm-1000ep --lam=0.0 --init=reve \
    --epochs=1000 --lr=1e-4 --save-every=100 --partition=gpuA40x4 submit

# probes
python -m eb_jepa.evaluation.clip_probe.probe \
    --checkpoint <exp>/latest.pth.tar --config <exp>/config.yaml \
    --split val --cv-splits 5 --output probe_val.json
python -m eb_jepa.evaluation.clip_probe.probe_traintest \
    --checkpoint <exp>/latest.pth.tar --config <exp>/config.yaml \
    --eval-split test --bootstrap 2000 --seed 42 --output probe_test.json
```

Job IDs: training 20627291 (300 ep), 20652094 (1000 ep), 20657782 (3000 ep),
20633076 (λ=0 scratch), 20652083/84 (λ=1 warm/scratch).
