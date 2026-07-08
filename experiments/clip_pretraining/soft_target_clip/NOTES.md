# soft_target_clip — Andonian-2022-style distillation from V-JEPA-2 similarity

Sibling to [`../scene_clip_fromscratch/`](../scene_clip_fromscratch/) and
[`../scene_clip_multimovie/`](../scene_clip_multimovie/). Tests replacing the
`scene_clip` recipe's **discrete scene-ID multi-positive label** with a
**continuous target-similarity soft label** derived from the (mean-centered)
V-JEPA-2 embeddings themselves — the "Option 4" line in
[`../embedding_feature_correlation/clip_design_observations_vjepa2.md`](../embedding_feature_correlation/clip_design_observations_vjepa2.md) §6.

> **Experiment concluded 2026-07-08. Final results in [`RESULTS_jul7.md`](RESULTS_jul7.md).**
> TL;DR — soft τ=0.05 wins on TP-only 400 ep (+0.0026 mean Δr² over vanilla CLIP,
> 2.6× seed std); ties vanilla on multi-movie 400 ep; and reaches R6-test
> Pearson r = 0.1517 (+0.099 vs random) — the best from-scratch checkpoint on
> record, +0.019 raw r over fresh500. Full analysis, per-feature breakdown,
> and reproducibility recipes in the RESULTS file.

## Objective (in one paragraph)

Andonian 2022 (*Robust Cross-Modal Representation Learning With Progressive
Self-Distillation*, CVPR) blends the hard CLIP diagonal with a soft target
distribution from an EMA teacher, gradually shifting weight to the teacher
as it becomes reliable. Our adaptation drops the EMA and the schedule
because the "teacher" here — V-JEPA-2 pairwise similarity — is **frozen and
reliable from step 0**. The blended target per row is
`t = (1 − α) · I + α · softmax(V V^T / τ_t)`, with `α = 0.5` and
`τ_t = 0.1` as the defaults. Loss is a symmetric cross-entropy against `t`.
Implementation: [`SoftTargetCLIPPretrain`](../../../eb_jepa/clip.py).

## Hypothesis

The `scene_clip` recipe's §7 near-duplicate-shot problem (4/36 scenes with
negative margin under the scene map, plus buffer-invisible false negatives
in the tail of the |Δt| distribution) is handled *automatically* by the
soft target: near-duplicate shots share high `V V^T` mass and are no longer
punished as negatives. Three possible outcomes:

1. **Soft target beats scene_clip** — the continuous similarity captures
   sub-scene structure (within-scene shot ordering) and inter-scene
   near-duplicates that the hard scene map cannot.
2. **Soft target matches scene_clip** — scene ID was already a good enough
   discretization of the underlying V-JEPA-2 similarity, and moving to
   continuous targets is a wash.
3. **Soft target underperforms** — the added multi-positive mass smears
   gradients and hurts convergence at typical batch sizes (B·T ≈ 512 tokens
   per contrastive matrix).

Prior on outcomes: I expect #1 or #2 — the §5 measurement of 12/54 shots
with negative margin is the specific pathology soft targets are designed
to remove.

## Design decisions

Fixed for this experiment, matching the §9 bottom-line recipe on everything
that isn't the loss:

- **Target**: `target_kind: per_window`, `mean_center: true` (dataset
  centers V-JEPA-2 by the training-set global mean once at init).
- **Hard component**: diagonal (identity), *not* scene multi-positive. The
  teacher `softmax(V V^T / τ_t)` already puts high mass on same-shot rows
  because same-shot windows share exact target vectors; layering scene IDs
  on top would double-count.
- **α = 0.5** (fixed). No progressive schedule.
- **τ_t = 0.1** (default). Sensitivity: 0.05 (sharper, closer to hard) vs
  0.2 (smoother, more many-to-many mass) as follow-ups if 0.1 wins.
- **Temporal buffer 2 s** kept. Masks cross-pairs from *both* the student
  softmax denominator and the teacher softmax denominator, so shot-cut-
  boundary neighbors (§3, centered cos ≈ 0.97) contribute to neither target
  nor prediction.
- **scene_ids passed but ignored** by the module (`del scene_ids` inside
  forward). Interface parity with `SceneCLIPPretrain` — the training loop
  can dispatch the same recipe-mode batch to either model.

## A/B setup

One submit script drives both arms, matched on encoder / optimizer / data.
Training is **multi-movie** on `task=[ThePresent, DespicableMe]` (the current
[`config/clip_pretrain.yaml`](../../../config/clip_pretrain.yaml) default,
matching the jul2 baseline in [`../scene_clip_multimovie/`](../scene_clip_multimovie/)):

- **Arm A** (baseline): `loss.mode: scene_clip`.
- **Arm B** (test): `loss.mode: soft_target_clip`, `soft_alpha: 0.5`,
  `soft_tau_teacher: 0.1`.

The submit script snapshots the base multi-movie config to the checkpoint
dir, patches `loss.mode` (and the soft-target knobs for arm B), then
snapshots per-movie variants (`config_TP.yaml`, `config_DM.yaml`) so each
movie is probed against a matching single-task config. Multi-movie plumbing
(per-task V-JEPA-2 recipes, scene-ID namespacing) already tested — see
[`../scene_clip_multimovie/NOTES.md`](../scene_clip_multimovie/NOTES.md).

Extra thing to watch under multi-movie: the teacher `V V^T` is computed
*within* each batch. Because `DataLoader` mixes recordings from both movies
into the same batch, cross-movie pairs will contribute to `q`. Their raw
centered-cosine will be near 0 by construction (per-movie centering, so
cross-movie targets are approximately orthogonal), which means the teacher
naturally down-weights them — a soft version of the scene-ID namespacing
that `scene_clip` uses. No explicit cross-movie masking needed.

## Metrics to watch

From the module's `loss_dict` — new signals specific to the soft-target
objective:

- `clip_teacher_entropy` — H(q) over the batch. High-entropy teacher = more
  multi-positive mass; low-entropy teacher ≈ hard label. Row-normalized so
  ln(N) is the ceiling.
- `clip_teacher_eff_positives` = exp(H(q)) — an interpretable "effective
  number of positives per anchor." On centered V-JEPA-2 with same-shot
  cos ≈ 0.79 and τ_t = 0.1 I expect ≈ 5–15 at B·T = 512.

Standard recipe val metrics (already logged by `evaluate_recipe`):
`val/{vision,clip}_{shot,scene}_auc`, `val/clip_nn_top1`,
`val/scene_collapse_ratio`. §8 warns that shot-mean + scene labels can
collapse EEG embeddings to scene-mean; soft targets should reduce this
risk because the target itself carries within-scene shot structure.

Downstream probe: standard CLIP-probe pipeline on ThePresent val.

## Files

- [`_submit_ab.py`](_submit_ab.py) — submit either arm to Delta. Snapshots
  config, patches `loss.mode`/`soft_alpha`/`soft_tau_teacher`, trains, then
  probes val on ThePresent.
- `RESULTS.md` — to be written after the first pair of runs.
