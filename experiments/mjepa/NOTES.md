# MJEPA-adapted — pre-registration

Adapting MJEPA (Teotia et al., [arXiv:2606.25225](https://arxiv.org/abs/2606.25225),
FAIR/NYU) to EEG ↔ movie. Written before any run; results go in `RESULTS.md`.

## Hypothesis

**Contrastive cross-modal objectives on this dataset are limited by the number
of distinct anchors, not by the loss.** `experiments/snr_scaling/PLAN.md`:

> ThePresent is 203.3 s = **101 distinct 2 s anchors**. The 71K training windows
> are 703 subjects × 101 moments. **The InfoNCE problem has ~101 classes.** This
> is almost certainly why every objective saturates at the same place.

Fourteen contrastive configs land in Δr² ∈ [+0.041, +0.043] regardless of loss,
depth, width, or compute. Every cross-modal loss in this repo is contrastive —
`eb_jepa/clip.py` contains zero `mse_loss`/`smooth_l1`/`l1_loss`.

MJEPA's cross-modal term is an **L1 regression to a continuous 1408-d target**,
which has no class-count ceiling. If the ~101-class diagnosis is right,
switching from discrimination to regression should break the saturation.

## What is being run

```
L_e2e = masked EEG token prediction                       (existing MaskedJEPA)
L_e2v = || mlp_ev(pool(z_eeg)) - sg(standardize(vjepa2)) ||_1
L_v2e = || mlp_ve(vjepa2) - sg(standardize(pool(z_eeg))) ||_1
total = lambda_intra * L_e2e (+ anti_collapse) + L_e2v + L_v2e
```

Adapted scope: V-JEPA-2 stays frozen precomputed data, so there is no movie
token stream and no change to `tokenize` / `pool_to_windows` / masking.

## Pre-registered predictions

| # | prediction | falsified if |
|---|---|---|
| 1 | λ=0 (pure cross-modal regression) beats 0.1517, the best from-scratch contrastive result | it lands at or below 0.1517 |
| 2 | λ=0 + REVE warm-start beats **0.2383**, the repo record | it lands at or below 0.2383 |
| 3 | λ>0 does **not** help: masked prediction has scored at/below random init in every prior run (0.0344 masked vs 0.0515 random) | λ=1.0 beats λ=0 by more than the bootstrap CI |

Prediction 3 is the one worth stating loudly, because MJEPA's own headline
ablation claims the opposite for audio-video (shared encoder *without*
cross-modal degrades below unimodal; adding it lifts both). Our setting differs
in that the "other modality" is a frozen teacher rather than a co-trained
stream, so their result may simply not transfer.

## Kill-switch protocol — DO NOT gate on this; run to completion

> **CORRECTION (2026-07-30, after arm 1 / job 20627291).** The protocol as
> originally written — "check within the first ~100 steps" — **produces a false
> negative on the arm this experiment exists to test.** Applied literally it
> would have killed the run. Recorded here because the mistake is more
> instructive than the thresholds.
>
> Arm 1's actual trajectory over 300 epochs (11 steps/epoch):
>
> | epoch | `ev` (floor 0.711) | `ev_gap` | `pooled_pr` |
> |---|---|---|---|
> | 3 | 0.759 | +0.0003 | 18.2 |
> | 48 | 0.697 | +0.066 | **1.24** ← trough |
> | 124 | 0.748 | +0.079 | 1.75 |
> | 199 | 0.636 | +0.207 | 2.55 |
> | 299 | **0.629** | **+0.242** | **2.73** |
>
> Three things every one of which defeats an early gate:
>
> 1. **`ev` sat exactly on the 0.711 floor for ~140 epochs**, then broke
>    through. At the 100-step mark (≈ epoch 9) it is indistinguishable from a
>    model that never learns. The separation appears ~1,600 steps in — **16×
>    past the original checkpoint.**
> 2. **The trajectory is collapse-then-recover.** `pooled_pr` falls 18.2 → 1.24
>    over epochs 12–50 as the encoder discards the REVE init, then rebuilds to
>    2.73. Read at the trough, the run looks terminally rank-collapsed.
> 3. **`pooled_pr` is not a proxy for usable representation.** The epoch-63
>    checkpoint, at `pr`≈1.3, already probes at val Δr² **+0.0202** over the
>    random REVE-shape baseline (0.0357 vs 0.0155), positive on all 12
>    features. Variance concentrated into few directions, but those directions
>    were movie-relevant. `pr` measures variance spread, not task content, and
>    here the two came apart.
>
> **Why gating was never necessary:** the measured cost is **10 s/epoch**, so a
> 300-epoch arm is **51 minutes** on one A40. There is nothing to save by
> killing early. Run every arm to completion and decide on the probe.
>
> **SECOND CORRECTION (2026-07-31, after the 3000-epoch arm).** I wrote here
> that "`ev_gap` in particular is trustworthy." **It is not.** See
> [RESULTS.md](RESULTS.md):
>
> | run | `ev` | `ev_gap` | `pr` | val Δr² |
> |---|---:|---:|---:|---:|
> | 1000 ep | 0.315 | +0.706 | 6.31 | **+0.0437** |
> | 3000 ep | 0.362 | +0.588 | 2.78 | **+0.0026** |
>
> The 3000-ep run has excellent training diagnostics and a representation at
> the random-noise floor — in fact *below* its own untrained REVE init
> (test r 0.0801 vs 0.0938). `ev`, `ev_gap` and `pr` are all **train-split**
> quantities; moving them the right way is fully compatible with destroying
> downstream performance.
>
> `ev_gap > 0` remains valid for the narrow thing it actually proves — a
> constant prediction gives *exactly* zero gap, since rolling the target is a
> permutation the L1 sum is invariant to, so `ev_gap > 0` rules out a
> degenerate encoder. It says nothing about whether the representation is
> **useful**, and it was the latter I was using it for.
>
> `pr` fails the same way: 300 ep (`pr`=2.73) and 3000 ep (`pr`=2.78) are
> indistinguishable on rank and differ 11× in Δr².
>
> **Only the probe is informative.** Log these as a post-hoc trace; never gate
> on them, and never report a run as healthy on their basis.

## Metric reference (log these; do not gate on them)

Targets are per-dim standardized, so "predict the mean" scores a **known floor**:
~0.711 for the V-JEPA-2 side (measured on ThePresent shot-means), ~0.798 for a
Gaussian. **`ev_loss` converged at its floor means the model learned nothing.**

| metric | healthy | failure |
|---|---|---|
| `ev_loss` | falls below 0.711 and keeps falling | parks at 0.711 |
| **`ev_gap`** = `L1(pred, roll(target,1)) − ev_loss` | **> 0 and increasing** | → 0 |
| `pooled_pr` (participation ratio) | ≫ 1 | → 1 (rank collapse) |
| `pooled_var_raw` | stable | → 0 (total collapse) |

All four are in the tqdm postfix (`ev`, `evgap`, `pr`) so they are readable
straight from the job log, not only from wandb. `ev_gap` is ~0 at init by
construction — the gap *opening* is the signal.

## Known limitations, stated up front

- **`L_v2e` gives the encoder no gradient.** The movie side is frozen data, so
  with stop-grad on the EEG target this term trains only `mlp_ve`. Kept as a
  free collapse detector, not as a second training signal.
- **P=2 patches per channel** at `patch_size=200` / `ws=2 s`, so the masked term
  degenerates toward channel-masking. A real caveat on the λ>0 arms only.
- ~~**From-scratch at REVE shape has no existing baseline.**~~ **Wrong** — one
  already existed at `clip_pretraining/scene_clip_from_checkpoint/probe_results/`
  (`probe_random_reve_shape_val.json`, `probe_traintest_random_reve_shape.json`).
  The fresh control run (job 20627304) reproduces it closely — val r²
  luminance 0.0309 vs 0.0330, contrast 0.0246 vs 0.0251 — so the duplicate was
  harmless and now serves as a same-session replication. Both use the full
  12-feature `SCALAR_FEATURES_DEFAULT`, which is what makes the comparison
  against the 0.2383 record valid; note that `probe.py` / `probe_traintest.py`
  ignore `--features` for dataset construction and always build on that
  constant, so a config-level `eval.feature_names` does **not** narrow the probe.
- **Ceiling.** ρ₁(CorrCA, val) = 0.045 → K=1 probe-r ceiling **0.213**, which the
  0.2383 record already exceeds. Single-trial headroom may be small or nil.
  Report `r / ceiling` either way; "objective engineering is exhausted, here is
  the proof" is a publishable outcome and this experiment is not designed to
  avoid it.

## Run matrix

Order: **0 → 1 → 2 → 3 → 4**. Arm 1 is the experiment; the rest is context.
λ=0.5 only if the endpoints separate.

| # | arm | init | λ | anti_collapse | lr | ep | compares against |
|---|---|---|---|---|---|---|---|
| 0 | `random_reve_shape` | random | — | — | — | 0 | new noise floor (free) |
| 1 | `l0_warm` | REVE | 0 | none | 1e-4 | 300 | **0.2383** (record) |
| 2 | `l0_scratch` | random | 0 | none | 5e-4 | 400 | 0.1517 + arm 0 |
| 3 | `l1_warm` | REVE | 1.0 | sigreg 0.05 | 1e-4 | 300 | arm 1 |
| 4 | `l1_scratch` | random | 1.0 | sigreg 0.05 | 5e-4 | 400 | arm 2 |

LR discipline from the record: **1e-4 or 3e-4 for warm-start, never 1e-5**
(the stale under-trained run); **300 ep, not 1000** (+0.198 → +0.110 at 999 ep).

## Baselines

| protocol | number | source |
|---|---|---|
| test mean r, record | **0.2383** | `scene_clip_from_checkpoint`, REVE warm-start lr=1e-4 300 ep |
| test mean r, best from-scratch | **0.1517** | `soft_target_clip` τ=0.05, TP-only, 400 ep |
| test mean r, random controls | 0.0527 / 0.0516 | `RESULTS_jul7.md` |
| test mean r, masked JEPA | 0.0344 | `probe_results_jul10` |
| test mean r, cross-subject | 0.0406 | this session, job 20604406 |
| K=1 probe-r ceiling | 0.213 | ρ₁(CorrCA, val) = 0.045 |

## Evaluation

`clip_probe/probe.py` (val, 5-fold) + `probe_traintest.py` (test,
`--bootstrap 2000`), submitted as separate A40 jobs. Same 293-rec val /
108-rec test sets as every prior result. `eval.auto_run` is deliberately
**off** — it costs 35–45 min and a timeout late in a long H200 job is expensive
to requeue.
