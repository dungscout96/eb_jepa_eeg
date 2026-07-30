# Cross-subject predictive JEPA — RESULTS (2026-07-30)

**Verdict: negative, and cleanly diagnosed.** The objective trains without collapse but
learns *zero* time-specific information. The probe number lands at test mean
r = **0.0406**, inside the same undifferentiated band as every other JEPA arm and
below random initialization (0.0515).

The diagnostics identify the mechanism precisely, and it is not a bug: the predictor
converges to the **position-conditional mean**, which is the Bayes-optimal answer when
the shared stimulus response is not recoverable from a single trial.

---

## 1. What was run

Predict subject B's target tokens from subject A's context at the same movie time.
Because B is drawn independently of A, `E[z_B | context_A] = E[shared_stimulus(t) | context_A]`,
so the subject fingerprint (~96% of variance) is marginalized out by construction.
Proposed at `experiments.md:73` in April; built and run here.

Implementation: `CrossSubjectJEPA` (`eb_jepa/jepa.py`) is `MaskedJEPA` on a flattened
`2B` batch with `torch.roll(targets, shifts=B, dims=0)` swapping the halves — symmetric
A→B / B→A in one predictor call, shared weights. Data comes from
`PairedSubjectJEPADataset` (`eb_jepa/datasets/paired.py`), which joins recordings on
`t_start`. Config is bit-identical to `lejepa_reve` in `model.*` and `masking.*`, so the
target swap is the only variable.

| run | job | partition | bs | rows | pooled SIGReg N | steps | wall |
|---|---|---|---:|---:|---:|---:|---|
| `xsubj-h200-bs32` | 20604406 | gpuH200x8 | 32 | 64 | **64** | 2,200 | 1:09:00 |
| `xsubj-a40-bs8` | 20572347 | gpuA40x4 | 8 | 16 | 16 | 8,800 | 3:55:32 |

Both completed 100 epochs with all five checkpoints and zero errors.

---

## 2. Probe results

`clip_probe/probe.py` (val, 5-fold GroupKFold-by-recording) and
`clip_probe/probe_traintest.py` (fit train / eval test), same 293-rec val and 108-rec
test sets as `probe_results_jul10`. **`--bootstrap 2000` was used here; all jul10 JSONs
have `bootstrap_iters: 0`.** JSONs in `probe_results/`.

Mean over the 12 scalar movie features:

| checkpoint | test mean r | val mean 5-fold R² |
|---|---:|---:|
| `random_lejepa` (untrained) | **0.0515** | 0.0186 |
| `jepa_laya_multi` | 0.0494 | 0.0153 |
| `jepa_lejepa_multi` | 0.0416 | 0.0189 |
| **`xsubj_h200_bs32` (N=64)** | **0.0406** | **0.0169** |
| `jepa_lejepa_single` (masked, direct counterpart) | 0.0344 | 0.0181 |
| `xsubj_a40_bs8` (N=16) | 0.0269 | 0.0065 |
| `random_laya` (untrained) | 0.0251 | 0.0192 |

Against the pre-registered success ladder:

1. **Minimum bar — beat `jepa_lejepa_single` (0.0344): nominally met** (+0.0062).
2. **Real win — beat `random_lejepa` (0.0515): not met** (−0.0109).

### 2.1 The nominal win is not real

Two independent reasons to reject it:

- **The metrics disagree in sign.** On test r cross-subject beats masked (+0.0062); on
  val R² it is *worse* (0.0169 vs 0.0181). A genuine effect does not flip across metrics.
- **The CIs swamp the delta.** Median per-feature 95% bootstrap CI width is **0.056**,
  about **9×** the +0.0062 gap. Every run here and in jul10 is single-seed.

The honest statement is that the whole 0.025–0.052 band is one undifferentiated cloud,
and the ranking within it carries no information.

### 2.2 The encoder is alive, though

**8 of 12 features have 95% CIs excluding zero** (`xsubj_h200_bs32`, test):

| feature | r | 95% CI |
|---|---:|---|
| entropy | +0.1095 | [+0.0663, +0.1561] |
| saturation_mean | +0.0598 | [+0.0099, +0.1130] |
| contrast_rms | +0.0583 | [+0.0130, +0.1064] |
| luminance_mean | +0.0545 | [+0.0010, +0.1102] |
| scene_natural_score | +0.0463 | [+0.0209, +0.0742] |
| narrative_event_score | +0.0454 | [+0.0247, +0.0644] |
| motion_energy | +0.0414 | [+0.0190, +0.0625] |
| face_area_frac | +0.0404 | [+0.0139, +0.0677] |
| edge_density | +0.0177 | [−0.0091, +0.0424] |
| depth_mean | +0.0146 | [−0.0120, +0.0421] |
| n_faces | +0.0117 | [−0.0162, +0.0397] |
| position_in_movie | −0.0124 | [−0.0783, +0.0584] |

Real low-level visual signal is present. It is just not *more* signal than a random
encoder provides. Note every `r2` is negative — a protocol property shared with the
jul10 random baselines, not specific to this arm.

---

## 3. Mechanism: why it failed

The training diagnostics answered this before the probe ran. New metrics added for this
objective (`eb_jepa/jepa.py`), all forwarded to W&B and to the tqdm postfix:

| diagnostic | A40 (N=16) | **H200 (N=64)** | reads as |
|---|---:|---:|---|
| `target_var_across_batch` | 0.000193 | **0.746** | across-*sample* variance |
| `target_var_across_pos` | 0.284 | 1.416 | across-*position* variance |
| `target_var` (pre-existing) | 0.283 | 1.554 | **flattens both — hides the collapse** |
| `pred_var_ratio` | 0.999 | 0.501 | →0 means constant predictor |
| `pred_target_cosim` | 0.999652 | 0.941442 | alignment to the matched target |
| `cosim_shuffled` | 0.999651 | 0.941386 | alignment to a *wrong subject and time* |
| `align_gap` | 1.6e-06 | 5.7e-05 | the headline: stimulus-specific info |
| `pred_loss_gap` | 4.2e-07 | −2.0e-05 | must be >0 and growing |

**At N=16 (A40) the encoder collapsed outright.** `target_var_across_batch = 1.9e-4`
against `target_var_across_pos = 0.284` — a 1,500× gap. The encoder learned a
position-dependent but *input-independent* function: every clip produced the same tokens.
This is a SIGReg-undersampling artifact (N=16 is 8× below the N=128 the config targets)
and should not be read as a property of the objective.

**At N=64 (H200) SIGReg did its job — and the objective still failed.**
`target_var_across_batch = 0.746` is healthy; representations stay spread across samples.
But `pred_target_cosim` and `cosim_shuffled` agree to four decimals (0.941442 vs
0.941386), and across all 2,200 steps `pred_loss_gap` ranged **−0.0032 to +0.0002** —
noise around zero. It never opened.

So the predictor is not constant (`pred_var_ratio = 0.50`); it predicts the
**position-conditional mean**, which is identical for every subject and every moment.
That is exactly the joint optimum flagged during planning as structurally invisible to
SIGReg: *targets spread isotropically while predictions carry no sample-specific
information*. `ac_loss`, `sigreg_loss`, and the pre-existing `target_var` all look
perfectly healthy while it happens.

> **Diagnostic lesson.** The pre-existing `target_var` reads a healthy 0.283 on the
> *fully collapsed* A40 run, because it flattens `(B·n_pred, D)` and conflates
> across-position with across-sample variance. Only `target_var_across_batch` catches it.
> Any future JEPA variant here should log the across-batch term.

---

## 4. Why the mean is the right answer: the 101-anchor ceiling

The pair index reports, for `task=ThePresent`:

```
Cross-subject pair index: 101 anchors, median 701 recordings/anchor,
subject-A pool 701/701 recordings (len=701 items/epoch)
```

**101 anchors.** `experiments/snr_scaling/PLAN.md` independently derives the same number
from the other direction — "ThePresent is 203.3 s = **101 distinct 2 s anchors**; the 71 K
training windows are `703 subjects × 101 moments`" — and names it as the likely reason
every objective saturates at the same place.

The cross-subject task therefore has at most **101 distinguishable targets**, each
observed through single-trial EEG at roughly −24 dB. Discriminating 101 targets at that
SNR is precisely the regime where regression-to-the-mean is optimal. The model did not
fail to find the signal; it correctly found that the conditional mean beats any attempt
at the signal. Swapping the *prediction target* cannot fix a problem that lives in the
**anchor count**.

This also predicts, correctly, that the pre-registered rescue levers would not have
helped: `pred_target_mode=all`, `sigreg coeff=0.1`, and `within_subject_weight=0.3` all
target *collapse*, and the N=64 run did not collapse.

---

## 5. Reproduce

```bash
# pretrain (dry-run without `submit`)
python experiments/jepa_pretraining/cross_subject/_submit.py xsubj-h200-bs32 \
    --partition=gpuH200x8 --batch-size=32 --auto-eval=false --sec-per-ep=200 submit

# probe (gpuA40x4, ~12 min val / ~30 min test)
PYTHONPATH=. uv run --group eeg python eb_jepa/evaluation/clip_probe/probe.py \
  --checkpoint $CKPT --config $CFG --split val --cv-splits 5 --encode-batch 64 \
  --output probe_results/xsubj_h200_bs32__probe_val.json
PYTHONPATH=. uv run --group eeg python eb_jepa/evaluation/clip_probe/probe_traintest.py \
  --checkpoint $CKPT --config $CFG --eval-split test --encode-batch 64 \
  --bootstrap 2000 --seed 42 \
  --output probe_results/xsubj_h200_bs32__probe_traintest.json
```

Unit tests (no cluster, no HBN data): `tests/test_cross_subject_jepa.py`,
`tests/unit/test_pair_index.py` — 49 tests.

---

## 6. Infrastructure findings

Four things measured along the way that outlive this experiment.

**(a) `--eval.auto_run=false` never worked.** A dot-notation CLI override arrives as the
*string* `"false"`, which is truthy, so the gate ran the full probe eval regardless. Any
past sweep that used this flag to skip eval was still paying for it — a plausible cause
of unexplained TIMEOUTs. Fixed in `jepa_pretrain.py` with the same coercion idiom already
used for `vicreg use_projector`.

**(b) Auto-eval costs 35–45+ min, not the ~15 min the submit scripts assumed.**
`eval.n_passes=20` across all three splits is `701+293+108` recordings × 20 =
**~22,040 clip encodes**, each a separate FIF open plus a batch-of-1 forward.
`probe_eval._embed_clip` seeds the RNG immediately before `dataset[rec_idx]` to make the
temporal offset deterministic, which is exactly what prevents batching.

**(c) Measured memory model.** Cross-subject at bs=B pushes 2B rows through the encoder,
so it costs ≈ masked JEPA at bs=2B. Measured on A40 (job 20571303), this exact geometry:

| bs | rows | peak GiB | % of 44.4 |
|---:|---:|---:|---:|
| 4 | 8 | 11.33 | 25.5% |
| 8 | 16 | 22.75 | 51.2% |
| 12 | 24 | 35.95 | 80.9% |
| 16 | 32 | OOM | — |

Linear at ~3.08 GiB per unit batch size. `_submit.py` now derives its OOM guard from this
model instead of a hardcoded threshold. **The original bs=64 was never viable** — it
predicts 196 GiB against H200's 139.8 GiB, and job 20567666 duly OOM'd. Note that
`fire.Fire` swallowed the exception, so SLURM reported `COMPLETED 0:0` after 2:26;
`sacct` state alone is not sufficient to confirm a training run.

**(d) Pooled SIGReg N is memory-bound at ≈68 on Delta.** `N = 2·batch_size·n_windows`, and
peak memory is proportional to that *same* product — so trading batch size for
`n_windows` buys no extra SIGReg samples. The N=128 this config targets is unreachable for
this encoder without gradient checkpointing, a smaller `embed_dim`, or multi-GPU sharding
(not currently supported). Any future single-stream run should state its N explicitly.

**(e) Coverage-driven window drops are a pure suffix trim.** Both V-JEPA-2 timestamp grids
are uniform 0.5 s with no gaps (ThePresent n=406 over [0, 202.5]; DespicableMe n=341 over
[0, 170.0]) and both drop paths are monotone in movie time. Confirmed empirically by
`subject-A pool 701/701`. So window index *i* already is the same movie moment in every
recording; the `t_start` join in `paired.py` is correctness insurance, not a bug fix, and
its anchor check is a tripwire against a future non-uniform grid.

---

## 7. What this does and does not establish

**Establishes.** Symmetric cross-subject token prediction, at pooled SIGReg N=64, 100
epochs / 2,200 steps, `masked/masked`, on ThePresent, produces representations carrying no
measurable time-specific predictive information, and a probe score indistinguishable from
the rest of the JEPA family. The failure mode is regression to the position-conditional
mean, not representational collapse.

**Does not establish.**

- That more optimizer steps would not help. This run had 2,200 steps. The A40 run had
  8,800 without the gap opening, but is confounded by N=16.
- That N=128 would behave differently. Untestable on Delta at this encoder size (§6d).
- Anything about multi-movie training, which would raise the anchor count — the variable
  §4 identifies as binding. ThePresent + DespicableMe gives ~186 anchors, still small.
- Anything about the `jepa_*_from_checkpoint` CLIP warm-start arms. **Do not cite the
  jul10 "JEPA warm-start is harmful" conclusion**: those arms ran 100 ep / lr=1e-5 against
  jul7's 400 ep / lr=1e-4, a confound already flagged in `snr_scaling/PLAN.md:216-224`.

**Recommended next step.** Not another JEPA target. §4 argues the binding constraint is
the ~101-anchor ceiling, which is a property of the data, not of the loss. Either raise
the anchor count (more movies) or move to an objective whose target is continuous rather
than one-of-N.
