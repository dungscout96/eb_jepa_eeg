# snr_scaling — experiment plan

**Paper thesis (revised 2026-07-30 after E0.1 and a source check of the
decoding literature; supersedes [paper/workshop_outline.md](../../paper/workshop_outline.md)):**

> EEG and MEG fail differently, and the difference dictates the strategy. On an
> identical single-trial cross-subject pipeline, MEG reaches 41 % top-1 segment
> retrieval and EEG reaches 5 % — an ~8× gap in **per-subject SNR**. But EEG's
> disadvantage is exactly where MEG's practical ceiling lies: MEG cohorts are
> tens of subjects, EEG cohorts are thousands. **The modality with the worse
> per-subject SNR is the one that can buy SNR with subjects.** We measure the
> cross-subject noise ceiling for stimulus-locked EEG, show four objectives
> already capture 83-88 % of it — so single-trial work is nearly exhausted —
> and show that *the objective determines how efficiently subjects convert into
> stimulus signal*.

The headline quantity is a **subject-scaling exponent** `dr / d log S` compared
*across objectives*, not an absolute *r*. See "Related work — the delta" below
for why the exponent, not the curve, has to be the contribution.

Calculator: [`scaling_calculator.py`](scaling_calculator.py) →
[`snr_scaling.png`](snr_scaling.png).

---

## Why EEG vs MEG is the framing, not a caveat

Verified against sources in
[`RESULTS.md` §1.2b](RESULTS.md). Défossez et al. (2023) run one architecture,
one contrastive objective, single-trial, cross-subject, over four naturalistic
speech datasets spanning both modalities:

| dataset | modality | top-1 | top-10 |
|---|---|---:|---:|
| Gwilliams | MEG | **41.3 %** | 70.7 % |
| Schoffelen | MEG | 36.8 % | 67.5 % |
| Brennan | **EEG** | **5.2 %** | 25.7 % |
| Broderick | **EEG** | **5.0 %** | 17.7 % |

This is the cleanest modality-controlled comparison in the literature: same
model, same loss, same protocol, ~8× gap. **The widely-quoted "41 % out of
1,000+ segments" is MEG.** Any claim that naturalistic neural decoding "works"
is, on inspection, a claim about MEG.

**The practical asymmetry is the whole argument.** MEG requires a magnetically
shielded room and a multi-million-dollar cryogenic instrument; cohorts are tens
of subjects and the modality does not scale to population studies, clinics, or
anything wearable. EEG is cheap, portable, and already collected at population
scale — HBN alone gives ~700 subjects on the same stimulus, and that is one
dataset. So the two modalities sit at opposite corners of a trade:

| | per-subject SNR | subjects obtainable |
|---|---|---|
| MEG | high (~8× EEG) | tens |
| EEG | low | thousands |

A method that converts *subjects* into *stimulus signal* is therefore worth far
more to EEG than to MEG — MEG has little left to gain on the axis it is already
good at, and cannot cheaply scale the axis it is weak on. **Cross-subject
aggregation is the modality-appropriate strategy for EEG specifically**, and
that is a stronger motivation than "EEG is noisy."

This also reframes what a negative EEG result means. Our
[v→e-at-chance retrieval finding](../clip_pretraining/soft_target_clip/RESULTS_jul7.md)
looked anomalous; against Défossez's EEG numbers (~5 % top-1, near floor) it is
the *expected* behaviour of single-trial EEG, and the anomaly would have been
succeeding. Cite their EEG rows, not their headline.

---

## Related work — the delta, and the reviewer objection to pre-empt

**Défossez et al. already publish a subject-scaling curve** (their Fig 3C):
top-10 accuracy versus number of training participants, ~1 to ~96 subjects,
rising with no clear saturation. Their subject-specific 1×1 conv is their
single largest ablation (removing it costs ~24 points, 70.7 % → 47.0 %). They
state it explicitly:

> "not only does our subject-specific layer improve decoding performance, but
> this performance increases with the amount of participants present in the
> training set"

**So "more subjects helps" is known and published.** If E1.1 is framed as a
subject-scaling curve, it is a re-derivation on EEG of a published MEG result
and a reviewer in this area will say so. What they do NOT do, verified by
source check:

- **No SNR or noise-ceiling framing.** "SNR", "noise ceiling", "averaging
  across subjects" do not appear in that sense. Their curve has no ceiling on
  its y-axis, so "how much of the achievable signal is captured" is unasked.
- **No subjects-vs-data-per-subject comparison** at matched total data. The
  exchange rate between the two currencies is untouched.
- **No test-time aggregation across subjects.** All evaluation is per-subject
  prediction; the K_test axis is absent.
- **No inter-subject correlation or shared-response analysis.** The mechanism
  we would claim — that a cross-subject target marginalises the fingerprint —
  has no counterpart.
- **The scaling curve is MEG-only.** Brennan was excluded from that analysis.

**Therefore the contribution is the exponent COMPARED ACROSS OBJECTIVES, not
the curve.** Défossez has one curve for one architecture. The claim we can make
and they cannot is that *the training objective sets how efficiently subjects
convert into stimulus signal* — with the falsifiable prediction that
within-subject masked prediction has slope ≈ 0 or negative (more subjects means
more fingerprint diversity to model) while a cross-subject target has positive
slope tracking `sqrt(R(K))`.

**Consequence for E1.1: the LeJEPA arm is load-bearing, not a courtesy
baseline.** Without a second objective there is no exponent comparison and
nothing survives the Fig 3C objection. Do not drop it to save compute.

---

## The analysis that motivates every experiment below

Model: `x_s(t) = g(t) + n_s(t)`, `rho1 = Var(g)/(Var(g)+Var(n))` = single-trial
inter-subject reliability. Averaging K subjects at the same movie moment gives
Spearman-Brown reliability `R(K) = K·rho1 / (1 + (K−1)·rho1)`. Because a frozen
V-JEPA-2 feature is a noiseless function of the movie, **`sqrt(R(K))` is a hard
ceiling on probe *r* for any encoder and any objective.**

Three consequences, all load-bearing:

**(a) MEASURED (E0.1 2026-07-30, corrected 2026-07-31): we are NEAR the ceiling.**
`rho1` is no longer an assumption. Cross-validated CorrCA on R5 val
(293 subjects, filters fit on 146 recordings and scored on the disjoint 147)
gives a combined 5-component reliability of **0.098**, i.e. a K=1 ceiling of
**0.313**.

| quantity | val (293 subj) | test (108 subj) |
|---|---:|---:|
| CorrCA held-out per component | [0.002, **0.045**, 0.040, 0.012, 0.006] | [0.002, **0.014**, 0.002, 0.011, 0.002] |
| max single component | 0.045 → ceiling 0.212 | 0.014 → ceiling 0.116 |
| **combined 5-component** | **0.098 → ceiling 0.313** | 0.030 → ceiling 0.172 |
| best from-scratch, SAME split | *r* 0.2584 → **CC_norm 0.826** | *r* 0.1517 → **CC_norm 0.882** |

**Use the val estimate.** The test estimate is downward-biased by fit-set
size: CorrCA there is fit on only 54 recordings across 128 channels, so its
filters generalise poorly. The ceiling is a property of the subject
population, not of which split it was estimated on.

**Use the SAME split on both sides.** An earlier version of this table paired
the *test* `r` (0.1517) with the *val* ceiling (0.313) and concluded 48 % of
ceiling with ~2× headroom. That was a split mismatch. Paired correctly the two
splits agree at **83–88 %**, and remaining single-trial headroom is only
**~1.15–1.2×**. The error was caught by E0.2, whose K=1 point on val
(*r* = 0.289) was far too high to be consistent with the 48 % figure.

**Verdict: the checkpoints are NEAR the single-trial ceiling.** The
four-objective tie is consistent with a ceiling effect after all. Two caveats
keep this from being a proof: 0.313 is a *lower* bound (only 5 components), and
no ceiling has been measured for DespicableMe, whose own plateau may still be
anchor-count or teacher-quality limited.

Consequence for the thesis: **single-trial objective work is close to
exhausted, which makes cross-subject aggregation not merely the better lever
but very nearly the only one.** E0.2 measures it at 2.6× (probe *r* 0.289 →
0.742 from K=1 to K=128) against ~1.15–1.2× left single-trial. That is a
cleaner argument than the previous "headroom exists and objectives can't reach
it", and it does not depend on the anchor-count explanation.

Two anomalies worth reporting in the paper, both robust across splits:

- **CorrCA component 1 does not generalise.** Its in-sample eigenvalue is by
  far the largest (0.201 val, 0.070 test) but its held-out ISC is ~0.002 in
  both. Components 2-3 carry the real reliability. Any analysis quoting the
  top eigenvalue as an ISC is quoting an artifact.
- **The delta/theta >> alpha split is confirmed but smaller than cited.**
  Measured best-channel δ/θ = 0.062 vs α = 0.029, a ratio of 2.1×.
  `experiments.md` § "Core Problem Identified" cites δ/θ ISC of 0.10-0.28 and α < 0.05; at 2 s
  window-level log power the true values are 2-5× lower. **Quote the measured
  numbers, not the cited ones.**

**(b) The two aggregation axes buy very different amounts.** At rho1 = 0.05:

- `K_test` (subjects averaged at evaluation) **raises the ceiling**: 0.224 →
  0.587 at K=10 → 0.851 at K=50. Worth **4–6×**.
- `K_train` (partners in the pretraining target) can only **approach** the
  K_test=1 ceiling of 0.224. Worth **at most ~1.5–2×**.

So the paper must report a **2-D** scaling law over (K_train, K_test). Reporting
only single-trial numbers understates the finding by 4×; reporting only
aggregated numbers hides that the objective still matters.

**(c) Subjects and anchors are different currencies.** ThePresent is 203.3 s =
**101 distinct 2 s anchors**. The 71 K training windows are
`703 subjects × 101 moments`. The InfoNCE problem has ~101 classes. Adding
subjects buys SNR per anchor and **zero** new anchors; adding movies buys
anchors and no SNR. This is almost certainly why every objective saturates at
the same place.

**Design-space placement** (approximate, verify before citing): HBN sits at
A≈101–186, S≈703; THINGS-EEG2 at A≈16,740, S≈10. **Opposite corners, and the
high-A × high-S corner is empty.** Naming that gap is a contribution in itself.

---

## Tier 0 — measurements. No training. Do these first.

These decide what the paper is allowed to claim. All are cheap and all reuse
existing infrastructure.

### E0.1 — Measure `rho1` empirically ★ highest value in the plan
Pairwise inter-subject correlation at 2 s resolution on R6, in three readout
spaces:
- **per-channel sensor space** (the classical ISC number, comparable to
  literature),
- **CorrCA-optimal multivariate** — reuse [corrca_study/](../corrca_study/).
  This is the right number, because a ridge probe on a 512-d embedding can
  exploit exactly this multivariate reliability, not the per-channel one,
- **per band** (δ/θ, α, β) — the δ/θ ≫ α split *is* the mechanism from
  [experiments.md § "Core Problem Identified"](../../experiments.md#core-problem-identified) and should be shown, not
  asserted.

Deliverable: the ceiling column of every table in the paper. Report `r/ceiling`
everywhere, never raw *r* alone.

### E0.2 — Empirical K-averaging curve (validates Spearman-Brown)
Build K-subject averages at matched movie times and probe them.
K ∈ {1, 2, 4, 8, 16, 32, 64, 108}; 108 is the R6 test maximum. Bootstrap over
which subjects are drawn. Run **both** aggregation points:
- **signal-space**: average raw EEG, then encode (classical ERP logic),
- **embedding-space**: encode each subject, then average embeddings (what a
  deployed system does).

The gap between the two curves is itself a result — it says whether the encoder
is approximately linear in the noise. Overlay the predicted `sqrt(R(K))` curve
from E0.1's rho1. **A measured curve landing on a predicted curve is the single
most persuasive figure available to this paper.**

Infra: `retrieval.py::_build_time_pool` and
[`paired.py`](../../eb_jepa/datasets/paired.py) already index across recordings
at matched movie time. Should be a short script, not a new pipeline.

### E0.3 — The (A × S) data-scaling surface, existing recipe
Retrain the jul7 soft-target recipe while subsampling:
- **S** (subjects) ∈ {50, 100, 200, 400, 703}
- **A** (distinct movie anchors) ∈ {13, 25, 50, 101}

Do **not** run the full 20-cell grid. Run an L-shape plus a diagonal (~11 runs):
vary S at full A, vary A at full S, then 3 diagonal points to test separability
of the two axes.

Two protocol requirements, both easy to get wrong:
- **Hold gradient steps constant, not epochs.** Otherwise data scale is
  confounded with optimization budget and the whole surface is uninterpretable.
- **Always evaluate on the full anchor set**, even when training on a subset,
  or retrieval chance levels shift between cells and nothing is comparable.

Deliverable: `r(A, S)` surface + fitted exponents. If `r` scales with `S` but
saturates in `A` (or vice versa), that single sentence is the paper's most
quotable result.

---

## Tier 1 — the method. Cross-subject JEPA as a subject-efficiency claim.

### E1.1 — Head-to-head subject-scaling curves ★ the money figure
Not "cross-subject JEPA beats LeJEPA." Instead: **both objectives get the same
S subjects; which converts subjects into SNR more efficiently?**

Run `r` vs `S ∈ {50, 100, 200, 400, 703}` for three arms, matched encoder
(`model.*`/`masking.*` are already bit-identical per
[cross_subject/config.yaml](../jepa_pretraining/cross_subject/config.yaml)),
matched gradient steps:
- random init (flat line, the control),
- LeJEPA within-subject masked prediction,
- cross-subject predictive JEPA.

Report the fitted slope `dr / d log S` per arm. The claim becomes *"the
cross-subject target has a subject-scaling exponent of X vs Y for masked
prediction"* — precise, and portable to datasets we don't have.

Prediction to state in advance (falsifiable, which reviewers reward): masked
prediction has slope ≈ 0 or negative, because more subjects means more
fingerprint diversity to model; cross-subject has positive slope tracking
`sqrt(R(K_train))`.

**Do not drop the LeJEPA arm.** Défossez et al. Fig 3C already publishes a
subject-scaling curve (MEG, ~1→96 subjects, rising, no saturation), so a single
rising curve on EEG is a re-derivation, not a finding. The contribution is
strictly the *difference in exponent between objectives*, which needs at least
two arms. With only the cross-subject arm this experiment does not survive
review — see "Related work — the delta" above.

**Three things to report alongside the slopes**, each closing a specific gap
that Fig 3C leaves open and that we can close:

1. **Slopes against the measured ceiling**, i.e. plot `CC_norm` vs `S`, not raw
   `r`. Their y-axis has no ceiling on it; ours does after E0.1. This converts
   "more is better" into "how much of the attainable signal each objective
   extracts per subject."
2. **The extrapolated subject count to reach a target `CC_norm`.** If the
   cross-subject exponent holds, state how many subjects each objective needs to
   reach, say, `CC_norm = 0.8`. This is the number a practitioner planning an
   EEG study actually wants, and it is the concrete form of the EEG-vs-MEG
   argument: MEG cannot reach that S, EEG can.
3. **An EEG-vs-MEG contextual row.** Défossez's Brennan/Broderick EEG numbers
   (~5 % top-1) are the honest reference point for what single-trial EEG does
   without cross-subject aggregation. Not a baseline we run — a citation that
   sets the scale.

### E1.2 — K_train partner-averaging sweep
Average the cross-subject target over K partners, K ∈ {1, 2, 4, 8, 16}. Overlay
the predicted `sqrt(R(K))` trend. Distinct from E0.2 (which aggregates at
*test*); together they populate the 2-D (K_train, K_test) law from consequence
(b).

### E1.3 — The double dissociation
On every checkpoint, probe **both** stimulus features and **subject traits**
(`evaluation/probe_eval.py`). The claim is stimulus readout ↑ *while*
subject-trait readout ↓ from a single change to the prediction target. One
table. This is the mechanism evidence and it is not optional.

### E1.4 — Collapse diagnostics as a reported result, not a footnote
At −24 dB the MSE gap between "recover the shared response" and "emit the
constant mean" is small, and SIGReg constrains only the *marginal* embedding
distribution — so "targets spread isotropically, predictions go constant" is a
valid optimum it cannot detect. Log `pred_loss_gap`, `pred_var_ratio`,
`target_var_across_batch` **from step 1, not epoch 100**. `pred_loss_gap` is ~0
at init by construction; its opening is the only evidence anything
time-specific was learned. Rescue order if it flattens: `pred_target_mode=all`
→ sigreg 0.05→0.1 → `within_subject_weight=0.3` → `context_mask_mode=full`.

---

## Tier 2 — reviewer-proofing. Cheap, and each closes a specific attack.

### E2.1 — Subject-trait probe on the existing LeJEPA/Laya/random checkpoints
Closes: *"your JEPA training is just broken."* Without it the jul10 table
(pretrained ≤ random init) reads as a bug; with it, it is the thesis. Cheap,
checkpoints already exist. **Blocking.**

### E2.2 — Matched rerun of the `*_from_jepa_checkpoint` CLIP arms
Current configs use `epochs=100, lr=1e-5`
([config_lejepa.yaml:74-77](../clip_pretraining/soft_target_clip_from_jepa_checkpoint/config_lejepa.yaml#L74-L77))
against jul7's `epochs=400, lr=1e-4`. The apparent collapse (val Δr² 0.019 vs
0.063; scene e→v 2.4–3.0× vs 5.1× chance) is **confounded and unusable as-is**.
Also log matched/dropped key counts from the `strict=False` load. **Blocking if
this result appears in the paper at all** — otherwise cut it.

### E2.3 — Held-out movie
Everything on DespicableMe as well as ThePresent. The known DM ceiling
(loss-independent at ~+0.038 Δr², jul7 §5.3) is the honest caveat and is better
volunteered than discovered.

### E2.4 — Verify recordings vs. subjects
Val has 293 recordings but R5 is ~136 subjects, so there are ~2 recordings per
subject. `paired.py` already enforces distinct-subject partners and prunes
anchors spanning <2 subjects — good. But **S in every scaling curve above must
count subjects, not recordings**, or the aggregation gain is inflated by
within-subject repeats. Confirm the counts before plotting anything.

---

## What each tier gets you

| run | without it | with it |
|---|---|---|
| **E0.1 ✅ done** | "*r*=0.15" invites "that's small" | **"83–88 % of a measured ceiling"** |
| **E0.2 ✅ done** | Spearman-Brown is an assumption | **validated to 3–20 %, erring conservatively; aggregation measured at 2.6×** |
| E0.3 | 4 objectives tie, unexplained | tie is *explained* by A=101 |
| E1.1 | a curve Défossez Fig 3C already published | an exponent **compared across objectives** |
| E1.3 | mechanism is asserted | mechanism is measured |
| E2.1 | jul10 table is a liability | jul10 table is the thesis |

**Minimum viable paper: E0.1 + E0.2 + E0.3 + E2.1.** A complete scaling-law
contribution using only the existing recipe and existing checkpoints — no new
method, and no exposure to cross-subject JEPA failing to train. Framed on the
EEG-vs-MEG asymmetry it stands alone: *the deployable modality is the one that
fails single-trial, and the only axis it can scale is subjects; here is the
ceiling, here is how far four objectives get, here is the exchange rate.*

E1.x upgrades it to measurement-plus-method — but **only if E1.1 keeps both
objective arms**. A single rising curve is Fig 3C on EEG.

**Order:** E2.1 → E0.3 → E1.1 → E1.2/E1.3 → E2.3.
E0.1 and E0.2 are complete ([`RESULTS.md`](RESULTS.md)); E2.1 is cheap and
gates whether the jul10 below-random table can be used at all.
