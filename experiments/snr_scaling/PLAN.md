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

Calculator: [`src/scaling_calculator.py`](src/scaling_calculator.py) →
[`figures/snr_scaling.png`](figures/snr_scaling.png).

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

**Why `A` is in the design at all.** `S` buys repeats of the same stimulus —
the Spearman-Brown axis E0.1/E0.2 already measured, where each subject shrinks
noise on an anchor's target. `A` buys *new stimulus conditions*: it reduces no
noise anywhere, it adds constraints and enlarges the covered region of V-JEPA-2
space. Those are different failure modes and `r` alone cannot separate them.
The thesis — EEG is SNR-limited, subjects are the currency — survives **only**
if `r` climbs in `S` while flattening in `A`. If it is the other way round, the
bottleneck was never noise, and the honest paper is about stimulus coverage.
`A` is the axis that can falsify the claim, which is why it is worth ~4 runs.
It also pre-empts the first reviewer objection: 101 anchors against
image-decoding datasets with thousands of distinct stimuli invites "your
ceiling is an anchor-count artifact," and a flat measured `r(A)` answers that
with a curve instead of a paragraph.

**An out-of-distribution `A` point already exists — jul2 multi-movie.**
The same subjects watch both HBN movies (98.9 % of R1–R4 TP subjects also have
DM; §2.9 of [RESULTS.md](RESULTS.md)), so
[`scene_clip_multimovie/autoresearch/`](../clip_pretraining/scene_clip_multimovie/autoresearch/RESULTS_autoresearch_jul2_multimovie.md)
is close to an `A`-only manipulation: **anchors +84 %** (101 → 186), **subjects
+7 %** (703 → 753 union). Result: TP per-domain Δr² went **+0.0505 → +0.0447**,
i.e. *down*, and jul2's own diagnostics already rule out the two easy
rebuttals — 2× wider and 2× deeper both come back tied (+0.0007, +0.0006), so
"the encoder was too small to exploit the new anchors" is tested and rejected.

Two caveats to state rather than let a reviewer find:
- The added anchors are **out-of-distribution** (DM is 25 fps animated vs TP
  live action), so this bounds *"do OOD anchors help?"*, not the
  in-distribution `A` slope. The within-TP sweep is still required.
- Steps are not matched: at ~2× data per epoch, multi-movie ep400 got ~1.6× the
  updates of single-movie ep500, and jul2 found >400-equivalent *degrades*.
  Both effects cut conservatively for the "anchors are not binding" reading.

This also lifts the `A ≤ 101` cap the movie length otherwise imposes: TP+DM
gives a real `A = 186` cell at fixed cohort, labelled OOD.

Together the two close the anchor-count objection from both sides —
*in-distribution anchors have saturated* **and** *out-of-distribution anchors do
not help* — which is a sharper claim than either alone, and converts jul2's
negative-transfer dead end into a result this paper needs.

### E0.4 — Depth as a second axis: does capacity change the subject-scaling curve? (in progress, 2026-08-13)
Every experiment above holds the encoder fixed at `encoder_depth=12` (e03's
architecture) and varies only `S`. kkokate independently trained the same S-grid
— `S ∈ {10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863}`, 3 draws each plus
the full-pool cell — at `encoder_depth=22`
(`/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling`, experiment tag
`e04_reve_scaling`), everything else held bit-identical to e03 except
`patch_size`/`patch_overlap` (200/20 vs 400/0, required by the deeper stack).
28 cells total (2 failed-and-abandoned reruns and 3 smoke-test dirs excluded;
one alternate-seed full-pool replicate, `e04_s1863_a101_seed7`, also excluded
from the main sweep pending a decision on whether to fold it in as a
robustness check).

**Question:** is the subject-scaling exponent from E1.1's framing a property of
the *objective*, or does it also shift with model capacity? A depth-22 encoder
could convert subjects into SNR more efficiently (more capacity to extract the
shared response) or less efficiently (more capacity to overfit
subject-specific fingerprint at low `S`) — the S-sweep at both depths is the
only way to tell them apart. `figures/depth-12-vs-22-subject-scaling.pdf` is
the first cut at this comparison.

**Protocol difference from e03's own published tables, deliberate:** e03's
`RESULTS.md` 2.12/2.13 numbers all evaluate a single fixed epoch (325) across
every cell. For the depth-22 arm we instead run per-cell smoothed epoch
selection — the same `select_epoch` logic as `select_and_probe_e03.py`
(rolling-mean-smoothed `val/clip_scene_auc`, argmax, snap to nearest saved
checkpoint) — via `src/select_e04.py`, writing
`experiments/snr_scaling/e04_selection.json`. Selected epochs range 75-375
across cells, so a single shared epoch would have been wrong for several of
them. If the depth-12 arm is re-compared against these numbers later, the
comparison should ideally use the same per-cell-selection protocol on both
arms rather than epoch 325 vs a selected epoch — currently that is *not* yet
the case, and is a caveat on any depth-12-vs-22 delta until e03 is
re-selected the same way.

Infra added this pass, all depth-22-specific (mirrors the e03 scripts one for
one, see each file's docstring for the exact deltas):
- `src/select_e04.py` — per-cell epoch selection (run on Delta; needs the
  wandb datastore files on local disk).
- `config/config_probe_DM_e04.yaml` — DespicableMe cross-task eval config at
  the depth-22 architecture (repo-relative, not copied into kkokate's
  checkpoint root — this pipeline only *reads* from
  `/work/hdd/bbnv/kkokate/...`, never writes there).
- `submit/_submit_traintest_e04.py`, `submit/_submit_retrieval_e04.py` — job
  submitters, `within` (ThePresent) and `cross` (DespicableMe) presets, both
  val+test for probe, val+test (within) / test-only (cross) for retrieval,
  matching e03's split convention.
- `src/aggregate_e04.py` — collects the raw JSONs into the S-curve tables.

Deliverable: `RESULTS_model_scaling.md` (in progress as of this entry) — full
12-feature Pearson-r probe and time/shot/scene top-{1,5,10} e2v/v2e retrieval,
val and test, plus DespicableMe cross-task transfer, across all 28 depth-22
cells. Output files land in `raw_results/` as `e04_tt_*`, `e04_retr_*`,
`xtask_tt_DM*_e04_*`, `xtask_retr_DMtest_e04_*`.

### E0.5 - Does subject scaling facilitate cross-task transfer?

Question: would model initialized with naturalistically trained data in one task perform better when finetuned in another task compared to training on that new task from scratch?

### E0.6 — Add the val release to the training pool: is the saturation real? ✅ DONE 2026-08-17

**Answer: no — the S=1863 drop was ONE BADLY-OPTIMISED RUN, and E0.4's headline
is retracted.** The decisive test is E0.4's own never-evaluated alternate-seed
replicate of the full-pool cell: that cell has exactly one possible cohort, so
it varies only `meta.seed`, and it moves 0.2531 -> **0.3092** (final training
loss 3.32 -> 2.62). E0.6's three-draw estimate from the larger pool, 0.3076 ±
0.0008, agrees from a different direction. Three draws at S=1863 from the 2156 pool give within-task probe
mean *r* = **0.3097 ± 0.0005**, above E0.4's S=1400 peak (0.2927 ± 0.0085) and
**+0.052 above its single-draw S=1863 (0.2579)**. The decisive contrast is the
same step in S measured two ways: **+0.0130 with three draws vs −0.0347 with
one**, negative on all five readouts in the single-draw case and positive on
three of five (never below −0.003) with three. The pools are near-exchangeable
at matched S=1400 (offset +0.0041, under half of E0.4's own between-draw sd),
so the comparison holds. Full detail in
[`RESULTS_add_val_set.md`](RESULTS_add_val_set.md).

**What it does not show:** S=2156 is again the whole pool, hence one draw, so it
inherits exactly the weakness this experiment was built to expose. Its +0.0052
over S=1863 is inside single-draw range and retrieval is flat across that step.
Read it as *"no evidence of decline at 2156"*, not *"still rising"*. Settling
that needs a pool above 2156 (R11, or DespicableMe-native cohorts) so S=2156
becomes drawable in triplicate.

**Three process lessons worth more than the result:**

1. **A run's `config.yaml` is not the run.** The first sweep was invalid because
   E0.4 warm-starts from `reve_base_eet_init.pth.tar` and uses seed 2026 via
   *CLI overrides that no config file records*. A byte-for-byte config diff came
   back clean while the arms differed in the most important way. Symptom: a
   uniform ~37 % deficit at matched S with a **completely flat** S curve — an
   under-trained encoder makes the axis under study stop mattering, which reads
   as a dramatic finding rather than a bug. Always check
   `wandb/latest-run/files/wandb-metadata.json` → `args`.
2. **This recipe collapses at initialisation on ~10 % of cells, per seed.**
   `ln(batch_size) = ln(64) = 4.1589` is InfoNCE's chance loss, and a failed
   cell sits at 4.02–4.16 **flat for all 400 epochs** — it never leaves the
   collapsed solution, while a healthy cell at the same S escapes within ~40
   epochs. Measured 3/31 here and 3/28 in E0.4, so budget for it. Detect with
   `_submit_e05_addval.py screen`, which reads *training loss* (no eval needed,
   so a dead cell costs no probe GPU) *relative to same-S siblings* (loss scales
   with cohort size: 0.70 at S=10 vs 2.8 at S=1863, so a fixed cutoff misses
   low-S failures). Reseed to 7 — E0.4's own convention — and **cap retries at
   three**, then report n=2 with the failure disclosed. Past a small fixed
   budget, reseeding until a cell trains stops being a fix for a known
   instability and becomes selection on the outcome. Full write-up in
   [`RESULTS_add_val_set.md`](RESULTS_add_val_set.md) § "Failure mode".

3. **Screening catches collapse; only REPLICATION catches partial failure.**
   The instability is a spectrum. Full collapse is easy to detect (loss pinned
   at ln 64 = 4.16, 1.67–3.09× the S-group median). But the retracted S=1863
   cell was a *partial* failure — loss 3.32, i.e. **1.22×**, below the 1.30×
   screen threshold and overlapping the worst healthy cell at 1.19×. The two
   distributions genuinely overlap, so no threshold separates them. A clean
   `screen` means "nothing collapsed", not "everything trained well". **Treat
   every n=1 cell as provisional**, however clean it looks — that is the real
   reason the S=1863 headline survived as long as it did, and the reason E0.6's
   fix had to be replicates rather than a better detector.

4. **Depth comparisons in this experiment are confounded by initialisation.**
   e03 (depth-12) trains from scratch; E0.4/E0.6 (depth-22) warm-start from
   `reve_base_eet_init.pth.tar`. At matched from-scratch init and S=1400 the two
   depths are within 0.007 (0.1918 vs 0.1853, the deeper one lower) while the
   warm start is worth **+0.111** — ~15x the depth difference. So
   `RESULTS_model_scaling.md` § 9's deltas are predominantly the initialisation;
   that section now carries a warning block. A clean depth ablation would hold
   init *and* patchification (400/0 vs 200/20) fixed and does not exist yet.

### Original design notes (E0.6, written 2026-08-14 before the runs)

E0.4's depth-22 arm rises monotonically with `S` through 1400 and then **drops**
at the full-pool S=1863 cell, across every independent readout. The drop cannot
be believed as stated for one structural reason: **S=1863 *is* the pool**
(R1–R4 + R7–R10), so it has exactly one possible draw, while every other point
on the curve is a mean over three. A single-draw cell is exactly where smoothed
epoch selection protects least — and §2.10 documents a prior case where
selection variance manufactured an apparent scaling artifact in this experiment.

**The manipulation is one config line.** Folding R5 — previously the val split,
293 ThePresent recordings — into `data.train_releases` takes the pool from 1863
to 2156. That buys two things a bigger pool alone would not:

1. **S=1863 becomes a drawable cell with three replicates**, because 1863 <
   2156 makes `max_subjects=1863` a real subsample rather than a no-op cap. If
   three independent draws land at the S=1400 level, the drop was draw or
   selection variance; if they reproduce it, it is a population-level effect.
   This is the direct test, and it is the reason to do this rather than simply
   preprocess another release.
2. **A new max-S point at S=2156**, the first observation past the previous
   ceiling of the data.

**Cells (7):** S=1400 × 3 draws (pool-stitch calibration), S=1863 × 3 draws
(the money cells), S=2156 × 1 draw (whole pool). Everything else held to E0.4:
depth-22, `meta.seed=2025`, 400 epochs, `epoch_size=703` so every cell runs the
same 4400 steps, `max_anchors=101`, `save_every=25`.

**Three protocol consequences, all of them costs of putting R5 in train:**

- **Test split only.** A val number for these cells would be measured on data
  their encoder saw. R6 is untouched by both arms and is the only split on
  which they are comparable.
- **Fixed epoch 375, not per-cell selection.** `val/clip_scene_auc` is now
  in-sample, so E0.4's smoothed-argmax selection is unavailable. 375 is where
  4 of 4 high-S E0.4 cells' own selection landed (S=1400 d11/d33 and S=1863 at
  375; S=1400 d22 at 350), so the arms are matched to within one 25-epoch
  interval. `save_every=25` keeps the grid on disk for later re-evaluation.
- **The probe head-fit pool is deliberately NOT changed** — both eval configs
  still declare `[R1..R4, R7..R10]`, so every cell in both arms fits its ridge
  head on the identical 1863/1832 recordings and only the *encoder's* cohort
  varies.

**The result that gates the rest:** the S=1400 calibration. Draws nest within a
pool, but adding R5 changes the list being permuted, so the two arms' S=1400
cohorts are not the same subjects. §2.11 already measured this kind of pool
effect once (R7–R10 subjects worth 8–15 % less than R1–R4 at matched count), so
a non-zero offset here is not hypothetical — and if there is one, every
S=1863/2156 comparison must be read through it rather than at face value.

Infra: `config/clip_pretrain_e05_addval.yaml` (frozen training config — E0.4's,
plus R5), `submit/_submit_e05_addval.py` (training + `sync`/`verify`),
`--arm addval` on `submit/_submit_traintest_e04.py` and
`submit/_submit_retrieval_e04.py` (both restrict to test and pin epoch 375),
`src/write_results_add_val_set.py`.

Deliverable: `RESULTS_add_val_set.md`. Checkpoints land in
`/work/hdd/bbnv/dtyoung/eb_jepa/e05_addval`; raw JSONs in `raw_results/` under
`e05_*` slugs, which cannot collide with the E0.4 arm's `e04_*` ones.

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
