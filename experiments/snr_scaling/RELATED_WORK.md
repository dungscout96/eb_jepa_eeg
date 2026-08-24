# RELATED_WORK — this sweep vs Banville et al. 2025 on subject scaling

Banville, Benchetrit, d'Ascoli, Rapin & King, *Scaling laws for decoding images
from brain activity*, arXiv:2501.15322 (Meta AI, 2025).

Their headline: decoding performance "scales log-linearly with the amount of
brain recording", but "this scaling law primarily depends on the amount of data
per subject" and "little decoding gain is observed by increasing the number of
subjects" — indeed "scaling the number of subjects yields limited improvement in
most cases and may even lead to a decrease in performance."

This experiment reports the opposite on the subject axis (`RESULTS.md` §2.11,
`RESULTS_model_scaling.md` §1). This file sets out why, and what the difference
does and does not license claiming.

**Numbers attributed to this sweep are generated from `raw_results/` by
`src/write_related_work.py`. Numbers attributed to Banville et al. are
hand-transcribed from the paper and marked (P).** Re-run the script after any
change to the sweep.

---

## TL;DR

1. **The two results may not conflict at all.** Their subject-scaling claim
   rests mainly on Grootswagers2022 — 48 EEG subjects — where doubling 24 → 48
   gave **+0.004 R, p = 0.49** (P). Our own curve over the same window
   (interpolated 20 → 50) gives **+0.0089** mean(12) r per doubling.
   Different metric spaces, but the same order of magnitude: small, and not
   resolvable at their n. Our claim is not that a doubling helps a lot.
2. **It is that the slope survives another 4.8 doublings.**
   Log-linear fit S=10..1400: **+0.0233 r per doubling, R² =
   0.951** on the within-task test probe — cumulatively
   0.1309 → 0.2927 (2.24x). Their subject axis
   stops at 48; ours runs 29x further.
3. **Neither design can resolve a single doubling; ours does not have to.**
   With 3 draws per cell the smallest attainable two-sided permutation p is
   0.10, so *no* adjacent-S step here reaches p<0.05 either. What carries
   the result is monotonicity across a 9-point ladder: 5 of the
   7 readouts increase on **every one** of the 8 intervals
   (sign-test p = 0.00391 each), across families that share no fitting
   step (§2.4). Cross-task zero-shot retrieval is the flat one, and serves as
   the negative control (§4).
4. **Three design differences plausibly make our slope genuinely steeper**
   (§3): we have no subject-specific layers, our subjects are repeats of one
   shared stimulus timeline, and our readout targets the inter-subject-shared
   component of the response.
5. **We agree with them where our data speaks to their regime** (§4):
   cross-task zero-shot alignment is flat in subjects, v→e retrieval is at
   chance at every S, and our S=1863 cell drops.

---

## 1. What each study actually varied

Hand-transcribed from the paper (P) alongside this sweep.

| | Banville et al. 2025 (P) | this sweep (e03/e04) |
|---|---|---|
| datasets | 8 public, 4 devices (EEG / MEG / 3T fMRI / 7T fMRI) | 1 (HBN), EEG only |
| subjects | 84 total; largest single cohort **48** (Grootswagers2022, EEG) | **1863** recordings, one protocol |
| data per subject | 0.92 h (Grootswagers) to **40.1 h** (Allen2022, 7T) | ~3.4 min, one movie — **fixed, not a free axis** |
| stimulus | rapid discrete natural images | continuous naturalistic movie, **same timeline for every subject** |
| target | DINOv2-giant 1536-d embedding, single trial, no test-time averaging | 12 stimulus features (RidgeCV probe) + V-JEPA-2 embedding retrieval |
| metric | feature-wise Pearson R; top-5 retrieval; reconstruction | mean(12) Pearson r; e2v/v2e top-{1,5,10} over pools of N=101/49/35 |
| subject conditioning | **subject-specific linear layers** (M/EEG and fMRI) | **none** — encoder is subject-agnostic, `norm_mode: per_recording` only |
| what the subject axis touches | the whole end-to-end decoder | **pretraining only** — the probe head is refit on the full 1863-recording pool at every S |
| replicates | varies; the 24→48 comparison is a t-test | 3 nested subject draws per S cell (except S=1863) |

The two designs overlap in exactly one corner — many subjects × little data
each — and their benchmark samples that corner with one dataset (Grootswagers,
48 × 0.92 h). That is also the only place their subject axis has room to move.

---

## 2. Measured slopes

All from `raw_results/`, per-cell smoothed-selection epochs (not epoch 325);
see `RESULTS_model_scaling.md` §Methodology. Column keys:

- **WT probe test** — within-task (ThePresent) probe, mean(12) Pearson r, test split
- **WT probe val** — within-task probe, mean(12) Pearson r, val split
- **WT retr scene** — within-task retrieval, scene-pool e2v top-1, test split
- **WT retr time** — within-task retrieval, time-pool e2v top-1, test split
- **XT probe test** — cross-task (DespicableMe) probe, mean(12) Pearson r, test split
- **XT retr scene** — cross-task zero-shot retrieval, scene-pool e2v top-1, test split
- **d12 probe test** — depth-12 (e03) within-task probe, mean(12) Pearson r, test split, epoch 325

Rows marked `*` involve S=1863, which is a **single draw with no replicate** and
drops below S=1400 on every readout; `RESULTS_model_scaling.md`'s headline flags
it as possibly a checkpoint-selection artifact pending a 3-draw replicate. It is
excluded from every fit below and shown only for completeness.

### 2.1 Absolute values (mean ± sd over draws)

| S | WT probe test | WT probe val | WT retr scene | WT retr time | XT probe test | XT retr scene | d12 probe test |
|---|---|---|---|---|---|---|---|
| 10 | 0.1309 ±0.0116 | 0.1666 ±0.0068 | 0.0567 ±0.0020 | 0.0147 ±0.0010 | 0.1182 ±0.0088 | 0.0674 ±0.0115 | 0.1078 |
| 20 | 0.1391 ±0.0048 | 0.1821 ±0.0031 | 0.0603 ±0.0016 | 0.0150 ±0.0008 | 0.1298 ±0.0058 | 0.0770 ±0.0139 | 0.1101 |
| 50 | 0.1509 ±0.0060 | 0.1989 ±0.0049 | 0.0715 ±0.0030 | 0.0181 ±0.0012 | 0.1463 ±0.0025 | 0.0655 ±0.0016 | 0.1178 |
| 100 | 0.1742 ±0.0051 | 0.2231 ±0.0122 | 0.0769 ±0.0084 | 0.0234 ±0.0027 | 0.1719 ±0.0043 | 0.0646 ±0.0041 | 0.1298 |
| 200 | 0.1978 ±0.0029 | 0.2520 ±0.0058 | 0.0863 ±0.0075 | 0.0307 ±0.0019 | 0.1876 ±0.0024 | 0.0686 ±0.0058 | 0.1532 |
| 400 | 0.2286 ±0.0038 | 0.2864 ±0.0054 | 0.1128 ±0.0051 | 0.0463 ±0.0014 | 0.2112 ±0.0045 | 0.0623 ±0.0065 | 0.1729 |
| 701 | 0.2518 ±0.0127 | 0.3128 ±0.0054 | 0.1334 ±0.0076 | 0.0490 ±0.0007 | 0.2320 ±0.0061 | 0.0710 ±0.0119 | 0.1865 |
| 1000 | 0.2775 ±0.0049 | 0.3336 ±0.0019 | 0.1309 ±0.0121 | 0.0581 ±0.0072 | 0.2460 ±0.0045 | 0.0669 ±0.0098 | 0.1886 |
| 1400 | 0.2927 ±0.0085 | 0.3506 ±0.0047 | 0.1472 ±0.0116 | 0.0652 ±0.0007 | 0.2542 ±0.0072 | 0.0701 ±0.0049 | 0.1918 |
| 1863 * | 0.2579 | 0.3105 | 0.1338 | 0.0523 | 0.2390 | 0.0634 | 0.1901 |

### 2.2 Gain per doubling of subjects

Δ between adjacent S cells, divided by log2 of the S ratio.

| S interval | doublings | WT probe test | WT probe val | WT retr scene | WT retr time | XT probe test | XT retr scene | d12 probe test |
|---|---|---|---|---|---|---|---|---|
| 10 -> 20 | 1.00 | 0.0083 | 0.0154 | 0.0036 | 0.0003 | 0.0116 | 0.0096 | 0.0023 |
| 20 -> 50 | 1.32 | 0.0089 | 0.0127 | 0.0084 | 0.0023 | 0.0125 | -0.0087 | 0.0058 |
| 50 -> 100 | 1.00 | 0.0233 | 0.0242 | 0.0055 | 0.0053 | 0.0255 | -0.0009 | 0.0120 |
| 100 -> 200 | 1.00 | 0.0236 | 0.0289 | 0.0094 | 0.0073 | 0.0157 | 0.0040 | 0.0234 |
| 200 -> 400 | 1.00 | 0.0308 | 0.0344 | 0.0265 | 0.0156 | 0.0236 | -0.0062 | 0.0197 |
| 400 -> 701 | 0.81 | 0.0287 | 0.0326 | 0.0254 | 0.0033 | 0.0257 | 0.0107 | 0.0168 |
| 701 -> 1000 | 0.51 | 0.0501 | 0.0406 | -0.0050 | 0.0178 | 0.0273 | -0.0081 | 0.0041 |
| 1000 -> 1400 | 0.49 | 0.0313 | 0.0349 | 0.0336 | 0.0146 | 0.0168 | 0.0067 | 0.0067 |
| 1400 -> 1863 * | 0.41 | -0.0843 | -0.0973 | -0.0325 | -0.0313 | -0.0368 | -0.0163 | -0.0042 |

The within-task columns do not decay toward the top of the range — the
per-doubling gain over 701 → 1400 is comparable to or larger than at 50 → 100.
That is the whole disagreement: a log-linear subject law that has not begun to
saturate by 1400 subjects.

Two reading caveats. The 400 → 701, 701 → 1000 and 1000 → 1400 intervals are
**less than one doubling wide** (0.81, 0.51, 0.49), so dividing by log2 inflates
their draw noise by up to ~2x — read those three against the fitted slope in
§2.3, not on their own. And `d12 probe test` is a **single draw at fixed epoch
325**, while every e04 column is a 3-draw mean at per-cell selected epochs; the
protocol difference is the same one `RESULTS_model_scaling.md` §Methodology
flags, so compare the d12 column's *shape*, not its level, against the others.

### 2.3 Log-linear fits

| readout | slope / doubling (S=10..1400) | R^2 | slope / doubling (incl. S=1863) | R^2 | total gain S=10->1400 |
|---|---|---|---|---|---|
| WT probe test | 0.0233 | 0.951 | 0.0219 | 0.935 | 0.1618 (2.24x) |
| WT probe val | 0.0264 | 0.971 | 0.0244 | 0.945 | 0.1839 (2.10x) |
| WT retr scene | 0.0130 | 0.928 | 0.0125 | 0.930 | 0.0905 (2.59x) |
| WT retr time | 0.0073 | 0.912 | 0.0068 | 0.900 | 0.0505 (4.43x) |
| XT probe test | 0.0198 | 0.987 | 0.0188 | 0.976 | 0.1359 (2.15x) |
| XT retr scene | -0.0003 | 0.031 | -0.0005 | 0.088 | 0.0027 (1.04x) |
| d12 probe test | 0.0136 | 0.955 | 0.0131 | 0.956 | 0.0840 (1.78x) |

Five readouts fit a log-linear subject law tightly (R² 0.91–0.99) with no
common fitting step between the probe and retrieval families. `XT retr scene` —
cross-task zero-shot alignment — has slope ≈ 0 and R² = 0.03, and is the
negative control: the pipeline does not manufacture a subject slope where there
is none.

### 2.4 What significance a 3-draw design can reach

Exact two-sided permutation test on the 3 draws per cell, primary readout
(within-task test probe, mean(12) r).

| S interval | mean(12) r, lo | hi | delta | exact two-sided perm p | n draws (lo, hi) |
|---|---|---|---|---|---|
| 10 -> 20 | 0.1309 | 0.1391 | +0.0083 | 0.400 | (3, 3) |
| 20 -> 50 | 0.1391 | 0.1509 | +0.0117 | 0.100 | (3, 3) |
| 50 -> 100 | 0.1509 | 0.1742 | +0.0233 | 0.100 | (3, 3) |
| 100 -> 200 | 0.1742 | 0.1978 | +0.0236 | 0.100 | (3, 3) |
| 200 -> 400 | 0.1978 | 0.2286 | +0.0308 | 0.100 | (3, 3) |
| 400 -> 701 | 0.2286 | 0.2518 | +0.0232 | 0.100 | (3, 3) |
| 701 -> 1000 | 0.2518 | 0.2775 | +0.0257 | 0.100 | (3, 3) |
| 1000 -> 1400 | 0.2775 | 0.2927 | +0.0152 | 0.100 | (3, 3) |

Sign test across the 8 intervals: 8/8 increase, one-sided p = 0.5^8 = 0.00391.

With n=3 per cell the minimum attainable two-sided p is
2/C(6,3) = 0.10, so **no adjacent-S step in this design can be
individually significant at 0.05** — the same power problem that produced
Banville et al.'s p = 0.49 at 24 → 48 (P). Note that 7 of the 8 intervals sit
*exactly* at that floor, which means the three draws at the higher S completely
separate from the three at the lower S — the maximum evidence the design can
produce per step.

The evidence here is therefore the ladder, not any single step — and it
replicates across readouts that share no fitting step (the retrieval columns
have no head to fit; see `RESULTS_model_scaling.md` §Methodology):

| readout | intervals up / total (S=10..1400) | sign-test one-sided p |
|---|---|---|
| WT probe test | 8/8 | 0.00391 |
| WT probe val | 8/8 | 0.00391 |
| WT retr scene | 7/8 | 0.03516 |
| WT retr time | 8/8 | 0.00391 |
| XT probe test | 8/8 | 0.00391 |
| XT retr scene | 4/8 | 0.63672 |
| d12 probe test | 8/8 | 0.00391 |

`XT retr scene` is the flat one, as expected from its §2.3 fit.

### 2.5 The window they actually tested

24 and 48 both fall inside our 20 → 50 interval, so the comparable quantity is
our locally-interpolated gain over one doubling there.

| readout | our gain over 1 doubling in the 20->50 window | S=20 | S=50 |
|---|---|---|---|
| WT probe test | +0.0089 | 0.1391 | 0.1509 |
| WT probe val | +0.0127 | 0.1821 | 0.1989 |
| WT retr scene | +0.0084 | 0.0603 | 0.0715 |
| WT retr time | +0.0023 | 0.0150 | 0.0181 |
| XT probe test | +0.0125 | 0.1298 | 0.1463 |
| XT retr scene | -0.0087 | 0.0770 | 0.0655 |
| d12 probe test | +0.0058 | 0.1101 | 0.1178 |

Against their **+0.004 R, p = 0.49** (P). Metric spaces differ (12 coarse
stimulus features vs a 1536-d DINOv2 embedding), so treat this as an
order-of-magnitude check, not a like-for-like delta. The check passes: in the
window they tested, our effect is also small.

---

## 3. Methodological differences, ranked by likely contribution

### 3.1 Subject-specific input layers

Their M/EEG brain module carries **per-subject linear projections**; the fMRI
module a subject-specific linear layer plus a timestep-wise spatial projection
(P). This encoder has none — no subject embedding, no per-subject layer,
nothing in `eb_jepa/architectures.py`, `eb_jepa/clip.py` or `eb_jepa/jepa.py`;
subject-level normalisation is `data.norm_mode: per_recording` alone.

This flips what a new subject buys. With per-subject layers, subject N+1 adds
parameters fit on that subject's data alone — 0.92 h in the 48-subject cohort —
and the shared trunk is *relieved* of having to be subject-invariant. With a
subject-agnostic encoder, additional subjects are the only pressure toward an
invariant representation, and subject-idiosyncratic nuisance averages out
instead of being absorbed into private parameters. Banville et al. attribute
their flat subject axis to "inter-individual differences when it comes to
neural representations" (P) — which is precisely what a subject-agnostic
encoder is forced to marginalise over and a per-subject front-end is not.

This is the difference to name first if the contrast is written up.

### 3.2 Our subjects are stimulus repeats; theirs are not

Every HBN subject watches the same ThePresent timeline. Under the decomposition
in `RESULTS.md` §1.1, `x_s(t) = g(t) + n_s(t)`, an extra subject is literally an
extra draw of the same `g(t)` at the same `t` — the replication axis HBN lacks
*within* subject, recovered across subjects. Subject count and effective SNR on
the shared response are the same quantity here (`RESULTS.md` §2.8: probe r rises
0.289 → 0.742 from K=1 to K=128 test-time subject aggregation).

Their paradigm is rapid discrete image presentation with per-trial targets.
Subjects overlap in image set (THINGS) but not in trial order, ISI or timeline,
so a new subject is a much looser replicate of the same latent signal.

### 3.3 There is no per-subject axis in HBN to compete with

Their strong term runs to 40.1 h/subject (Allen2022, 7T) (P). HBN gives ~3.4 min
of one movie per subject; the per-subject axis is pinned near zero and cannot be
scaled. So this sweep cannot test their law, and their benchmark barely tests
ours.

Where the two axes *can* be traded here, we measured it and the trade goes the
other way: `RESULTS.md` §2.10 finds Δr² flat in anchors (**−0.007** over
A=50→101) and subject-heavy beating anchor-heavy **4/4, median 1.30x** at
matched (subjects × anchors) budget; §2.6 finds that pooling more seconds within
a subject gains ~0.015 and then degrades, against an oracle-segment bound
confirming that is the ceiling rather than a blocking artifact.

### 3.4 The subject axis touches pretraining only

`probe_traintest.py` refits a fresh RidgeCV head on the full training pool —
1863 recordings for ThePresent, 1832 for DespicableMe — regardless of the S its
encoder was pretrained with. So `WT probe test` isolates *how good a
representation learned from S subjects is*, with decoder fitting data held
constant. Their models are trained end-to-end on the subject set, so subject
count is confounded with the decoder's own fitting data.

Representation learning is where subject diversity should pay off; a supervised
decoder with a per-subject front-end has structurally less to gain from it.

### 3.5 Readout granularity — and this experiment already shows it decides the answer

Our targets are 12 coarse stimulus features (luminance, contrast, motion
energy, ...) plus retrieval over pools of N=101/49/35. Those are the high-ISC,
inter-subject-shared components. Theirs is a 1536-d DINOv2 embedding per single
image trial — fine-grained identity, far more subject-idiosyncratic.

`RESULTS.md` §2.11 makes the point internally, without needing a second paper:
over 701 → 1863 the scalar probe gains +2–5% while e→v scene retrieval gains
+18–26%. **Saturation is a property of the readout, not of the data.** Their
metric sits at the granularity where the smallest subject effect is expected.

### 3.6 Cohort homogeneity

HBN is one acquisition protocol and one 129-channel montage across all 1863
recordings; their benchmark spans 8 datasets and 4 devices, and their stated
mechanism for the flat subject axis is inter-individual variability (P). The
same effect is visible here but small: `RESULTS.md` §2.11 finds R7–R10 subjects
worth **8–15% less** than R1–R4 at matched count — enough to slow the slope, not
to kill it.

### 3.7 Architecture interacts with the subject exponent

The fitted slopes in §2.3 differ by architecture: **+0.0233 r per
doubling at depth-22 vs +0.0136 at depth-12** (1.7x).
Read that as indicative only — the two columns use different checkpoint-selection
protocols (§2.2 caveat). The matched-epoch version is
`RESULTS_model_scaling.md` §9.8: depth-22 beats depth-12 on 35/35 probe rows,
and the advantage *grows* ~4–5x from S=10 (Δ≈0.02–0.03) to S=1400 (Δ≈0.08–0.10)
— the deeper encoder converts subjects into probe r more efficiently, i.e. the
subject-scaling exponent is architecture-dependent. Banville et al. explicitly
leave this open: "the systematic exploration of increasingly large architectures
remains an open question" (P), and did not vary model scale against subject
scale. A subject axis measured at one capacity does not transfer to another.

---

## 4. Where this sweep agrees with them

State these when writing up the contrast; they bound the claim.

- **Cross-task zero-shot alignment is flat in subjects.** `XT retr scene` above,
  and `RESULTS_model_scaling.md` §4.1–4.3: time-pool e2v@1 moves 0.99x → 1.52x
  chance from S=10 to S=1400, noise elsewhere. Subjects do not buy transferable
  alignment.
- **v→e retrieval is at chance at every S tested.** `RESULTS.md` §2.13: 1.00x at
  S=10 and 1.00x at S=1863. 186x more subjects moves it not at all — the
  modality-gap failure is not a data-quantity problem.
- **Our top cell drops.** S=1863 falls below S=1400 on every readout, which is
  their "may even lead to a decrease in performance" (P) — if it survives the
  3-draw replicate flagged in `RESULTS_model_scaling.md`.
- **Their per-subject law is untested here** and nothing in this experiment
  contradicts it.

---

## 5. What the contrast licenses claiming

Not "scaling subjects works for EEG". The defensible form is narrower:

> On a shared continuous stimulus, with a subject-agnostic encoder and a readout
> targeting the inter-subject-shared response, decoding quality scales
> log-linearly in subject count to at least 1400 subjects
> (+0.0233 mean(12) r per doubling, R² = 0.951) — over a
> range 29x beyond the largest cohort in Banville et al. 2025, and on
> an axis their datasets cannot vary. It does not buy cross-task zero-shot
> alignment, and it says nothing about their hours-per-subject law, which HBN
> cannot test.

Two claims to avoid:

- "Banville et al. are wrong about subjects." In the window they tested, our own
  curve gains +0.0089 per doubling (§2.5) — we would likely have concluded
  the same thing from 48 subjects.
- "The subject axis saturates / does not saturate" without naming the metric.
  §2.2 and `RESULTS.md` §2.11 both show the answer flips between probe and
  retrieval.

---

## Provenance

- Generated by `src/write_related_work.py` from `raw_results/`; loaders are
  imported from `src/write_results_model_scaling.py` so both docs read the
  sweep identically.
- Sweep, selection protocol and caveats: `RESULTS_model_scaling.md`.
  Within-ThePresent record, anchors-vs-subjects trade and K-averaging:
  `RESULTS.md`. Cross-movie transfer: `RESULTS_cross_task.md`.
- Paper: Banville, Benchetrit, d'Ascoli, Rapin & King, "Scaling laws for
  decoding images from brain activity", arXiv:2501.15322 (2025).
  https://arxiv.org/abs/2501.15322
