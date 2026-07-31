# snr_scaling — RESULTS

Measured noise ceilings for stimulus-locked EEG readout on HBN movie-watching.
Establishes what fraction of the achievable signal the CLIP/JEPA checkpoints in
[`soft_target_clip`](../clip_pretraining/soft_target_clip/RESULTS_jul7.md) and
[`cs_aligner`](../clip_pretraining/cs_aligner/RESULTS_jul9.md) actually capture,
so results can be reported as *r*/ceiling rather than raw *r*.

Forward-looking experiment design lives in [`PLAN.md`](PLAN.md); this file is
the record of what has been measured. Commits `b4492fe`, `6dd992d`.

---

## TL;DR

1. **The single-trial ceiling on R5 val is *r* = 0.313.** Cross-validated
   CorrCA gives a combined 5-component reliability of `rho1 = 0.0982`; the
   Spearman-Brown / Schoppe ceiling is `sqrt(rho1) = 0.313`.
2. **The best checkpoint sits at 48% of it** (Schoppe CC_norm = 0.484 for
   from-scratch soft τ=0.05; 0.547 for REVE warm-start; 0.168 for random init).
3. **"We are already at the ceiling" is FALSIFIED.** There is ~2× single-trial
   headroom, and 0.313 is itself a lower bound — only 5 CorrCA components were
   computed. The saturation of four objectives within ±0.003 Δr²
   ([jul7 §4.1](../clip_pretraining/soft_target_clip/RESULTS_jul7.md)) is
   therefore **not** a ceiling effect and needs a different explanation.
4. **Three independent estimators agree** to within ~0.005 across all bands —
   Sahani-Linden explainable variance, Hsu-style split-half, and mean pairwise
   ISC. The ceiling is a property of the data, not of one estimator.
5. **δ/θ ≫ α is confirmed but 2–5× smaller than the repo has been citing.**
   Measured best-channel δ/θ = 0.062 vs α = 0.029 (2.1×);
   [`experiments.md` § "Core Problem Identified"](../../experiments.md#core-problem-identified) cites 0.10–0.28 vs
   <0.05. **Quote the measured numbers.**
6. **Test-time subject aggregation is worth far more than any objective.**
   Ceiling rises 0.313 → 0.722 at K=10 → 0.919 at K=50. Single-trial work is
   capped at ~2×; aggregation offers ~3×.
7. **Subjects buy SNR; seconds do not.** Measured directly on Top-K retrieval
   (§2.6): scene Top-1 goes 0.217 → 0.571 with subject averaging, but pooling
   more seconds *within* a subject gains ~0.015 and then degrades — and the
   oracle-segment upper bound confirms that is the ceiling, not a blocking
   artifact. This closes the free alternative to collecting subjects.
8. **The retrieval metric hides a collapse** (§2.7): the model answers "scene 0"
   — the black title card — for 40.6 % of test windows while it is correct 5.0 %
   of the time. Report the modal-answer share alongside Top-K.
9. Two artifacts found and fixed that would have corrupted the headline: a flat
   reference channel faking ISC ≈ 0.4, and CorrCA component 1 failing to
   generalise despite the largest in-sample eigenvalue.

---

## §1. Setup

### 1.1 What is being measured

Model the response of subject *s* at movie moment *t* as
`x_s(t) = g(t) + n_s(t)`, with `g` the shared stimulus response and `n` the
subject fingerprint plus noise, independent across subjects. Then

```
rho1 = Var(g) / (Var(g) + Var(n))
R(K) = K*rho1 / (1 + (K-1)*rho1)          (Spearman-Brown)
r(K) <= sqrt(R(K)) * corr(g, y)  <=  sqrt(R(K))
```

Because a frozen V-JEPA-2 movie feature is a *deterministic* function of the
stimulus, it carries no measurement noise, so `sqrt(R(K))` bounds the
correlation achievable by **any encoder under any objective**. Two facts make
this tractable: mean pairwise inter-subject correlation estimates `rho1`
exactly under this decomposition (no extra assumptions), and Schoppe's
`CC_max = sqrt(N*SP/(N*SP+NP))` is algebraically the same quantity.

### 1.2 The trial-axis caveat — read before citing

Sahani & Linden (2003), Hsu, Borst & Theunissen (2004) and Schoppe et al.
(2016) all assume **repeated presentations of the same stimulus to the same
subject**. HBN gives exactly one presentation per subject, so we use
**subjects as the repeat axis**: the "signal" is the cross-subject-shared
response, and the subject fingerprint is absorbed into the noise term.

That is the correct choice for bounding *stimulus* decoding — subject-specific
structure is precisely what must not count — and it makes these estimators
commensurable with ISC. But it means these are **cross-subject, not
cross-trial, ceilings**, and they are consequently far lower than a
within-subject repeated-trial ceiling.

### 1.2b Comparability to the neural-decoding literature — checked against sources

An earlier draft of this file claimed that NICE/THINGS-EEG, Défossez et al.
and Benchetrit et al. are all "trial-averaged image decoding" and therefore not
comparable. **That was wrong for two of the three.** Verified against the
papers:

| work | modality | stimulus | repeats | averaged? | training |
|---|---|---|---|---|---|
| NICE / THINGS-EEG (Song et al.) | EEG | discrete images | 4 train / **80 test** | **Yes, explicitly** | **per-subject** |
| Benchetrit et al. 2024 | MEG | discrete images | 12 test | **Both reported** | cross-subject |
| Défossez et al. 2023 | MEG **and EEG** | continuous speech | none used | **No — single-trial** | cross-subject |

- **NICE is genuinely trial-averaged and within-subject.** "We averaged all EEG
  repetitions of one image to ensure the signal-to-noise ratio"; the test split
  is 200 concepts × 1 image × **80 repetitions**. Averaging is load-bearing:
  10 repeats give 9.9 % top-1, stabilising above 13.0 % only after ~25.
  Subject-dependent training gives 13.8 % top-1 vs **6.2 % subject-independent**.
  Not comparable to single-trial cross-subject HBN — the original claim holds
  here.
- **Benchetrit et al. report both.** THINGS-MEG test is 200 images × 12
  repetitions, and Table 1 gives PixCorr 0.069 no-average, 0.079 per-trial
  average, 0.088 per-subject average. Averaging helps modestly; the headline
  does not depend on it. It is not comparable to our numbers chiefly because it
  is **MEG** and event-related, not because of averaging.
- **Défossez et al. is single-trial, and it is speech, not images.** "A 'sample'
  is a 3 s window of brain recording with its associated speech
  representation." No trial averaging. Trained across subjects with a
  subject-specific 1×1 convolution. Listing it as trial-averaged image decoding
  was wrong on both counts.

**Défossez et al. is therefore the closest methodological analogue to this
work** — cross-subject, single-trial, continuous naturalistic stimulus,
contrastive retrieval — and its modality breakdown is directly relevant:

| dataset | modality | top-1 | top-10 |
|---|---|---:|---:|
| Gwilliams | MEG | **41.3 %** | 70.7 % |
| Schoffelen | MEG | 36.8 % | 67.5 % |
| Brennan | **EEG** | **5.2 %** | 25.7 % |
| Broderick | **EEG** | **5.0 %** | 17.7 % |

The widely-quoted "41 % out of 1,000+ segments" is **MEG**. The same
architecture, same objective, same single-trial protocol yields **~5 % top-1 on
EEG** — an ~8× modality gap. That is independent evidence for the thesis here:
the binding constraint is EEG SNR, not the training objective. It is a stronger
argument than the averaging one, and unlike the averaging one it survives
checking.

The honest comparability statement is therefore: **NICE-style numbers are not
comparable because they are trial-averaged and within-subject; MEG results are
not comparable because MEG has far higher SNR than EEG; Défossez's EEG results
ARE comparable in protocol, and they are low — consistent with our ceiling.**

### 1.3 Data and protocol

| | val (R5) | test (R6) |
|---|---:|---:|
| recordings | 293 | 108 |
| unique subjects | 293 | 108 |
| shared anchors | 101 (202.0 s) | 101 |
| channels after flat-channel drop | 128 | 128 |

ThePresent, 2 s non-overlapping windows at 200 Hz. Recordings are joined on
**movie time** (`t_start_recordings`) — the same key
[`paired.py`](../../eb_jepa/datasets/paired.py) and `retrieval.py` use — and
the anchor set is the intersection across recordings, so every pairwise
correlation is computed on identical movie moments. Per-recording per-channel
z-scoring matches `norm_mode=per_recording` in the training configs, so the
ceiling is commensurable with the probe numbers it bounds.

Three readout spaces, because `rho1` depends on what the downstream model may
read: per-channel waveform, per-channel per-band window-level log power (the
probe's unit of analysis), and cross-validated CorrCA (the multivariate
reliability a ridge on a 512-d embedding can exploit).

---

## §2. Results

### 2.1 The ceiling

`rho1` candidates on R5 val, and the ceiling each implies:

| estimate | rho1 | K=1 | K=10 | K=50 |
|---|---:|---:|---:|---:|
| waveform, mean over channels | 0.0096 | 0.098 | 0.297 | 0.571 |
| waveform, best channel | 0.0228 | 0.151 | 0.435 | 0.734 |
| α band, best channel | 0.0289 | 0.170 | 0.479 | 0.773 |
| δ/θ band, best channel | 0.0620 | 0.249 | 0.631 | 0.876 |
| CorrCA, best held-out component | 0.0450 | 0.212 | 0.566 | 0.838 |
| **CorrCA, combined 5 components** | **0.0982** | **0.313** | **0.722** | **0.919** |

**Quote the combined CorrCA row.** A ridge probe on a 512-d embedding pools
multiple components rather than reading one, so the single-component figure is
a lower bound. Components are combined by SNR addition
(`SNR_i = rho_i/(1-rho_i)`, `R = ΣSNR/(1+ΣSNR)`).

### 2.2 Where the checkpoints sit

Against the val ceiling of 0.313, using Schoppe CC_norm = CC_abs / CC_max.
Observed *r* from [jul7 §4.3](../clip_pretraining/soft_target_clip/RESULTS_jul7.md):

| checkpoint | test mean *r* | CC_norm | % of ceiling |
|---|---:|---:|---:|
| random init | 0.0527 | 0.168 | 17% |
| fresh500 scene_clip | 0.1328 | 0.424 | 42% |
| vanilla CLIP 400 ep | 0.1459 | 0.466 | 47% |
| **soft τ=0.05 400 ep** | **0.1517** | **0.484** | **48%** |
| **REVE warm-start** | **0.1715** | **0.547** | **55%** |

**~2× headroom remains at K=1**, and this is a lower bound on the headroom
since only 5 components were computed.

### 2.3 Three estimators agree

R5 val, best channel per band. `SL(std)` is Sahani-Linden after per-subject
z-scoring; `split-half` is the Hsu-style estimator; `ISC` is mean pairwise
inter-subject correlation:

| band | SL(raw) | SL(std) | split-half | ISC | CC_max K=1 |
|---|---:|---:|---:|---:|---:|
| δ/θ | 0.0349 | **0.0609** | **0.0654** | **0.0620** | 0.247 |
| α | 0.0100 | **0.0283** | **0.0286** | **0.0289** | 0.168 |
| β | 0.0041 | **0.0296** | **0.0315** | **0.0302** | 0.172 |
| broadband | 0.0114 | **0.0510** | **0.0549** | **0.0521** | 0.226 |

Three independent routes — variance decomposition, resampling, correlation —
land within ~0.005. They fail in different ways, so agreement is meaningful
evidence rather than a tautology.

**The `SL(raw)` column is not an error; it is a measurement.** Sahani-Linden is
scale-*sensitive* across repeats (a high-variance subject inflates total power
and deflates SP/TP) while Pearson is scale-*invariant*. The gap therefore
quantifies between-subject amplitude heterogeneity — largest in β, where raw
SL is deflated 7× (0.0041 vs 0.0296). After per-subject z-scoring the two
coincide *exactly*: with `Var(x_n)=1`, `SP = (N*Var(mean)-1)/(N-1) = rho`.

**Quote the scale-invariant column.** This repo trains with
`norm_mode=per_recording`, so per-subject amplitude is not information the
encoder can use; charging it as noise would understate the ceiling. Quote
`SL(raw)` only for a model fed un-normalised amplitudes.

### 2.4 Band structure confirms the mechanism — at a smaller magnitude

Window-level log power, mean over channels (best channel in parentheses):

| band | val | test |
|---|---:|---:|
| δ/θ | 0.0396 (0.0620) | 0.0184 (0.0354) |
| β | 0.0164 (0.0302) | 0.0139 (0.0306) |
| α | 0.0139 (0.0289) | 0.0095 (0.0159) |
| broadband | 0.0315 (0.0521) | 0.0120 (0.0188) |

δ/θ exceeds α by 2.8× at the channel mean and 2.1× at the best channel,
consistently across splits. This **confirms** the mechanism asserted in
[`experiments.md` § "Core Problem Identified"](../../experiments.md#core-problem-identified) — stimulus signal lives
in δ/θ, while α dominates variance but is subject-specific.

**It also corrects the magnitude.** That file cites δ/θ ISC of 0.10–0.28 and
α < 0.05. At 2 s window-level log power the true values are 2–5× lower. The
qualitative claim survives; the numbers should be replaced with measured ones
wherever they appear.

### 2.5 Val and test disagree, and val is right

| | val | test |
|---|---:|---:|
| CorrCA fit / scored recordings | 146 / 147 | 54 / 54 |
| combined reliability | 0.0982 | 0.0297 |
| ceiling | 0.313 | 0.172 |
| CC_norm, soft τ=0.05 | 0.484 | 0.880 |

The test ceiling is **downward-biased by fit-set size**: CorrCA there estimates
128-channel covariances from 54 recordings, so its filters generalise poorly
and the held-out ISC understates the true reliability. The ceiling is a
property of the subject population, not of the split it was estimated on — so
**use the val estimate**. Expect a reviewer to notice both numbers; state this
explicitly rather than reporting only one.

### 2.6 Aggregation curves on the retrieval metric — subjects work, seconds do not

§4.2's K_test argument rests on a Spearman-Brown *extrapolation* of the ceiling.
This measures the aggregation effect directly on a downstream metric — e→v
Top-K retrieval from
[`scene_clip_from_checkpoint` §3.10](../clip_pretraining/scene_clip_from_checkpoint/RESULTS.md)
— by pooling more EEG into the **query** while holding the candidate pool
byte-identical. Same centroids, same N, so chance stays `K/N` and every row is
comparable. Implementation:
[`aggregation_curves.py`](aggregation_curves.py).

Three regimes, in increasing order of what they assume:

- **temporal-k** — average `k` consecutive windows within one recording,
  non-overlapping, label = majority group of the block. Assumes nothing about
  where shots or scenes begin: the honest "more seconds of EEG" curve.
- **oracle-seg** — average every window of one (recording, group). Uses
  ground-truth boundaries, so it is the *upper bound* on temporal-k, not a
  deployable number.
- **n-subjects** — average over `n` recordings for the same group (oracle
  segment). One recording = one subject watching the film once.

**VAL** (M = 29,593 windows, 293 recordings), e→v Top-1 / Top-5 / Top-10:

| query | shot (N=49) | scene (N=35) |
|---|---|---|
| 1 window (2 s) — the §3.10 number | 0.167 / 0.439 / 0.586 | 0.217 / 0.526 / 0.675 |
| 2 windows (4 s) | 0.183 / 0.469 / 0.635 | 0.232 / 0.568 / 0.716 |
| 4 windows (8 s) | 0.176 / 0.469 / 0.627 | 0.241 / 0.580 / 0.732 |
| 8 windows (16 s) | 0.183 / 0.385 / 0.534 | 0.208 / 0.503 / 0.666 |
| 16 windows (32 s) | 0.207 / 0.298 / 0.419 | 0.159 / 0.512 / 0.676 |
| **oracle segment** (upper bound on the above) | 0.159 / 0.484 / 0.641 | 0.230 / 0.604 / 0.741 |
| 2 subjects | 0.212 / 0.676 / 0.833 | 0.284 / 0.764 / 0.881 |
| 4 subjects | 0.258 / 0.757 / 0.902 | 0.356 / 0.884 / 0.946 |
| 8 subjects | 0.303 / 0.869 / 0.961 | 0.429 / 0.949 / 0.993 |
| 16 subjects | 0.321 / 0.929 / 0.996 | 0.503 / 0.990 / **1.000** |
| all 293 subjects | 0.347 / **1.000** / **1.000** | **0.571** / **1.000** / **1.000** |

TEST (M = 10,908, 108 recordings) has the same shape at lower absolute values:
scene Top-1 0.149 → 0.159 (oracle segment) → 0.429 (all 108 subjects); shot
Top-1 0.114 → 0.114 → 0.327.

Two findings, with different status:

1. **Subject aggregation: confirmatory.** Scene Top-1 nearly triples (0.217 →
   0.571) and Top-5 saturates at 1.000 by ~16 subjects. This is §4.2's claim
   reproduced on a second, independent metric, and it is *empirical* rather than
   Spearman-Brown-extrapolated — so it partially discharges **E0.2**. Only
   partially: it measures a downstream retrieval metric, not `rho1(K)` or
   `r(K)`, so the extrapolation itself remains unvalidated on its own terms.

2. **Temporal aggregation: new, and negative.** Pooling more seconds from *one*
   subject buys almost nothing — k=2 gains ~0.015 Top-1, k=4 is flat, and
   beyond that Top-5/Top-10 degrade. Critically the **oracle-segment row lands
   at the k=2–4 value**, so this is not a failure of the blocking scheme: it is
   the ceiling on any scheme that averages EEG within a segment.

Finding 2 matters because it closes the cheapest imaginable alternative to
collecting subjects. Every subject already contributes 203 s of EEG, so if
seconds substituted for subjects the SNR would be free. They do not.

**Why — and what is *not* evidence here.** The mechanism is that segments are
short: mean shot 3.7 s (1.9 windows), mean scene 5.6 s (2.8 windows). That mean
is **not independent evidence** for the anchor-count argument in §4.1 — with a
fixed 203 s film it is forced arithmetically, `203 / 54 shots = 3.8 s`,
`203 / 36 scenes = 5.6 s`. "Few anchors" and "short anchors" are one fact in two
costumes. What *is* independent is the skew: median shot 2.9 s, median scene
3.7 s, so **70 % of shots and 56 % of scenes contain two windows or fewer** —
tighter than the mean implies. And the consequence itself had to be measured:
a model with temporal context could have kept improving on longer input
regardless of where labels fall, and it does not.

The deeper reason is that the two axes are not the same operation. Averaging
windows *within* a segment averages different stimulus content, blurring the
thing being identified; averaging *across subjects at the same moment* averages
the same content over independent noise. Only the second accumulates signal —
which is exactly the `x_s(t) = g(t) + n_s(t)` model in
[`PLAN.md`](PLAN.md), now observed end-to-end on a retrieval metric.

**Caveats, in order of how badly they would bite.** The subject rows use
ground-truth segment boundaries to decide what to average, as does oracle-seg —
they answer "if you knew the boundaries". The subject rows have only `N` queries
(35 or 49), one per group; `n ≤ 16` are means over 20 random draws and are
stable, but the all-subjects row is a single draw, so read 1.000 as "no errors
in 35 queries", not as a converged rate. A group's query and candidate are
averaged over the same window set on their respective sides — no information
crosses modalities, but the correspondence is maximally favourable. The k=16
collapse is a labelling artifact, not a signal: a 32 s block spans several
scenes, so its majority label is close to meaningless.

### 2.7 The retrieval metric collapses onto one attractor

Not an aggregation result, but it surfaced from the same runs and is invisible
in the §3.10 tables. At the scene level the model names **scene 0 — the 10 s
black title card — as its top-1 for 40.6 % of test windows and 31.6 % of val
windows**, though scene 0 is the correct answer only 5.0 % of the time (uniform
would be 2.9 %). Shot level is the same story on shot 0: 33.5 % test / 28.7 %
val predicted vs 4.0 % true.

Temporal aggregation makes it slightly **worse** (test scene 41 % → 48 % at
k=16); subject aggregation reduces it but does not remove it (41 % → 26 % test,
32 % → 31 % val). So it is not merely per-window noise — a systematic component
survives averaging over 108 subjects. Anything that reports Top-K on this
checkpoint should report the modal-answer share alongside it; a large share of
first guesses landing on one pool entry is not visible in Top-K.

---

## §3. Two artifacts that would have corrupted the headline

### 3.1 A flat reference channel faking ISC ≈ 0.4

The first run reported `waveform_best_channel` ISC = 0.377 (test) / 0.416
(val), while every other channel sat at ≤0.023.

Channel index 128 is **`Cz`, the EGI HydroCel reference**, stored as exact
zeros (std ≈ 3.45e-29) and **bit-identical across every recording** — the same
denormal residue of the same referencing operation. Per-recording z-scoring
against an epsilon floor (`np.maximum(sd, 1e-8)`) divided that shared pattern
by 1e-8 and amplified it into a signal every subject has in common.

This was not merely one bad table row: **CorrCA maximises across-subject
correlation, so a channel with ISC 0.4 is exactly what it selects**, and the
first run's ceiling was fit on data containing it. Fixed by dropping
numerically-flat channels *before* normalisation, via a ratio test that
generalises to any flat or interpolated channel rather than hardcoding an
index. `tests/test_isc_estimator.py` reproduces the spurious ≈1.0 ISC and pins
the guard.

### 3.2 CorrCA component 1 does not generalise

| split | in-sample eigenvalues | held-out component ISC |
|---|---|---|
| val | [**0.2014**, 0.0582, 0.0421, 0.0178, 0.0135] | [**0.0019**, 0.0450, 0.0402, 0.0121, 0.0056] |
| test | [**0.0701**, 0.0413, 0.0290, 0.0276, 0.0206] | [**0.0021**, 0.0135, 0.0019, 0.0114, 0.0015] |

Component 1 has by far the largest in-sample eigenvalue in both splits and a
held-out ISC of ~0.002 in both. Components 2–3 carry the real reliability.

Consequences: **quoting a CorrCA eigenvalue as an ISC quotes an artifact** —
the optimism gap is up to 100×, which is why filters must be fit and scored on
disjoint subject halves. And index 0 is not a usable summary statistic; the
reported quantities are the max component and the SNR-additive combination.

---

## §4. What this means

### 4.1 The saturation is not a ceiling effect

The strongest prior hypothesis — that four objectives tie because they all sit
at the noise ceiling — is falsified. At 48% of ceiling there is roughly 2×
single-trial headroom that vanilla CLIP, scene_clip, soft-target and
CS-Aligner all fail to capture, despite spanning multi-positive masks,
distillation and distributional alignment.

This **sharpens** the thesis rather than weakening it. The claim becomes *"~2×
headroom demonstrably exists and four independent objectives capture none of
it"*, which points at a cause outside the loss function. The leading candidate
is the anchor-count argument: ThePresent contains only **101 distinct 2 s
stimulus anchors**, so 71 K training windows are `703 subjects × 101 moments`
and the contrastive problem has ~101 classes. Subjects buy SNR per anchor and
zero new anchors. See [`PLAN.md`](PLAN.md) §(c) and experiment E0.3.

### 4.2 Aggregation beats objectives, by a lot

The ceiling at K=1 is 0.313; at K=10 it is 0.722 and at K=50, 0.919. Objective
work is bounded by ~2× improvement; test-time subject aggregation offers ~3×
on top of that, and K_train aggregation in the objective is the mechanism for
approaching the K=1 ceiling. Any scaling law reported for this pipeline should
be **2-D over (K_train, K_test)** — single-trial-only numbers understate the
available effect by ~3×.

§2.6 measures this end-to-end rather than extrapolating it: on Top-K retrieval,
subject averaging takes scene Top-1 from 0.217 to 0.571 and saturates Top-5 at
1.000 by ~16 subjects. The magnitude is consistent with the Spearman-Brown
prediction above, on a metric that never enters the ceiling calculation.

It also rules out the obvious cheap substitute. The aggregation axis has to be
**subjects**, not time: pooling more seconds within one subject gains ~0.015
Top-1 and then degrades, with the oracle-segment upper bound confirming that is
a true ceiling. So "aggregation beats objectives" cannot be cashed in by simply
using longer EEG windows on the subjects already collected — it requires
subjects, which is what makes the (anchors × subjects) surface in §4.1 the
right design space rather than an (anchors × seconds) one.

### 4.3 Report r/ceiling, not r

`r = 0.15` invites "that is small." `CC_norm = 0.48` against a measured,
cross-validated, triple-cross-checked ceiling is a defensible claim with a
literature-standard name attached (Schoppe et al. 2016). Every probe table in
the paper should carry the normalised column.

---

## §5. Reproducibility

Measurement (both splits, ~4 min on one A40; needs no GPU but this account has
no CPU allocation — see the neurolab skill, pitfall #7):

```bash
ssh delta "srun --account=bbnv-delta-gpu --partition=gpuA40x4-interactive \
    --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=16 --mem=64g \
    --time=01:00:00 bash /u/dtyoung/run_isc.sh"
```

Single split directly:

```bash
PYTHONPATH=. uv run --group eeg python experiments/snr_scaling/measure_isc.py \
    --split=val --task=ThePresent --n-components=5 --seed=2025 \
    --output=experiments/snr_scaling/isc_val_ThePresent.json
```

Batch submission (`_submit_isc.py`) packs both splits into one job. Note it
defaults to a **gpu** partition deliberately: `sinfo` lists Delta's `cpu`
partition but this account holds only `bbnv-delta-gpu`, so a `cpu` submit is
rejected.

The CorrCA subject split is seeded; re-runs reproduce the numbers exactly.

Analytic design calculator (no cluster needed):

```bash
uv run --group eeg python experiments/snr_scaling/scaling_calculator.py
```

Aggregation curves (§2.6, §2.7). Runs locally in seconds — it consumes the
shared-space export rather than a checkpoint, so it needs no GPU and no cluster:

```bash
# Once, on Delta: write the shared-space export (see demo/README.md)
python demo/_submit_export.py interactive

# Then locally, per split:
PYTHONPATH=. uv run --group eeg python \
    experiments/snr_scaling/aggregation_curves.py \
    --npz demo/data/demo_val.npz \
    --output experiments/snr_scaling/aggregation_val_ThePresent.json
```

Subject draws are seeded (`--seed`, default 0); the `n ≤ 16` rows are means over
20 draws and move by ~±0.02 across seeds, so quote them to two decimals.

---

## §6. Artifacts

**Results:**
- `isc_val_ThePresent.json`, `isc_test_ThePresent.json` — per-channel waveform
  and band ISC, cross-validated CorrCA, all three literature ceilings,
  Spearman-Brown tables, and CC_norm against the measured probe results.
- `snr_scaling.png` / `.pdf` — ceiling vs K, and the (anchors × subjects)
  design space.
- `aggregation_val_ThePresent.json`, `aggregation_test_ThePresent.json` — §2.6
  temporal / oracle-segment / n-subject curves at shot and scene level, plus the
  §2.7 modal-answer shares. Produced from the shared-space export, so the
  checkpoint provenance travels inside the file.

**Code:**
- [`measure_isc.py`](measure_isc.py) — the measurement.
- [`aggregation_curves.py`](aggregation_curves.py) — §2.6 / §2.7 curves.
  Depends on `demo/export_retrieval_npz.py` for *data* only; it imports nothing
  from `demo/`.
- [`noise_ceiling.py`](noise_ceiling.py) — Sahani-Linden, split-half, Schoppe.
- [`scaling_calculator.py`](scaling_calculator.py) — analytic design calculator.
- [`_submit_isc.py`](_submit_isc.py) — Delta submission.
- [`tests/test_isc_estimator.py`](../../tests/test_isc_estimator.py) (13),
  [`tests/test_noise_ceiling.py`](../../tests/test_noise_ceiling.py) (40) —
  rho1 recovery against planted ground truth, Spearman-Brown agreement with
  direct averaging, the CC_max ≡ Spearman-Brown identity, the standardised-SL
  ≡ ISC identity, and regression tests for both artifacts in §3.

---

## §7. Open

- **E0.2** — empirical K-averaging curve, to validate the Spearman-Brown
  extrapolation the K_test argument now leans on more heavily. **Partially
  addressed by §2.6**: the K curve is measured end-to-end on Top-K retrieval and
  agrees in magnitude, but `rho1(K)` / `r(K)` themselves are still unmeasured,
  which is what would validate the extrapolation on its own terms.
- **E0.3** — the (anchors × subjects) scaling surface, now the leading
  explanation for the objective saturation.
- **E2.1** — subject-trait probes on the LeJEPA / Laya / random checkpoints.
- More than 5 CorrCA components, to tighten the ceiling from below.
- DespicableMe, and a ceiling for the multi-movie regime.

Thesis framing in [`PLAN.md`](PLAN.md) and
[`paper/workshop_outline.md`](../../paper/workshop_outline.md) still describes
this line as "objective-limited vs SNR-limited" and **has not been updated**
for §4.1. That rewrite needs a decision on how hard to lean on the
anchor-count explanation.

---

*E0.1 of [`PLAN.md`](PLAN.md) is complete. E0.2 is partially addressed by §2.6.
Next: E0.2 proper (`rho1(K)`), then E0.3.*
