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
   [`experiments.md:47-70`](../../experiments.md#L47-L70) cites 0.10–0.28 vs
   <0.05. **Quote the measured numbers.**
6. **Test-time subject aggregation is worth far more than any objective.**
   Ceiling rises 0.313 → 0.722 at K=10 → 0.919 at K=50. Single-trial work is
   capped at ~2×; aggregation offers ~3×.
7. Two artifacts found and fixed that would have corrupted the headline: a flat
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
within-subject repeated-trial ceiling. Trial-averaged image-decoding results
(NICE / THINGS-EEG, Défossez et al., Benchetrit et al.) are **not comparable**
to these numbers. This asymmetry is itself an argument for the paper: that
literature buys SNR through repetition, which naturalistic continuous viewing
does not afford.

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
[`experiments.md:47-70`](../../experiments.md#L47-L70) — stimulus signal lives
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

---

## §6. Artifacts

**Results:**
- `isc_val_ThePresent.json`, `isc_test_ThePresent.json` — per-channel waveform
  and band ISC, cross-validated CorrCA, all three literature ceilings,
  Spearman-Brown tables, and CC_norm against the measured probe results.
- `snr_scaling.png` / `.pdf` — ceiling vs K, and the (anchors × subjects)
  design space.

**Code:**
- [`measure_isc.py`](measure_isc.py) — the measurement.
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
  extrapolation the K_test argument now leans on more heavily.
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

*E0.1 of [`PLAN.md`](PLAN.md) is complete. Next: E0.2.*
