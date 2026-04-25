# Input-Space Predictability Decomposition (Littwin Test)

**Date:** 2026-04-25
**Script:** [scripts/input_variance_decomposition.py](../scripts/input_variance_decomposition.py)
**Submission:** [scripts/run_input_vardecomp_delta.py](../scripts/run_input_vardecomp_delta.py)

## 1. Motivation

Littwin et al.'s analysis of JEPA-style objectives predicts that the encoder
representation will allocate variance in proportion not to *static input
variance* per source, but to *context→target predictability* per source. A
subject DC offset has trivial static variance (one scalar per subject) yet
R² = 1.0 from context (constants are perfectly predictable); pure noise
has high static variance but R² ≈ 0. The two quantities coincide in many
cases but diverge in informative ones.

Earlier sections of [docs/variance_decomposition_report.md](variance_decomposition_report.md)
characterize the *representation-side* variance hierarchy (η²_subj,
η²_stim, principal angles, effective rank). To make the JEPA mechanism
claim tight, we need the *input-side* counterpart — both static variance
and predictability decomposition under the same four preprocessing
conditions used in training.

## 2. Methodology

For each clip read as `[n_windows, C, T]` float EEG, we compute three
per-channel feature vectors:

- `X_full`: feature on the whole clip → static variance decomposition.
- `X_ctx`: feature on the first half along time → context.
- `X_tgt`: feature on the second half → target.

Then per condition:

1. **Static decomposition** of `X_full` into subject / stimulus / residual
   via the same nested-ANOVA used on representations:
   `Var_total = Var_subject + Var_stimulus + Var_residual`.
2. **OLS regression** `X_tgt ≈ W · X_ctx + b` on the flat `[S·K, D]`
   matrices.
3. **Decomposition of the prediction** `ŷ = W·X_ctx + b` into the same
   three components → **predictability**: `Var_k(ŷ) / Var_total(ŷ)`.
4. **R² per source**: `1 − Var_k(ε) / Var_k(X_tgt)` where `ε = X_tgt − ŷ`.

Two predictability measures are reported per source k:

- **`predictability_k = Var_k(ŷ) / Var_k(X_tgt)`** — fraction of source-k
  variance captured by the prediction. This is the Littwin-relevant number.
- **`r2_k = 1 − Var_k(ε) / Var_k(X_tgt)`** — sanity-check identical to the
  above only when `Cov_k(ŷ, ε) = 0` (which OLS sample-orthogonality does
  not guarantee at the source-decomposed level).

### Conditions (mirroring training arms)

| condition | `norm_mode` | CorrCA | training counterpart |
|---|---|---|---|
| `raw_global` | global | no | SIGReg/VICReg baselines |
| `per_rec` | per_recording | no | retrain_perrec arm |
| `corrca_global` | global | yes | ablation (untested in training) |
| `corrca_per_rec` | per_recording | yes | CorrCA training |

### Features tested

| feature | dimension | EEG interpretation | output dir |
|---|---|---|---|
| `rms` | C | per-channel RMS amplitude | `outputs/input_variance_decomp/` |
| `bandpower` | 5×C | log-power in 5 standard bands (δ/θ/α/β/γ) via Welch PSD | `outputs/input_variance_decomp_bp/` |
| `alpha_phase` | 2×C | (cos φ, sin φ) of 8-13 Hz Hilbert phase at clip midpoint | `outputs/input_variance_decomp_phase/` |

All runs: K=32 clips/rec, val split, 293 subjects, seed=2025.

## 3. Results

### 3.1 RMS (channel amplitude)

| condition | static η²_subj | predicted η²_subj | R²_total | R²_subj | R²_stim |
|---|---:|---:|---:|---:|---:|
| raw_global | 0.632 | 0.651 | 0.715 | 0.808 | 0.721 |
| per_rec | **1.000*** | 0.350 | 0.614 | 0.674 | 0.410 |
| corrca_global | 0.478 | 0.509 | 0.348 | 0.444 | 0.240 |
| corrca_per_rec | 0.723 | 0.626 | 0.388 | 0.413 | 0.453 |

`*` Per-rec norm forces channel RMS toward 1 by construction → static
decomposition is degenerate (all variance trivially "between subjects").
RMS is the wrong feature to assess subject-vs-stimulus structure under
per-rec norm.

### 3.2 Bandpower (5 bands × C channels, log-power)

| condition | static η²_subj | predicted η²_subj | R²_total | R²_subj | R²_stim |
|---|---:|---:|---:|---:|---:|
| raw_global | 0.809 | 0.800 | 0.715 | 0.987 | 0.639 |
| per_rec | **0.914** | 0.853 | 0.614 | 0.990 | 0.708 |
| corrca_global | 0.776 | 0.786 | 0.348 | 0.974 | 0.340 |
| corrca_per_rec | 0.884 | 0.825 | 0.388 | 0.981 | 0.406 |

Spectral shape survives per-rec normalization (per-rec controls overall
amplitude but not band ratios), so bandpower gives a non-degenerate
decomposition under per-rec.

**Key bandpower findings:**
- Subject variance is essentially saturated at 80-90% under all
  conditions — every subject's spectral shape is highly recognizable.
- Subject is **near-perfectly predictable** from context (R²_subj 0.97-0.99).
  Within-recording stationarity makes subject identity trivially recoverable.
- Stimulus is partially predictable (0.34-0.71). CorrCA reduces stimulus
  predictability roughly by half — likely because projection from 129→k
  channels eliminates information the OLS could otherwise use.
- Static and predicted source fractions agree (~80% subject in both).
  Bandpower does not give a clean discriminator — under this feature, the
  variance and predictability hypotheses both predict subject-dominated
  representations.

### 3.3 Alpha phase (cos/sin of Hilbert phase at clip midpoint)

| condition | static (subj/stim/res) | predicted (subj/stim/res) | R²_total | R²_subj | R²_stim |
|---|---|---|---:|---:|---:|
| raw_global | 0.051 / 0.005 / **0.944** | 0.346 / 0.003 / 0.651 | **0.047** | 0.373 | 0.059 |
| per_rec | 0.052 / 0.005 / **0.944** | 0.343 / 0.003 / 0.654 | **0.047** | 0.374 | 0.061 |
| corrca_global | 0.034 / 0.004 / **0.963** | 0.038 / 0.003 / 0.959 | **0.001** | 0.001 | 0.008 |
| corrca_per_rec | 0.034 / 0.003 / **0.963** | 0.040 / 0.003 / 0.957 | **0.001** | 0.006 | 0.005 |

**This is the cleanest discriminator we obtained.**

- Phase static variance is **noise-dominated** (94-96% residual,
  3-5% subject, <1% stimulus). Nontrivial total variance, mostly noise.
- Phase predictability is **near zero** (R²_total = 0.047 raw, 0.001
  with CorrCA) — alpha phase wanders pseudo-randomly across the multi-
  second context-target gap.
- Within the tiny predictable fraction, subject share is inflated
  (35% predicted vs 5% static) — alpha frequency differences across
  subjects give a small phase bias that OLS can recover.
- **CorrCA destroys whatever phase predictability exists** (R² drops 50×
  vs no-CorrCA). The CorrCA filters maximize inter-subject correlation,
  which has nothing to do with phase consistency, so the projection
  loses phase information.

### 3.4 Predictability comparison across features

| feature | R²_total raw | R²_total per_rec | R²_subj raw | R²_stim raw |
|---|---:|---:|---:|---:|
| RMS | 0.715 | 0.614 | 0.808 | 0.721 |
| Bandpower | 0.715 | 0.614 | **0.987** | 0.639 |
| Alpha phase | **0.047** | **0.047** | 0.373 | 0.059 |

Alpha phase is **~15× less predictable** from context to target than
either RMS or bandpower, despite all three being EEG-canonical features.

## 4. Mapping to representation results

Our existing representation η² decomposition (see
[docs/variance_decomposition_report.md](variance_decomposition_report.md)
§§3, 6, 7, 8) decomposes 64-d embeddings into subject / stimulus /
residual variance buckets. It does **not** decompose into
phase / amplitude / bandpower buckets. The Littwin prediction "encoder
encodes more bandpower than phase" is therefore on an axis our existing
numbers do not directly probe.

What the existing numbers *do* say, consistently with Littwin:

1. **Subject variance dominates representation (~45-74%)** across most
   trained encoders → consistent with input bandpower's near-perfectly-
   predictable subject signal (R²_subj ≈ 0.99).
2. **Stimulus variance is tiny in representations (<4%)** → consistent
   with input bandpower stimulus being only partially predictable AND
   accounting for a tiny share of input variance to begin with.
3. **Per-rec norm doesn't collapse representation subject variance** →
   consistent with input bandpower showing subject variance survives
   per-rec norm (only overall amplitude is removed).
4. **CorrCA-VICReg has high representation η²_subj despite CorrCA's
   stated goal of suppressing subject-fingerprint channels** →
   consistent with input bandpower showing 88% subject under
   `corrca_per_rec` even after the spatial projection.

These are Littwin-compatible but not Littwin-specific — they would also
follow from a hypothetical "encoder encodes whatever has most variance"
rule, since variance and predictability happen to align under bandpower.

## 5. The proper Littwin discriminator (open)

Bandpower gives a feature where static variance and predictability
align → the variance hypothesis and the Littwin hypothesis make the
same prediction. **Alpha phase gives a feature where they sharply
diverge**: nontrivial static variance, near-zero predictability.

Littwin's specific prediction: the encoder should encode bandpower-like
structure but NOT phase-like structure, even though phase has comparable
or larger absolute static variance in the input.

To test this on the representation side requires a separate analysis:

1. Extract embeddings `z[s, k]` for all (subject, clip) pairs (val split,
   K=32 — already on disk for many checkpoints).
2. Extract input-feature vectors `φ_phase[s, k]` and `φ_bp[s, k]` from the
   same clips — already in `outputs/input_variance_decomp_phase/*/features.npz`
   and `outputs/input_variance_decomp_bp/*/features.npz`.
3. For each input feature, fit OLS `z ≈ W · φ + b`; report R²_repr_from_φ.
4. Compare `R²_repr_from_bandpower` to `R²_repr_from_phase`.

**Strong Littwin support** would be: `R²_repr_from_bandpower ≫
R²_repr_from_phase` despite phase having comparable or larger input
variance.

This regression takes seconds on already-saved data and would close the
loop on Littwin's mechanism claim.

## 6. Methodological lessons

1. **Feature choice matters enormously.** RMS is degenerate under per-rec
   norm. Bandpower works but doesn't discriminate static vs predictability.
   Alpha phase is the only feature in our set that gives a clean
   variance/predictability divergence, and only because phase is
   pseudo-randomly distributed in time.

2. **`Var_k(ŷ)/Var_k(y)` and `1 − Var_k(ε)/Var_k(y)` differ at the
   source-decomposed level.** OLS gives sample-level orthogonality
   (`Cov(ŷ, ε) = 0`) but not source-decomposed orthogonality
   (`Cov_k(ŷ, ε) ≠ 0` in general). The first measure is the literal
   "fraction of source-k variance captured by prediction" that Littwin's
   theory cares about; the second is a useful sanity check that
   coincides in iid-like settings.

3. **Per-recording z-norm flattens amplitude, not spectral shape.** The
   common intuition that per-rec strips subject identity at input
   applies to RMS but not to bandpower (subject-characteristic alpha
   peaks, beta ratios, etc. survive normalization). This explains why
   subject-trait probes still work on encoders trained with per-rec
   norm — the subject signal is preserved at input.

4. **CorrCA's spatial projection eliminates phase information.** R²_subj
   drops 50× from no-CorrCA to CorrCA under alpha phase. CorrCA's
   filter optimization (maximize inter-subject correlation) removes
   within-subject phase consistency.

## 7. Reproducing

```bash
# Run all 4 conditions, one feature at a time:
python scripts/input_variance_decomposition.py \
    --feature=alpha_phase \
    --output_dir=outputs/input_variance_decomp_phase \
    --n_windows=4 --window_size_seconds=4 --n_clips_per_rec=32 \
    --corrca_filters=/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz

# Reanalyze (recompute decompositions from saved features.npz, no re-read):
python scripts/input_variance_decomposition.py \
    --reanalyze_dir=outputs/input_variance_decomp_phase

# Selftest (synthetic toy cases, no dataset needed):
python scripts/input_variance_decomposition.py --selftest
```

Delta submission for all 4 conditions in one job:
[scripts/run_input_vardecomp_delta.py](../scripts/run_input_vardecomp_delta.py).
Toggle `FEATURE` between `rms`, `bandpower`, `alpha_phase` in that script.
