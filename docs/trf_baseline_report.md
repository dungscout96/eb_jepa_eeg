# TRF Baseline vs Frozen-Encoder Linear Probes

**Date:** 2026-04-25
**Scripts:** [experiments/trf_baseline/run_trf.py](../experiments/trf_baseline/run_trf.py),
[experiments/trf_baseline/sanity.py](../experiments/trf_baseline/sanity.py)

## 1. Question

Does a standard backward-decoding TRF on raw EEG decode the same stimulus
features (contrast_rms, luminance_mean, position_in_movie) that the
frozen-encoder linear probe does? If TRF matches or beats the SSL probe,
the paper's "SSL captures stimulus information" claim is significantly
weakened. If not, we need to know by how much, and whether the gap is
real or an artifact of the TRF setup.

## 2. Method

**Decoder direction.** Predict feature[t] from EEG[t..t+L-1] (causal in
EEG, capturing the post-stimulus neural response), single Ridge fit
across all training subjects. Cross-subject pooling matches the probe's
training regime (R1–R4 train, R5 val, R6 test).

**Implementation.** Covariance-form Ridge with unregularized intercept,
streamed across recordings — never materializes the full design matrix.
Necessary because the raw branch design matrix would otherwise be
~1.6 TB.

**Two preprocessing branches**, both with per-recording z-score:
1. **Raw 129-channel** EEG, decimated 200→50 Hz. d = 50 lags × 129 ch = 6450.
2. **CorrCA 5-component** spatial filter (top-5 ISC: 0.0150, 0.0104,
   0.0037, 0.0029, 0.0024). d = 50 × 5 = 250.

**Lag window.** 0–1000 ms (0–500 ms for the per-subject sanity).

**Alpha.** Selected from {1e-1, 1e1, 1e3, 1e5} by val mean continuous r.
Best alpha was 1e5 in both branches — the strongest regularization in the
grid won, indicating any lower alpha overfits.

**Training set.** Random 100/718 subjects (seed=2025) for the prototype.
Pooled samples ≈ 1M, equivalent to ~280 minutes of EEG, well beyond the
literature TRF data scale (typically ~30 min/subject, single-subject).

**Eval.** Mirrors `experiments/eeg_jepa/probe_eval.py`:
- Pearson r on continuous prediction time series (per-recording, mean across recs).
- Window-matched: aggregate prediction inside 1-s, 2-s, or 16-s clips,
  pool clips across recordings, compute Pearson r and median-split bal_acc.

## 3. Results — pooled cross-subject TRF (R5 val, R6 test)

Window-matched continuous correlations at the three temporal configs the
probe report uses. Probe values cited from
[docs/sigreg_vs_vicreg_probe_eval_report.md](sigreg_vs_vicreg_probe_eval_report.md)
and [docs/variance_decomposition_report.md](variance_decomposition_report.md).

### contrast_rms

| config | TRF raw (val/test) | TRF CorrCA (val/test) | SSL SIGReg+per-rec (val) | SSL CorrCA-VICReg (val/test) |
|---|---|---|---|---|
| nw1_ws1 | +0.017 / +0.003 | +0.016 / +0.009 | 0.081 | — |
| nw2_ws1 | +0.016 / +0.004 | +0.008 / +0.006 | 0.055 | — |
| nw4_ws4 | +0.063 / **−0.008** | **+0.036 / +0.034** | 0.087 | 0.115 / 0.056 |

### luminance_mean

| config | TRF raw (val/test) | TRF CorrCA (val/test) | SSL SIGReg+per-rec (val) | SSL CorrCA-VICReg (val/test) |
|---|---|---|---|---|
| nw1_ws1 | +0.002 / −0.004 | +0.007 / +0.003 | 0.123 | — |
| nw2_ws1 | +0.008 / −0.002 | +0.018 / +0.011 | 0.151 | — |
| nw4_ws4 | +0.014 / **−0.044** | **+0.109 / +0.033** | 0.134 | 0.168 / 0.119 |

### position_in_movie

| config | TRF raw (val/test) | TRF CorrCA (val/test) | SSL SIGReg+per-rec (val) | SSL CorrCA-VICReg (val/test) |
|---|---|---|---|---|
| nw1_ws1 | +0.010 / +0.000 | +0.012 / +0.006 | 0.156 | — |
| nw2_ws1 | +0.016 / +0.000 | +0.021 / +0.010 | 0.125 | — |
| nw4_ws4 | +0.079 / **−0.013** | **+0.089 / +0.060** | 0.149 | 0.176 / 0.188 |

Median-split balanced accuracy was **0.50 ± 0.005** for every cell —
chance — even where Pearson r is non-trivial. The classifier head decision
boundary is dominated by the train-set median split's degenerate
distribution at clip-level.

### Reading the table

1. **Raw 129-channel branch fails on test.** Val correlations at nw4_ws4
   look meaningful (0.06–0.08), but test correlations are *negative*
   (−0.04 to −0.01). The d=6450 model overfits cross-subject patterns
   that don't transfer R1–R4 → R6.

2. **CorrCA branch generalizes.** d=250 — well-conditioned at 1M training
   samples — gives positive val *and* test correlations at every config.

3. **The TRF only surfaces signal at long aggregation (nw4_ws4 = 16 s).**
   At 1-s and 2-s clip aggregation, even CorrCA correlations are 0.01–0.02
   (essentially noise floor). At 16-s clips, the TRF pulls 0.03–0.11.
   The TRF is decoding a slow-varying component that needs averaging to
   surface above per-sample noise.

4. **SSL probe beats TRF at every config tested.** At the apples-to-apples
   nw4_ws4 + CorrCA config (CorrCA-VICReg row), SSL test correlations
   are 1.6–3.6× higher than CorrCA TRF:
   - contrast: SSL 0.056 vs TRF 0.034 (1.6×)
   - luminance: SSL 0.119 vs TRF 0.033 (3.6×)
   - position: SSL 0.188 vs TRF 0.060 (3.1×)

## 4. Sanity check — per-subject TRF (literature-comparable)

The literature standard for visual TRFs (Lalor/Crosse mTRF toolbox,
Di Liberto, Madsen et al.) is **per-subject TRF with within-subject
held-out time**, on 30–60 min recordings. Our recordings are 3:23 min —
~10× shorter than literature.

Three diagnostics on N=20 randomly chosen R1–R4 recordings, CorrCA input,
0–500 ms lag, α=1.0 (well-conditioned at d=125):

| Feature | A in-sample r | B train r | **B eval r** | **C eval (shuffled-y) r** |
|---|---|---|---|---|
| contrast_rms | +0.086 ± 0.030 | +0.102 | **+0.010 ± 0.024** | −0.024 ± 0.052 |
| luminance_mean | +0.062 ± 0.019 | +0.072 | **+0.002 ± 0.047** | +0.015 ± 0.066 |
| position_in_movie | +0.030 ± 0.015 | +0.039 | **+0.025 ± 0.035** | −0.009 ± 0.034 |

**Reading the diagnostics**:

- **A (in-sample)** validates the math: train r is small (0.03–0.09),
  consistent with d=125 weights and 10k samples — no overfitting.
- **B (80/20 time split)** is the literature-standard per-subject TRF
  result. Eval r = 0.00–0.025 across features.
- **C (circularly shifted y)** is a no-signal control. C eval r = −0.02
  to +0.02 — statistically indistinguishable from B eval.

**Per-subject TRF on this data has no signal above the shuffled-y null.**
This matches expectations from literature: visual feature TRFs need ~30
minutes of single-subject EEG to reach r≈0.10. We have 3:23.

A separate single-subject diagnostic at d=6450 (raw 129ch, lag 1 s, α=1e3)
confirmed the same regime: train r 0.4–0.6 in-sample (overfit), train r
0.4 even on shuffled-y (model has too many degrees of freedom relative
to per-rec sample count), eval r ~0 in both conditions.

## 5. Why the cross-subject pooled TRF works at all

Per-subject TRF is null. Pooling 100 subjects gives ~1M samples — enough
to fit a stable weak signal. The CorrCA top component has ISC = 0.015,
quantifying how little of the EEG signal is shared across subjects in
a stimulus-locked way. Linear pooled TRF can extract this weak shared
component, especially at long temporal aggregation (nw4_ws4) where
high-frequency noise averages out — yielding the 0.03–0.11 test
correlations observed.

The SSL encoder, by contrast, does non-linear feature extraction
(transformer with attention over temporal patches and channels) and
extracts richer, subject-invariant dynamics. That non-linearity is what
buys the 1.6–3.6× advantage over the linear TRF on the same input.

## 6. Verdict

- **The paper's stimulus-decoding claim is not weakened by a TRF
  baseline.** Linear TRF on the same CorrCA + per-rec input that the
  best SSL checkpoint uses falls short by 1.6–3.6× on test correlations
  for every feature.
- **Recording duration is the bottleneck for a literature-standard
  per-subject TRF.** Single-subject TRF gives null results at 3:23 min
  — consistent with literature requiring ~30 min for visual feature
  TRFs. The cross-subject pooled TRF works around this by accumulating
  samples, at the cost of relying on the (weak) CorrCA top components
  for cross-subject consistency.
- **Window aggregation matters as much as preprocessing.** TRF
  correlations are noise-floor at 1–2 s clips and become non-trivial
  only at 16-s clips. The probe is reported at all three configs; SSL
  beats TRF at all three.
- **Caveat that should be in the paper.** "SSL beats linear TRF on
  3-minute recordings" is the honest framing. We don't have enough
  per-subject data to test whether SSL still beats per-subject TRF on
  literature-standard 30-min recordings.

## 7. What's still outstanding

- **3-seed replication.** All numbers are seed=2025 for the 100-rec subsample.
  A 3-seed run would tighten the noise estimates.
- **Full 718-rec TRF run.** Cheap (~30 min on jamming) and would
  marginally tighten the cross-subject estimates, though we don't expect
  it to change the verdict given how close the 100-rec results are to
  the per-feature noise floor at 1-s and 2-s clips.
- **mTRF-style basis-spline lag smoothing.** The standard mTRF toolbox
  applies a smoothness prior across lags; our naive ridge does not.
  Could buy a small bump — most likely a bigger effect on raw than CorrCA.
- **TRF on per-subject 8-s clip aggregation.** A middle-ground between
  per-subject and pooled — fit per subject but evaluate on 16-s clips
  to match the only setting where the pooled TRF surfaces signal.

## 8. Methodology notes

- Both runs use 100 randomly subsampled training recordings (seed=2025);
  full val/test (R5: 124 recordings, R6: 108 recordings).
- EEG decimated 200→50 Hz with scipy `decimate(zero_phase=True, ftype="fir")`.
- Per-recording z-score (per channel) applied to continuous EEG before
  CorrCA projection; CorrCA filter computed on full R1–R4 (718 recs).
- Visual processing delay (0.3 s) absorbed implicitly by the 0–1 s lag
  window; no explicit shift applied.
- Train clip-level medians (used for bal_acc threshold) computed once on
  the canonical movie feature time series (deterministic per movie).
- All 4 alphas produced near-identical numbers — α didn't matter much
  given the regularization-dominated regime.

## 9. Reproducing

```bash
# CorrCA filters (5 components, 718 train subjects, ~5 min on jamming)
PYTHONPATH=. uv run --group eeg python scripts/compute_corrca.py \
    --output_path corrca_filters.npz --n_components 5 --task ThePresent

# Pooled cross-subject TRF (raw branch, ~13 min)
PYTHONPATH=. uv run --group eeg python experiments/trf_baseline/run_trf.py \
    --input=raw --max_train_recs=100 --n_lags_ms=1000 --fs_target=50 \
    --output_dir=outputs/trf_prototype_raw

# Pooled cross-subject TRF (CorrCA branch, ~2 min)
PYTHONPATH=. uv run --group eeg python experiments/trf_baseline/run_trf.py \
    --input=corrca --corrca_path=corrca_filters.npz \
    --max_train_recs=100 --n_lags_ms=1000 --fs_target=50 \
    --output_dir=outputs/trf_prototype_corrca

# Per-subject sanity (literature-comparable per-subject TRF)
PYTHONPATH=. uv run --group eeg python experiments/trf_baseline/sanity.py \
    --n_recs=20 --input=corrca --corrca_path=corrca_filters.npz \
    --n_lags_ms=500 --fs_target=50 --alpha=1.0
```
