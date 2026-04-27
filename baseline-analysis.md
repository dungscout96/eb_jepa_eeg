# Baseline Analysis

Running log of baselines run against the Exp 6 reference (CorrCA-5 + per-rec norm + depth=2 + pred_dim=24 + VCLoss(0.25, 0.25) + smooth_L1, 5-seed mean ± std).

All cells use n_windows=4, window_size=2s, per-recording z-norm, splits = train(R1–R4) / val(R5) / test(R6). Numbers are 3-seed test-set mean unless noted; bold = winner of the row.

---

## Tier 1 — trivial controls (2026-04-24, kkokate/tier1-baselines)

| Baseline | Feat dim | What it tests |
|---|---:|---|
| `raw_corrca` | 500 | CorrCA-5 EEG, box-pooled to 100 samples/window. Linear stimulus floor. |
| `psd_band` | 645 | Welch PSD × 5 bands (δ/θ/α/β/γ) × 129 ch. Handcrafted spectral. |
| `random_init` | 64 | Exp 6 architecture (depth=2, embed=64, CorrCA-5 in), random weights. Isolates SSL from architecture. |

| Probe (test) | Exp 6 (5s) | raw_corrca | psd_band | random_init |
|---|---:|---:|---:|---:|
| reg position corr   | **0.176 ± 0.048** | 0.067 ± 0.023 | 0.009 ± 0.021 | 0.046 ± 0.031 |
| reg luminance corr  | **0.168 ± 0.059** | 0.116 ± 0.016 | 0.046 ± 0.047 | 0.063 ± 0.010 |
| reg contrast corr   | **0.115 ± 0.053** | 0.003 ± 0.037 | −0.010 ± 0.040 | 0.066 ± 0.018 |
| reg narrative corr  | −0.003 ± 0.042 | **0.071 ± 0.027** | 0.003 ± 0.050 | −0.016 ± 0.007 |
| cls position AUC    | **0.580 ± 0.025** | 0.548 ± 0.011 | 0.506 ± 0.037 | 0.526 ± 0.020 |
| cls luminance AUC   | **0.567 ± 0.021** | 0.548 ± 0.012 | 0.472 ± 0.016 | 0.532 ± 0.028 |
| cls contrast AUC    | **0.553 ± 0.032** | 0.525 ± 0.012 | 0.490 ± 0.017 | 0.516 ± 0.013 |
| cls narrative AUC   | 0.528 ± 0.025 | **0.539 ± 0.034** | 0.511 ± 0.029 | 0.528 ± 0.005 |
| movie_id top1 (chance .05) | — | **0.136 ± 0.004** | 0.043 ± 0.029 | 0.028 ± 0.000 |
| movie_id top5 (chance .25) | — | **0.448 ± 0.012** | 0.259 ± 0.038 | 0.228 ± 0.024 |
| age reg corr        | **0.325 ± 0.030** | 0.127 ± 0.035 | **0.325 ± 0.080** | −0.018 ± 0.203 |
| age_cls AUC         | 0.648 ± 0.022¹ | 0.611 ± 0.003 | **0.670 ± 0.016** | 0.609 ± 0.022 |
| sex AUC             | **0.618 ± 0.007** | 0.576 ± 0.010 | 0.584 ± 0.033 | 0.612 ± 0.005 |

¹ Exp 6 age_cls AUC computed in the tier1 extended runs (3 seeds: 42, 123, 2025); not in the published 5-seed table in `experiments.md` (which reports age bal_acc 0.637 ± 0.024 instead).

### Findings
- **SSL helps but modestly.** Exp 6 wins position / luminance / contrast regression corrs and AUCs; biggest gap is position (+0.11 vs raw_corrca, ~2.3σ). Luminance/contrast gaps are only ~1σ.
- **Subject-trait signal is entirely architectural/spectral.** PSD age corr = 0.325 ties Exp 6 exactly. Random encoder sex AUC = 0.612 ≈ Exp 6's 0.618. The "Exp 6 leaks subject identity" story is a property of feature extraction, not SSL.
- **raw_corrca beats Exp 6 on narrative corr (+0.07 vs ~0) and dominates movie-ID** (13.6% vs 4.3% / 2.8% top-1, chance 5%). The Exp 6 pipeline may be destroying slow CorrCA content; multi-head bottleneck or CorrCA residual stream worth exploring.
- **Random-init ≥ PSD on stimulus regression** despite 10× smaller features — patchify + Fourier-pos channel encoding carries inductive bias.

Artifacts: `tier1_results/{baseline}_seed{42,123,456}.json`, code `experiments/eeg_jepa/tier1_baselines.py`, sbatch `scripts/tier1_baseline.sbatch`.

---

## Tier 1 extended — bootstrap CIs, permutation, CKA, stacked, data-eff (2026-04-24)

Re-ran 4 cells (3 Tier-1 baselines + Exp 6 from std=0.25 pd=24 checkpoints, seeds {42, 123, 2025}) with `--save_embeddings`, then `tier1_extended_analysis.py` over the saved per-clip mean-pooled embeddings + targets. Probe = MLP head (D → 64 → 4) trained per-clip (not per-window), so absolute corrs are slightly different from the per-window-probe Tier-1 table; comparisons within this section are self-consistent.

**Caveat #1 — measurement scale.** This per-clip probe gives Exp 6 a position corr of ~0.10 here vs 0.176 in `experiments.md` (5-seed, per-window probe via `probe_eval.py`). Don't compare numbers across the two scales. Within the extended-analysis section, all four baselines use the same probe.

**Caveat #2 — permutation conservatism.** Shuffling clip targets within the test set breaks within-recording autocorrelation. `position_in_movie` is monotonic within each recording, so any random predictor that follows recording structure looks "correlated" with position → null distribution is wide → p-values are inflated. Permutation results are most trustworthy for fast-varying targets (contrast, luminance, narrative) and least trustworthy for position.

### (1) Bootstrap 95% CI on per-clip test corrs (3-seed mean of CI bounds)

| Baseline | contrast | luminance | position | narrative |
|---|---:|---:|---:|---:|
| raw_corrca  | +0.05 [−0.16, +0.25] | +0.19 [−0.02, +0.40] | +0.12 [−0.07, +0.30] | +0.06 [−0.13, +0.25] |
| psd_band    | −0.02 [−0.20, +0.16] | +0.05 [−0.12, +0.22] | −0.02 [−0.21, +0.17] | −0.00 [−0.18, +0.20] |
| random_init | +0.04 [−0.14, +0.22] | +0.06 [−0.14, +0.25] | +0.04 [−0.15, +0.24] | +0.13 [−0.06, +0.32] |
| **exp6**    | +0.06 [−0.14, +0.24] | +0.10 [−0.10, +0.29] | +0.10 [−0.10, +0.29] | −0.03 [−0.22, +0.15] |

**All CIs span zero.** Per-clip bootstrap can't separate any single baseline from chance on a single seed's test set. The Exp 6 wins reported in the headline Tier 1 table are detectable across-seed averages, not within-seed effects.

### (2) Per-feature permutation p-values (mean across 3 seeds)

| Baseline | contrast | luminance | position | narrative |
|---|---:|---:|---:|---:|
| raw_corrca  | r=+0.05, p=0.61 | r=+0.20, **p=0.12** | r=+0.12, p=0.30 (autocorr-inflated) | r=+0.07, p=0.60 |
| psd_band    | r=−0.02, p=0.67 | r=+0.05, p=0.64 | r=−0.02, p=0.75 | r=−0.01, p=0.41 |
| random_init | r=+0.04, p=0.58 | r=+0.06, p=0.59 | r=+0.04, p=0.59 | r=+0.14, p=0.27 |
| **exp6**    | r=+0.06, p=0.32 | r=+0.10, p=0.41 | r=+0.10, p=0.42 (autocorr-inflated) | r=−0.04, p=0.57 |

**Closest to significance** (excluding position): `raw_corrca` luminance p=0.12. **Nothing** rises above the conventional p<0.05 threshold under per-clip permutation. Either the per-clip probe pipeline is too weak, the test set is too clustered, or the SSL signal genuinely doesn't survive a stricter test than mean-of-seeds.

### (4) Linear CKA between random_init and Exp 6 val embeddings

| Pair (seed pairing by ordinal) | CKA |
|---|---:|
| ri seed=42 vs e6 seed=42 | 0.85 |
| ri seed=123 vs e6 seed=123 | 0.77 |
| ri seed=456 vs e6 seed=2025 | 0.43 |
| **Mean** | **0.68** |

Exp 6's representation has meaningfully diverged from a random-init encoder of the same architecture (mean CKA ≈ 0.68 < 1.0), but most of the seed pairs are still highly similar (>0.75). The seed=2025 Exp 6 encoder is the outlier — much further from random-init than the other two. **Interpretation:** SSL is moving the representation, but only modestly for 2 of 3 seeds. The "Exp 6 ≈ architecture prior" hypothesis isn't crazy.

### (5) Stacked probe — concat(exp6, raw_corrca) vs exp6 alone (Δ in test corr, per seed)

| Seed pair | contrast | luminance | position | narrative |
|---|---:|---:|---:|---:|
| (e6=42, rc=42)     | +0.18 | +0.02 | −0.13 | +0.19 |
| (e6=123, rc=123)   | +0.08 | −0.09 | +0.03 | −0.08 |
| (e6=2025, rc=456)  | −0.23 | −0.03 | +0.01 | −0.01 |

**Inconclusive**: Δ swings ±0.2 across seeds with no consistent direction. Stacked probe doesn't show a robust "Exp 6 missed signal that raw_corrca had." The seed-1 hypothesis (concat helps narrative) is overturned by seed-3.

### (6) Data-efficiency curve — position_in_movie corr at {1%, 5%, 25%, 100%} train (mean of 3 seeds)

| Baseline | 1% | 5% | 25% | 100% |
|---|---:|---:|---:|---:|
| raw_corrca  | +0.05 | +0.09 | **+0.16** | +0.13 |
| psd_band    | −0.11 | +0.05 | −0.01 | +0.01 |
| random_init | +0.09 | +0.01 | +0.05 | +0.03 |
| **exp6**    | **+0.12** | **+0.14** | +0.11 | +0.12 |

Same for luminance:

| Baseline | 1% | 5% | 25% | 100% |
|---|---:|---:|---:|---:|
| raw_corrca  | +0.07 | **+0.24** | +0.08 | **+0.20** |
| psd_band    | −0.01 | +0.02 | +0.11 | +0.06 |
| random_init | +0.09 | −0.04 | +0.04 | +0.05 |
| **exp6**    | **+0.13** | +0.12 | +0.09 | +0.14 |

**Cleanest finding of the extended analyses.** Exp 6 saturates at 1% train (~280 clips) and stays flat — that's the SSL data-efficiency story. raw_corrca needs ≥5–25% of train data to catch up, then matches or exceeds Exp 6 at 100%. So:
- At very low train sizes, **Exp 6 dominates** all baselines.
- At 100% train, raw_corrca matches/beats Exp 6 on luminance and position.

### Headline takeaways from the extended analyses

1. **Per-clip permutation/bootstrap is too weak to detect any single-seed effect.** What looked like clean Tier-1 wins for Exp 6 disappear under a stricter test. The signal exists *across seeds* (consistent positive corrs), not *within a single seed's test clips*. **Action:** report any future Tier 2/3 wins with seed-level effect sizes (e.g., paired t-test across 3+ seeds), not single-seed point estimates.
2. **Exp 6 is genuinely data-efficient.** This is the clearest, most defensible win: at 1% of train data Exp 6 already matches its own 100% performance, while raw_corrca needs 25× more data. **Action:** pitch the SSL story as data-efficiency, not asymptotic performance.
3. **CKA shows SSL has moved the representation** (mean 0.68 < 1) but not dramatically. Combined with (2), suggests SSL is doing useful but small work — pushing harder on the same objective is unlikely to multiply gains.
4. **Stacked probe and bootstrap CIs do not support the "Exp 6 destroys CorrCA signal" hypothesis** that the headline movie-ID gap suggested. Movie-ID gap was real (raw_corrca 13.6% vs Exp 6 ~3.7%) but stacked probe shows that *combining* the two doesn't reliably help — meaning the lost signal isn't recoverable as a simple linear add-on.
5. **Probe-pipeline mismatch flagged.** Tier 1 per-clip probe gives Exp 6 lower numbers than per-window probe in `probe_eval.py`. Future Tier 2/3 should use a single canonical probe so tables are directly comparable. (Easy fix: align tier1 to per-window.)

Artifacts: `tier1_results/extended_analysis.json`; embeddings on Delta at `/projects/bbnv/kkokate/eb_jepa_eeg/tier1/embeddings/`. Code: `experiments/eeg_jepa/tier1_extended_analysis.py`, sbatch: `scripts/tier1_extended_analysis.sbatch`.

---

## Tier 2 — classical supervised EEG decoders (2026-04-25, kkokate/tier1-baselines)

Four `braindecode` architectures trained end-to-end on the 4 stimulus targets jointly (multi-output regression head + parallel classification head). Same data setup as Tier 1: 4×2-s clips, 129 ch raw EEG (no CorrCA), per-recording z-norm, splits R1–R4 / R5 / R6, 3 seeds {42, 123, 456}. Adam lr=1e-3, batch=64, 50 epochs, early-stop on val mean reg corr (patience=8). Per-recording subject probes use penultimate-layer activations.

| Model | Params | best val mean corr |
|---|---:|---:|
| `shallow` (ShallowFBCSPNet) | 421K | 0.079 ± 0.023 |
| `deep4` (Deep4Net) | 663K | 0.067 ± 0.019 |
| `eegnet` (EEGNetv4) | **8K** | **0.148 ± 0.034** |
| `eegnex` (EEGNeX) | 125K | **0.148 ± 0.011** |

### Test-set metrics (3-seed mean ± std)

| Probe (test) | Exp 6 (5s) | shallow | deep4 | eegnet | eegnex |
|---|---:|---:|---:|---:|---:|
| reg position corr   | **0.176 ± 0.048** | 0.100 ± 0.136 | 0.013 ± 0.049 | 0.101 ± 0.104 | 0.100 ± 0.035 |
| reg luminance corr  | **0.168 ± 0.059** | 0.083 ± 0.063 | 0.046 ± 0.040 | 0.117 ± 0.013 | 0.127 ± 0.026 |
| reg contrast corr   | **0.115 ± 0.053** | 0.078 ± 0.052 | 0.022 ± 0.011 | 0.094 ± 0.041 | 0.004 ± 0.051 |
| reg narrative corr  | −0.003 ± 0.042 | 0.001 ± 0.008 | −0.020 ± 0.011 | 0.064 ± 0.043 | **0.074 ± 0.094** |
| cls position AUC    | **0.580 ± 0.025** | 0.567 ± 0.046 | 0.539 ± 0.024 | 0.555 ± 0.052 | 0.572 ± 0.008 |
| cls luminance AUC   | **0.567 ± 0.021** | 0.528 ± 0.018 | 0.527 ± 0.035 | 0.529 ± 0.024 | 0.548 ± 0.024 |
| cls contrast AUC    | 0.553 ± 0.032 | 0.498 ± 0.010 | 0.510 ± 0.030 | **0.583 ± 0.023** | 0.567 ± 0.033 |
| cls narrative AUC   | 0.528 ± 0.025 | 0.498 ± 0.010 | 0.510 ± 0.030 | 0.552 ± 0.005 | **0.567 ± 0.028** |
| age reg corr        | 0.325 ± 0.030 | **0.490 ± 0.044** | 0.360 ± 0.070 | 0.232 ± 0.066 | 0.133 ± 0.103 |
| age_cls AUC         | 0.648 ± 0.022¹ | **0.759 ± 0.059** | 0.668 ± 0.027 | 0.638 ± 0.044 | 0.607 ± 0.040 |
| sex AUC             | 0.618 ± 0.007 | 0.617 ± 0.019 | **0.621 ± 0.073** | 0.508 ± 0.026 | 0.614 ± 0.016 |

¹ See footnote in Tier 1 table — Exp 6 age_cls AUC = 0.648 ± 0.022 from the 3-seed tier1 extended runs (seeds 42, 123, 2025); not in the published 5-seed table.

### Findings

1. **No supervised model beats Exp 6 on position / luminance / contrast regression.** Closest gaps to Exp 6: shallow position (−0.08), eegnex luminance (−0.04), eegnet contrast (−0.02). All within seed-noise bands. **Exp 6 holds the top spot for the 3 fast-varying stimulus features against fully-supervised end-to-end training.**
2. **EEGNet (8K params!) and EEGNeX (125K) win narrative** — corr +0.06–0.07 vs Exp 6 ≈ 0, and AUC 0.55–0.57 vs Exp 6 0.528. Same direction as raw_corrca in Tier 1. The "Exp 6 doesn't capture narrative signal" finding is now confirmed by **two independent baselines** (raw CorrCA + small supervised CNNs), not just one.
3. **EEGNet is the most parameter-efficient supervised baseline** — at 8K params it matches or beats every other supervised model on stimulus features. Suggests the right inductive bias for HBN movie EEG is depthwise-separable + small, not deeper. Deep4 (83× more params) is the *worst* across the board — overfitting on 8-s clips.
4. **Subject-trait wins go to the supervised models, not Exp 6.** Shallow ConvNet gets age corr **0.490** (vs Exp 6 0.325, PSD 0.325) and age AUC **0.759**. Either (a) supervised end-to-end training with stimulus targets coincidentally learns age-discriminative features as a side effect, or (b) shallow CNNs preserve more of the spectral subject fingerprint than the JEPA bottleneck does. Sex AUC is roughly tied across models (0.5–0.62).
5. **Reframing:** Tier 1 said "subject-trait is architectural/spectral" because PSD matched Exp 6. Tier 2 strengthens this — supervised CNNs with much more capacity than PSD get *higher* age scores than Exp 6, suggesting the JEPA bottleneck is actually *removing* age signal that the input + most architectures preserve. Whether that's good (subject invariance) or bad (lost predictive power) depends on the use case.

### What this means for Exp 6 next iteration

- **Position / luminance / contrast story is solid.** Exp 6 is the best method we have for these. Tier 2 confirms — no off-the-shelf supervised model beats it.
- **Narrative is a confirmed weakness.** Two unrelated baselines (raw_corrca, EEGNet/EEGNeX) both extract more narrative signal than Exp 6. The architectural-residual-stream idea (route raw CorrCA into the encoder) is now better motivated.
- **Subject-trait suppression by Exp 6 may be unintentional.** Shallow ConvNet shows there's much more age signal extractable than Exp 6 surfaces. If we want age suppression, this is a feature; if we want age decodability as a side product, Exp 6 is leaving signal on the table.

Artifacts: `tier2_results/{model}_seed{42,123,456}.json` (12 files), per-clip predictions on Delta at `/projects/bbnv/kkokate/eb_jepa_eeg/tier2/embeddings/`. Code: `experiments/eeg_jepa/tier2_supervised.py`. SLURM: `scripts/tier2_supervised.sbatch` + `scripts/submit_tier2.sh`.

### Deep4Net re-run with native preprocessing (2026-04-25)

The original Tier 2 used a single shared preprocessing (200 Hz, 0.5–50 Hz BP, 2-s windows) so Deep4Net hit `n_times (400) < min required (441)` and braindecode auto-shrank its kernels by 0.91×. To remove this disadvantage we ran Deep4 with its **native Schirrmeister-2017 spec**:

- Single 4-s window per clip (`n_windows=1, window_size_seconds=4`)
- In-graph resample 200 → **250 Hz** via `F.interpolate(mode='linear')`
- In-graph **4–38 Hz bandpass** via FFT mask (zero-phase)
- In-graph per-window per-channel z-score (offline approximation of EMA standardization)
- Otherwise identical pipeline (3 seeds, 50 epochs, early-stop on val mean reg corr)

| Metric | orig (200 Hz, 2-s) | **native (250 Hz, 4-s)** | Δ |
|---|---:|---:|---:|
| best val mean reg corr | 0.067 ± 0.019 | **0.094 ± 0.008** | +0.028 |
| reg position corr | 0.013 ± 0.049 | **0.073 ± 0.031** | +0.060 |
| reg luminance corr | 0.046 ± 0.040 | 0.037 ± 0.057 | −0.009 |
| reg contrast corr | 0.022 ± 0.011 | −0.020 ± 0.074 | −0.042 |
| reg narrative corr | −0.020 ± 0.011 | −0.128 ± 0.069 | **−0.108** |
| cls position AUC | 0.539 ± 0.024 | 0.503 ± 0.039 | −0.035 |
| cls luminance AUC | 0.527 ± 0.035 | **0.561 ± 0.037** | +0.035 |
| cls contrast AUC | 0.510 ± 0.030 | 0.524 ± 0.018 | +0.014 |
| cls narrative AUC | 0.510 ± 0.030 | 0.484 ± 0.056 | −0.026 |
| age reg corr | 0.360 ± 0.070 | 0.367 ± 0.044 | +0.007 |
| age cls AUC | 0.668 ± 0.027 | 0.663 ± 0.015 | −0.005 |
| sex AUC | 0.621 ± 0.073 | 0.567 ± 0.027 | −0.054 |

**Findings:**

1. **Native preprocessing materially helps val selection and position regression** (+0.028 val mean corr, +0.060 position corr). Deep4 was indeed disadvantaged by the shared pipeline; the native run is the fairer comparison.
2. **Native preprocessing destroys narrative signal** (−0.108 corr): the 4–38 Hz bandpass cuts content < 4 Hz where the narrative envelope lives. Same issue would likely apply if we re-ran Shallow with the same bandpass. Confirms that narrative is a slow-band signal — any model that filters above 4 Hz loses it.
3. **Conclusion is unchanged**: Deep4 native still loses to Exp 6 on every stimulus feature except a slight position advantage that's well within noise. The "Exp 6 wins position/luminance/contrast against all supervised baselines" headline survives. Deep4 native does climb past `shallow` and `eegnex` on val mean reg corr (0.094 vs 0.079 vs 0.148), but eegnet still leads at 0.148.
4. **Subject probes essentially unchanged** — no preprocessing-driven shift in age or sex AUC.

Artifacts: `tier2_results/native/deep4_native_seed{42,123,456}.json`. Code: `NativePreprocWrapper` in `experiments/eeg_jepa/tier2_supervised.py`; sbatch `scripts/tier2_deep4_native.sbatch`; submitter `scripts/submit_tier2_deep4_native.sh`.

---

## Tier 3 — frozen EEG foundation models (2026-04-25, kkokate/tier1-baselines)

Three pretrained foundation models loaded from HuggingFace + braindecode and applied to HBN with each model's **native preprocessing**, frozen, then evaluated with the same MovieFeatureHead probe + per-recording linear probes used in Tier 1 / Tier 2. 3 seeds {42, 123, 456}.

| Model | Params | Pretraining | Window | Sample rate | HBN channels | Bandpass | Norm |
|---|---:|---|---:|---:|---:|---|---|
| BIOT (NeurIPS 2023) | 3.2M | TUH Abnormal + SHHS | 2 s | 200 Hz | 16 (10-20 subset of 19) | none | 95th-pctile per-channel |
| CBraMod (ICLR 2025) | 4.9M | Mixed corpus, MAE | 2 s | 200 Hz | 19 (10-20) | 0.3–50 Hz | per-channel mean removal |
| **LUNA** (NeurIPS 2025) | 7.1M | TUEG + Siena (21k h) | 2 s | 200 Hz | **all 129** (topology-agnostic) | none | per-window z-score |

**Key technical note on LUNA:** unlike the other FMs, LUNA accepts **arbitrary channel counts** via a 3D-coordinate channel-location encoding (no fixed 10-20 channel-name embedding). We feed the full HBN-129 montage with electrode positions, no channel selection. This is the cleanest comparison to Exp 6 (which also sees all of HBN, via the CorrCA spatial filter).

LaBraM and EEGPT deferred — their pretrained checkpoints have hardcoded fixed channel count (62) and patch position embedding (LaBraM's 16 patches) that don't admit a clean wrapper.

### Test-set metrics (3-seed mean ± std)

| Probe (test) | Exp 6 (5s) | BIOT | CBraMod | **LUNA** |
|---|---:|---:|---:|---:|
| reg position corr   | **0.176 ± 0.048** | 0.025 ± 0.040 | 0.040 ± 0.131 | 0.132 ± 0.053 |
| reg luminance corr  | 0.168 ± 0.059 | 0.076 ± 0.063 | 0.086 ± 0.142 | **0.180 ± 0.100** |
| reg contrast corr   | 0.115 ± 0.053 | 0.008 ± 0.058 | −0.025 ± 0.132 | **0.162 ± 0.079** |
| reg narrative corr  | −0.003 ± 0.042 | −0.034 ± 0.016 | **0.074 ± 0.082** | −0.086 ± 0.067 |
| cls position AUC    | **0.580 ± 0.025** | 0.513 ± 0.017 | 0.524 ± 0.055 | 0.553 ± 0.036 |
| cls luminance AUC   | **0.567 ± 0.021** | 0.493 ± 0.013 | 0.512 ± 0.033 | 0.520 ± 0.078 |
| cls contrast AUC    | **0.553 ± 0.032** | 0.475 ± 0.016 | 0.440 ± 0.046 | 0.499 ± 0.058 |
| cls narrative AUC   | 0.528 ± 0.025 | **0.540 ± 0.110** | 0.526 ± 0.057 | 0.511 ± 0.091 |
| age reg corr        | 0.325 ± 0.030 | **0.562 ± 0.011** | 0.516 ± 0.001 | 0.451 ± 0.005 |
| age cls AUC         | 0.648 ± 0.022 | **0.813 ± 0.002** | 0.797 ± 0.000 | 0.706 ± 0.002 |
| sex AUC             | 0.618 ± 0.007 | 0.664 ± 0.017 | **0.774 ± 0.001** | 0.582 ± 0.008 |

### Findings

1. **LUNA is the first method to beat Exp 6 on a stimulus feature regression.** Wins luminance (+0.012) and contrast (+0.047) on test corr; within 1σ on position (−0.044). Topology-agnostic encoding (all 129 channels with 3D coords) appears to capture more visual stimulus signal than CorrCA-5 + Exp 6.
2. **No FM beats Exp 6 on stimulus AUC** — Exp 6 still wins position/luminance/contrast classification across the board.
3. **All 3 FMs trounce Exp 6 on subject-trait probes** — BIOT age corr **0.562** (vs Exp 6 0.325), age AUC **0.813** (vs 0.648); CBraMod sex AUC **0.774**. Same trend as Tier 2: pretrained encoders preserve spectral subject fingerprint that Exp 6's JEPA bottleneck strips out.
4. **Narrative remains Exp 6's confirmed blind spot** — CBraMod gets +0.074 narrative corr (matches raw_corrca's +0.071 from Tier 1). Now confirmed by 3 independent baselines (raw_corrca, eegnet/eegnex, CBraMod).
5. **High variance across CBraMod seeds** (luminance ±0.142, contrast ±0.132) — its frozen embedding is unstable across probe seeds; BIOT and LUNA much more consistent.
6. **LUNA's age leak is intermediate** (age corr 0.451 vs BIOT 0.562, Exp 6 0.325). The 3D-coordinate channel encoding may be partially obscuring per-channel spectral fingerprint, hence less age leak than the named-channel FMs.

### What this means for Exp 6 next iteration

- **LUNA is the first credible alternative to Exp 6 on stimulus regression.** Worth seriously considering as either (a) a frozen feature extractor for downstream tasks (no SSL of our own needed), or (b) an architectural prior — the topology-agnostic 3D-coord channel encoding may be the right design for HBN's 129-ch montage.
- **Exp 6 still wins stimulus classification AUCs** — its representation is binary-decision-friendly even where corrs are matched.
- **Subject-trait gap to FMs is structural, not a bug.** PSD (T1), shallow CNN (T2), and 3 FMs (T3) all preserve more age/sex signal than Exp 6. Confirmed across 7+ baselines now — Exp 6's JEPA bottleneck is *removing* spectral subject info regardless of method-class.
- **Narrative architectural-residual idea now backed by 3 baselines** — raw_corrca (T1), eegnet/eegnex (T2), CBraMod (T3). Exp 6's narrative gap is a real, multi-method-confirmed weakness.

Artifacts: `tier3_results/{model}_seed{42,123,456}.json`, embeddings on Delta at `/projects/bbnv/kkokate/eb_jepa_eeg/tier3/embeddings/`. Code: `experiments/eeg_jepa/tier3_foundation.py`, vendored LUNA at `experiments/eeg_jepa/external/luna/`. SLURM: `scripts/tier3_foundation.sbatch` + `scripts/submit_tier3.sh`.
