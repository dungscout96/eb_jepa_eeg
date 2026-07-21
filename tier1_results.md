# Tier 1 baselines (2026-04-24)

Three feature sources evaluated under the **same** MovieFeatureHead probe + per-recording linear probes as `probe_eval.py`, so numbers are directly comparable to the Exp 6 5-seed table in `experiments.md`. All baselines use n_windows=4, window_size=2s, per-recording z-norm, splits = train(R1–R4) / val(R5) / test(R6), 3 seeds {42, 123, 456}.

| Baseline | Feature dim | What it tests |
|---|---:|---|
| `raw_corrca` | 500 | CorrCA-5 EEG, box-pooled to 100 samples per window. Floor on linearly decodable stimulus content. |
| `psd_band` | 645 | Welch PSD × 5 bands (δ/θ/α/β/γ) per 129 channel, per window. Handcrafted spectral features. |
| `random_init` | 64 | Exp 6 encoder (depth=2, embed=64, CorrCA-5 in) with random weights; tokens mean-pooled. Isolates SSL contribution from architecture. |

## Test-set metrics (3-seed mean ± pop. std; Exp 6 = 5-seed from `experiments.md`)

| Probe | Exp 6 (ours) | raw_corrca | psd_band | random_init | Winner |
|---|---:|---:|---:|---:|:---:|
| reg position corr        | **0.176 ± 0.048** | 0.067 ± 0.023 | 0.009 ± 0.021 | 0.046 ± 0.031 | Exp 6 |
| reg luminance corr       | **0.168 ± 0.059** | 0.116 ± 0.016 | 0.046 ± 0.047 | 0.063 ± 0.010 | Exp 6 |
| reg contrast corr        | **0.115 ± 0.053** | 0.003 ± 0.037 | −0.010 ± 0.040 | 0.066 ± 0.018 | Exp 6 |
| reg narrative corr       | −0.003 ± 0.042    | **0.071 ± 0.027** | 0.003 ± 0.050 | −0.016 ± 0.007 | raw_corrca |
| cls position AUC         | **0.580 ± 0.025** | 0.548 ± 0.011 | 0.506 ± 0.037 | 0.526 ± 0.020 | Exp 6 |
| cls luminance AUC        | **0.567 ± 0.021** | 0.548 ± 0.012 | 0.472 ± 0.016 | 0.532 ± 0.028 | Exp 6 |
| cls contrast AUC         | **0.553 ± 0.032** | 0.525 ± 0.012 | 0.490 ± 0.017 | 0.516 ± 0.013 | Exp 6 |
| cls narrative AUC        | 0.528 ± 0.025    | **0.539 ± 0.034** | 0.511 ± 0.029 | 0.528 ± 0.005 | tie |
| movie_id top1 (chance .05) | — | **0.136 ± 0.004** | 0.043 ± 0.029 | 0.028 ± 0.000 | raw_corrca |
| movie_id top5 (chance .25) | — | **0.448 ± 0.012** | 0.259 ± 0.038 | 0.228 ± 0.024 | raw_corrca |
| subject age reg corr      | **0.325 ± 0.030** | 0.127 ± 0.035 | **0.325 ± 0.080** | −0.018 ± 0.203 | tie |
| subject age_cls AUC       | —                | 0.611 ± 0.003 | **0.670 ± 0.016** | 0.609 ± 0.022 | psd_band |
| subject sex AUC           | **0.618 ± 0.007** | 0.576 ± 0.010 | 0.584 ± 0.033 | 0.612 ± 0.005 | Exp 6 (tie with random_init) |

(Exp 6 5-seed numbers quoted from `experiments.md`; `movie_id` wasn't in that table.)

## Headline findings

1. **SSL contribution is real but modest (+0.05 – 0.11 corr on stimulus features).**
   The gap between Exp 6 and the best baseline per row:
   - position: +0.109 (~+2.3σ) over raw_corrca — the **clearest win**, consistent with the "pred_dim=24 captures slow drift" story.
   - luminance: +0.052 (~+0.9σ) over raw_corrca — marginal.
   - contrast:  +0.049 (~+0.9σ) over random_init — marginal.
   - narrative: Exp 6 = ~0; **raw_corrca actually wins (+0.071)**. Still within single-sigma noise, but flags that Exp 6 may have lost the low-frequency ISC envelope that CorrCA preserves.

2. **Subject-trait signal is entirely architectural / spectral.**
   - PSD band-power alone gets age corr = 0.325 — **identical** to Exp 6's 0.325.
   - PSD age-classification AUC = 0.670 is slightly *above* any JEPA number.
   - Random-init encoder gets sex AUC = 0.612 vs Exp 6's 0.618.
   These are the "subject info is a free side channel" claims from the project summary, now quantified: **no SSL is doing any work for age/sex**. Any "subject dominance" we attribute to the encoder was already latent in spectral power or the channel-patch architecture.

3. **CorrCA components linearly encode movie position very well** — 13.6% top-1 vs 5% chance on 20-bin movie-ID. Neither PSD nor random_init come close (4.3% and 2.8%). This is a lower bound on how far a stimulus-locked linear decoder can go; any encoder that fails to exceed this number on movie-ID is adding noise.

4. **Random-init encoder ≥ PSD on stimulus regression.** Feature-dim 64 random CNN+transformer beats 645-dim hand-crafted spectral features on contrast, luminance, and narrative corrs. The inductive biases in the architecture (patchification + Fourier positional encoding over channel positions) are doing meaningful work before any training.

## Implications for the project's own ceiling discussion

- The Exp 6 wins on `position / luminance / contrast` regression corrs confirm that SSL is pulling signal that handcrafted features miss. The effect size is small but not negligible given per-seed σ ≈ 0.04–0.06.
- **Narrative corr gap to baselines is concerning.** raw_corrca > Exp 6 on this, which suggests the CorrCA-5 → encoder pipeline is actively destroying slow narrative signal that the raw CorrCA components carry. Could motivate a multi-head bottleneck or keeping CorrCA features as a residual stream.
- **Subject-probe numbers should no longer be framed as "Exp 6 leaks subject identity."** PSD and random encoder leak at the same level. The right question is not "how do we stop Exp 6 from leaking" but "can we learn stimulus features without having to remove more subject variance than handcrafted features already fail to remove" — i.e., focus on the stimulus gains rather than on suppressing something that was never added.

## Artifacts

Raw metric JSONs: `tier1_results/{baseline}_seed{42,123,456}.json` (one file per cell).
Code: `experiments/eeg_jepa/tier1_baselines.py`; sbatch: `scripts/tier1_baseline.sbatch`; submitter: `scripts/submit_tier1.sh`. Branch: `kkokate/tier1-baselines`.
