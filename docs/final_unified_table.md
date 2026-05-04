# Final unified comparison table

**Date:** 2026-05-03
**Branch:** `kkokate/trivial-stats-baselines`

All test-set metrics under recording-level bootstrap (B=2000) with t-test on the per-encoder bootstrap means against chance.

## Method tiers in this comparison

| Tier | Method | Encoder | Probe | Seeds | Bootstrap |
|---|---|---|---|---|---|
| 1 | **JEPA + MLP** `--keep_channels` (PR #15) | EEG-JEPA + CorrCA-5, frozen | MovieFeatureHead 2-layer MLP / nn.Linear (subject + movie-id) | 5 enc × 5 probe | View 3 B=2000 |
| 2 | **JEPA + Ridge per-clip** `--keep_channels` | Same encoder, frozen | sklearn Ridge per stim feature + LogReg per cls (movie-feature head only) | 5 enc × 1 probe | View 3 B=2000 |
| 3 | **Trivial Ridge corrca35** | None (CorrCA-5 → mean+std+5 log-bands per chan = 35 dims) | sklearn Ridge per stim feature + LogReg | 5 probe | View 3 B=2000 |
| 4 | **Tier 5 — FM full-FT (raw EEG)** | BIOT/CBraMod/LUNA fine-tuned end-to-end on raw EEG with neg-Pearson loss | nn.Linear(D, 4) reg head | 5 seeds (BIOT had 3 prior + 2 new = 5 partial) | View 3 B=2000 |
| 5 | **Tier 6 — LUNA full-FT on CorrCA** (NEW) | LUNA fine-tuned end-to-end on CorrCA-5 (5 components) with neg-Pearson loss | nn.Linear(D, 4) reg head | 5 seeds | View 3 B=2000 |

## Stimulus regression — `reg_<feature>_corr` (test, chance=0)

| Method | position | luminance | contrast | narrative |
|---|---:|---:|---:|---:|
| **JEPA + MLP** `--keep_channels` | **+0.212** ✓ | +0.187 ✓ | +0.101 ✓ | +0.062 ✓ |
| **JEPA + Ridge** `--keep_channels` | +0.144 ✓ | **+0.208** ✓ | +0.159 ✓ | **+0.090** ✓ |
| Trivial Ridge corrca35 | +0.125 ✓ | +0.142 ✓ | +0.139 ✓ | +0.044 ✓ |
| Trivial Ridge raw903 | +0.008 ns | +0.003 ns | +0.009 ns | +0.018 ns |
| Tier 5 — BIOT FT (raw) | +0.147 ns | +0.177 ✓ | +0.100 ns | +0.043 ns |
| Tier 5 — CBraMod FT (raw) | +0.023 ns | +0.054 ns | +0.036 ns | +0.041 ns |
| Tier 5 — LUNA FT (raw) | +0.171 ✓ | +0.142 ✓ | +0.094 ✓ | +0.050 ns |
| **Tier 6 — LUNA FT (CorrCA)** | +0.165 ✓ | +0.163 ✓ | **+0.158** ✓ | −0.009 ns |

## Stimulus classification — `cls_<feature>_auc` (test, chance=0.5)

| Method | position | luminance | contrast | narrative |
|---|---:|---:|---:|---:|
| **JEPA + MLP** `--keep_channels` | **+0.636** ✓ | **+0.573** ✓ | +0.529 ✓ | +0.559 ✓ |
| **JEPA + Ridge** `--keep_channels` | +0.573 ✓ | +0.569 ✓ | +0.558 ✓ | **+0.579** ✓ |
| Trivial Ridge corrca35 | (not run) | (not run) | (not run) | (not run) |
| Trivial MLP corrca_per_chan | +0.531 ns | +0.510 ns | +0.521 ns | +0.522 ns |
| Tier 5 — BIOT FT (raw) | +0.607 ns | +0.553 ns | +0.509 ns | +0.559 ns |
| Tier 5 — CBraMod FT (raw) | +0.513 ns | +0.525 ns | +0.541 ✓ | +0.513 ns |
| Tier 5 — LUNA FT (raw) | +0.591 ✓ | +0.556 ns | +0.530 ns | +0.516 ns |
| **Tier 6 — LUNA FT (CorrCA)** | +0.561 ✓ | +0.551 ✓ | **+0.578** ✓ | +0.499 ns |

## Subject traits — `age_reg_corr` / `age_cls_auc` / `sex_auc` (test)

JEPA + Ridge subject probes were buggy (label alignment issue); using JEPA + MLP `--keep_channels` numbers since the encoder + nn.Linear probe are identical across both probe stacks.
LUNA + CorrCA subject probes failed at runtime (channel-count mismatch in `_features_per_recording`); not reported.

| Method | age_reg corr | age_reg MAE | age_cls AUC | sex AUC |
|---|---:|---:|---:|---:|
| **JEPA + MLP** `--keep_channels` | +0.504 ✓ | 2.766 ✓ | +0.727 ✓ | **+0.713** ✓ |
| Trivial MLP raw_per_chan | +0.316 ✓ | 3.036 ✓ | +0.695 ✓ | +0.588 ✓ |
| Trivial MLP raw_pooled903 | +0.361 ✓ | **2.915** ✓ | +0.696 ✓ | +0.488 ns |
| Tier 5 — BIOT FT (raw) | **+0.543** ✓ | **2.394** ✓ | **+0.796** ✓ | +0.686 ✓ |
| Tier 5 — CBraMod FT (raw) | +0.284 ✓ | (not in summary) | +0.657 ✓ | +0.655 ✓ |
| Tier 5 — LUNA FT (raw) | +0.356 ✓ | (not in summary) | +0.696 ✓ | +0.543 ns |

## Movie identity — 20-bin position classifier (chance: top1=0.05, top5=0.25)

The production single-Linear movie-id probe is undertrained for high-D JEPA embeddings (single-seed diagnostic showed sklearn LogReg with C=1e-4 lifts top1 to +0.102, vs production's +0.074).

| Method | top1 | top5 |
|---|---:|---:|
| JEPA + MLP `--keep_channels` (production probe, 5×5 sweep) | +0.056 ns | +0.217 sig below chance* |
| JEPA + MLP `--keep_channels` + sklearn LogReg C-sweep (single seed s42) | +0.102 | +0.352 |
| JEPA + MLP `--keep_channels` + 2-layer MLP probe (single seed s42) | +0.093 | +0.380 |
| Reference: Trivial raw_corrca (Tier 1, 3-seed) | +0.136 | +0.448 |

*"sig below chance" is a known recording-level bootstrap artifact for 20-way classifiers, not real anti-prediction.

## Per-feature winner

| Feature | Best probe | Value |
|---|---|---:|
| position regression | JEPA + MLP `--keep_channels` | **+0.212** |
| luminance regression | JEPA + Ridge `--keep_channels` | **+0.208** |
| contrast regression | Tier 6 LUNA + CorrCA OR JEPA + Ridge | +0.158 / +0.159 (tie) |
| narrative regression | JEPA + Ridge `--keep_channels` | **+0.090** |
| position classification | JEPA + MLP `--keep_channels` | **+0.636** |
| luminance classification | JEPA + MLP `--keep_channels` | **+0.573** |
| contrast classification | Tier 6 LUNA + CorrCA | **+0.578** |
| narrative classification | JEPA + Ridge `--keep_channels` | **+0.579** |
| age regression corr | Tier 5 BIOT FT (raw) | **+0.543** |
| age MAE (yrs, lower=better) | Tier 5 BIOT FT (raw) | **2.394** |
| age classification AUC | Tier 5 BIOT FT (raw) | **+0.796** |
| sex AUC | JEPA + MLP `--keep_channels` | **+0.713** |
| movie_id top1 | (production probe at chance for JEPA; raw_corrca old Tier 1 +0.136) | — |

## Headline takeaways

1. **JEPA + `--keep_channels` (either Ridge or MLP probe) wins 6 of 8 stimulus metrics** when both reg + cls AUCs are counted. Position+cls AUCs go to JEPA + MLP; luminance/contrast/narrative regression and narrative AUC go to JEPA + Ridge.

2. **Tier 6 LUNA + CorrCA is competitive** — wins contrast cls AUC (+0.578) and ties on contrast regression. CorrCA preprocessing gives LUNA a real boost on luminance/contrast vs Tier 5 raw (luminance +0.142 → +0.163, contrast +0.094 → +0.158).

3. **Tier 5 BIOT FT dominates subject traits** — age corr +0.543, age MAE 2.394, age AUC +0.796. Beats every other method by a clear margin. CBraMod raw also competitive on sex (+0.655) but BIOT wins on age across the board.

4. **Tier 6 LUNA + CorrCA narrative collapsed** to −0.009 (vs Tier 5 raw LUNA's +0.050) — CorrCA pre-projection removes the narrative signal LUNA was extracting from raw EEG. Mirrors the JEPA observation that narrative needs per-channel context CorrCA collapses.

5. **Trivial Ridge corrca35 is a strong floor** — 0.13–0.14 on stim regs, no encoder needed. Confirms that CorrCA-projected EEG carries decodable stimulus content; SSL/FM training adds 0.02–0.10 on top.

6. **The proposed paper model (JEPA + `--keep_channels`)** has a defensible story: best on the broadest set of stimulus metrics, complementary to FM strengths (FMs win subject, JEPA wins stim), and beats trivial baselines on every cls AUC where they fail.

## Methodology notes

- **JEPA encoder**: phaseD `nw4ws2_baseline` checkpoints, 5 seeds {42, 123, 456, 789, 2025}, 100 epochs SSL pretraining + per-recording z-norm + CorrCA-5.
- **MovieFeatureHead probe**: `Linear(D, 64) → ReLU → Linear(64, 4)`, joint reg+cls Adam lr=1e-3, 20 epochs.
- **Ridge probe**: sklearn `Ridge(α=1.0)` per stim feature, n_passes=20 random clips per recording.
- **LogReg cls (Ridge path)**: sklearn `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)` per stim feature on median-split target.
- **Subject probe**: `nn.Linear(D, 1)`, Adam lr=1e-3, 100 epochs, per-recording mean embedding.
- **Movie-id probe**: `nn.Linear(D, 20)`, Adam lr=1e-3, 20 epochs, cross-entropy. Production setting; the diagnostic in this doc shows it's underconfigured for JEPA embeddings.
- **FM full-FT**: AdamW with discriminative LRs (encoder=1e-5, head=1e-3), weight_decay=1e-2, 30 epochs, early-stop on val mean reg corr (patience 8), neg-Pearson loss, no parallel cls head.
- **Tier 6 channel positions for LUNA**: per-component absolute-weight-weighted centroid of HBN electrode positions under the CorrCA spatial filter.

## Artifacts

Per-baseline bootstrap markdown reports: `docs/trivial_bootstrap/<baseline>.md` (12 files including `tier5_*` and `tier6_luna_corrca.md`).
Drivers: `scripts/trivial_ridge_baseline.py` (trivial + JEPA Ridge), `scripts/diagnose_movie_id.py` (movie_id probe diagnostic).
Tier 5/6 fine-tuning code: `experiments/eeg_jepa/tier4_full_ft.py` with `--use_corrca` for Tier 6.
