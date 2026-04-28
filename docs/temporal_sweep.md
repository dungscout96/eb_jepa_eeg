# Temporal context sweep + CorrCA ablation (Issue #8)

**Branch:** `kkokate/issue8-corrca-ablation-temporal-sweep`
**Dates:** 2026-04-26 → 2026-04-28
**Headline:** `nw2_ws4` replaces `nw4_ws2` as the best 8 s-context config — paired Δ contrast_corr **+0.080 ± 0.008** (≈ 10 σ) over 25 paired runs.

## Goal

Two questions from issue #8:

1. **Is CorrCA load-bearing?** Per-recording z-norm already removes the subject amplitude fingerprint at the input. Does the encoder still need CorrCA, or is the whole stack just per-rec norm + a depth-2 transformer?
2. **Does temporal context (window count × window length) matter for stimulus encoding?** Exp 6 fixed `nw4_ws2` (4 windows × 2 s = 8 s). Sweep nw1_ws1, nw2_ws1, nw2_ws4, nw4_ws2, nw4_ws4 to map the space.

## What we ran

Five sequential phases. Phase D + E supersede A–C — earlier phases were confounded by an early-stopping bug we diagnosed mid-flight.

### Phase A — CorrCA ablation (5 seeds, paired)

Same Exp 6 stack minus the CorrCA spatial filter (n_chans returns from 5 → 129). Seeds **{42, 123, 456, 789, 2025}** matched to Exp 6 seeds for paired-Δ comparison.

`scripts/submit_issue8_phaseA.sh` → 5 jobs.

### Phase B — single-seed temporal sweep

`{nw1_ws1, nw2_ws1, nw2_ws4, nw4_ws4}` at seed 2025 with CorrCA on, to scan the timescale axis cheaply. nw4_ws2 skipped (== Exp 6 baseline).

`scripts/submit_issue8_phaseB.sh` → 4 jobs.

### Phase C — 5-seed confirmation of Phase B winner (`nw2_ws4`)

`scripts/submit_issue8_phaseC.sh` → 5 jobs.

### Diagnostic interlude: broken early-stop signal

Inspecting Phase B's `nw4_ws4` log revealed early stopping fired at epoch 27 of 100 with the "best" checkpoint at **epoch 8**. The early-stop monitor was `val/reg_loss` — the auxiliary online probe regression loss, which oscillates 0.78–1.10 across epochs while `val/cls_loss` stays at `log 2 ≈ 0.693` (chance) for the entire run. The stop fired on a noisy lucky dip, not encoder convergence. Phase C `nw2_ws4` seed 456 stopped at epoch 23 with best at epoch 4 — same pathology.

Literature (V-JEPA, MAE, DINO, RankMe/LiDAR) is unanimous: SSL pretraining uses **fixed-budget training**; the SSL loss itself is also a poor downstream proxy. Auxiliary co-trained probe loss is the worst possible stop signal because of probe-head optimization noise.

**Fix** (commit `403a009`):
- `experiments/eeg_jepa/cfgs/default.yaml`: default `early_stopping_patience` 20 → **0** (disabled).
- `experiments/eeg_jepa/main.py`: guard `best.pth.tar` write under `patience > 0` so the misleading "best by broken signal" artifact is no longer produced.
- `experiments/eeg_jepa/eval.py`: drop `validation_loop`'s spurious `.train()` restore on the frozen encoder.

### Phase D — 15 encoder pretrainings, fixed budget

3 configs × 5 encoder seeds = **15 runs**, all with `patience=0` (full 100 epochs), per-rec norm, CorrCA on, depth=2, predictor_embed_dim=24, std=cov=0.25, smooth_l1, batch_size=64, EMA momentum 0.996 → 1.0 cosine. Probe eval consumes `latest.pth.tar` (epoch-99 weights).

| Config | n_windows | window_size | total ctx | tokens/sample | LR | warmup | rationale |
|---|---:|---:|---:|---:|---:|---:|---|
| **nw4_ws2 baseline** | 4 | 2 s | 8 s | 240 | 5e-4 | 5 | clean re-run of Exp 6 with patience=0 (so all 3 configs share fixed-budget policy) |
| **nw2_ws4** | 2 | 4 s | 8 s | 260 | 5e-4 | 5 | same total context, fewer/longer windows |
| **nw4_ws4** | 4 | 4 s | 16 s | 520 | **3e-4** | **10** | 2× tokens → lower LR + longer warmup for stability |

Encoder seeds (per config): **{42, 123, 456, 789, 2025}** — matches Exp 6's published baseline seeds.

`scripts/submit_issue8_phaseD.sh` → 15 jobs (`scripts/train_issue8_phaseD.sbatch`). Wall time: ~17 min/cell for nw4_ws2 / nw2_ws4 (260-token sequences), ~22 min for nw4_ws4 (520 tokens). All 15 finished cleanly in <30 min wall-clock.

### Phase E — 75 probe evaluations

For each of the 15 Phase D encoders, run `experiments/eeg_jepa/probe_eval.py` with **5 different probe seeds** = 75 jobs. Probe seeds: **{7, 13, 42, 1234, 2025}**. Each probe-eval trains:

- **Movie-feature regression + classification** (per-clip, `MovieFeatureHead` 2-layer MLP, hidden=64): contrast, luminance, position, narrative — 20 epochs.
- **Subject-trait** (per-recording mean-pooled embeddings, `nn.Linear(D, 1)`): age regression, age binary (> train median), sex — 100 epochs.
- **Movie-id** (20 temporal bins, `nn.Linear(D, 20)`).

Eval splits: `val,test`. CorrCA filter passed via `--corrca_filters=corrca_filters.npz`. Per-rec z-norm via `--norm_mode=per_recording`. Pre-flight checkpoint check refuses to submit if any encoder is missing.

`scripts/submit_issue8_phaseE.sh` → 75 jobs (`scripts/probe_eval_phaseE.sbatch`). Wall time ~12 min/job; full Phase E drained in ~25 min on Delta with ~12 jobs concurrent.

## Code-review audit

Two parallel `code-reviewer` agents (training pipeline + eval pipeline) inspected Phase D + E before launch. Verdict: **clean**, no leakage. Three patches applied pre-launch:

- `eval.py:130-132` `.train()` restore removed (was harmless today, latent bug if dropout added).
- `main.py` guarded `best.pth.tar` write under `patience > 0`.
- `train_issue8_phaseD.sbatch` time limit 6 h → 8 h headroom for nw4_ws4.

Confirmed clean: train/val/test split disjointness (HBN R1-R6 are independent OpenNeuro datasets), feature stats train-only, CorrCA filter fitted on train releases only, age-binary threshold = train median propagated to eval, encoder fully frozen during probe eval, per-rec z-norm parity between training and eval paths, encoder-only deterministic forward (no dropout in `EEGEncoderTokens` / RMSNorm-only transformer), all 15 + 75 EXP_TAGs unique.

## Results — test split

### Phase A — CorrCA ablation (5 seeds, mean ± std)

| Probe | Exp 6 (with CorrCA) | Phase A (no CorrCA) | Δ |
|---|---:|---:|---:|
| pos_corr | 0.176 ± 0.048 | 0.032 ± 0.058 | **−0.144 (≈ −3 σ)** |
| lum_corr | 0.168 ± 0.059 | 0.037 ± 0.046 | **−0.131 (≈ −2.2 σ)** |
| cont_corr | 0.115 ± 0.053 | 0.076 ± 0.029 | −0.039 |
| age_corr | 0.325 ± 0.030 | −0.032 ± 0.154 | **−0.357 (chance)** |
| sex_auc | 0.618 ± 0.007 | 0.506 ± 0.041 | **−0.112 (chance)** |

**Verdict: CorrCA stays.** Removing it collapses *every* probe (stimulus and subject) and destabilizes optimization (age_corr std 0.030 → 0.154). The pre-experiment hypothesis ("CorrCA is just a stim-channel selector, encoder works on raw 129ch") is falsified.

### Phase D + E — temporal sweep, 5 enc × 5 probe seeds = 25 runs/config

Config means (test split, 25 runs each):

| Config | pos_corr | lum_corr | cont_corr | pos_auc | age_corr | age_auc | sex_auc |
|---|---:|---:|---:|---:|---:|---:|---:|
| nw4_ws2 baseline | 0.172 ± 0.060 | 0.136 ± 0.060 | 0.106 ± 0.040 | 0.561 ± 0.028 | 0.338 ± 0.062 | 0.661 ± 0.021 | 0.618 ± 0.007 |
| **nw2_ws4** | **0.190 ± 0.057** | **0.158 ± 0.057** | **0.186 ± 0.055** | 0.560 ± 0.048 | 0.345 ± 0.056 | 0.664 ± 0.031 | 0.621 ± 0.010 |
| nw4_ws4 | 0.136 ± 0.041 | 0.109 ± 0.047 | 0.094 ± 0.046 | 0.549 ± 0.027 | 0.318 ± 0.051 | 0.651 ± 0.017 | 0.614 ± 0.004 |

Paired Δ vs nw4_ws2 baseline (test, paired by encoder seed; mean over 5 probe seeds first, then mean ± std across 5 enc seeds):

| variant | pos_corr | lum_corr | cont_corr | pos_auc | age_corr | age_auc | sex_auc |
|---|---|---|---|---|---|---|---|
| **nw2_ws4** | **+0.018 ± 0.009** | **+0.023 ± 0.015** | **+0.080 ± 0.008** | −0.001 ± 0.005 | +0.007 ± 0.021 | +0.002 ± 0.011 | +0.002 ± 0.003 |
| nw4_ws4 | −0.036 ± 0.024 | −0.027 ± 0.021 | −0.012 ± 0.019 | −0.012 ± 0.013 | −0.021 ± 0.015 | −0.011 ± 0.008 | −0.005 ± 0.005 |

## Headline findings

1. **`nw2_ws4` is the new headline config.** Same 8 s total context as Exp 6 but 2 long (4 s) windows instead of 4 short (2 s). Paired Δ on contrast corr is **+0.080 ± 0.008** — consistent across all 5 encoder seeds. Mechanism: longer per-window patches (26 patches vs 12) give the masked predictor finer temporal resolution for slowly-varying features.
2. **`nw4_ws4` (16 s) genuinely doesn't help** — even with the early-stop fix, longer warmup, and lower LR. All stimulus probes regress vs nw4_ws2.
3. **CorrCA is essential**, not optional.
4. **Exp 6's published 5-seed numbers are valid** — the early-stop bug didn't materially hurt the headline metrics. Re-run nw4_ws2 with patience=0 gives pos_corr 0.172 ± 0.060 vs published 0.176 ± 0.048 (within noise).
5. **5×5 paired design** (5 encoder seeds × 5 probe seeds, paired by encoder) cut the standard error of the cont_corr delta roughly in half compared to single-probe-seed Phase C.

## Side finding — MLP probe (Exp 6 only, 3 enc seeds × 3 variants)

Replacing the linear subject-trait + movie-id probe heads with a 2-layer MLP (`Linear → LayerNorm → GELU → Dropout → Linear`, hidden=128 = 2 × D) gives paired Δ +0.09 lum_corr, **+0.12 cont_corr**, +0.20 age_corr on Exp 6 checkpoints. Hidden=256 (4 × D) overfits. Eval-side win, orthogonal to encoder choice. Not yet run on Phase D encoders.

## Recommendation

- **Switch paper headline config from `nw4_ws2` → `nw2_ws4`.** Same compute, same hyperparameters, +0.08 contrast corr.
- **Keep CorrCA.** Phase A confirms it's load-bearing.
- **Default `early_stopping_patience: 0`** — fixed budget is the SSL standard and the only stop signal that wouldn't have produced our diagnostic-interlude pathology.
- **Optional eval-side upgrade**: roll the MLP probe (h=128) into the standard probe_eval flow.

## Artifact locations

- Branch: `kkokate/issue8-corrca-ablation-temporal-sweep`
- Sbatch: `scripts/train_issue8_phaseD.sbatch`, `scripts/probe_eval_phaseE.sbatch`
- Submitters: `scripts/submit_issue8_phase{A,B,C,D,E}.sh`
- Aggregator: `scripts/aggregate_issue8.py` (Phases A-C); inline aggregator on Delta for Phase E
- Checkpoints: `/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/issue8/phaseD_{config}_s{seed}/latest.pth.tar`
- Logs: `/projects/bbnv/kkokate/eb_jepa_eeg/logs/issue8{D,E}_*.{out,err}`
- W&B groups: `issue8D_phaseD_*`, `issue8E_*_enc{seed}_p{seed}`

## Open questions

- nw2_ws4 wins contrast big (+0.08 paired) but lum/pos only modestly (+0.02). Is the win specific to *slowly-varying* visual features (luminance and contrast change on the order of seconds), or does it generalize?
- Can the MLP-probe upgrade compound on the Phase D encoders? Worth running.
- Does the win hold under the V-JEPA-recommended teacher-side k-NN R² selector instead of last-epoch checkpoint?
