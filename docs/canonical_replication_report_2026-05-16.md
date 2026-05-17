# Canonical 5-seed replication — nw=2 ws=4 (2026-05-16)

Spec-faithful replication of the locked canonical protocol against the
post-refactor (`refactor/eeg-only-library`) codebase. All 12 canonical
metrics significantly above chance (p<0.05, two-sided one-sample t-test).

## Headline table

L1 = single raw Pearson r / AUC / top-k accuracy on the full flat
(108 × 20 = 2160) test array.
L3 = mean ± 1σ of L2 means across 5 encoder seeds.
✓ = p<0.05 vs metric-specific chance.

| Metric | L1 (5-seed mean ± σ) | L3 (5-seed mean ± σ) | t-test |
|---|---|---|---|
| `reg_luminance_mean_corr` | +0.1875 ± 0.0075 | +0.1882 ± 0.0076 | ✓ (p=6.3e-7) |
| `reg_contrast_rms_corr` | +0.1976 ± 0.0188 | +0.1979 ± 0.0184 | ✓ (p=1.8e-5) |
| `reg_position_in_movie_corr` | +0.1885 ± 0.0209 | +0.1893 ± 0.0209 | ✓ (p=3.5e-5) |
| `reg_narrative_event_score_corr` | +0.0369 ± 0.0158 | +0.0369 ± 0.0162 | ✓ (p=0.0070) |
| `cls_luminance_mean_auc` | +0.5616 ± 0.0095 | +0.5615 ± 0.0096 | ✓ (p=1.4e-4) |
| `cls_contrast_rms_auc` | +0.5479 ± 0.0128 | +0.5481 ± 0.0127 | ✓ (p=0.0011) |
| `cls_position_in_movie_auc` | +0.5980 ± 0.0108 | +0.5984 ± 0.0108 | ✓ (p=3.4e-5) |
| `cls_narrative_event_score_auc` | +0.5246 ± 0.0125 | +0.5247 ± 0.0123 | ✓ (p=0.011) |
| `age_reg_corr` | +0.5828 ± 0.0383 | +0.5841 ± 0.0385 | ✓ (p=4.5e-6) |
| `sex_auc` | +0.6639 ± 0.0069 | +0.6639 ± 0.0068 | ✓ (p=7.1e-7) |
| `movie_id_top1` | +0.0993 ± 0.0044 | +0.0992 ± 0.0043 | ✓ (p=1.4e-5) (chance=0.05) |
| `movie_id_top5` | +0.3812 ± 0.0050 | +0.3814 ± 0.0052 | ✓ (p=5.7e-7) (chance=0.25) |

L1 and L3 agree to ≤0.001 across all metrics — bootstrap mean tracks the
raw point estimate with no resampling artifact. σ across seeds is small
(0.005 to 0.04) — encoder-training variance only, since probe_seed=42 is
fixed across encoder seeds.

## Protocol

### Encoder (MaskedJEPA)

| Component | Value |
|---|---|
| Class | `EEGEncoderTokens` ([eb_jepa/architectures.py:278](../eb_jepa/architectures.py#L278)) |
| `embed_dim` | 64 |
| `depth` | 2 transformer blocks |
| `heads` × `head_dim` | 4 × 16 |
| `patch_size` / `patch_overlap` | 50 / 20 samples (250 ms / 100 ms @ 200 Hz) |
| `mlp_dim_ratio` | 2.66 (GEGLU FFN) |
| Positional encoding | 4D Fourier on (chan.x, chan.y, chan.z, time_index) |
| Norm | RMSNorm in attention/FFN; LayerNorm in pos MLP |
| Predictor bottleneck | 64 → 24 → 64 |
| Output token grid | `[B, 5×2×26, 64]` per 8 s clip |

### Pretraining

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr=5e-4, cosine to 1e-6 |
| Warmup | 5 epochs |
| Epochs | 100 |
| Batch size | 64 |
| `n_windows × window_size_seconds` | 2 × 4 s |
| `data.norm_mode` | `per_recording` |
| `data.corrca_filters` | `corrca_filters.npz` (129 → 5, fit train-only) |
| `data.task` | `ThePresent` |
| `loss` | `smooth_l1(pred, target.detach()) + VCLoss(std=0.25, cov=0.25)` |
| `early_stopping_patience` | 0 (full 100 epochs) |
| Encoder seeds | {42, 123, 456, 789, 2025} |

### Eval (canonical, deterministic)

| Stage | Implementation |
|---|---|
| Probe heads | `sklearn.Ridge(α=1)` (reg+age), `LogReg(C=1, lbfgs)` (cls+sex), multinomial LR (`max_iter=2000`) for movie_id 20-bin |
| n_passes | 20 per recording |
| Train iteration | outer pass × inner shuffled-recording (`train_order=True`), `probe_seed=42` |
| Val/test iteration | rec × passes (sequential) so saved arrays reshape to `(n_rec, 20)` |
| Bootstrap | recording-level, B=2000, 95% CI |
| L1 | single raw Pearson r / AUC / top-k on full flat (2160,) test |
| L2 | bootstrap mean ± 95% CI per seed |
| L3 | mean ± 1σ of L2 means across 5 encoder seeds |
| t-test | one-sample two-sided of 5 L2 means against per-metric chance (0, 0.5, 0.05, 0.25) |

### Splits

train=701, val=293, test=108. The `(108, 20)` test reshape matches the
spec's 108-recording test set exactly. val is captured but not reported
(protocol audit only).

## Files produced

Per-seed (on Delta):

- `/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/canonical_nw2ws4_s{42,123,456,789,2025}/latest.pth.tar`
- `…/canonical_preds/preds_seed{seed}.npz` — flat per-clip predictions/targets, keys `test_*_pred`, `test_*_target`, `test_movie_id_probs`, `test_age_reg_pred`, `test_sex_prob`, `test_rec_ids`
- `…/canonical_L2.json` — per-seed L1 + L2 (bootstrap mean & 95% CI per metric)

5-seed aggregate:

- `/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/canonical_nw2ws4_L3/canonical_nw2ws4_L3.json`

## Pipeline code

| File | Purpose |
|---|---|
| [eb_jepa/evaluation/probe_eval_canonical.py](../eb_jepa/evaluation/probe_eval_canonical.py) | sklearn Ridge/LogReg heads, n_passes=20, probe_seed=42 deterministic clip starts |
| [eb_jepa/evaluation/bootstrap_canonical.py](../eb_jepa/evaluation/bootstrap_canonical.py) | (n_rec, 20) reshape, B=2000 recording-level bootstrap, emits `{L1, L2}` JSON |
| [scripts/aggregate_and_print.py](../scripts/aggregate_and_print.py) | L3 across seeds + t-test vs chance |
| [experiments/eeg_jepa/sbatch/canonical_replication.sbatch](../experiments/eeg_jepa/sbatch/canonical_replication.sbatch) | one-seed end-to-end (pretrain → probe → bootstrap) |
| [experiments/eeg_jepa/sbatch/canonical_probe_only.sbatch](../experiments/eeg_jepa/sbatch/canonical_probe_only.sbatch) | probe + bootstrap against existing checkpoint |
| [experiments/eeg_jepa/sbatch/canonical_aggregate.sbatch](../experiments/eeg_jepa/sbatch/canonical_aggregate.sbatch) | L3 aggregator (sbatch dep on the 5 per-seed jobs) |
| [experiments/eeg_jepa/sweeps/canonical_5seed.py](../experiments/eeg_jepa/sweeps/canonical_5seed.py) | full 5-seed neurolab sweep |
| [experiments/eeg_jepa/sweeps/canonical_5seed_probe_only.py](../experiments/eeg_jepa/sweeps/canonical_5seed_probe_only.py) | probe-only re-run sweep |

Commits on `refactor/eeg-only-library`:

- `a2ceb4e` — initial canonical pipeline
- `5746147` — drop per-job git pull (race fix)
- `2e35cea` — cwd guard + propagate inner exit code
- `ac17ec6` — drop `LogReg(multi_class=)` (sklearn ≥1.5) + probe-only re-run path

## Bugs hit and fixed during the run

1. **Concurrent `git fetch` race.** Five Delta jobs racing on
   `.git/refs/*` broke the `&&` chain; jobs exited <10s with
   sacct State=COMPLETED. Fix: drop per-job pull (pull once before
   submit) at `2e35cea`.
2. **Inner exit code masked.** neurolab wrapper's trailing echo
   returned 0 regardless of inner failure. Fixed by ending the outer
   command in `|| exit $?`.
3. **Wrong cwd inside sbatch.** `cd "$SLURM_SUBMIT_DIR"` resolves to
   `$HOME` under neurolab, breaking the relative `experiments/eeg_jepa/train.py`
   path. Replaced with an explicit guard that fails loud if cwd is
   wrong.
4. **sklearn API drift.**
   `LogisticRegression(multi_class="multinomial")` was removed in
   sklearn ≥1.5; lbfgs auto-handles multinomial when y has >2 classes.
   Pretraining completed cleanly for all 5 seeds before this crashed
   probe_eval; recovered by re-running the probe-only stage against
   existing checkpoints (`ac17ec6`).

## SLURM provenance

Successful sweep (after recovery):

| Stage | Job IDs | Wall time |
|---|---|---|
| Pretrain (5 seeds) | 18295900-04 | ~25 min – 1h 6min |
| Probe + bootstrap (5 seeds, after sklearn fix) | 18296367-71 | ~13 min each |
| L3 aggregate | 18296372 | ~6 s |

W&B project: `sccn/eb_jepa`, groups `canonical_nw2ws4` (pretrain) and
`canonical_nw2ws4_probe` (probe eval).

## Spec-faithfulness vs prior canonical-baselines reference

`docs/canonical_baselines_2026-05-04.md` referenced in the spec text
does not exist in this repo. This run is the first canonical
replication on `refactor/eeg-only-library`. Numbers here become the
new reference for that branch; cross-checks against the pre-refactor
canonical_baselines should be done by re-running the pre-refactor
codebase on the same data and comparing — out of scope for this
report.
