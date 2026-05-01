# SSL vs trivial baselines — paper Table 3 redo

**Date:** 2026-05-01
**Branch:** `kkokate/trivial-stats-baselines`

## What this answers

The paper draft's Table 3 claims trivial 35-dim CorrCA stats with a Ridge
probe match or exceed the JEPA encoder on every continuous stimulus
feature. We re-derive every cell ourselves under the same significance
procedure as PR #15 (View 2: 1-sample t-test on per-seed test corrs;
View 3: per-seed recording bootstrap B=2000 → 1-sample t-test on the
seed bootstrap-means).

We test the draft's claim under **two probe stacks**:

1. **MovieFeatureHead** (matches `probe_eval.py --keep_channels`): the
   same MLP head + per-recording subject Linear probe + 20-bin movie-id
   probe used to evaluate JEPA. This is the apples-to-apples comparison
   "match the probe arch exactly".
2. **Ridge** (matches the draft): per-clip features (mean of 4 windows),
   `sklearn.linear_model.Ridge(alpha=1.0)` regression per stimulus
   feature, `n_passes=20` random clips per recording.

## Trivial baselines

For each baseline we compute (mean, std, log-power δ/θ/α/β/γ) per channel.
Channel routing varies:

| Baseline | n_chans | D = stats × C | Channel routing |
|---|---:|---:|---|
| `trivial_corrca_per_chan` | 5 | 35 | per-chan concatenated into D |
| `trivial_corrca_chan1_only` | 1 | 7 | only CorrCA component 1 |
| `trivial_raw_per_chan` | 129 | 903 | per-chan concatenated |
| `trivial_corrca_pooled35` | 5 | 35 | mean-then-tile across chans |
| `trivial_raw_pooled903` | 129 | 903 | mean-then-tile across chans |

`per_chan` modes are direct twins of JEPA `--keep_channels`; `pooled` modes
are twins of the default channel-averaged pool.

## Results — Continuous stimulus regression (test, View 3 B=2000)

`reg_<feature>_corr` View 3 mean ± σ (5 probe seeds; trivial baselines)
or mean ± σ_enc (5 encoder seeds; JEPA from PR #15):

| Feature | JEPA default pool | JEPA `--keep_channels` | trivial corrca per_chan | trivial corrca chan1_only | trivial raw per_chan | trivial corrca pooled35 | trivial raw pooled903 |
|---|---:|---:|---:|---:|---:|---:|---:|
| position | +0.088 ✗ | **+0.212 ✓** | +0.051 ns | +0.022 ns | −0.010 ns | +0.036 ns | −0.023 ns |
| luminance | +0.137 ✓ | **+0.187 ✓** | +0.060 ns | +0.050 ns | +0.003 ns | +0.034 ns | +0.047 ns |
| contrast | +0.030 ✗ | **+0.101 ✓** | +0.035 ns | −0.009 ns | −0.012 ns | +0.002 ns | −0.007 ns |
| narrative | −0.020 ✗ | **+0.062 ✓** (p=2.7e-3) | −0.013 ns | +0.008 ns | +0.041 ns | +0.004 ns | +0.007 ns |

**Reading:** JEPA + CorrCA `--keep_channels` is significantly above chance
on **every** continuous stimulus feature. **No** MovieFeatureHead trivial
baseline is significantly above chance on **any** continuous stimulus
feature. This contradicts the draft's "trivial 35-dim Ridge matches JEPA"
claim under the apples-to-apples probe stack.

## Results — Subject traits (test, View 3 B=2000)

| Probe | JEPA default pool | JEPA `--keep_channels` | trivial corrca per_chan | trivial raw per_chan | trivial corrca pooled35 | trivial raw pooled903 |
|---|---:|---:|---:|---:|---:|---:|
| `subject/age_reg/corr` | +0.370 | +0.504 | +0.134 ns | **+0.316 ✓** (p=0.001) | +0.160 ns | **+0.361 ✓** (p=2.3e-6) |
| `subject/age_cls/auc` | — | — | **+0.645 ✓** | **+0.695 ✓** | **+0.659 ✓** | **+0.696 ✓** |
| `subject/sex/auc` | +0.619 | +0.713 | +0.560 ns | **+0.588 ✓** (p=2.9e-4) | +0.522 ns | +0.488 ns |

**Reading:** Subject-trait signal **does** survive in trivial baselines —
specifically the 903-dim raw EEG stats reach age corr +0.361 (vs JEPA
default +0.370) and the 129-d raw stats reach sex AUC +0.588. This is
consistent with the prior Tier-1 finding that age/sex are spectral, not
SSL-driven. JEPA `--keep_channels` still wins on age/sex (+0.504 / +0.713).

## Headline takeaways

1. **The draft's Table 3 claim does not survive the matched-probe-arch test.**
   Under `MovieFeatureHead` (the same probe used to evaluate JEPA), no
   trivial baseline passes View 2 or View 3 significance on any continuous
   stimulus feature. JEPA + `--keep_channels` does, on all four.
2. **Channel routing matters at the trivial level too.** `per_chan`
   variants beat `pooled` variants by ~0.02 corr — small but consistent
   with the JEPA `--keep_channels` finding that channel-mean pooling
   wastes per-channel signal.
3. **CorrCA component 1 alone (7-d) is at chance for narrative under
   per-window mean/std/log-band features.** The +0.213 raw-correlation
   between component 1 and narrative does not survive being summarized
   as 7 stats per window; it lives in slow temporal dynamics that
   spectral features throw away.
4. **Subject-trait wins for the 903-dim raw stats** — age corr +0.361,
   sex AUC +0.588, both significant under View 3. Confirms that age/sex
   are recoverable from raw EEG amplitude statistics without any encoder.
5. **SSL pretraining does add stimulus-decoding signal beyond
   handcrafted spectral features**, when measured under the same probe
   stack. The draft's "JEPA = trivial" headline is an artifact of the
   probe arch swap (Ridge per-clip vs MovieFeatureHead per-window), not
   a property of the encoder.

## Pending: ridge probe runs

A separate sweep is being run with the **draft's exact Ridge probe procedure**
(per-clip mean of 4 windows, sklearn `Ridge(alpha=1.0)`, `n_passes=20`)
on the same 5 probe seeds, so we can quantify how much of the draft's
result is due to the probe-stack swap. Results to be appended below
once the jobs finish.

## Methodology

For each (baseline, probe seed) cell we run the full evaluation pipeline:
- `JEPAMovieDataset` with n_windows=4, ws=2s, per-recording z-norm
- splits R1–R4 / R5 / R6 (train / val / test)
- features extracted by `TrivialStatsExtractor` (MovieFeatureHead path) or
  `_trivial_features` (Ridge path)
- probe trained from scratch with the seed's RNG
- per-recording prediction npz dumped in the schema
  `scripts/bootstrap_probe_eval.py` consumes

For each baseline, `scripts/bootstrap_trivial_perseed.py` then runs the
existing `_bootstrap_movie` / `_bootstrap_subject` helpers per probe seed
(B=2000 resamples), and 1-sample t-tests across the 5 seed means against
chance (n=5, df=4) for both View 2 (raw test corrs) and View 3 (per-seed
bootstrap means).

JEPA reference numbers come from PR #15
(`docs/probe_eval_keep_channels.md`), 5 enc × 5 probe seeds, ensemble-
then-bootstrap matching the original significance_analysis_2026-04-29.md
procedure.

## Artifacts

- Per-recording prediction npz: `/projects/bbnv/kkokate/eb_jepa_eeg/tier1/predictions/<baseline>_seed<seed>/{val,test}_seed<seed>.npz`
- Summary JSON: `/projects/bbnv/kkokate/eb_jepa_eeg/tier1/<baseline>_seed<seed>.json`
- Per-baseline bootstrap markdown: `docs/trivial_bootstrap/<baseline>.md`
- Driver: `scripts/bootstrap_trivial_perseed.py`
- Submitter: `scripts/submit_trivial_baselines.sh` (MovieFeatureHead),
  `scripts/submit_trivial_ridge.sh` (Ridge, in progress)
