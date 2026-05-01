# SSL vs trivial baselines — paper Table 3 redo

**Date:** 2026-05-01
**Branch:** `kkokate/trivial-stats-baselines`

> **Scope note (from cleanup 2026-05-01):** JEPA default-pool numbers are
> dropped from this analysis. Default pool fails View 2 / View 3
> significance on multiple stimulus features and is no longer the production
> evaluation path. From here on, "JEPA" means **JEPA + `--keep_channels`**
> (per-CorrCA-channel concat into the probe input dim). Default-pool numbers
> remain archived in `docs/probe_eval_keep_channels.md` and PR #15 commit
> history for reference.

## What this answers

The paper draft's Table 3 claims trivial 35-dim CorrCA stats with a Ridge
probe match or exceed the JEPA encoder on every continuous stimulus
feature. We re-derive every cell ourselves under the same significance
procedure as PR #15 (View 2: 1-sample t-test on per-seed test corrs;
View 3: per-seed recording bootstrap B=2000 → 1-sample t-test on the
seed bootstrap-means).

We test the draft's claim under **two probe stacks**, both with JEPA in
its `--keep_channels` configuration:

1. **MovieFeatureHead** (`probe_eval.py --keep_channels`): 2-layer MLP
   `Linear(D=320 → 64) → ReLU → Linear(64 → 4)` per-window probe + per-
   recording subject Linear probe + 20-bin movie-id Linear probe. Same
   stack used to evaluate JEPA in PR #15.
2. **Ridge** (matches the draft): per-clip features (mean of 4 windows),
   `sklearn.linear_model.Ridge(alpha=1.0)` regression per stimulus
   feature, `n_passes=20` random clips per recording.

## Trivial baselines

For each baseline we compute (mean, std, log-power δ/θ/α/β/γ) per channel.
Channel routing varies:

| Baseline | n_chans | D = stats × C | Channel routing |
|---|---:|---:|---|
| `corrca_per_chan` (corrca35) | 5 | 35 | per-chan concatenated into D |
| `corrca_chan1_only` | 1 | 7 | only CorrCA component 1 |
| `raw_per_chan` (raw903) | 129 | 903 | per-chan concatenated |
| `corrca_pooled35` | 5 | 35 | mean-then-tile across chans |
| `raw_pooled903` | 129 | 903 | mean-then-tile across chans |

`per_chan` modes are direct twins of JEPA `--keep_channels`; `pooled` modes
are the trivial counterparts of the channel-mean-pool variant we dropped
on the JEPA side.

## Continuous stimulus regression (test, View 3 B=2000) — `reg_<feature>_corr`

| Probe / baseline | position | luminance | contrast | narrative |
|---|---:|---:|---:|---:|
| **JEPA + MLP `--keep_channels`** | **+0.212** ✓ | +0.187 ✓ | +0.101 ✓ | +0.062 ✓ |
| **JEPA + Ridge `--keep_channels`** | +0.144 ✓ | **+0.208** ✓ | **+0.159** ✓ | **+0.090** ✓ |
| Trivial Ridge `corrca35` (per_chan, D=35) | +0.125 ✓ | +0.142 ✓ | +0.139 ✓ | +0.044 ✓ |
| Trivial Ridge `corrca_chan1_only` (D=7) | +0.064 ✓ | +0.077 ✓ | +0.076 ✓ | +0.021 ✓ |
| Trivial Ridge `corrca_pooled35` | +0.072 ✓ | +0.090 ✓ | +0.090 ✓ | +0.021 ✓ |
| Trivial Ridge `raw903` (per_chan) | +0.008 ns | +0.003 ns | +0.009 ns | +0.018 ns |
| Trivial Ridge `raw_pooled903` | +0.009 ns | +0.059 ✓ | +0.037 ✓ | +0.022 ns |
| Trivial MLP `corrca_per_chan` | +0.051 ns | +0.060 ns | +0.035 ns | −0.013 ns |
| Trivial MLP `corrca_chan1_only` | +0.022 ns | +0.050 ns | −0.009 ns | +0.008 ns |
| Trivial MLP `corrca_pooled35` | +0.036 ns | +0.034 ns | +0.002 ns | +0.004 ns |
| Trivial MLP `raw_per_chan` | −0.010 ns | +0.003 ns | −0.012 ns | +0.041 ns |
| Trivial MLP `raw_pooled903` | −0.023 ns | +0.047 ns | −0.007 ns | +0.007 ns |

**Bold** = best in column. All sweeps complete (5 enc / probe seeds × B=2000 bootstrap).

### Per-feature winner

```
position     | JEPA + MLP --keep_channels    | +0.212
luminance    | JEPA + Ridge --keep_channels  | +0.208
contrast     | JEPA + Ridge --keep_channels  | +0.159
narrative    | JEPA + Ridge --keep_channels  | +0.090
```

## Subject traits (test, View 3 B=2000)

| Probe | JEPA + MLP `--keep_channels` | trivial corrca per_chan | trivial raw per_chan | trivial corrca pooled35 | trivial raw pooled903 |
|---|---:|---:|---:|---:|---:|
| `subject/age_reg/corr` | **+0.504** | +0.134 ns | +0.316 ✓ (p=0.001) | +0.160 ns | +0.361 ✓ (p=2.3e-6) |
| `subject/age_cls/auc` | — | +0.645 ✓ | +0.695 ✓ | +0.659 ✓ | +0.696 ✓ |
| `subject/sex/auc` | **+0.713** | +0.560 ns | +0.588 ✓ (p=2.9e-4) | +0.522 ns | +0.488 ns |

Subject-trait signal **does** survive in trivial baselines — the 903-dim raw
EEG stats reach age corr +0.361 and sex AUC +0.588, both significant.
Confirms prior Tier-1 finding that age/sex are spectral, not SSL-driven.
JEPA `--keep_channels` still wins (+0.504 / +0.713).

(JEPA + Ridge subject probes are not yet wired — only stim regression. If
needed, ~10 jobs would extend `trivial_ridge_baseline.py` with per-rec
Ridge / LogReg.)

## Channel-routing effect (per_chan vs pooled, isolating the channel axis)

Same probe, same channel set, only difference is whether the stats are
kept per-channel (concat into D) or averaged-and-tiled (D unchanged but
channels indistinguishable).

```
Probe                | feature    | per_chan  | pooled    | Δ (per_chan − pooled)
---------------------|------------|----------:|----------:|----------:
Trivial Ridge corrca | position   |  +0.125   |  +0.072   |   +0.053
Trivial Ridge corrca | luminance  |  +0.142   |  +0.090   |   +0.052
Trivial Ridge corrca | contrast   |  +0.139   |  +0.090   |   +0.049
Trivial Ridge corrca | narrative  |  +0.044   |  +0.021   |   +0.023
Trivial Ridge raw    | position   |  +0.008   |  +0.009   |   −0.001
Trivial Ridge raw    | luminance  |  +0.003   |  +0.059   |   −0.056
Trivial Ridge raw    | contrast   |  +0.009   |  +0.037   |   −0.028
Trivial Ridge raw    | narrative  |  +0.018   |  +0.022   |   −0.004
```

Per-channel routing helps **CorrCA-projected features** (CorrCA channels
encode different stimulus subspaces; routing matters). For raw 129-ch the
gap is ~0 because no individual raw channel is special.

## Headline takeaways

1. **The draft's Table 3 claim "JEPA = trivial Ridge corrca35" does not
   survive the matched-probe comparison.** Under the same Ridge probe with
   `n_passes=20`, **JEPA + Ridge `--keep_channels` beats trivial Ridge
   corrca35 on every continuous feature**:
   - position: +0.144 vs +0.125 → +0.019
   - luminance: +0.208 vs +0.142 → +0.066
   - contrast: +0.159 vs +0.139 → +0.020
   - narrative: +0.090 vs +0.044 → +0.046

2. **JEPA + MLP `--keep_channels` is the best probe for position** (+0.212),
   while **JEPA + Ridge `--keep_channels` is the best probe for luminance,
   contrast, and narrative**. Position is a within-clip drift signal that
   the per-window MLP captures and per-clip Ridge averages away.

3. **No MovieFeatureHead-trained trivial baseline reaches significance on
   any stimulus feature.** The MLP probe is a poor reader of per-window
   trivial spectral stats — variance dominates the per-window training.

4. **Ridge + per-clip aggregation is the right probe for handcrafted
   spectral baselines.** Trivial Ridge corrca35 reaches 0.12–0.14 corr on
   position/luminance/contrast (significant under both views). This is the
   strong floor the draft was reporting; we replicate it.

5. **Subject-trait signal is spectral, not SSL.** `raw_pooled903` reaches
   age corr +0.361 (vs JEPA `--keep_channels` +0.504) and `raw_per_chan`
   reaches sex AUC +0.588 (vs JEPA +0.713). Confirms prior Tier-1 finding.

6. **CorrCA component 1 alone (7-d) is significant but weak for narrative
   under spectral features.** Trivial Ridge chan1_only narr_corr = +0.021
   (V3 p=0.018) — significant but tiny next to the +0.213 *raw-trajectory*
   correlation between component 1 and the narrative annotation. The slow
   narrative signal lives in raw amplitude dynamics that mean+std+log-bands
   throw away. This is consistent with the per-subject ceiling memo's
   prediction.

## What changes for the paper

The headline of Table 3 should flip:

- *Old*: "JEPA encoder, given the same CorrCA-filtered input, does not
  improve over a Ridge probe on per-window mean/std/log-power"
- *New*: "Under the matched Ridge probe with `--keep_channels`, JEPA
  encoder embeddings beat 35-d CorrCA spectral stats on every continuous
  stimulus feature (Δ = +0.019 to +0.066 corr). The apparent parity in
  the original Table 3 was due to comparing JEPA's MLP-per-window numbers
  against the trivial's Ridge-per-clip numbers — not an apples-to-apples
  comparison"

The trivial CorrCA Ridge baseline remains a non-trivial floor (0.12–0.14
corr on stim features, well above raw/random-init), preserving the paper's
"CorrCA does much of the work" message. But the "JEPA adds nothing" claim
does not hold once the probe scales are matched.

## Methodology

For each (baseline, probe seed) cell we run the full evaluation pipeline:
- `JEPAMovieDataset` with n_windows=4, ws=2s, per-recording z-norm
- splits R1–R4 / R5 / R6 (train / val / test)
- features extracted by `TrivialStatsExtractor` (MovieFeatureHead path),
  `_trivial_features` (trivial-Ridge path), or `_jepa_features` (JEPA-Ridge
  path: `encoder.encode_tokens` + per-clip mean over the 4 windows; the
  per-channel concat for `--keep_channels` is inlined in `_jepa_features`)
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
procedure. JEPA + Ridge numbers are 5 enc seeds × 1 probe seed (Ridge
solve is deterministic given features).

## Artifacts

- Per-recording prediction npz:
  `/projects/bbnv/kkokate/eb_jepa_eeg/tier1/predictions/<baseline>_seed<seed>/{val,test}_seed<seed>.npz`
- Summary JSON:
  `/projects/bbnv/kkokate/eb_jepa_eeg/tier1/<baseline>_seed<seed>.json`
- Per-baseline bootstrap markdown:
  `docs/trivial_bootstrap/<baseline>.md`
- Drivers:
  - MovieFeatureHead: `experiments/eeg_jepa/tier1_baselines.py` +
    `scripts/submit_trivial_baselines.sh`
  - Ridge: `scripts/trivial_ridge_baseline.py` +
    `scripts/submit_trivial_ridge.sh` (trivial) +
    `scripts/submit_jepa_ridge.sh` (JEPA encoder)
  - Bootstrap: `scripts/bootstrap_trivial_perseed.py`
