# Phase 1 — Encoder Diagnostic Findings (2026-05-02)

**Anchor:** `phaseD_nw4ws2_baseline_s42` (one enc seed, 5 probe seeds per
condition, all metrics on the test split). Raw per-condition table:
[`phase1_diagnostics.md`](phase1_diagnostics.md). Raw per-seed dict:
[`phase1_results.json`](phase1_results.json).

> **Caveat — variance.** This is one encoder seed × five probe seeds (n=5).
> The 5-enc × 5-probe Phase D published result is the right anchor for absolute
> numbers; here we look at *relative* shape across diagnostic conditions.
> Differences below are reported as the rank/sign pattern across taps, not as
> absolute headline values.

## Layout of the matrix

18 conditions = 12 (`layer × tower × routing`) + 5 (`per-channel attribution`)
+ 1 (`prepred`).

| Code | Meaning |
|---|---|
| `<layer>_<tower>_<routing>` | full-pool tap |
| `layer ∈ {patch_embed, block0, final}` | depth — block0 = after first transformer block; final = after both |
| `tower ∈ {stu, tea}` | student (context) vs EMA target encoder |
| `routing ∈ {mp, kc}` | mean-pool (default) vs `--keep_channels` |
| `final_stu_chN` | mean-pool, but slice only encoder channel N |
| `final_stu_prepred` | final-layer student tokens projected through `predictor.input_proj` (24-d bottleneck) |

## Findings

### 1. The transformer is not adding information for stimulus probes

For every continuous stimulus probe, **block0 ≈ final** (within seed noise),
and `patch_embed_*_kc` already gets close to `final_*_kc`:

| Probe (kc routing, student) | patch_embed | block0 | final |
|---|---:|---:|---:|
| reg_narrative_event_score_corr | +0.030 | +0.027 | +0.026 |
| reg_position_in_movie_corr     | +0.091 | +0.115 | +0.124 |
| reg_luminance_mean_corr        | +0.112 | +0.134 | +0.129 |
| reg_contrast_rms_corr          | +0.071 | +0.077 | +0.072 |

For narrative the patch-embed alone already gets the entire signal. For
position/lum/contrast the first transformer block adds ~30–50 %; the second
block adds zero. **Block 1 (the second of two transformer blocks) is dead
weight for stim regression.**

### 2. The teacher EMA tower offers no advantage

For every probe, `*_tea_*` is within seed noise of `*_stu_*`. The EMA target
is not a cleaner representation than the live student. The "predict against a
shifting target" hypothesis from the Phase-1 plan is rejected on this data.

### 3. The 24-d predictor bottleneck preserves all stimulus information

`final_stu_prepred` is competitive with `final_stu_mp` on every metric:

| Probe | final_stu_mp | final_stu_prepred | Δ |
|---|---:|---:|---:|
| reg_narrative_event_score_corr | -0.002 | **+0.033** | +0.035 |
| reg_position_in_movie_corr     | +0.190 | +0.158 | -0.032 |
| reg_luminance_mean_corr        | +0.175 | +0.173 | -0.002 |
| reg_contrast_rms_corr          | +0.115 | **+0.137** | +0.022 |
| cls_position_in_movie_auc      | +0.575 | +0.566 | -0.010 |
| cls_contrast_rms_auc           | +0.547 | **+0.574** | +0.026 |

Prepred even *beats* the un-projected final on narrative and contrast. This
rules out the predictor bottleneck as the lossy stage. **Tightening or widening
the 24-d bottleneck is unlikely to recover signal.**

### 4. Per-channel attribution shows specialized routing

Single-channel mean-pool at `final, student`:

| Channel | narr | pos | lum | contrast | age_corr | sex_auc |
|---:|---:|---:|---:|---:|---:|---:|
| **ch0** | -0.011 | +0.089 | +0.093 | +0.070 | +0.325 | +0.605 |
| **ch1** | -0.012 | **+0.178** | **+0.152** | **+0.129** | +0.279 | +0.585 |
| **ch2** | -0.011 | +0.125 | +0.113 | +0.092 | +0.348 | **+0.650** |
| **ch3** | +0.018 | +0.147 | +0.119 | +0.082 | +0.365 | +0.593 |
| **ch4** | **+0.031** | +0.146 | +0.142 | +0.096 | +0.360 | +0.648 |

Each stim feature peaks in a different channel. Position/luminance/contrast
cluster in **ch1**; narrative is in **ch3–ch4**; sex is in **ch2** (and ch4).
The default mean-pool was diluting all of these by ~5×, exactly the failure
mode that motivated `--keep_channels`. **`keep_channels` is structurally
correct — channels are not redundant.**

### 5. The encoder *does* learn subject traits

Patch-embed alone gives chance-level age/sex; the encoder lifts age_corr from
≈ 0 to +0.50 and sex_auc from ≈ 0.50 to +0.71:

| Probe | patch_embed_stu_kc | block0_stu_kc | final_stu_kc |
|---|---:|---:|---:|
| subject/age_reg/corr | +0.031 | +0.471 | **+0.498** |
| subject/age_cls/auc  | +0.543 | +0.699 | **+0.725** |
| subject/sex/auc      | +0.549 | +0.680 | **+0.710** |

For traits, both transformer blocks contribute (block0 → final still adds
~+0.03 corr). The encoder is doing real work here, just not for stim features.

### 6. Mean-pool vs keep_channels — non-uniform

| Probe | best mp | best kc | winner |
|---|---:|---:|---|
| reg_narrative_event_score_corr | -0.002 (final_stu) | +0.027 (final_stu) / +0.030 (block0_tea) | **kc** |
| reg_position_in_movie_corr     | **+0.190** (final_stu) | +0.124 (final_stu) | mp (high seed variance) |
| reg_luminance_mean_corr        | **+0.175** (final_stu) | +0.129 (final_stu) | mp (high seed variance) |
| reg_contrast_rms_corr          | +0.120 (block0_stu) | +0.083 (final_tea) | mp |
| subject/age_reg/corr           | +0.382 (final_stu) | **+0.498** (final_stu) | **kc** |
| subject/sex/auc                | +0.624 (final_stu) | **+0.710** (final_stu) | **kc** |
| movie_id/top5_acc              | +0.252 (block0_tea) | **+0.311** (final_stu) | **kc** |

For subject traits and identity, kc dominates. For continuous stim regression
on this single enc seed, mp is competitive — but the published 5-enc-seed
Phase-D result has kc winning on narrative and pos. The single-enc-seed mp
"win" here is most likely seed-luck noise, not a real signal flip.

## What this means for Phase 2

### Promote
- **Encoder depth ablation: depth ∈ {1, 2, 3}.** Block 1 in depth=2 adds
  nothing for stim probes; testing depth=1 is the cheapest way to confirm
  *and* save train/inference compute. Depth=3 tests whether more capacity
  could carry stim info that depth=2 doesn't reach (rather than truncate).
- **Re-target the masked-prediction objective at block 0** (predict
  intermediate-layer teacher). I-JEPA-v2-style. If block 0 carries stim
  signal and block 1 doesn't, predicting at block 0 may force the encoder
  to *concentrate* useful info there instead of dispersing across two layers.
- **Multi-scale / longer patches.** The patch-embed alone gets +0.030 narrative
  before any transformer. The inductive bias of the patch embedding is doing
  the lift. Sweep patch_size ∈ {50, 100, 200} (currently 50) — this directly
  tests the hypothesis that narrative needs longer-time-scale features at the
  embedding boundary.
- **Per-channel routing inside the encoder.** Channels are specialized;
  inter-channel attention may be averaging the channel-specific signal.
  Ablations: (a) split the encoder into 5 per-channel towers + a single
  cross-channel attention layer at the end, (b) restrict attention to
  intra-channel for the first block.

### Demote / drop from Phase 2
- ~~Predictor bottleneck width (24 → 32 → 48)~~. Prepred preserves stim
  signal — the bottleneck is not the lossy stage.
- ~~Teacher EMA momentum schedule~~. Teacher ≈ student here.
- ~~Predicting from intermediate teacher layer~~ replaced by the more direct
  "re-target the prediction objective" item above.
- ~~Stop-gradient placement~~ — same reasoning.

### Open question (not addressable from Phase 1)
- Why does the encoder learn age/sex but not new narrative information beyond
  patch-embed? The masked-prediction objective on per-recording-normalized
  CorrCA-5 may be paying for stable subject identity rather than
  stimulus-driven dynamics. The next experiment that would settle this is a
  **per-clip normalization** ablation: if the encoder *can't* use slow
  recording-level statistics, does it shift to stim signal? This belongs in
  Phase 2 alongside the depth ablation.

## Memos to update

- `project_per_subject_ceilings_2026-04-30`: keep as-is; Phase 1 results
  affirm the memo's prediction that narrative is concentrated in a single
  CorrCA component (here, encoder ch4) and that channel-routing matters.
- New memo for the Phase-1 finding that **block 1 is dead for stim** and
  **prepred preserves stim signal** — relevant for any future Phase-2
  planning conversation.
