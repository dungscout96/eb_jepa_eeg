# Variance Decomposition of Frozen JEPA Embeddings

**Date:** 2026-04-22 (updated)
**Script:** [scripts/variance_decomposition.py](../scripts/variance_decomposition.py)
**Embeddings:** per-clip mean-pooled encoder tokens, evaluated on the val split.

## 1. What the decomposition measures

For each trained encoder we extract embeddings `Z[S, K, D]` — `S=293` subjects
(recordings), `K` clips per subject sampled via `np.linspace(0, n_clips-1, K)`
(roughly time-aligned movie positions), `D=64` embedding dims. The two-way
decomposition

```
Var_total  = Var_subject + Var_within                       (law of total variance)
Var_within = Var_stimulus + Var_residual                    (3-way nested ANOVA,
                                                             assumes clips aligned
                                                             across subjects)
η²_subj = Var_subject / Var_total
η²_stim = Var_stimulus / Var_total
stim/within = Var_stimulus / Var_within
```

Null for iid Gaussian embeddings: `η²_subj ≈ 1/K`, `η²_stim ≈ 1/S` (so at K=32,
S=293: null η²_subj ≈ 0.031, null η²_stim ≈ 0.0034).

In addition we compute:
- **Effective rank** (Shannon exp-entropy) of `C_subject` and `C_within` —
  how many dimensions carry subject/within variance.
- **Principal angles** between the top-k subject subspace and top-k
  within-subject subspace. Near-0° = entangled; near-90° = orthogonal/separable.

## 2. K=4 → K=32: K=4 was misleading

Early results at K=4 clips/rec overstated η²_subj because with only 4 samples
per subject the per-subject mean carries significant within-subject noise.
Re-running with K=32 showed substantial drops, especially for VICReg.

| config | K=4 η²_subj | **K=32 η²_subj** | Δ |
|---|---:|---:|---:|
| SIGReg nw1_ws1 (1.0) | 0.342 | **0.157** | −0.185 |
| SIGReg nw2_ws1 (0.1) | 0.489 | **0.348** | −0.141 |
| SIGReg nw4_ws4 (0.1) | 0.450 | **0.371** | −0.079 |
| VICReg+proj nw2_ws1 (1.0) | 0.450 | **0.061** | **−0.389** |
| VICReg+proj nw4_ws4 (1.0) | 0.326 | **0.194** | −0.132 |
| VICReg+proj nw4_ws4 (0.1) | 0.392 | **0.278** | −0.114 |

**Takeaway:** K=4 inflated subject fraction; the cleanest baseline number for
VICReg nw2_ws1 (1.0) dropped from 0.45 to 0.06 — essentially null. All
further analysis uses K=32.

## 3. K=32 results on the "best subject-trait" checkpoints

Six `latest.pth.tar` checkpoints from [docs/best_subject_trait_checkpoints.md](best_subject_trait_checkpoints.md).
All seed=2025, epoch 99/100 (pre-early-stopping era).

| checkpoint | η²_subj | η²_stim | stim/within | eff_rank(C_subj) | eff_rank(C_within) | angle@k=5 (mean) |
|---|---:|---:|---:|---:|---:|---:|
| SIGReg nw1_ws1 (1.0) | 0.157 | 0.0054 | 0.0064 | 4.67 | 6.72 | 32.1° |
| SIGReg nw2_ws1 (0.1) | 0.348 | 0.0047 | 0.0073 | 2.58 | 5.25 | 16.9° |
| SIGReg nw4_ws4 (0.1) | 0.371 | 0.0041 | 0.0065 | 4.42 | 5.64 | 25.1° |
| VICReg+proj nw2_ws1 (1.0) | 0.061 | 0.0068 | 0.0072 | 3.36 | 3.17 | 10.5° |
| VICReg+proj nw4_ws4 (1.0) | 0.194 | 0.0060 | 0.0074 | 2.30 | 3.09 | 11.4° |
| VICReg+proj nw4_ws4 (0.1) | 0.278 | 0.0078 | 0.0108 | 1.90 | 3.32 | 19.3° |

### Findings

- **SIGReg encodes more subject variance than matched-config VICReg.** At nw2_ws1
  the gap is huge (0.348 vs 0.061); at nw4_ws4 SIGReg(0.1) is also clearly
  higher than VICReg variants.
- **SIGReg distributes subject info across more dimensions.** eff_rank of
  `C_subject` for SIGReg sits at 4-5 effective dims; VICReg sits at 2-3.
- **Principal angles are larger for SIGReg.** At the top-5 level SIGReg
  subject vs within-subject subspaces sit 17-32° apart; VICReg sits 10-19°.
  SIGReg's subject and stimulus geometry is more separable than VICReg's.
- **Stimulus-locked variance is tiny for all baselines.** stim/within ≈
  0.006-0.011, only 2-3× the iid null (0.0034). Translated: **in these
  baselines, the within-subject bucket is dominated by noise, not stimulus
  response.** This is consistent with the probe_eval report where movie
  probes sit at chance for every baseline.

## 4. CorrCA: measurement matched to training normalization

The CorrCA VICReg(0.25) nw4_ws2 best.pth.tar (kkokate's exp6) was trained
with `--data.norm_mode=per_recording`. Evaluating with the default global
norm produced a distribution-shift artifact where stimulus probes and
variance decomposition both looked like baseline chance. Re-running with
`--norm_mode=per_recording` reversed the picture.

| metric | **buggy (global eval)** | **fixed (per-rec eval)** |
|---|---:|---:|
| Var_total | 4.24 | 3.58 |
| Var_subject | 1.03 | **2.10** |
| Var_within | 3.21 | 1.48 |
| Var_stimulus | 0.024 | 0.038 |
| **η²_subj** | 0.243 | **0.587** |
| η²_stim | 0.0056 | **0.0105** |
| **stim/within** | 0.0074 | **0.0254** |
| eff_rank(C_subject) | 1.67 | 1.65 |
| eff_rank(C_within) | 3.56 | **7.67** |
| top principal angle (k=1) | 4.9° | **79.8°** |
| angles@k=5 | [3.2° mean] | **[79.8, 39.9, 11.4, 3.5, 0.8]** |

### CorrCA vs best baseline (same K=32 methodology)

| metric | best baseline (K=32) | **CorrCA (fixed)** |
|---|---:|---:|
| η²_subj (highest of baselines) | 0.371 (SIGReg nw4_ws4) | **0.587** |
| stim/within (highest of baselines) | 0.0108 (VICReg(0.1) nw4_ws4) | **0.0254** |
| top-1 principal angle (largest) | ~50° (SIGReg nw1_ws1, individual) | **79.8°** |
| eff_rank(C_within) | 6.72 (SIGReg nw1_ws1) | **7.67** |

**CorrCA wins on every axis.** Per-recording norm strips subject baseline
from the input, so the encoder reconstructs subject identity from dynamics
alongside stimulus — producing representations where both signals are
stronger *and* structurally separated (top subject direction nearly
orthogonal to top within-subject direction).

The probe_eval numbers independently confirm this (from a separate run):
- movie-feature regression correlations on test: position 0.188, luminance
  0.119, contrast 0.056, narrative −0.007 (matches colleague's 5-seed
  aggregate of 0.176, 0.168, 0.115, −0.003)
- position AUC 0.574, luminance 0.535, contrast 0.492, narrative 0.558
- subject: test age_cls AUC 0.652, sex AUC 0.614, age corr 0.293

Variance decomposition geometry (elevated subject *and* stimulus variance,
near-orthogonal top dimensions) and probe decodability agree.

## 5. Methodological lessons

1. **K=4 is not enough.** The per-subject mean is too noisy; η²_subj gets
   inflated by O(1/K) residual variance. Use K=32 or larger.
2. **Variance decomposition is not a probe substitute.** Early analysis
   claimed "CorrCA doesn't encode more stimulus" based on stim/within. The
   real story required matching eval norm to training norm, after which
   both variance decomposition *and* probe_eval agreed that CorrCA has real
   stimulus signal.
3. **The 3-way decomposition is conservative.** `Var_stimulus` is measured
   as between-clip-position variance pooled across subjects. Clips are only
   approximately time-aligned (`np.linspace(0, n_clips-1, K)` per recording);
   any slack gets lumped into `Var_residual`. So stim/within under-estimates
   the true stimulus share, especially if recordings vary in length.
4. **Principal angles are the cleanest disentanglement metric.** η²_subj
   alone doesn't tell you whether subject and stimulus share directions;
   the top-1 principal angle does. CorrCA's 79.8° vs baselines' ~10-50°
   is the clearest single-number contrast across regularizers.

## 6. Retrained-with-early-stopping A/B: global vs per-rec norm

14 new checkpoints (7 configs × 2 normalization arms) were trained with
`--optim.early_stopping_patience=20` (matching CorrCA's training setup), so
each encoder comes from its own val-peak `best.pth.tar` rather than epoch 99
`latest.pth.tar`. This gives a clean A/B on input normalization.

| config | arm | η²_subj | η²_stim | stim/within | eff_rank(subj) | eff_rank(within) | angle@k=5 |
|---|---|---:|---:|---:|---:|---:|---:|
| **sigreg0.1 nw4_ws4** | **per-rec** | **0.526** | **0.0373** | **0.0788** | 1.13 | 2.25 | 15.8° |
| sigreg0.1 nw4_ws4 | global | 0.452 | 0.0045 | 0.0082 | 2.92 | 4.56 | 23.6° |
| sigreg0.1 nw2_ws1 | per-rec | 0.056 | 0.0086 | 0.0091 | 3.60 | 3.98 | 19.8° |
| sigreg0.1 nw2_ws1 | global | 0.061 | 0.0069 | 0.0073 | 2.17 | 2.47 | 17.8° |
| sigreg1.0 nw1_ws1 | per-rec | 0.036 | 0.0093 | 0.0097 | 3.72 | 3.24 | 22.7° |
| sigreg1.0 nw1_ws1 | global | 0.188 | 0.0075 | 0.0092 | 1.71 | 4.15 | 17.9° |
| vc1.0 proj nw2_ws1 | per-rec | 0.031 | 0.0079 | 0.0082 | 6.48 | 3.76 | **30.6°** |
| vc1.0 proj nw2_ws1 | global | 0.051 | 0.0069 | 0.0073 | 2.39 | 2.48 | 16.1° |
| vc1.0 proj nw4_ws4 | per-rec | 0.098 | 0.0080 | 0.0088 | 4.34 | 2.96 | 15.4° |
| vc1.0 proj nw4_ws4 | global | 0.100 | 0.0067 | 0.0074 | 2.78 | 2.43 | 8.1° |
| vc0.1 proj nw4_ws4 | per-rec | 0.050 | 0.0074 | 0.0078 | 5.07 | 2.46 | 14.9° |
| vc0.1 proj nw4_ws4 | global | 0.086 | 0.0067 | 0.0073 | 2.23 | 2.31 | 15.0° |
| vc0.1 noproj nw4_ws4 | per-rec | 0.188 | 0.0097 | 0.0119 | 3.58 | 3.70 | 10.3° |
| vc0.1 noproj nw4_ws4 | global | 0.213 | 0.0056 | 0.0071 | 1.98 | 3.15 | 22.1° |

### Findings

1. **SIGReg(0.1) nw4_ws4 + per-rec is the standout.** stim/within = 0.0788
   is the highest across every checkpoint we've measured — ~3× CorrCA-VICReg
   (0.0254) and ~10× baseline configs. η²_subj = 0.526 is also high. The
   per-rec arm lifts stim/within by ~10× over the same config under global
   norm, matching the mechanism observed on CorrCA.
2. **Per-rec norm doesn't help configs with null η²_subj.** For most
   VICReg variants, per-rec norm leaves stim/within essentially unchanged.
   The "stripping subject baseline unlocks stimulus" effect requires the
   encoder to have *learned* a strong subject-encoding direction to begin
   with; if η²_subj is already near the iid null under global, there's
   nothing for per-rec to unlock.
3. **VICReg(0.1) noproj nw4_ws4** is the only VICReg configuration where
   per-rec produces a meaningful stim/within bump (0.0071 → 0.0119). It's
   also the one whose probe_eval report showed the highest VICReg age-cls
   AUC (0.636).
4. **best.pth.tar (early-stopped at val peak) ≠ latest.pth.tar (epoch 99).**
   Compared to the K=32 latest.pth.tar baseline in §3 above:
   - sigreg0.1 nw4_ws4: 0.371 → 0.452 (up, best still carries subject
     variance)
   - sigreg0.1 nw2_ws1: 0.348 → 0.061 (down dramatically — val peak is
     much earlier than epoch 99)
   - vc0.1 proj nw4_ws4: 0.278 → 0.086 (down)
   - vc1.0 proj nw4_ws4: 0.194 → 0.100 (down)
   Val-peak checkpoints generally have less subject variance than
   continued-training ones. Subject-trait probe advantages reported on
   latest.pth.tar reflect an overtraining phase where the encoder has
   further memorized subject fingerprints beyond the val-stimulus-loss
   peak.

### Paired probe_eval summary (val split)

Movie-feature regression correlations (higher = more stimulus-decodable):

| config | per-rec pos | global pos | per-rec lum | global lum | per-rec con | global con |
|---|---:|---:|---:|---:|---:|---:|
| SIGReg nw1_ws1 (1.0) | **0.156** | 0.023 | **0.123** | 0.058 | **0.081** | −0.012 |
| SIGReg nw2_ws1 (0.1) | **0.125** | −0.122 | **0.151** | −0.077 | 0.055 | −0.033 |
| SIGReg nw4_ws4 (0.1) | **0.149** | 0.046 | **0.134** | 0.014 | **0.087** | −0.016 |
| VICReg proj nw2_ws1 (1.0) | −0.042 | −0.055 | −0.033 | −0.030 | 0.019 | −0.050 |
| VICReg proj nw4_ws4 (1.0) | −0.081 | 0.002 | 0.003 | 0.051 | 0.034 | 0.048 |
| VICReg proj nw4_ws4 (0.1) | −0.105 | −0.051 | 0.027 | 0.034 | 0.037 | 0.045 |
| VICReg noproj nw4_ws4 (0.1) | **0.157** | 0.074 | **0.132** | 0.102 | **0.103** | 0.056 |

- **SIGReg stimulus decodability jumps to 0.12-0.16 under per-rec** on
  all three configs — a clear, large effect exactly where the variance
  decomposition says stim/within rose. The "SIGReg + per-rec" cell of the
  config matrix is the only one where stimulus signal is linearly
  decodable off the variance-norm baseline.
- **VICReg+proj stays near zero** under both arms — consistent with its
  near-null η²_subj and no change under per-rec.
- **VICReg noproj jumps like SIGReg does** — noproj preserves more raw
  encoder structure, per-rec then extracts stimulus from it. This is the
  non-SIGReg case where both probe and variance decomposition agree.

## 7. SIGReg + per-rec + CorrCA (fills the missing matrix cell)

kkokate's pre-existing `exp6_sigreg` checkpoint — the only available
SIGReg + per-rec + CorrCA combination — gives one data point in the
remaining cell.

**Checkpoint:** SIGReg(1.0) nw4_ws2 + per-rec + CorrCA, seed=2025,
`/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/eeg_jepa/dev_2026-04-15_21-58/.../best.pth.tar`.

**Variance decomposition:**

| metric | exp6_sigreg (nw4_ws2) | CorrCA VICReg (nw4_ws2) | SIGReg+per-rec (nw4_ws4, no CorrCA) |
|---|---:|---:|---:|
| Var_total | 802.1 | 3.58 | (similar order) |
| η²_subj | **0.726** | 0.587 | 0.526 |
| η²_stim | 0.0032 | 0.0105 | **0.0373** |
| stim/within | 0.0116 | 0.0254 | **0.0788** |
| eff_rank(C_subj) | 1.05 | 1.65 | 1.13 |
| eff_rank(C_within) | 1.37 | 7.67 | 2.25 |
| top-1 angle | 67.6° | **79.8°** | (~63° from k=5 mean) |

SIGReg + CorrCA pushes η²_subj to 0.726 (highest across all 18
checkpoints measured) but **collapses the representation** — effective
ranks are ~1 on both subject and within-subject sides, and stim/within
drops relative to the no-CorrCA SIGReg+per-rec variant. SIGReg's
slicing objective drives the encoder to a near-1-dimensional subject
fingerprint when given CorrCA-filtered input. The geometric advantage of
CorrCA-VICReg (wide within-subject subspace, ~7 effective stimulus dims)
does *not* carry over — CorrCA + SIGReg is *more* compressed than either
piece alone.

**probe_eval (val split):** position 0.190, luminance 0.154, contrast
0.109 — comparable to CorrCA-VICReg on stimulus; sex AUC 0.801 (highest
of any checkpoint), age AUC 0.551. But **test-set stimulus corrs collapse
to ~0.00**, suggesting the val-peak `best.pth.tar` over-fit to val
stimulus rather than generalizing. Test sex AUC drops to 0.649 (still
above chance). The train→val→test gap is larger than any other checkpoint.

### The 3 new SIGReg + CorrCA configs (still evaluating)

Training complete (all 3 early-stopped between epoch 27 and 51):
- `sigreg1.0_nw1_ws1_corrca` — best epoch 7
- `sigreg0.1_nw2_ws1_corrca` — best epoch 7
- `sigreg0.1_nw4_ws4_corrca` — best epoch 6

Probe_eval and variance_decomp on these are not yet run; once they are,
the full 2×2×3 config matrix (SIGReg/VICReg × per-rec/global × ±CorrCA)
will have data at every cell.

## 8. What's still outstanding

- **Probe_eval + variance_decomp on the 3 new SIGReg + CorrCA configs.**
  Will tell us whether SIGReg + CorrCA's rank-1 collapse is specific to
  `exp6_sigreg`'s nw4_ws2 setup or a general property.
- **Test-set vs val-set generalization.** The exp6_sigreg best.pth.tar
  has val pos corr 0.190 / test pos corr 0.004 — a huge gap. Whether
  this is an overfitting artifact, a val/test distribution shift, or
  something about SIGReg+CorrCA specifically is worth investigating.
- **Multi-seed replication.** All numbers are seed=2025. A 3-seed run of
  the standout configs (sigreg0.1 nw4_ws4 per-rec; CorrCA-VICReg) would
  tighten the claims.
- **Absolute-time-aligned clips.** `_embed_per_clip` sampling is
  per-recording linspace. An absolute-timestamp-aligned version would
  sharpen the `Var_stimulus` estimate and likely raise all stim/within
  numbers by a consistent factor.

## 9. Reproducing

```bash
# Extract embeddings + decomposition for a single checkpoint (K=32, val split)
PYTHONPATH=. uv run --group eeg python scripts/variance_decomposition.py \
    --checkpoint=/path/to/best.pth.tar \
    --n_windows=4 --window_size_seconds=4 \
    --n_clips_per_rec=32 \
    --norm_mode=per_recording \
    --output_dir=outputs/variance_decomp_k32

# Re-aggregate (or reanalyze) existing embeddings without re-embedding
python scripts/variance_decomposition.py --reanalyze_dir=outputs/variance_decomp_k32

# Self-test (no checkpoint needed)
python scripts/variance_decomposition.py --selftest
```

For batch submissions on Delta see
[scripts/eval_retrained_perrec_delta.py](../scripts/eval_retrained_perrec_delta.py)
and [scripts/eval_retrained_global_delta.py](../scripts/eval_retrained_global_delta.py).
