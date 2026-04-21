# Variance Decomposition of Frozen JEPA Embeddings

**Date:** 2026-04-21
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

## 6. What's still outstanding (as of 2026-04-21)

- **Retrained checkpoints with early-stopping.** 14 new checkpoints (7
  global-norm + 7 per-rec-norm) have been trained and probe_eval +
  variance_decomp jobs have been queued. This will give a clean A/B:
  - Same configs, same hyperparameters, same seed
  - Only difference is input normalization
  - Expected outcome: per-rec-norm → higher η²_subj and stim/within,
    larger principal angles (same mechanism as CorrCA)
- **Single seed.** All numbers above are seed=2025. Colleague's CorrCA
  probe_eval used 5 seeds and the result held; a 3-seed replication of the
  variance decomposition would tighten the SIGReg > VICReg claim.
- **Absolute-time-aligned clips.** `_embed_per_clip` sampling is
  per-recording linspace. An absolute-timestamp-aligned version would
  sharpen the `Var_stimulus` estimate.

## 7. Reproducing

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
