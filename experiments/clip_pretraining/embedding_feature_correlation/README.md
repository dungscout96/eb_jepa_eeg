# V-JEPA-2 embedding ↔ scalar movie feature correlation

Sanity check: V-JEPA-2 frame embeddings should explain a large fraction of
variance in the existing scalar movie features (luminance, contrast, motion,
faces, depth, etc.) because both are derived from the same frames. If the
embeddings *do not* recover these basic signals, something is wrong with the
extraction or alignment.

## Method

- Align: each V-JEPA-2 clip (2 Hz, 0.5 s span) is paired with the mean of the
  per-frame parquet features falling in `[t, t + 0.5 s)`.
- Regression: per scalar target, fit `RidgeCV` (alphas log-spaced 1e-2…1e4) over
  L2-normalized embeddings; report 5-fold CV R².
- Classification (`shot_id`): multinomial logistic regression; report
  5-fold CV accuracy vs. majority-class chance.

## Run

```bash
PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run.py
```

Outputs `results.csv` and `results.png` alongside this README.

## Expected outcome

- Low-level descriptors (luminance, contrast, edge_density, motion) → R² > 0.5
- Shot_id → accuracy ≫ chance (53 boundaries, ~54 classes, chance ≈ 5%)
- `position_in_movie` → near-perfect R² (V-JEPA-2 is told the clip's content
  but not its time; any signal must come from gradual scene-level drift)

---

# Does averaging the embeddings *by shot* preserve the signal?

One script per encoder:

- `run_shot_averaged_vjepa2.py` — V-JEPA-2 clip-level (1408-d, 2 Hz, 406 clips).
- `run_shot_averaged_clip.py`   — OpenAI CLIP frame-level (512-d, 24 Hz, 4877 frames).

```bash
PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_shot_averaged_vjepa2.py
PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_shot_averaged_clip.py
```

Outputs land with the encoder prefix: `vjepa2_results_shot_averaged.{csv,png}`
and `clip_results_shot_averaged.{csv,png}`.

## Three measurements (all 5-fold CV ridge R²)

| column | n | what it measures |
|---|---|---|
| `r2_clip` | 406 clips | per-clip embedding → per-clip feature (baseline) |
| `r2_broadcast` | 406 clips | each clip's embedding **replaced by its shot-mean**, still predicting the per-clip feature |
| `r2_shot` / `r2_shot_pca` | 54 shots | one mean embedding + mean feature per shot (raw 1408-d, and PCA-20) |

The **broadcast control** is the instrument that isolates the quantity of
interest: it discards within-shot embedding detail but keeps n=406, so any drop
vs `r2_clip` is the *pure cost of averaging the embedding*. The direct 54-shot
regression conflates that with a 7.5× drop in sample size.

## Result (The_Present, 54 shots, median 6 clips/shot)

- **Broadcast ≈ per-clip**: mean R² 0.792 → **0.757** (−0.035). Shot-mean
  embeddings retain essentially all the predictive signal. Several targets even
  *improve* under averaging (`n_objects` 0.52→0.74, `n_faces` 0.72→0.82,
  `depth_std` 0.87→0.91) — the mean denoises clip-level jitter.
  → **Averaging embeddings by shot preserves adequate signal.**
- The direct 54-shot fit (`r2_shot`, mean −0.45) is **underpowered, not a real
  signal loss**: with 54 samples vs 1408 dims and heavy-tailed targets
  (`spatial_freq_energy`, `face_area_frac`) individual CV folds blow up to large
  negative R². PCA-20 doesn't rescue it — the problem is n, not dimensionality.
- Slow, scene-level targets still survive even at the raw shot level:
  `position_in_movie` R²≈0.66, `luminance_mean`≈0.50.

**Bottom line:** use the shot-mean embedding freely — it keeps the signal. Just
don't evaluate it on only 54 shot-level points; quantify against the per-clip /
broadcast setup where the sample size is honest.

## OpenAI CLIP result (4877 frames, 54 shots, median 71 frames/shot)

| measurement | mean R² over 16 targets |
|---|---|
| per-frame baseline | 0.869 |
| broadcast shot-mean (n=frames) | **0.781** (−0.088) |
| raw 54-shot fit | −0.02 |
| PCA-20 54-shot fit | −0.08 |

- CLIP's per-frame baseline is **higher** than V-JEPA-2's per-clip (0.869 vs
  0.792) because CLIP frames map to per-frame features without temporal averaging.
- The broadcast cost is **larger** for CLIP than V-JEPA-2 (−0.088 vs −0.035):
  CLIP carries more within-shot signal (η² = 0.71 vs 0.89), so averaging
  discards more.
- Most of the cost is in features that legitimately vary *within* a shot:
  `motion_energy` 0.56 → 0.35, `spatial_freq_energy` 0.98 → 0.74,
  `scene_natural_score` 1.00 → 0.68, `scene_open_score` 1.00 → 0.78. Low-level
  per-frame appearance (luminance, entropy, saturation) survives almost
  untouched (≤ 0.01 R² drop).
- Shot-level (n=54) is much less catastrophic than V-JEPA-2's — several
  features have *positive* shot-level R² (`n_faces` 0.74, `scene_open` 0.74,
  `edge_density` 0.50). CLIP's 512-d + stronger raw separability survives n=54
  better than V-JEPA-2's 1408-d.

## V-JEPA-2 after mean-centering — does shot-averaging still work?

The variance analysis showed mean-centering is the meaningful preprocessing for
V-JEPA-2's cosine geometry (raw cosines are crammed near 1.0; centering takes
AUC 0.88 → 0.92). `run_shot_averaged_vjepa2_centered.py` checks whether
shot-averaging still preserves the feature signal after that centering.

```bash
PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_shot_averaged_vjepa2_centered.py
```

| mean R² (16 targets) | raw | mean-centered | Δ |
|---|---|---|---|
| per-clip baseline | 0.792 | 0.791 | −0.001 |
| broadcast shot-mean | 0.757 | 0.754 | −0.003 |
| **embedding-averaging cost** (broadcast − per-clip) | **−0.035** | **−0.037** |

**Mean-centering is regression-invariant.** Ridge with `fit_intercept=True`
(sklearn's default) learns its own offset, so subtracting the global mean
upfront doesn't change R² — the tiny residual delta (~0.003) is only from the
raw baseline's per-dim std-scaling, which the centered-only version skips. So:

> The raw shot-averaging numbers ARE the mean-centered shot-averaging numbers.
> Mean-centering matters for the cosine objective (CLIP head); it doesn't
> affect downstream regression-based readouts.

**Implication for CLIP design:** mean-center the V-JEPA-2 targets for the
contrastive metric — you get the AUC 0.88 → 0.92 improvement essentially for
free, with no impact on how well shot-averaged embeddings preserve the feature
signal. Full PCA-whitening would over-correct (it up-weights low-variance
within-shot directions, making shot-averaging cost more); centering is the
right operating point.

## V-JEPA-2 vs CLIP, shot-averaging

| | V-JEPA-2 | OpenAI CLIP |
|---|---|---|
| per-unit baseline | 0.792 | **0.869** |
| broadcast shot-mean | 0.757 | 0.781 |
| broadcast loss | **−0.035** | −0.088 |
| shot-level PCA-20 | −0.54 | **−0.08** |

**Takeaway:** Shot-mean embeddings preserve the signal for *both* encoders, but
CLIP loses more (≈10% of its baseline) because more of its predictive power
lives at the frame timescale. If the downstream EEG pairing is at shot
granularity:

- V-JEPA-2 is the more natural target — it already averages a 0.5 s clip and
  shot-mean barely hurts.
- For CLIP, keep features that vary within a shot (motion, scene scores) at
  finer granularity than shot-mean, or budget the ~10% loss.

---

# Embedding geometry for the CLIP objective

Two scripts, one per video encoder:

- `run_embedding_variance_vjepa2.py` — V-JEPA-2 *clip*-level (1408-d, ~2 Hz, 406 clips)
- `run_embedding_variance_clip.py`   — OpenAI CLIP *frame*-level (512-d, ~24 Hz, 4877 frames)

Both quantify how dissimilar embeddings are *within* a shot vs *across* shots,
in cosine space, to decide what counts as a positive vs negative pair.

```bash
PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_embedding_variance_vjepa2.py
PYTHONPATH=. .venv/bin/python experiments/embedding_feature_correlation/run_embedding_variance_clip.py
```

Outputs land alongside each script with the encoder prefix
(`vjepa2_*` / `clip_*`): summary text, per-shot CSV, 4-panel PNG.

## What it measures

1. **Variance decomposition** — η² = fraction of (mean-centered) embedding
   variance that is between-shot vs within-shot.
2. **Pairwise cosine** — same-shot (positives) vs different-shot (negatives),
   with d-prime and ROC-AUC, computed both on **raw** and **mean-centered**
   embeddings.
3. **Temporal confound** — same- vs different-shot cosine at matched |time gap|,
   so the shot effect is separated from "nearby in time".
4. **Per-shot margin** — within-shot compactness vs cosine to the nearest other
   shot centroid.
5. **NN purity** — fraction of clips whose top-1/5 nearest neighbor is in the
   same shot (in-batch negative hardness).

## Result (The_Present, 406 clips, 54 shots)

| | raw cosine | mean-centered |
|---|---|---|
| same-shot (pos) | 0.999 ± 0.002 | 0.793 ± 0.250 |
| diff-shot (neg) | 0.991 ± 0.013 | −0.008 ± 0.529 |
| d-prime | 0.87 | **1.94** |
| ROC-AUC | 0.88 | **0.92** |

- **η² = 0.89** — shot identity explains 89% of embedding variance (F≈55). The
  structure is very strong; it's just hidden in raw cosine.
- **Raw V-JEPA-2 cosine is anisotropic**: a dominant shared component puts every
  pair at ~0.97–1.0. Mean-centering opens the range (negatives drop to ≈0) and
  is the single most impactful preprocessing step.
- **Temporal decay is real and crosses shot boundaries**: at gap <1 s even
  different-shot pairs sit at 0.95; within a matched gap, same-shot is ~0.2–0.5
  cosine above different-shot — so the shot effect is genuine, not just adjacency.
- **NN purity 0.83 / 0.70** (top-1/5) — ~17% of clips' nearest neighbor is in
  another shot: hard negatives exist but aren't the norm.
- **12/54 shots overlap a neighbor shot** (negative margin) — a handful of
  near-duplicate adjacent shots.

## V-JEPA-2 design implications

- **Center / batch-norm the V-JEPA-2 targets before contrasting** (or subtract
  the global mean), else cosine has almost no dynamic range and you lean entirely
  on a tiny learned temperature.
- **Shot-level positives are well-posed**: same-shot clips (and clip ↔ shot-mean)
  are reliably closer than cross-shot pairs after centering. Cross-shot pairs are
  mostly clean negatives.
- **Exclude temporally adjacent clips (gap ≲ 2 s) from the negative pool** — they
  are near-identical even across a shot cut and would be false negatives. Note a
  "same-shot positive" is partly just "nearby in time".
- **A moderate temperature + many in-batch negatives** beats hard-negative mining
  here, given ~17% genuinely-hard NN and a dozen overlapping shots; consider
  soft / temperature-scaled targets rather than hard 1-vs-rest labels for the
  near-duplicate shots.

## OpenAI CLIP result (The_Present, 4877 frames, 54 shots)

| | raw cosine | mean-centered |
|---|---|---|
| same-shot (pos) | 0.899 ± 0.067 | 0.614 ± 0.241 |
| diff-shot (neg) | 0.736 ± 0.092 | −0.024 ± 0.265 |
| d-prime | **2.01** | **2.52** |
| ROC-AUC | **0.92** | **0.95** |

- **η² = 0.71, F = 223** — strong shot structure, but more within-shot variation
  than V-JEPA-2 (which compresses 0.5 s into one vector). At 24 Hz, CLIP retains
  frame-level micro-motion.
- **CLIP is much less anisotropic than V-JEPA-2.** Raw cosine already has real
  dynamic range (0.74 → 0.90), and raw AUC = 0.92 is honest. Centering still
  helps (+0.03 AUC, d-prime 2.01 → 2.52), but it's not the decisive step it is
  for V-JEPA-2.
- **Top-1 NN purity ≈ 1.00**: the nearest neighbor of nearly every CLIP frame is
  in its own shot — shot-level positives are essentially noise-free in CLIP space.
- **24 Hz amplifies the temporal-adjacency confound.** Same-shot pairs at <0.1 s
  sit at cosine 0.94 (centered); same-shot pairs at 16–32 s have dropped to 0.31.
  And same-shot is *far* above different-shot at every matched gap (centered:
  0.94 vs −0.10 at <0.1 s, 0.49 vs 0.05 at 8–16 s), so shot identity is a real
  signal on top of adjacency.

## Comparison: V-JEPA-2 vs OpenAI CLIP

| | V-JEPA-2 (clips, 1408-d) | OpenAI CLIP (frames, 512-d) |
|---|---|---|
| samples / Hz | 406 @ 2 Hz | 4877 @ 24 Hz |
| η² (between/total) | **0.89** | 0.71 |
| raw cos same-shot | 0.999 ± 0.002 | 0.899 ± 0.067 |
| raw cos diff-shot | 0.991 ± 0.013 | 0.736 ± 0.092 |
| raw AUC / d-prime | 0.88 / 0.87 | **0.92 / 2.01** |
| centered AUC / d-prime | 0.92 / 1.94 | **0.95 / 2.52** |
| centering AUC gain | **+0.04** | +0.03 |
| top-1 NN purity (centered) | 0.83 | **1.00** |
| shots overlapping a neighbor | 12 / 54 | 10 / 54 |

**Bottom line:** Yes, both encoders gain a few AUC points from mean-centering,
but they need it for very different reasons:

- **V-JEPA-2 *requires* centering.** Raw cosines are crammed into 0.97–1.0; the
  full discriminative signal sits in a narrow off-mean component. Without
  centering you depend entirely on temperature to expand that range.
- **CLIP does not require centering.** Raw cosine has real dynamic range and
  already gives AUC = 0.92 / d-prime = 2.0. Centering is a free small win; not
  centering is acceptable.
- **CLIP positives are cleaner than V-JEPA-2 positives** (NN purity 1.00 vs
  0.83, raw d-prime 2.0 vs 0.9). The price is a stronger 24 Hz adjacency
  confound — sub-sample frames to ~1–2 Hz (or use a clip-aware sampler) before
  contrasting, or trivial "next-frame" positives will dominate.

So if the EEG↔video CLIP head consumes V-JEPA-2 targets, **whiten / mean-center
them first**. If it consumes OpenAI CLIP targets, raw cosine is fine; just
control the frame rate when forming pairs.
