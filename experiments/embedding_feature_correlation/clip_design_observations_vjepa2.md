# CLIP design observations — V-JEPA-2 targets

Observations from `experiments/embedding_feature_correlation/` on what the
V-JEPA-2 clip embedding space (1408-d, ~2 Hz, 406 clips of *The_Present*) means
for the EEG ↔ video CLIP-style contrastive objective. Each section ties an
empirical measurement to a design choice.

All AUC / d-prime numbers below treat **cosine similarity itself** as the
score (no learned classifier) and same-shot vs different-shot as the label.
That's the right framing for CLIP because CLIP's own metric is cosine — we want
properties of the metric, not of a downstream head.

---

## 1. The embedding space is strongly structured by shot

| | value |
|---|---|
| η² (between-shot / total variance, mean-centered) | **0.89** |
| F-statistic | **54.6** |
| n clips / n shots | 406 / 54 |

Shot identity explains ~89% of (angular) embedding variance. F ≈ 55 means the
between-shot signal is 55× the per-dof within-shot noise — a very large
descriptive effect size. This is the load-bearing fact:

> Shot-level contrastive positives are well-posed for V-JEPA-2. Same-shot
> clips genuinely live in distinct neighborhoods.

See [`run_embedding_variance_vjepa2.py`](run_embedding_variance_vjepa2.py).

---

## 2. Raw cosine is anisotropic — mean-center before contrasting

| | raw | mean-centered |
|---|---|---|
| same-shot (positives) | 0.999 ± 0.002 | **0.793 ± 0.250** |
| diff-shot (negatives) | 0.991 ± 0.013 | **−0.008 ± 0.529** |
| d-prime | 0.87 | **1.94** |
| ROC-AUC | 0.88 | **0.92** |

Raw V-JEPA-2 embeddings carry a dominant shared component, so **every** pairwise
cosine sits in [0.97, 1.0]. The discriminative geometry is real but compressed
into a tiny sliver of the cosine range — you can see this in the left panel of
`vjepa2_embedding_variance.png`.

Subtracting the global mean (one vector, computed once over the training set)
opens the dynamic range: negatives drop to ≈ 0, positives sit at ≈ 0.8, and
ROC-AUC moves from 0.88 → 0.92.

**Design implication:**

- **Mean-center the V-JEPA-2 targets before computing the contrastive loss.**
  This is essentially free: one vector subtraction at preprocessing time.
- For ridge / linear readouts, mean-centering is **regression-invariant** —
  it changes the cosine metric but not how well a linear model fits. So you
  can center freely without affecting any of the existing R² baselines.

See `run_shot_averaged_vjepa2_centered.py` — mean-centered shot-averaging
matches raw shot-averaging to within 0.003 mean R².

---

## 3. Same- vs different-shot is real even at matched time gaps

Same-shot pairs are temporally adjacent by definition, so we have to verify the
"shot effect" isn't just a "nearby in time" effect. Mean-centered cosines binned
by |Δt|:

| time gap | same-shot cos | different-shot cos | n diff pairs |
|---|---|---|---|
| 0–1 s | 0.97 | 0.96 | 53 |
| 1–2 s | 0.87 | 0.82 | 255 |
| 2–4 s | 0.76 | 0.53 | 960 |
| 4–8 s | 0.78 | 0.32 | 2,592 |
| 8–16 s | 0.77 | 0.27 | 5,498 |
| 16–32 s | 0.66 | 0.10 | 11,007 |
| > 32 s | — | < 0.10 | many |

Two facts:

1. **Even cross-shot pairs at < 2 s are very similar** (≥ 0.82 cosine). At a
   shot boundary the frames before/after the cut are almost identical to
   V-JEPA-2.
2. **At every gap where same-shot pairs exist, same-shot cosine clearly exceeds
   different-shot cosine** — by 0.2–0.5 cosine points. So the shot effect is
   genuine on top of adjacency, not a confound of it.

**Design implication:**

- **Exclude temporally adjacent clips from the negative pool.** Pairs within
  ~2 s of each other should not be treated as hard negatives — they will
  inject label noise (false negatives).
- Be aware that a "same-shot positive" is partly just "nearby in time" — if
  you want the positive to carry shot-identity signal rather than adjacency
  signal, sample positives across a wider time gap within the shot.

### How big a temporal buffer? (and why it's not enough on its own)

We can quantify the trade-off explicitly. Treat any cross-shot pair with
centered cosine > 0.5 as a "candidate false negative" and ask: how many does
a buffer of size T eliminate, and at what cost in legitimate negatives?

| buffer | FN removed (of 18,445) | legit neg removed | purity of removals | efficiency (FN / legit) |
|---|---|---|---|---|
| < 1 s | 52 (0.3 %) | 1 (0.00 %) | 0.98 | 52× |
| **< 2 s** | **283 (1.5 %)** | **25 (0.04 %)** | **0.92** | **11×** |
| < 4 s | 927 (5.0 %) | 341 (0.56 %) | 0.73 | 2.7× |
| < 8 s | 2,154 (11.7 %) | 1,706 (2.82 %) | 0.56 | 1.3× |
| < 16 s | 4,488 (24.3 %) | 4,870 (8.0 %) | 0.48 | 0.9× |

The CDF of FN time-gaps tells the limiting story: only 25 % of false negatives
have Δt < 16 s; the **median FN has Δt ≈ 42 s**, the 95th percentile is
≈ 138 s. So:

- **A 2-second buffer is a free win**: 92 % of what it removes is real false
  negatives, and it catches the shot-cut-boundary cases (centered cosine
  > 0.95). Apply it unconditionally.
- **A larger buffer rapidly stops paying off.** By 8 s purity drops below 60 %
  and you're sacrificing legitimate negatives roughly 1-for-1. The cure is
  worse than the disease.
- **A buffer alone cannot fix the false-negative problem.** Most false
  negatives are visually similar shots scattered across the movie, not next to
  the positive in time. They need a content-based deduplication step
  (see §6 and §7).

---

## 4. Shot-averaging the embedding preserves the signal

The key question for shot-level EEG pairing: if we collapse every clip inside a
shot to one **mean embedding**, does it still carry the visual content?

Three measurements (5-fold CV ridge R² across 16 scalar movie features):

| | mean R² |
|---|---|
| per-clip baseline (n=406) | 0.792 |
| **broadcast shot-mean (n=406)** | **0.757** (−0.035) |
| direct shot-level fit (n=54, raw) | −0.45 (small-n CV artifact) |
| direct shot-level fit (n=54, PCA-20) | −0.54 (small-n CV artifact) |

The **broadcast control** is the instrument: it replaces each clip's embedding
with its shot-mean while keeping n=406, so the only thing that changes is the
embedding. The drop is only −0.035 mean R² — and several features *improve*
under averaging because shot-mean denoises (`n_objects` 0.52 → 0.74,
`n_faces` 0.72 → 0.82).

The direct 54-shot fit looks catastrophic but isn't real signal loss —
1408-d ridge on 54 samples blows up on heavy-tailed targets (`face_area_frac`,
`spatial_freq_energy`). The broadcast number is the trustworthy read.

**Design implication:**

- **Shot-mean V-JEPA-2 embeddings are a clean target for the CLIP head.** They
  retain ~95% of the per-clip predictive signal and denoise the noisiest few
  features.
- This is consistent with η² = 0.89: most embedding variance is between shots,
  so collapsing within shots discards little.
- The corollary: V-JEPA-2 already absorbs within-shot motion into its 0.5 s
  clip aggregation, so there's less to lose by averaging further. If you ever
  swap to OpenAI CLIP (per-frame) as a target, the broadcast cost roughly
  triples (−0.088), so the same recommendation does not transfer directly.

See `run_shot_averaged_vjepa2.py`, `run_shot_averaged_vjepa2_centered.py`.

---

## 5. Negative-sampling difficulty is moderate

| | value |
|---|---|
| top-1 NN purity (mean-centered) | 0.83 |
| top-5 NN purity | 0.67 |
| shots with negative margin (own − nearest other centroid) | **12 / 54** |
| mean per-shot margin (centered) | 0.049 |

For ~17% of clips, the nearest neighbor in cosine space is in *another* shot —
these are the genuinely hard negatives. And 12 of 54 shots sit closer to a
neighbor shot's centroid than to their own clip spread — i.e., a dozen pairs
of shots are near-duplicates (likely consecutive shots of the same scene).

**Design implication:**

- **Many in-batch negatives + moderate temperature beats hard-negative mining.**
  Mining would surface the same near-duplicate-shot edges over and over and
  push the model to discriminate genuinely confusable shots, which is unlikely
  to be useful for EEG ↔ video alignment.
- For the ~12 near-duplicate shot pairs, consider **soft / temperature-scaled
  targets** rather than hard 1-vs-rest labels. A pure InfoNCE loss treats them
  as wrong, even though the underlying video content is essentially the same.
- Sanity-check the loss on these specific shots if it's available — the model
  shouldn't be forced to a margin it can't physically achieve.

---

## 6. Alternatives to shot-ID membership

The motivation for replacing shot-ID is in §5: top-1 NN purity is 0.83 (17 %
genuinely-hard NN), 12 / 54 shots overlap a neighbor, and the temporal-buffer
analysis above can only address ~5 % of false negatives. If you want to mine
hard negatives without surfacing false ones, you need a membership definition
cleaner than shot identity. Five options, ordered from least to most invasive
change:

### Option 1 — Shot ID + temporal buffer (cheapest fix)

Keep `same_shot` as the positive criterion but add two filters:

- Exclude cross-shot pairs with |Δt| < 2 s from the negative pool (§3 buffer
  analysis: 92 % purity, 11× efficiency).
- Sample positives across the shot's duration, not adjacent frames, so the
  positive carries shot-identity signal and not just adjacency.

Pros: zero new infrastructure. Cons: doesn't address near-duplicate shot pairs
that aren't adjacent in time (3 of the 12 are returning locations, separated
by tens to hundreds of seconds — no buffer can fix them).

### Option 2 — Visual-cluster membership

Cluster the centered V-JEPA-2 clip embeddings into K groups (K < n_shots so
near-duplicate shots merge). Use cluster ID as membership. Hard negatives
become *nearest-other-cluster* pairs — visually similar but no longer false
negatives.

Pros: data-driven, removes the dozen near-duplicate-shot pairs. Cons: cluster
boundaries have no semantic meaning, less interpretable than shot.

### Option 3 — Scene / locale membership (coarser than shot)

Special case of Option 2 with K close to the original shot count: merge only
the most-similar shot pairs into "scenes" using centroid cosine. Several shots
of the same conversation, same location, or returning location collapse to one
scene; the rest stay separate.

This gives a hierarchy you can exploit:

| pair relation | role |
|---|---|
| same shot, far in time | clean positive |
| same scene, different shot | weak positive / semi-positive |
| different scene, near in time | hard negative |
| different scene, far in time | easy negative |

Quantified empirically in §7.

### Option 4 — Continuous similarity targets (give up hard membership)

Drop the binary same/different membership and use centered V-JEPA-2 cosine as
a soft target (SoftCLIP, DistillCLIP, or KL between the EEG-side and
V-JEPA-2-side similarity matrices).

This sidesteps the membership problem entirely — near-duplicate pairs simply
get a high target similarity, the model is not punished for predicting them
similar, and "hard" cases become "pairs where EEG and V-JEPA-2 disagree."

Pros: removes label noise; objective lives on the underlying continuous
structure. Cons: less InfoNCE-like; needs the V-JEPA-2 similarity matrix
precomputed (cheap — 406² floats).

### Option 5 — Content-based / multi-feature membership

Define membership using the per-frame movie features the regressions already
target (`n_faces`, `depth_mean`, `scene_natural_score`, etc.). Two clips are
positives if they're close in this feature space, regardless of shot ID.

Most "semantic" definition; robust to shot-cut artifacts (a face close-up cut
to a face wide-shot of the same person becomes a positive). Cons: depends on
feature-extractor quality.

---

## 7. Scene merge: empirical results for Option 3

`run_scene_merge_analysis.py` implements Option 3: agglomerative average-
linkage clustering of shot centroids in centered cosine space, with a tunable
merge threshold. At the default threshold (centered cos > 0.90) the 54 shots
collapse into **36 scenes**, with 12 multi-shot scenes that match the
originally-identified near-duplicate pairs from §5:

```
scene  0: shots [0, 1]              scene 21: shots [29, 30]
scene  3: shots [8, 9]              scene 23: shots [35, 36]
scene  4: shots [10, 11]            scene 24: shots [37, 38]
scene 10: shots [16, 18]            scene 32: shots [46, 47]
scene 17: shots [25, 26, 27]        scene 33: shots [48, 49, 50, 52]
... plus a few others
```

The frame grid (`vjepa2_scenes_frame_grid.png`) confirms these are
semantically sensible — alternating angles of the same conversation, returning
locations. Scene 33 in particular catches the returning-location case (shots
48-50 and 52 are the kid playing with the dog at the end, with shot 51 as a
separate intervening shot) that no temporal buffer could fix.

### Geometry: clearly improved at scene level

| metric | shot (K=54) | **scene (K=36)** |
|---|---|---|
| η² (between-group / total) | 0.816 | 0.799 (lower — expected with fewer groups) |
| F-statistic | 29.5 | **42.1** |
| same-group cosine (centered) | 0.793 | **0.811** |
| different-group cosine (centered) | −0.008 | **−0.025** |
| d-prime | 1.94 | **2.08** |
| ROC-AUC | 0.916 | **0.936** |
| NN top-1 purity | 0.825 | **0.899** |
| NN top-5 purity | 0.696 | **0.800** |
| groups with negative margin | **12 / 54** | **4 / 36** |

AUC +0.02, NN top-1 +0.07, false-negative count from 12 → 4 — the original
problem from §5 is mostly resolved. The four remaining negative-margin scenes
are genuine content edges that no clustering threshold separates without
over-merging.

### Feature signal: scene-mean is more expensive than shot-mean

| broadcast level | mean R² | loss vs per-clip |
|---|---|---|
| broadcast-shot (n=406) | 0.757 | **−0.035** |
| **broadcast-scene** (n=406) | **0.492** | **−0.300** |

Scene-mean averages across multiple shots, which discards within-scene visual
variation. Features that survive (`position_in_movie` 0.99,
`luminance_mean` 0.84) are slow / scene-level; features that collapse
(`spatial_freq_energy` 0.93 → −0.01, `motion_energy` 0.64 → 0.29) are
within-scene-variable.

### Threshold trade-off (for tuning)

| threshold | scenes | negative-margin | AUC | NN top-1 | bcast-scene loss |
|---|---|---|---|---|---|
| 0.75 | 19 | 1/19 | **0.950** | **0.933** | −0.453 |
| 0.85 | 29 | 3/29 | — | 0.914 | −0.388 |
| **0.90** | **36** | **4/36** | **0.936** | **0.899** | **−0.300** |
| 0.92 | 40 | 8/40 | 0.934 | 0.865 | −0.268 |
| 0.95 | 49 | 10/49 | 0.920 | 0.840 | −0.106 |

0.90 is the natural choice: it matches the 12 originally-flagged
negative-margin shots and pulls NN top-1 above 0.9, with broadcast loss kept
manageable. Looser thresholds (0.75) maximize geometry but destroy the
broadcast signal; tighter ones (0.95) don't fix the original problem.

### Key design move: decouple membership from embedding

The crucial result from §7 + §4: **scene membership is great for the
contrastive label, but scene-mean is a bad embedding target.** Don't conflate
the two:

- Use **scene ID** for the contrastive loss labels (cleaner positives, no
  false negatives within a scene, hard negatives = "same scene, different
  shot" pairs).
- Use **shot-mean (or per-clip)** as the actual embedding target (−0.035
  broadcast loss vs −0.300 for scene-mean).

These are independent: the label tells the loss *what to push together*, the
embedding tells it *what to push*.

See [`run_scene_merge_analysis.py`](run_scene_merge_analysis.py),
[`vjepa2_scenes_overview.png`](vjepa2_scenes_overview.png),
[`vjepa2_scenes_frame_grid.png`](vjepa2_scenes_frame_grid.png).

---

## 8. Embedding granularity given scene labels: per-clip vs shot-mean vs scene-mean

Once scene ID is the contrastive label, the remaining design knob is the
*embedding target* on the V-JEPA-2 side. Three choices:

### Per-clip mean-centered (n=406 unique)

**Pros**

- Maximum target diversity — 406 distinct points, the richest InfoNCE
  vocabulary the data supports.
- Full feature signal preserved (R² 0.79 baseline, §4). EEG can in principle
  learn within-shot variation (motion, momentary visual changes).
- No information coercion at the target side.

**Cons**

- Within-scene positive ambiguity. A scene contains ~10–20 clips with distinct
  embeddings. Standard InfoNCE (one positive per anchor) picks arbitrarily;
  multi-positive / SupCon averages over them, and the *effective* target the
  EEG anchor is pulled toward is the scene-mean of those per-clip embeddings —
  you pay the per-clip variance cost without changing the destination.
- High target variance. Within-shot V-JEPA-2 jitter that EEG cannot predict
  becomes noisy gradient signal.
- Risk of overfitting V-JEPA-2 idiosyncrasies. The shot-averaging analysis
  (§4) showed shot-mean *improved* some features (`n_objects` 0.52→0.74,
  `n_faces` 0.72→0.82), implying per-clip carries denoisable noise.

### Shot-mean centered (n=54 unique, broadcast to 406)

**Pros**

- Matches V-JEPA-2's natural granularity (η²=0.89 — between-shot variance
  dominates). You keep what V-JEPA-2 represents, drop only the noise.
- All clips in a shot share an exact target → no within-shot ambiguity.
  Different shots within the same scene give *distinct but close* targets →
  the contrastive loss naturally sees multi-positive structure at the right
  granularity, without needing a special loss.
- Denoises (§4): broadcast loss only −0.035 mean R²; several features improve.
- Within-scene shot structure still teachable: the EEG can learn which shots
  in a scene are visually closer to each other.

**Cons**

- Loses within-shot motion: `motion_energy` 0.64 → 0.32 broadcast. If EEG
  carries reliable motion-onset signal you've cut that readout channel.
- 54 unique targets is small for very large effective batch sizes.
- Inherits whatever bias V-JEPA-2's shot-aggregation has.

### Scene-mean centered (n=36 unique, broadcast)

**Pros**

- Cleanest label alignment: one target per scene_id. No within-label
  ambiguity at all.
- Lowest-variance target — averaging across ~11 clips per scene.
- Same vocabulary size as label set — contrastive task is well-posed
  trivially.

**Cons**

- Discards 30 % of feature signal (broadcast loss −0.30, §7).
- Multi-shot scenes collapse: e.g. scene 33 = shots [48, 49, 50, 52], all
  distinct visuals, all become one point. EEG cannot learn within-scene
  structure at all.
- Tiny target vocabulary (36) → contrastive signal saturates quickly.
- Bottleneck mismatch: if downstream EEG decoding needs sub-scene resolution,
  you've cut it off at the source.

### Interaction with the loss formulation

| loss | per-clip target | shot-mean target | scene-mean target |
|---|---|---|---|
| **InfoNCE** (1 positive / anchor) | arbitrary positive choice; high variance | clean — multi-positive structure emerges naturally from within-scene different-shot pairs | trivial — one target per label |
| **SupCon / multi-positive** | averages → implicit scene-mean target with explicit per-clip variance penalty | "scene-mean of shot-means" — preserves the shot hierarchy | same as InfoNCE |
| **Soft / sim-matrix targets** | preserves full continuous structure | preserves shot-level structure | step-function-like targets |

The natural pairing: **shot-mean + InfoNCE** gives implicit multi-positive
structure without a fancy loss. **Per-clip + SupCon** has the same expected
target as scene-mean with extra variance. **Scene-mean + anything** is the
cleanest formulation but the most lossy.

### Default recommendation

**Shot-mean centered embeddings + scene IDs for labels + 2 s temporal buffer +
InfoNCE.** This operating point maximizes (label cleanliness) × (embedding
richness):

- Scene labels remove false negatives (AUC 0.92 → 0.94, top-1 NN 0.83 → 0.90).
- Shot-mean retains 95 % of the feature signal (and denoises 6 / 16 features).
- Within-scene different-shot pairs supply multi-positive structure for free.

### When to deviate

- **Per-clip targets** if you have a hypothesis that EEG carries within-shot
  motion (the marginal R² on `motion_energy` is +0.32 over shot-mean).
- **Per-clip targets** if you're using a similarity-matrix / soft-label loss
  — it's the loss family designed to consume continuous target structure.
- **Scene-mean targets** if the EEG signal is very noisy at clip resolution
  and the variance reduction from ~11-clip averaging dominates the bias cost.

### The collapse-to-scene-mean risk

With shot-mean targets + scene labels, the contrastive label tells the loss
that two shots in the same scene should produce similar EEG embeddings — but
the V-JEPA-2 shot-mean targets for those shots are *distinct* points. The
EEG embedding can satisfy "same scene" by sitting between the shot-mean
targets, which is exactly the scene-mean. If you find EEG embeddings
collapsing to scene-mean despite using shot-mean targets, that's the
diagnosis: you've effectively trained scene-mean alignment with extra steps.

Two responses: (a) accept it — you decided to learn at scene granularity —
or (b) drop scene IDs back to shot IDs as labels, accepting the false-negative
cost quantified in §5 / §7.

---

## 9. Bottom-line recipe for the V-JEPA-2 → EEG CLIP head

1. **Preprocess the V-JEPA-2 targets**: compute the global mean vector on the
   training clips, subtract it. No whitening, no per-dim scaling (§2).
2. **Embedding granularity**: **shot-mean** as the default target (§8). It
   matches V-JEPA-2's natural between-shot variance structure, preserves 95 %
   of feature signal, and denoises a third of features. Use per-clip targets
   only if you have a specific hypothesis about within-shot signal (e.g.,
   motion) or you're switching to a similarity-matrix / soft-label loss.
   **Do not use scene-mean as the embedding** — it discards 30 % of feature
   signal (§7, §8).
3. **Membership for the contrastive label**: scene ID, not shot ID (§7). Build
   the scene map once via
   [`run_scene_merge_analysis.py`](run_scene_merge_analysis.py) at threshold
   0.90 and use the resulting `vjepa2_scenes_map.csv`. Hard negatives become
   "same scene, different shot" pairs — visually adjacent without being false
   positives.
4. **Loss formulation**: plain InfoNCE works with the shot-mean + scene-ID
   combination (§8) — within-scene different-shot pairs naturally produce
   multi-positive structure because the shot-means are distinct but close. No
   need for SupCon / multi-positive unless you switch to per-clip targets.
5. **Temporal-buffer filter on negatives**: drop pairs with |Δt| < 2 s
   regardless of scene (§3 buffer analysis: 92 % purity, free win). This
   handles the shot-cut-boundary cases the scene map doesn't.
6. **Many in-batch negatives, moderate temperature, no hard mining.** With
   scene-level membership the negative pool is already cleaner; mining on top
   would over-fit the residual hard cases.
7. **Diagnostic to watch**: are EEG embeddings collapsing toward scene-mean
   despite using shot-mean targets? If yes, the loss is winning the
   "same-scene → same-EEG" battle at the expense of within-scene shot
   structure (§8, "collapse-to-scene-mean risk"). Either accept it (you've
   chosen scene-granularity learning) or fall back to shot-ID labels with
   the false-negative cost.
8. **Sanity-check numbers**: η² = 0.89, raw AUC 0.88, centered AUC 0.92,
   scene-level AUC 0.94, NN top-1 = 0.90. A CLIP run that produces effective
   AUC much below 0.92 after a few epochs has a preprocessing / temperature
   issue, not a model issue.

## Pointers

- Variance / geometry: [`run_embedding_variance_vjepa2.py`](run_embedding_variance_vjepa2.py),
  [`vjepa2_embedding_variance.png`](vjepa2_embedding_variance.png),
  [`vjepa2_embedding_variance_summary.txt`](vjepa2_embedding_variance_summary.txt)
- Shot-averaging: [`run_shot_averaged_vjepa2.py`](run_shot_averaged_vjepa2.py),
  [`vjepa2_results_shot_averaged.png`](vjepa2_results_shot_averaged.png)
- Centering sanity check: [`run_shot_averaged_vjepa2_centered.py`](run_shot_averaged_vjepa2_centered.py)
- Scene merge (Option 3): [`run_scene_merge_analysis.py`](run_scene_merge_analysis.py),
  [`vjepa2_scenes_overview.png`](vjepa2_scenes_overview.png),
  [`vjepa2_scenes_frame_grid.png`](vjepa2_scenes_frame_grid.png),
  [`vjepa2_scenes_map.csv`](vjepa2_scenes_map.csv)
- Movie features the regressions target: [`run.py`](run.py),
  [`results.csv`](results.csv)
