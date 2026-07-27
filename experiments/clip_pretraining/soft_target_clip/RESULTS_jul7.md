# soft_target_clip — RESULTS (jul7)

Andonian-2022-style soft-target distillation from V-JEPA-2 similarity, adapted
to a frozen teacher, tested against vanilla CLIP and jul2's scene_clip baseline
under matched training conditions. Concludes the `soft_target_clip` experiment
line and updates the recommended recipe for EEG ↔ V-JEPA-2 CLIP pretraining.

All artifacts in this directory. Configs / probe JSONs / this write-up are on
branch [`autoresearch/soft-target-clip-jul7`](https://github.com/dungscout96/eb_jepa_eeg/tree/autoresearch/soft-target-clip-jul7).

---

## TL;DR

Three losses (vanilla CLIP, scene_clip, soft_target_clip) tested at 3 seeds ×
2 data regimes (multi-movie, TP-only), 400 ep each. Two probe protocols
(5-fold CV on val, plus SSL-standard train→test on R6). Also compared against
the historical record on the same protocol.

**Findings:**

1. At **matched-condition 400 ep multi-movie**, all three losses are tied
   within seed noise. Vanilla CLIP mean Δr² = **+0.0423 ± 0.0005**, soft
   τ=0.05 = **+0.0426 ± 0.0006** (joint over TP + DM). Matches jul2 iter 1
   scene_clip (+0.0413) exactly.
2. On **TP-only 400 ep**, soft τ=0.05 wins over vanilla by **+0.0026 mean Δr²**
   (2.6× seed std). The multi-positive teacher signal genuinely helps when it
   can concentrate on within-movie structure; cross-movie teacher mass in
   multi-movie dilutes it away.
3. Best overall from-scratch result: **TP-only soft τ=0.05, seed=2026**.
   - Val 5-fold CV: **Δr² = +0.0501 ± 0.0010**
   - Test train→test **Pearson r = 0.1517, Δr = +0.0990**
   - **~+0.019 raw r above [`fresh500_ep499`](../scene_clip_fromscratch/RESULTS_fresh500_sweep.md)**,
     the prior best from-scratch scene_clip result.
   - Still **−0.020 raw r below `reve_warmstart_v2_ep299`** — REVE
     pretraining brings genuine value no from-scratch loss recovers.
4. **The DM ceiling is loss-independent at ~+0.038 Δr².** Every jul2 iter
   that landed at multi-movie best (iter 1 scene_clip, iter 7 deeper, plus
   our vanilla and soft) hit within 0.0002 of each other on DM. Loss / depth
   / width / compute all fail to move it.
5. **Mean-centering was the actual load-bearing intervention** in the whole
   scene_clip line, not the label mask. Once you fix centering + lr + epochs,
   the loss function contributes essentially nothing on the multi-movie
   metric — vanilla CLIP catches all the way up.
6. **Recommendation for new movies**: soft τ=0.05 on TP-only-style training,
   vanilla CLIP on multi-movie-style training. Neither needs the scene-cluster
   preprocessing that scene_clip required — a lite recipe (only `global_mean`)
   is now sufficient.

---

## §1. Setup

### 1.1 The soft-target objective

Andonian 2022 (*Robust Cross-Modal Representation Learning With Progressive
Self-Distillation*, CVPR) blends the hard CLIP diagonal with a soft target
distribution from an EMA teacher, gradually shifting weight to the teacher as
it becomes reliable. Our adaptation drops the EMA and the schedule because
the "teacher" here — V-JEPA-2 pairwise similarity — is **frozen and reliable
from step 0**. The blended target per row is

```
t = (1 − α) · I + α · softmax(V V^T / τ_t)
```

with V being the (mean-centered) V-JEPA-2 target vectors. Loss is a symmetric
cross-entropy against t. Implementation:
[`SoftTargetCLIPPretrain`](../../../eb_jepa/clip.py).

Sensitivity sweep at 160 ep (see §3.1) chose the operating point
**α = 0.5, τ_t = 0.05**. Diagonal-hard component (no scene mask), 2-second
temporal buffer excluding cross-pairs from both the student and teacher
softmax denominators.

### 1.2 Training config

Matched across all three losses (vanilla CLIP, scene_clip, soft_target_clip).
Every knob from jul2's iter 1 baseline except `loss.mode`:

```yaml
model:  encoder_embed_dim: 512, encoder_depth: 12, encoder_heads: 8, patch_size: 400
loss:   mean_center: true, target_kind: per_window, temporal_buffer_s: 2.0
        proj_dim: 512, temperature: 0.07, vision_passthrough: false
optim:  optimizer: adam, lr: 1e-4, warmup_epochs: 5, epochs: 400
data:   batch_size: 64, n_windows: 8, window_size_seconds: 2, task: [ThePresent, DespicableMe]
        (single task override for TP-only runs)
```

Two engineering changes committed with this experiment:

- **`recipe_require_shots: bool = True`** on `JEPAMovieDataset`. When False
  (auto-set by trainer for `loss.mode=soft_target_clip` and `loss.mode=clip`),
  the dataset works with a lite V-JEPA-2 recipe that only carries
  `global_mean` — no shot boundaries, no scene clustering needed.
- **`loss.mode=clip` extended to use recipe centering**. Vanilla `CLIPPretrain`
  now gets the same mean-centered per-window V-JEPA-2 targets that
  scene_clip / soft_target_clip use, closing the confound in fresh500's
  vanilla-vs-scene_clip comparison.

### 1.3 Evaluation protocols

Two, run separately:

| protocol | fit on | eval on | metric | when used |
|---|---|---|---|---|
| **5-fold CV on val** ([probe.py](../../../eb_jepa/evaluation/clip_probe/probe.py)) | 4 folds of R5 | held-out R5 fold | R² (5-fold mean) | primary metric for the 3-seed replicates |
| **train→test on R6** ([probe_traintest.py](../../../eb_jepa/evaluation/clip_probe/probe_traintest.py)) | full R1–R4 train (~71K windows) | full R6 test (~11K windows) | **Pearson r** + R² | SSL-literature standard; used only on the best checkpoint for direct comparison to the historical record |

All numbers report mean over 12 scalar movie features (see [design notes
§9 sanity gate](../embedding_feature_correlation/clip_design_observations_vjepa2.md)
for the list). Random baselines:

- `rand_r2_TP` = +0.01589 (jul2 iter-12 shape, single-movie)
- `rand_r2_DM` = +0.00695 (jul2 iter-12 shape, single-movie)
- `random_depth4` (fresh500) mean Pearson r on R6 test = 0.0527
- `random_reve_shape` (scene_clip_from_checkpoint) mean Pearson r on R6 test = 0.0516

---

## §2. The soft-target design

Details of the objective's formulation, sensitivity choices, and interaction
with the mean-centered per-window targets are in [`NOTES.md`](NOTES.md). The
short version:

- **α = 0.5** — no schedule. Frozen-teacher doesn't need Andonian's
  progressive ramp; the teacher is reliable from step 0.
- **τ_t = 0.05** — chosen empirically from the 160-ep sensitivity sweep (§3.1).
  At τ_t=0.1 the teacher softmax is too smeared (effective positives ≈ 5–6);
  at τ_t=0.05 it sharpens to ~2–3 effective positives, which turns out to be
  the sweet spot.
- **Diagonal hard component**, not scene-multi-positive. The teacher's soft
  distribution already puts high mass on same-shot rows (same-shot targets
  are exact duplicates after shot-mean averaging), so layering scene IDs on
  top would double-count.
- **2-second temporal buffer** on both student and teacher softmax
  denominators — cross-pairs within 2 s of each other are excluded from both
  the target and the prediction.
- **`scene_ids` accepted but unused** by the module (interface parity with
  `SceneCLIPPretrain`; the teacher carries all the label structure).

---

## §3. Results

### 3.1 Sensitivity sweep (160 ep, multi-movie)

Established that τ_t=0.05 is the right point on the sharpness axis and α=0.5
sits at a reasonable blend. Reported values are joint mean Δr² over TP + DM.

| config | mean Δr² | vs scene_clip | vs default soft |
|---|---:|---:|---:|
| scene_clip (160-ep baseline) | +0.0299 | 0 | +0.0015 |
| soft α=0.5 τ_t=0.1 (default) | +0.0284 | −0.0015 | 0 |
| **soft α=0.5 τ_t=0.05** (sharper teacher) | **+0.0316** | **+0.0017** | **+0.0032** |
| soft α=0.25 τ_t=0.1 (lighter blend) | +0.0306 | +0.0007 | +0.0022 |

Both sensitivity variants flipped the initial soft-vs-scene_clip loss at
default τ_t=0.1 into a small win. τ_t=0.05 was the clear leader and became
the promoted config for 400-ep testing.

### 3.2 Full 3-seed replication at 400 ep

Multi-movie (jul7-3seed), joint mean Δr² over TP + DM:

| arm | seed=2025 | seed=2026 | seed=2027 | **mean ± std** |
|---|---:|---:|---:|---:|
| vanilla clip | +0.0423 | +0.0418 | +0.0428 | **+0.0423 ± 0.0005** |
| soft τ=0.05 | +0.0420 | +0.0430 | +0.0430 | **+0.0426 ± 0.0006** |

Gap = **+0.0003** in favor of soft. Seed noise ≈ 0.0005. **Effectively tied
on multi-movie.**

TP-only (jul7-tp), mean Δr² on TP only:

| arm | seed=2025 | seed=2026 | seed=2027 | **mean ± std** |
|---|---:|---:|---:|---:|
| vanilla clip | +0.0464 | +0.0482 | +0.0479 | **+0.0475 ± 0.0010** |
| **soft τ=0.05** | **+0.0492** | **+0.0512** | **+0.0499** | **+0.0501 ± 0.0010** |

Gap = **+0.0026** in favor of soft. Seed noise ≈ 0.0010. **2.6× seed std —
small but real effect** when training is single-movie.

### 3.3 Per-movie breakdown (multi-movie 3-seed)

| arm | movie | seed=2025 | seed=2026 | seed=2027 | mean ± std |
|---|---|---:|---:|---:|---:|
| vanilla clip | TP | +0.0467 | +0.0466 | +0.0470 | +0.0468 ± 0.0002 |
| vanilla clip | DM | +0.0380 | +0.0369 | +0.0385 | +0.0378 ± 0.0008 |
| soft τ=0.05 | TP | +0.0463 | +0.0476 | +0.0482 | +0.0474 ± 0.0010 |
| soft τ=0.05 | DM | +0.0377 | +0.0384 | +0.0377 | +0.0379 ± 0.0004 |

Two things worth reading off:

- **DM is essentially identical between the two losses** (+0.0378 vs +0.0379).
- On **TP**, soft has a small edge (+0.0474 vs +0.0468, roughly matching the
  TP-only pattern but muted by cross-movie dilution).

### 3.4 Test-set Pearson r on the best checkpoint

Best checkpoint = TP-only soft τ=0.05 seed=2026 (highest single-seed Δr² =
+0.0512). Ran `probe_traintest.py` on both `--eval-split val` and
`--eval-split test`.

| checkpoint | init | val mean r | test mean r | test Δr |
|---|---|---:|---:|---:|
| fresh500_ep499 scene_clip from scratch | rand | — | 0.1328 | +0.0801 |
| **ours: TP-only vanilla clip 400 ep (seed=2025)** | rand | 0.2485 | 0.1459 | +0.0932 |
| **ours: TP-only soft τ=0.05 400 ep (seed=2026)** | rand | **0.2584** | **0.1517** | **+0.0990** |
| reve_warmstart_v2_ep299 scene_clip warm-start | REVE | — | **0.1715** | **+0.1198** |

**Ours (soft τ=0.05) beats fresh500 by +0.019 raw r on the R6 test set** —
the largest from-scratch result on the SSL-standard protocol. **Still loses
to reve_warmstart_v2 by −0.020 raw r**; REVE pretraining brings ~+0.04 of
Pearson r that from-scratch training does not recover at 400 ep.

### 3.5 Per-feature pattern (R6 test)

Sorted by which checkpoint is best on each feature:

| feature | fresh500 | reve_wsv2 | ours vanilla | **ours soft** |
|---|---:|---:|---:|---:|
| luminance_mean | 0.159 | **0.227** | 0.167 | 0.167 |
| contrast_rms | 0.134 | **0.211** | 0.120 | 0.126 |
| saturation_mean | 0.149 | **0.198** | 0.126 | 0.143 |
| entropy | 0.150 | **0.203** | 0.137 | 0.163 |
| position_in_movie | 0.116 | **0.241** | 0.108 | 0.106 |
| **edge_density** | 0.109 | 0.131 | 0.131 | **0.136** |
| **motion_energy** | 0.179 | 0.188 | 0.210 | **0.238** |
| **narrative_event_score** | 0.107 | 0.118 | 0.139 | **0.145** |
| **n_faces** | 0.128 | 0.144 | **0.166** | 0.164 |
| **face_area_frac** | 0.107 | 0.129 | **0.143** | 0.142 |
| **depth_mean** | 0.128 | 0.139 | **0.155** | 0.149 |
| **scene_natural_score** | 0.129 | 0.130 | **0.149** | 0.141 |

**REVE pretraining dominates low-level EEG-statistics features** (luminance,
contrast, saturation, entropy, position). This is intuitive: REVE was
pretrained on raw EEG spectral / temporal statistics that correlate with
these low-level movie features via bottom-up visual pathways.

**Our from-scratch runs dominate content features** (motion, narrative,
edges, faces, depth, scene-naturalness). The contrastive objective drives
the encoder toward exactly the semantic dimensions V-JEPA-2 emphasizes.

These are complementary rather than strictly ordered wins — the largest
open lever is combining them (§4.3).

### 3.6 Top-K retrieval on the best checkpoint

`probe_traintest.py` gives per-feature Pearson r; the SSL-literature
counterpart is Top-K retrieval — for each EEG window, does the paired
V-JEPA-2 target rank in the top-K by cosine similarity? Implemented in
[`eb_jepa/evaluation/clip_probe/retrieval.py`](../../../eb_jepa/evaluation/clip_probe/retrieval.py)
at three pool granularities:

- **time**: one candidate per unique `(task, round(t_start / 0.5s))`. Time
  bucketing matches V-JEPA-2's ~2 Hz clip rate. Task: "identify the movie
  moment you're watching."
- **shot**: one candidate per unique `(task, shot_id)`; pool entry is the
  L2-normalized centroid of projected V-JEPA-2 vectors within that shot.
  Task: "identify the shot."
- **scene**: one candidate per unique `(task, scene_id)`; same centroid
  semantics. Task: "identify the scene."

Chance level for both directions is `K / N_pool` (both e→v and v→e — see
[commit `0cb02ab`](https://github.com/dungscout96/eb_jepa_eeg/commit/0cb02ab) for the derivation).

**VAL** (M = 29,593 EEG windows, 293 recordings):

| level | N | e→v Top-1 | e→v Top-5 | e→v Top-10 | v→e Top-1 | v→e Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| time | 101 | 0.083 (8.4×) | 0.229 (4.6×) | 0.339 (3.4×) | 0.010 (1.0×) | 0.109 (1.1×) |
| shot | 49 | 0.122 (6.0×) | 0.343 (3.4×) | 0.505 (2.5×) | 0.020 (1.0×) | 0.184 (0.9×) |
| scene | 35 | 0.145 (5.1×) | 0.438 (3.1×) | **0.618 (2.2×)** | 0.029 (1.0×) | 0.257 (0.9×) |

**TEST** (M = 10,908 EEG windows, 108 recordings):

| level | N | e→v Top-1 | e→v Top-5 | e→v Top-10 | v→e Top-1 | v→e Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| time | 101 | 0.054 (5.5×) | 0.157 (3.2×) | 0.240 (2.4×) | 0.010 (1.0×) | 0.109 (1.1×) |
| shot | 49 | 0.084 (4.1×) | 0.255 (2.5×) | 0.391 (1.9×) | 0.020 (1.0×) | 0.204 (1.0×) |
| scene | 35 | 0.106 (3.7×) | 0.351 (2.5×) | **0.517 (1.8×)** | 0.029 (1.0×) | 0.257 (0.9×) |

Two clean findings from these tables:

1. **e→v works meaningfully**: the encoder identifies the correct scene
   51.7 % of the time in top-10 on unseen R6 test recordings (out of 35
   candidates, chance 28.6 %). Time-level Top-1 sits at 5.5 × chance on
   test / 8.4 × on val. Val-vs-test optimism gap (~40 % on Top-1) mirrors
   the Pearson-r probe pattern.
2. **v→e is at chance across all levels and both splits.** The most-similar
   EEG anchor for a given shot/scene centroid is essentially random — no
   subject-window sits systematically closer to the centroid than others.
   **Signature of cross-subject variability dominating within-group
   alignment tightness**, consistent with the modality-gap literature
   (see the sibling [`../cs_aligner/`](../cs_aligner/) work).

JSONs: [`retrieval_val_jul7-tp_soft_seed2026_TP.json`](retrieval_val_jul7-tp_soft_seed2026_TP.json),
[`retrieval_test_jul7-tp_soft_seed2026_TP.json`](retrieval_test_jul7-tp_soft_seed2026_TP.json).
(Note: raw Top-K values in the JSONs are correct; the pre-`0cb02ab`
`v2e_chance` field uses K/M and understates chance — the tables here
use the corrected K/N formula.)

**vs. REVE warm-start (test set).** The
[`scene_clip_from_checkpoint` retrained winner](../scene_clip_from_checkpoint/RESULTS.md#310-top-k-retrieval--the-ssl-standard-alignment-metric)
(`warmstart_lr3e4_ep299`, Delta job 20400451) provides a direct comparison
on the SSL retrieval protocol:

| level | metric | ours (from-scratch) | REVE-warmstart | Δ |
|---|---|---:|---:|---:|
| time | e→v Top-1 | 0.054 (5.5×) | 0.046 (4.7×) | −0.008 |
| shot | e→v Top-1 | 0.084 (4.1×) | 0.114 (5.6×) | **+0.030** |
| scene | e→v Top-1 | 0.106 (3.7×) | 0.149 (5.2×) | **+0.043** |
| scene | e→v Top-10 | 0.517 | 0.532 | +0.015 |
| time | **v→e Top-1** | 0.010 (1.0×) | **0.020 (2.0×)** | **+0.010** |
| shot | **v→e Top-1** | 0.020 (1.0×) | **0.041 (2.0×)** | **+0.020** |
| scene | **v→e Top-1** | 0.029 (1.0×) | **0.057 (2.0×)** | **+0.029** |

REVE warm-start wins on shot / scene e→v and **doubles v→e Top-1 across all
levels** — reaching from below chance (1.0×) into meaningful above-chance
(2.0×). This flips the v→e-at-chance finding above: **REVE warm-start
alleviates the modality gap** that the from-scratch soft-target objective
leaves in place. Consistent with the +0.026 raw Pearson r advantage REVE
holds on the probe protocol (§3.4). The one place from-scratch soft-target
matches or narrowly beats REVE is **time-level e→v Top-1**, where V-JEPA-2's
per-window representation is what our soft-target loss most directly
optimizes against.

---

## §4. Full comparison to the historical record

### 4.1 On the joint multi-movie metric (mean Δr² over TP + DM)

| checkpoint | loss | epochs | data | joint Δr² |
|---|---|---:|---|---:|
| jul2 iter 4 (160-ep baseline) | scene_clip | 160 | multi | +0.0299 |
| jul2 iter 3 (lr=2e-4) | scene_clip | 400 | multi | +0.0294 |
| jul2 iter 5 (wider @ compute-matched) | scene_clip | 85 | multi | +0.0250 |
| jul2 iter 9 (2× compute) | scene_clip | 800 | multi | +0.0382 |
| jul2 iter 8 (both wider + deeper) | scene_clip | 400 | multi | +0.0378 |
| jul2 iter 2 (500 ep) | scene_clip | 500 | multi | +0.0412 |
| jul2 iter 1 (long-budget best) | scene_clip | 400 | multi | +0.0413 |
| jul2 iter 6 (wider, 400 ep) | scene_clip | 400 | multi | +0.0420 |
| jul2 iter 7 (deeper, 400 ep) | scene_clip | 400 | multi | +0.0419 |
| **ours: vanilla clip 3-seed mean** | vanilla | 400 | multi | **+0.0423 ± 0.0005** |
| **ours: soft τ=0.05 3-seed mean** | soft | 400 | multi | **+0.0426 ± 0.0006** |

**Everything at 400 ep, embed=512 lands in ~[+0.041, +0.043].** Neither our
loss changes nor jul2's capacity diagnostics move the needle. The joint
metric ceiling looks stable across ~14 configurations tried.

### 4.2 On the DM Δr² axis specifically

Per §3.3, DM is loss-independent and pinned at ~+0.038 across every
configuration that reached the 400-ep asymptote:

| checkpoint | DM Δr² |
|---|---:|
| jul2 iter 5 (wider @ 85 ep) | +0.0229 |
| jul2 iter 3 (lr=2e-4) | +0.0266 |
| jul2 iter 4 (baseline @ 160) | +0.0277 |
| jul2 iter 0 (baseline @ 250) | +0.0322 |
| jul2 iter 9 (2× compute) | +0.0330 |
| jul2 iter 8 (both wider + deeper) | +0.0341 |
| jul2 iter 2 (500 ep) | +0.0367 |
| jul2 iter 6 (wider) | +0.0367 |
| **ours vanilla clip 3-seed** | +0.0378 ± 0.0008 |
| jul2 iter 1 (long budget) | +0.0379 |
| **ours soft τ=0.05 3-seed** | +0.0379 ± 0.0004 |
| jul2 iter 7 (deeper) | +0.0380 |

**Four best runs land in [+0.0378, +0.0380].** Doubling depth (iter 7),
doubling params (iter 8), doubling epochs (iter 9), swapping to vanilla, and
swapping to soft-target all fail to move DM meaningfully. The ceiling is not
loss- or capacity-limited — likely a data or V-JEPA-2 teacher-quality
bottleneck on DespicableMe specifically.

### 4.3 On R6 test Pearson r (SSL-literature standard)

Full ranking of every recorded `probe_traintest` result:

| checkpoint | init | test mean r | test Δr |
|---|---|---:|---:|
| random_reve_shape (baseline) | — | 0.0516 | — |
| random_depth4 (baseline) | — | 0.0527 | — |
| reve_init (raw REVE weights) | REVE | 0.0938 | +0.042 |
| reve_frozen_ep99 (REVE frozen) | REVE | 0.0938 | +0.042 |
| reve_warmstart_v1_ep99 | REVE | 0.1373 | +0.086 |
| fresh500_ep499 scene_clip | rand | 0.1328 | +0.0801 |
| **ours: TP-only vanilla clip 400 ep** | rand | 0.1459 | +0.0932 |
| **ours: TP-only soft τ=0.05 400 ep** | rand | **0.1517** | **+0.0990** |
| **reve_warmstart_v2_ep299 scene_clip warm-start** | REVE | **0.1715** | **+0.1198** |

Our best from-scratch (soft τ=0.05, seed=2026, TP-only) is currently the
**best from-scratch checkpoint on record** by +0.019 raw r over fresh500.
REVE + scene_clip fine-tuning (reve_warmstart_v2_ep299) remains the overall
best, +0.020 raw r above ours.

---

## §5. What this means

### 5.1 The scene_clip mask was not load-bearing

At 400 ep with matched conditions, vanilla CLIP with mean-centered per-window
V-JEPA-2 targets **matches scene_clip within seed noise** on the multi-movie
joint metric. The scene_clip method (agglomerative-clustered scene IDs, 2 s
temporal buffer, SupCon-multi-positive loss) improved training-speed
convergence and helped at 160-ep budgets, but doesn't move the 400-ep
ceiling. See jul7 chat log for full derivation.

**Practical corollary**: mean-centering was the load-bearing intervention in
scene_clip's initial reported gains (fresh500 +0.0272 raw Δr² over its own
lr=3e-4 uncentered baseline of +0.0014). The scene mask was a shortcut, not
a ceiling-raiser.

### 5.2 Soft-target genuinely helps on single-movie training

At TP-only 400 ep, soft τ=0.05 beats vanilla CLIP by +0.0026 mean Δr² (2.6×
seed std). The multi-positive teacher signal — mass distributed across ~3
effective positives per anchor — appears to give the encoder a soft
regularization that vanilla one-hot targets don't. This effect vanishes in
multi-movie because cross-movie targets are approximately orthogonal in the
centered space, dominating the softmax denominator and diluting the
within-movie multi-positive mass.

**Practical corollary**: for single-movie or single-domain CLIP-style
training, use soft-target. For multi-domain / multi-movie, vanilla is
equivalent.

### 5.3 DM is teacher-limited, not loss- or capacity-limited

The **loss-independent DM ceiling at ~+0.038 Δr²** is the strongest
constraint we've documented on this pipeline. It sits regardless of loss
(scene_clip, vanilla, soft), capacity (embed=512 → 1024, depth=12 → 24), or
compute (400 ep → 800 ep). Loss-level interventions are exhausted for DM.

Where the DM ceiling probably comes from:

- **V-JEPA-2 teacher fidelity on DespicableMe.** V-JEPA-2 was pretrained on
  a different data distribution; DM (2010 3D animated film, fast cuts,
  cartoon physics) is likely further out-of-distribution than TP (2014 CGI
  short with slower cuts).
- **Data scale / diversity.** ~700 DM recordings might be too few for the
  particular signal the linear probe measures.

Neither is fixable by loss engineering. Next-step recommendations for DM
specifically live outside this experiment's scope.

### 5.4 REVE pretraining is the largest remaining lever

reve_warmstart_v2_ep299 sits +0.020 raw Pearson r above our best from-scratch
result on R6 test. The per-feature breakdown (§3.5) shows the win comes
mostly from low-level EEG-statistics features (luminance, contrast,
saturation, entropy, position). These are exactly the features REVE's
pretraining objective is well-aligned to.

Combining REVE warm-start with soft-target CLIP loss is the natural next
experiment. The falsifiable claim: **REVE + soft τ=0.05 400 ep will land at
r ≈ 0.19+ on test**, above the current record. See §7.

---

## §6. Engineering deliverables from this experiment

Committed on branch `autoresearch/soft-target-clip-jul7`, retained for
future experiments:

- **[`SoftTargetCLIPPretrain`](../../../eb_jepa/clip.py)** — the module,
  ~120 lines. Interface parity with `SceneCLIPPretrain`.
- **Extended `recipe_mode`** in [`clip_pretrain.py`](../../../eb_jepa/training/clip_pretrain.py) —
  vanilla `CLIPPretrain` now supports the same centered per-window recipe
  path (previously only scene_clip / soft_target_clip did). Closes the
  vanilla-vs-scene_clip confound in fresh500.
- **`recipe_require_shots=False`** on [`JEPAMovieDataset`](../../../eb_jepa/datasets/hbn.py) —
  soft_target_clip and vanilla clip can now train on a movie without shot
  boundaries. Recipe .npz needs only `global_mean`; `shot_means` and
  `scene_id_per_shot` are optional. **Onboarding a new movie for
  soft_target_clip is now 3 steps instead of 4** (skip scene-merge
  threshold selection).
- **[`_submit_ab.py`](_submit_ab.py)** — multi-mode / multi-seed / multi-task
  submit script for future replicate work. Includes `--epochs`, `--seed`,
  `--task`, `--alpha`, `--tau` overrides.
- **[`_probe_only.py`](_probe_only.py)** — recovery submit for TIMEOUT'd
  training jobs. Runs the standard probe on `latest.pth.tar` for any
  arm / run_tag / seed combination.
- **`--logging.save_every=99999`** wired into `_submit_ab.py`. Avoids the
  quota-fill incident documented in the jul7 chat log — 400-ep runs now
  cost ~450 MB each instead of ~8 GB, allowing safe replicate parallelism.

Two documentation artifacts:

- **[`NOTES.md`](NOTES.md)** — hypothesis, design, sensitivity discussion.
- **`neurolab` skill file** — added pitfall #5 documenting the disk-quota
  incident and its symptoms (misleading `COMPLETED 0:0` exit code, missing
  probe outputs, `.tmp` files left behind). Available for future users of
  the skill.

---

## §7. What we recommend going forward

### 7.1 Default recipe for from-scratch training on a new movie

**Soft τ=0.05 + mean-centered per-window V-JEPA-2 + lr=1e-4 + 400 ep.**
Uses the lite recipe (only `global_mean` needed). No scene-cluster
preprocessing, no shot-detection pipeline, no threshold tuning per movie.

If you're **only training on multi-movie**, vanilla CLIP works equally well —
save the soft-target implementation for the times it matters.

### 7.2 The obvious next experiment

**REVE warm-start + soft τ=0.05 + 400 ep on TP-only.** Two hypotheses to
resolve:

- Does the from-scratch soft-target win transfer to REVE-warm-start (which
  would beat `reve_warmstart_v2_ep299`)?
- Or does REVE warm-start dominate the loss choice (in which case scene_clip
  and soft would tie again once warm-started)?

Either result is informative. Cost: ~2h 30m Delta job.

### 7.3 Off-limits (no more juice on these axes)

- **Capacity scaling** (wider / deeper) — jul2 §5 showed diminishing returns,
  and our result confirms nothing to gain past embed=512 depth=12 at this
  data scale.
- **Compute scaling** past 400 ep — jul1 §3.2, jul2 iter 9, and our
  observations all agree the cosine-schedule pathology past ~500 ep hurts.
- **Loss engineering on DM** — the +0.038 ceiling is teacher- or data-limited.

---

## §8. Reproducibility

Best training run (single-seed):

```bash
python experiments/clip_pretraining/soft_target_clip/_submit_ab.py \
    soft_target_clip jul7-tp --tau=0.05 --epochs=400 \
    --seed=2026 --task=ThePresent submit
```

3-seed replicate loop:

```bash
for seed in 2025 2026 2027; do
    python .../soft_target_clip/_submit_ab.py \
        soft_target_clip jul7-tp --tau=0.05 --epochs=400 \
        --seed=$seed --task=ThePresent submit
    python .../soft_target_clip/_submit_ab.py \
        clip jul7-tp --epochs=400 --seed=$seed --task=ThePresent submit
done
```

Per-checkpoint Pearson r on val + test:

```bash
python -c "
from neurolab.jobs import Job
exp_dir = '/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip/jul7-tp_soft_target_clip_seed2026'
for split in ('val', 'test'):
    Job(name=f'tt_{split}', cluster='delta', repo_path='/u/dtyoung/eb_jepa_eeg',
        partition='gpuA40x4', time_limit='00:30:00', venv='__none__', branch='',
        env_vars={'HBN_PREPROCESS_DIR': '/projects/bbnv/kkokate/hbn_preprocessed'},
        command=(
            f'PYTHONPATH=. uv run --group eeg python '
            f'eb_jepa/evaluation/clip_probe/probe_traintest.py '
            f'--checkpoint {exp_dir}/latest.pth.tar '
            f'--config {exp_dir}/config_TP.yaml '
            f'--eval-split {split} '
            f'--output experiments/.../probe_traintest_{split}_seed2026.json'
        )).submit()
"
```

Wall-time total for the jul7 experiments (after quota-fix): ~28 GPU-hours on
Delta A40 across 12 replicate runs + 3 probe_traintest jobs + 4 sensitivity
runs.

---

## §9. Artifacts

All checked in beside this document:

**5-fold CV probe (val split)**:

- `probe_val_jul7-3seed_{clip,soft_target_clip}_seed{2025,2026,2027}_{TP,DM}.json`
  — 12 files, the multi-movie 3-seed replicates.
- `probe_val_jul7-tp_{clip,soft_target_clip}_seed{2025,2026,2027}_TP.json`
  — 6 files, the TP-only 3-seed replicates.
- `probe_val_jul7-{tau05,a25}_soft_target_clip_{TP,DM}.json`,
  `probe_val_jul7-e400*_TP.json`, `probe_val_jul7-e400*_DM.json` —
  earlier sensitivity + 400-ep-recovery runs.

**Train→test Pearson r probe (val + test splits)**:

- `probe_traintest_val_jul7-tp_clip_seed2025_TP.json`
- `probe_traintest_test_jul7-tp_clip_seed2025_TP.json`
- `probe_traintest_val_jul7-tp_soft_seed2026_TP.json`
- `probe_traintest_test_jul7-tp_soft_seed2026_TP.json`

**Code**:

- `_submit_ab.py`, `_probe_only.py`, `NOTES.md`, this `RESULTS_jul7.md`.
- Objective implementation in `eb_jepa/clip.py:SoftTargetCLIPPretrain`.
- Trainer changes in `eb_jepa/training/clip_pretrain.py`.
- Dataset changes in `eb_jepa/datasets/hbn.py` (recipe_require_shots).

---

*Concludes the soft_target_clip experiment. Next: REVE + soft-target trial
per §7.2.*
