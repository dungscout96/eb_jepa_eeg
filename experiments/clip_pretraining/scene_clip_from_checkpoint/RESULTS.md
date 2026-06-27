# scene_clip warm-started from REVE — Current Results

EEG ↔ V-JEPA-2 alignment via the §9 `scene_clip` recipe (supervised-contrastive
InfoNCE), with the EEG encoder **warm-started from braindecode's pretrained
REVE-base** (`brain-bzh/reve-base`) instead of random initialization. Single-task
on HBN *ThePresent*. This document records the sweep ending 2026-06-26 along
with the methodology used to validate it.

Code paths:

- Recipe / scene_clip loss: [`eb_jepa/clip.py`](../../../eb_jepa/clip.py) (`SceneCLIPPretrain`)
- Trainer (`meta.encoder_init_from` for warm-start, `meta.freeze_encoder` for projector-only): [`eb_jepa/training/clip_pretrain.py`](../../../eb_jepa/training/clip_pretrain.py)
- Config: [`config/clip_pretrain_from_reve.yaml`](../../../config/clip_pretrain_from_reve.yaml)
- REVE→EET adapter (one-time HF download + key rename): [`prepare_reve_checkpoint.py`](prepare_reve_checkpoint.py)
- Probes: [`eb_jepa/evaluation/clip_probe/probe.py`](../../../eb_jepa/evaluation/clip_probe/probe.py) (CV-by-recording) and [`probe_traintest.py`](../../../eb_jepa/evaluation/clip_probe/probe_traintest.py) (ImageNet-style train→test + bootstrap CI)
- Experiment plan / hypothesis: [`NOTES.md`](NOTES.md)

All probe JSONs are checked in under [`probe_results/`](probe_results/).

---

## TL;DR

**Locked-in winner (post-2026-06-26 LR sweep): `warmstart_lr3e4_ep299`** —
REVE-base warm-start + per_window targets + `proj_dim=512` + **`lr=3e-4`** +
warmup=10 + AdamW + wd=0.05 + 300 epochs.

Bootstrap-CI'd train→eval (R1-R4 fit; B=2000 by eval-recording):

| run | val Δr [CI] | test Δr [CI] |
|---|---|---|
| random_REVE_shape | — | — |
| lr=1e-5 (initial recipe "winner") | +0.116 [+0.099, +0.133] | +0.131 [+0.099, +0.162] |
| lr=3e-5 | +0.155 [+0.137, +0.171] | +0.166 [+0.131, +0.197] |
| lr=1e-4 | +0.198 [+0.178, +0.217] | +0.198 [+0.158, +0.235] |
| **lr=3e-4** ⭐ | **+0.218 [+0.197, +0.238]** | **+0.197 [+0.155, +0.234]** |
| lr=5e-4 | +0.195 [+0.175, +0.216] | — |
| ⟵ fresh500_ep499 (from-scratch baseline) | — | +0.080 [+0.005, +0.144] |

**+146 % over the from-scratch winner on test.** Every per-feature CI on test
is strictly above the matched random baseline (12 / 12 features).

**Honest note on the lr=1e-4 → lr=3e-4 comparison**: lr=3e-4 wins on val by
+0.020 Δr but **ties with lr=1e-4 on test** (+0.197 vs +0.198). The val
advantage didn't transfer. Two reads: (a) we slightly val-overfit by
selecting from a 5-LR sweep on val; (b) test (n=108 rec, ~3× wider CIs than
val n=293) doesn't have the resolution to detect a +0.02 Δr improvement.
Either way, the recipe is **robust to LR ∈ [1e-4, 3e-4]** — pick 3e-4 if
you trust the val selection, 1e-4 if you prefer the original CLIP-from-
scratch winner. Both clearly beat lr=1e-5.

The big methodological correction is in §3.1 below. Briefly: the initial
hypothesis ("REVE-base is pretrained, so fine-tune at 1/10 from-scratch LR")
was the standard fine-tuning prior but **wrong for this setup**. The REVE
encoder needs at least the same `lr=1e-4` that won the from-scratch sweep.

§3.2 covers a second surprise: **longer training at lr=1e-4 hurts**. The
1000-epoch run lands at val Δr ≈ +0.11 across ep500–999 — *worse than the
lr=1e-4 ep299 sibling at +0.198*. Schedule length matters as much as peak
LR. Don't extend the schedule.

---

## §1. Methodology

### 1.1 Data

- HBN EEG, 129 channels, preprocessed to 200 Hz, *ThePresent* movie. Splits:
  R1-R4 train, R5 val, R6 test (subject-disjoint by HBN release wave).
- Vision teacher: V-JEPA-2 (1408-d, ~2 Hz, 406 clips of *ThePresent*),
  precomputed and frozen.

### 1.2 The warm-start

REVE-base (`brain-bzh/reve-base`) is a 22-layer transformer pretrained on broad
multi-site EEG by braindecode. Its architecture is bit-for-bit identical to
`EEGEncoderTokens` except for one key rename
(`to_patch_embedding.0.*` → `to_patch_embedding.*`) and a classification head
to drop. The offline adapter [`prepare_reve_checkpoint.py`](prepare_reve_checkpoint.py)
converts REVE's state_dict to a `.pth.tar` that the existing
`meta.encoder_init_from` flag loads directly — **no trainer code change needed**.

Encoder shape: `embed_dim=512, depth=22, heads=8, head_dim=64, patch_size=200,
patch_overlap=20, freqs=4, mlp_dim_ratio=2.66`. ~50 M params, ~3× the
fresh-from-scratch baseline (35 M).

### 1.3 Probe protocols

| protocol | fit on | eval on | metric | when to trust |
|---|---|---|---|---|
| **CV-on-val** | 4 folds of R5 | held-out R5 fold | R² (5-fold mean) | quick dev iteration |
| **probe_traintest** | full R1-R4 (~700 rec, ~71k windows) | full R6 (~108 rec, ~11k windows) | **Pearson r** + bootstrap CI by R6 recording (B=2000) | matches SSL literature. **Primary metric.** |

Per-feature Pearson r resamples the 108 test recordings with replacement
(cluster bootstrap — windows within a recording are not independent). Reports
median + 2.5 / 97.5 percentiles. The 95 % CI is the headline.

---

## §2. Variants tested

All probed against the matched random-REVE-shape baseline
(encoder-architecture-controlled). Sorted by Δr.

All test Δr's are paired against `probe_tt_random_reve_shape_boot` (mean r
=+0.040). Sorted by val Δr where available, falling back to test Δr.

| name | knobs | epochs | val Δr | test Δr | wandb |
|---|---|---|---|---|---|
| REVE_alone | encoder init only, no CLIP training | 0 | — | +0.042 | — |
| frozen_simple | freeze encoder, train clip_head (1 res block) | 100 | — | +0.042 | [qxe6g9gv](https://wandb.ai/sccn/eb_jepa/runs/qxe6g9gv) |
| frozen_complex | freeze encoder, train clip_head (3 res blocks) | 100 | — | +0.042 | [de5vxx5x](https://wandb.ai/sccn/eb_jepa/runs/de5vxx5x) |
| warmstart_v1 | unfrozen, **shot_mean** targets, lr=1e-5 | 100 | — | +0.086 | [xio8z1zm](https://wandb.ai/sccn/eb_jepa/runs/xio8z1zm) |
| v3a (proj=1408, vfreeze) | per_window, vision_passthrough, lr=1e-5 | 300 | — | +0.092 | [w2tw9k0r](https://wandb.ai/sccn/eb_jepa/runs/w2tw9k0r) |
| v3b (proj=1024, sym) | per_window, sym, lr=1e-5 | 300 | — | +0.098 | [kng2cyl5](https://wandb.ai/sccn/eb_jepa/runs/kng2cyl5) |
| warmstart_v2_300 | per_window, proj=512, lr=1e-5 | 300 | +0.116 | +0.131 | [8f4kly1d](https://wandb.ai/sccn/eb_jepa/runs/8f4kly1d) |
| warmstart_v2_500 | same as v2_300, longer | 500 | — | +0.130 | [zk60hpt5](https://wandb.ai/sccn/eb_jepa/runs/zk60hpt5) |
| lr=3e-5 | per_window, proj=512, lr=3e-5 | 300 | +0.155 | +0.166 | [jo34f2xl](https://wandb.ai/sccn/eb_jepa/runs/jo34f2xl) |
| lr=1e-4 @ 1000 ep (ep500–999) | per_window, proj=512, **stretched schedule** | 1000 | +0.110–0.120 | — | [f5tmobz1](https://wandb.ai/sccn/eb_jepa/runs/f5tmobz1) |
| **lr=1e-4 @ 300 ep** | per_window, proj=512, lr=1e-4 | 300 | +0.198 | +0.198 | [zd928rf0](https://wandb.ai/sccn/eb_jepa/runs/zd928rf0) |
| lr=5e-4 | per_window, proj=512, lr=5e-4 | 300 | +0.195 | — | [q3frbndw](https://wandb.ai/sccn/eb_jepa/runs/q3frbndw) |
| **lr=3e-4** ⭐ | per_window, proj=512, lr=3e-4 | 300 | **+0.218** | **+0.197** | [i4nu07v9](https://wandb.ai/sccn/eb_jepa/runs/i4nu07v9) |

---

## §3. Key findings

### 3.1 Learning rate was the most undertuned knob — lr=3e-4 is the locked-in winner

The original recipe used `lr=1e-5` based on the standard fine-tuning prior
("use 1/10 of from-scratch LR when starting from a pretrained checkpoint").
A late-sweep LR ablation showed that prior was wrong here:

| LR | val Δr [CI] | test Δr [CI] |
|---|---|---|
| 1e-5 (original "winner") | +0.116 [+0.099, +0.133] | +0.131 [+0.099, +0.162] |
| 3e-5 | +0.155 [+0.137, +0.171] | +0.166 [+0.131, +0.197] |
| 1e-4 | +0.198 [+0.178, +0.217] | +0.198 [+0.158, +0.235] |
| **3e-4** ⭐ | **+0.218 [+0.197, +0.238]** | **+0.197 [+0.155, +0.234]** |
| 5e-4 | +0.195 [+0.175, +0.216] | — |

**The val curve has a clear peak at lr=3e-4** (inverted-U: 1e-5 → 3e-5 →
1e-4 → 3e-4 → 5e-4 climbs then descends). On **test**, however, lr=3e-4
and lr=1e-4 land within 0.001 of each other (+0.197 vs +0.198) — the val
advantage of lr=3e-4 did not transfer at the resolution of the n=108
test set. Either (a) we slightly val-overfit by selecting from 5 LRs on
val, or (b) test CIs (~3× wider than val due to smaller n) miss a
genuine +0.02 difference. Treat **lr ∈ [1e-4, 3e-4]** as the recipe's
LR sweet spot.

Going from lr=1e-5 → lr=3e-4 buys +0.102 absolute val Δr (+88 % relative)
or +0.066 test Δr (+50 %). The "1/10 from-scratch LR" prior cost roughly
half of the available gain.

**The val column was added retrospectively to validate that the new
"winner" wasn't a test-set overfitting artifact.** The original sweep
probed only on test (every checkpoint, every hyperparameter), which is the
textbook definition of test-set overfitting via selection. After noticing
this, every LR variant was re-probed on val (R5) with the same B=2000
bootstrap protocol. Both val and test ranked the four LRs identically
(1e-5 < 3e-5 < 1e-4 ≤ 3e-4); the only place val and test diverged was the
size of the lr=3e-4 vs lr=1e-4 gap. Methodology going forward:
hyperparameter selection probes val only; test is touched once per
final-reported configuration.

**Per-feature gains from the original lr=1e-5 → lr=3e-4 winner** (val
column, where the largest improvements are):

| feature | lr=1e-5 r | lr=3e-4 r | gain |
|---|---|---|---|
| motion_energy | +0.213 | +0.406 | +91 % |
| narrative_event_score | +0.180 | +0.281 | +56 % |
| scene_natural_score | +0.213 | +0.335 | +57 % |
| n_faces | +0.207 | +0.296 | +43 % |
| depth_mean | +0.213 | +0.309 | +45 % |

The features REVE was *weakest* on (motion, faces, depth, narrative) gain
the most. Higher LR lets the encoder actually move toward V-JEPA-2's
geometry rather than staying near REVE-init.

### 3.2 The LR–schedule-length interaction: longer training HURTS at the same peak LR

A 1000-epoch run at lr=1e-4 (cosine to lr_min=1e-7, save_every=50) was run
to test the "more training helps" hypothesis. It does not — in fact it's
substantially **worse** than the 300-epoch sibling:

| run | val Δr (B=2000) |
|---|---|
| lr=1e-4 @ 300 ep, ep299 | +0.198 |
| lr=1e-4 @ 1000 ep, ep500 | +0.120 |
| lr=1e-4 @ 1000 ep, ep700 | +0.115 |
| lr=1e-4 @ 1000 ep, ep999 | +0.110 |

The 1000-ep run hovers at the lr=1e-5 baseline level (+0.116) across the
entire ep500–999 range. **Same peak LR, very different cosine schedule:**

- 300-ep cosine: lr decays from 1e-4 to lr_min=1e-7 by ep299 — encoder
  effectively frozen by the end, baked-in early-stopping.
- 1000-ep cosine: lr decays from 1e-4 over 1000 epochs — at ep500 the
  encoder is still training at lr≈5e-5, well into a "contrastively-
  overfit" regime.

Train-side diagnostics confirm overfit: train top1_e2v rises from 0.21
(ep300) → 0.41 (ep700) → 0.49 (ep990) while val AUC peaks at ep700
(0.86) then degrades to 0.78 by ep999. Train loss drops from 1.6 → 0.13
while val R² drops in parallel. **The contrastive objective continues
to overfit the train distribution but the encoder's transferable
structure degrades** — the AUC ≠ R² decoupling at its sharpest, now
visible as a train/val decoupling.

**Implication**: schedule length matters as much as peak LR. With
lr=1e-4 the 300-epoch cosine schedule is essentially providing implicit
early-stopping via fast LR decay; extending to 1000 ep removes that
regularization. The lr=3e-4 + 300 ep operating point pushes the same
total "LR×epochs" budget into a shorter window at higher peak — and
wins. Do not extend the schedule under this recipe.

**Per-feature: the high-level visuals explode.** REVE's pre-existing
strengths (luminance, contrast, position) improve modestly (~+20-30 % r);
the features REVE was *weak* on (motion, faces, depth, scene_natural,
narrative) gain **+45 to +71 % r**:

| feature | lr=1e-5 r | lr=1e-4 r | gain |
|---|---|---|---|
| motion_energy | +0.188 | **+0.286** | +52 % |
| scene_natural_score | +0.130 | **+0.222** | +71 % |
| narrative_event_score | +0.118 | **+0.201** | +70 % |
| edge_density | +0.131 | **+0.214** | +63 % |
| depth_mean | +0.139 | **+0.205** | +47 % |
| n_faces | +0.144 | **+0.209** | +45 % |
| face_area_frac | +0.129 | +0.176 | +37 % |
| entropy | +0.203 | +0.278 | +37 % |
| saturation_mean | +0.198 | +0.264 | +33 % |
| contrast_rms | +0.211 | +0.262 | +24 % |
| luminance_mean | +0.227 | +0.279 | +23 % |
| position_in_movie | +0.241 | +0.264 | +10 % |

The higher LR lets the encoder actually move toward V-JEPA-2's geometry
rather than staying near REVE-init. This corrects (or at least
re-interprets) several other findings in this doc; see §3.5 and §5.

### 3.2 Interaction with REVE's depth: same LR as from-scratch, despite 5× more layers

Standard fine-tuning wisdom says deeper networks need *lower* LR (more layers
compound gradient updates, more pretrained knowledge to potentially destroy).
REVE-base has **depth=22, embed=512** (~50 M params). The from-scratch
winning encoder (fresh500) was **depth=4, embed=1024** (~35 M params). One
might expect REVE to need 1/3-1/10 the LR.

Instead, **both win at lr=1e-4**. Same LR for both architectures despite
the 5× depth ratio and 1.4× param-count ratio. Three plausible reasons:

1. **REVE's pretraining basin is broad and stable.** It saw orders of
   magnitude more EEG than HBN provides, so its weights are well-conditioned
   — lr=1e-4 doesn't easily push them out of the useful regime.
2. **The contrastive loss has a depth-independent "effective LR" target.**
   With L2-normalized projector outputs and a learnable temperature, the
   per-sample loss magnitude is bounded by `O(log K)` in K = batch size,
   regardless of encoder depth. The "right" LR for this loss may depend
   more on bs / proj_dim / temperature than on encoder shape.
3. **EEG signal is the constraint, not encoder capacity.** Either encoder
   has plenty of capacity to absorb the V-JEPA-2 alignment signal at the
   ~700-recording training scale. The bottleneck is gradient magnitude
   reaching the encoder, not over-fitting deep layers.

Practical implication: when extending the recipe to other pretrained EEG
encoders, **don't reflexively shrink LR for deeper models**. Start at the
from-scratch winning LR; only shrink if you observe instability.

### 3.3 Warm-start beats from-scratch by a wide margin

At the now-corrected LR:

| | mean r | mean Δr | epochs | wall (Delta A40) |
|---|---|---|---|---|
| fresh500_ep499 (from scratch) | +0.133 | +0.080 | 499 | 3 h 22 min |
| **warmstart_lr1e4_ep299** | **+0.238** | **+0.198** | **299** | **1 h 09 min** |

Mean r improves by **+0.105 absolute (+79 % relative)**, mean Δr improves by
**+148 %**. In ~1/3 the wall time. REVE's broad EEG pretraining is the
right starting point even though it never saw movie-watching data.

### 3.4 Statistical significance (per-feature, bootstrap B=2000)

**12 / 12 features have CI lo > matched random point estimate** — every
feature's Pearson r is significantly above random at the 95 % level. No
feature inversions:

| feature | warmstart_v2_ep299 r [95 % CI] | rand_REVE r [95 % CI] |
|---|---|---|
| luminance_mean | +0.227 [+0.186, +0.265] | +0.048 [−0.002, +0.101] |
| contrast_rms | +0.211 [+0.174, +0.247] | +0.020 [−0.022, +0.067] |
| saturation_mean | +0.198 [+0.160, +0.235] | +0.028 [−0.017, +0.076] |
| entropy | +0.203 [+0.164, +0.242] | +0.057 [+0.021, +0.101] |
| **position_in_movie** | **+0.241 [+0.197, +0.283]** | −0.005 [−0.073, +0.063] |
| motion_energy | +0.188 [+0.158, +0.216] | +0.064 [+0.038, +0.089] |
| n_faces | +0.144 [+0.120, +0.166] | +0.061 [+0.036, +0.086] |
| depth_mean | +0.139 [+0.111, +0.166] | +0.038 [+0.013, +0.061] |
| face_area_frac | +0.129 [+0.108, +0.150] | +0.051 [+0.025, +0.076] |
| edge_density | +0.131 [+0.102, +0.159] | +0.036 [+0.003, +0.066] |
| scene_natural_score | +0.130 [+0.099, +0.161] | +0.035 [+0.004, +0.068] |
| narrative_event_score | +0.118 [+0.093, +0.140] | +0.050 [+0.031, +0.068] |

### 3.5 REVE's strength/weakness profile + how CLIP fills the gap

REVE-alone (no CLIP) per-feature Δr vs random:

- **Strong (matched or exceeded fresh500)**: luminance, contrast, saturation,
  entropy, position_in_movie. These are *low-level perceptual* features REVE
  picked up from broad EEG pretraining.
- **Weak or below random**: n_faces (−0.010), scene_natural (−0.011), depth
  (+0.004), motion (+0.004). REVE never saw movie-watching data, so its
  features have no representation of these *high-level visual* signals.

The CLIP warm-start fills exactly the high-level gap. Per-feature gain from
REVE-alone → warmstart_v2_300:

| feature | REVE-alone | warmstart_v2_300 | gain |
|---|---|---|---|
| motion_energy | +0.068 | +0.188 | **+0.120** |
| n_faces | +0.048 | +0.144 | +0.096 |
| depth_mean | +0.059 | +0.139 | +0.080 |
| face_area_frac | +0.052 | +0.129 | +0.077 |
| scene_natural_score | +0.048 | +0.130 | +0.082 |
| narrative_event_score | +0.049 | +0.118 | +0.069 |

Meanwhile, the low-level features REVE was already strong on stay strong
(REVE-alone luminance +0.151 → warmstart +0.227 = small improvement). The
warm-start preserves REVE's pre-existing strengths *and* adds the V-JEPA-2-
specific high-level signal CLIP can teach.

### 3.6 At lr=1e-5, training saturates at ~300 epochs — but this may be LR-specific

**Caveat**: this finding was established at the now-superseded lr=1e-5. At
the new lr=1e-4 the val AUC was still climbing sharply at ep299 (0.76 →
0.83 over the last 100 epochs), so the saturation point likely moves with
LR. The lr=1e-4 @ 1000-epoch run currently in flight (job 19713613) will
tell us the actual saturation epoch at the new LR.

Below is the lr=1e-5 saturation evidence for reference:

500-ep run (ep400 = AUC peak, ep499 = latest) was indistinguishable from the
300-ep run on probe R²:

| checkpoint | mean r | mean Δr | mean CI |
|---|---|---|---|
| warmstart_v2_300_ep299 | +0.1715 | +0.131 | [+0.139, +0.202] |
| warmstart_v2_500_ep400 | +0.1714 | +0.131 | [+0.138, +0.203] |
| warmstart_v2_500_ep499 | +0.1700 | +0.130 | [+0.137, +0.201] |

All three CIs overlap nearly completely. Per-feature differences are within
bootstrap noise (±0.005 r). **The recipe saturates at ~300 epochs.** Extended
training is wasted compute. (Train top1_e2v peaks at ep450 per W&B, but this
within-distribution train metric decouples from downstream R²;
intermediate-epoch checkpoints were not preserved.)

### 3.7 At lr=1e-5, bigger projector hurts — but this is now entangled with LR and needs re-test

**Caveat**: same as above. The proj_dim ablation was run at lr=1e-5, where
the encoder barely moves. The original hypothesis was "a small projector
bottlenecks gradient flow through the encoder, forcing the encoder to
fine-tune". Under that hypothesis, with the higher lr=1e-4 the encoder
moves regardless of projector size, and bigger projectors might no longer
hurt — they might even help by giving the contrastive loss a larger target
space. **Re-running the proj_dim sweep at lr=1e-4 is the obvious next
experiment.**

Below is the lr=1e-5 evidence for reference (winning combo at the time):

| run | proj_dim | vision_passthrough | mean Δr |
|---|---|---|---|
| **warmstart_v2_300** | **512** | **false (sym)** | **+0.131** |
| v3b | 1024 | false (sym) | +0.098 |
| v3a | 1408 | true (vfreeze) | +0.092 |
| ⟵ fresh500_ep499 (proj=1024) | 1024 | false (sym) | +0.093 |

Every feature drops when going from proj=512 → 1024 → 1408. No exceptions.

**Why** (hypothesis): with REVE's 22 layers being fine-tuned at lr=1e-5
(gentle), most of the alignment learning happens in the projector. A small
projector (matched to encoder embed_dim = 512) bottlenecks the gradient such
that the encoder must shift to satisfy the loss. A bigger projector is
expressive enough to do alignment entirely in the head, leaving the encoder
largely unchanged from REVE-init — and **the linear probe sees encoder
outputs, not projected outputs**, so encoder fine-tuning is exactly what
matters.

Consistent with v3b (proj=1024) landing at +0.098 — essentially identical to
fresh500_ep499 (also proj=1024, +0.093). **The REVE warm-start advantage only
manifests at proj=512.** With proj=1024 there's no measurable benefit from
the better init.

### 3.8 Frozen-encoder ablation confirms encoder fine-tuning matters

Probing **encoder** outputs of frozen runs gives bit-identical numbers to
REVE-alone (encoder weights unchanged):

| run | mean r |
|---|---|
| REVE_alone | +0.094 |
| frozen_simple | +0.094 |
| frozen_complex | +0.094 |

This is methodologically expected — a probe of `encoder.pool_to_windows()` is
blind to anything happening in `clip_head`. AUCs differ across these runs
(simple=0.68, complex=0.60 final shot-AUC), but that gain lives in the
projector and is invisible to a linear probe on encoder outputs.

**Practical implication**: the +0.131 (at lr=1e-5) or +0.198 (at lr=1e-4)
Δr lift from REVE-alone is *all* encoder fine-tuning. Freezing the encoder
defeats the purpose of the recipe.

---

## §4. Why this works (theoretical reading)

The result is consistent with the multi-view-redundancy / InfoMin framework
(Tian et al. 2020; Tosh-Krishnamurthy-Hsu 2020): contrastive learning
preserves the shared mutual information between modalities. REVE's
pretraining on broad EEG data gives the encoder a rich representation of
general EEG content; the contrastive step refines *which subspace* of that
content aligns with V-JEPA-2's visual structure. From scratch, the encoder
has to learn both "how to represent EEG" and "how to align it with V-JEPA-2"
simultaneously, on relatively limited training data (~700 R1-R4 recordings).
With REVE doing the first half, the contrastive loss can spend all its
budget on the second half.

The "small projector wins" finding *at lr=1e-5* sharpens this view: the
encoder is the storage location that matters for downstream transfer (the
probe reads encoder outputs). A small projector forces the contrastive loss
to flow gradient *through* the encoder rather than absorbing it in a large
projection head. But the LR ablation reframes this: with lr=1e-4 the
encoder moves regardless of projector size, so the small-projector
"gradient-bottleneck" benefit may disappear at the new LR. The right way
to think about both findings together is **"the encoder needs to actually
move to learn V-JEPA-2 alignment, and both LR and projector size affect
how much it moves"** — lr=1e-5 with a small projector is one operating
point that gets useful movement; lr=1e-4 with any projector likely gets
more movement, but we haven't checked the proj_dim sweep at the new LR yet.

---

## §5. Open questions / next experiments

1. **Higher LR — how far can we push?** lr=1e-4 beat lr=1e-5 by +51 % Δr.
   lr=3e-4 and lr=5e-4 at 300 epochs are queued (jobs 19713611, 19713612).
   When does going higher stop helping?
2. **Longer at lr=1e-4** — val AUC was still rising at ep299 (0.76 → 0.83
   over the last 100 epochs at lr=1e-4). A 1000-epoch run with
   `save_every=50` is queued (job 19713613); we'll probe ep400, ep500,
   ep1000 (or whichever epoch range plateaus) afterward.
3. **Re-run proj_dim sweep at lr=1e-4** — the "proj=512 wins" finding was
   established at lr=1e-5 where gradient flow through the encoder was the
   bottleneck. At lr=1e-4 the encoder moves freely; bigger projectors may
   no longer hurt and might help (more contrastive-target room).
4. **Same recipe on DespicableMe** alone (single-task) — does the warm-start
   advantage generalize to a second movie? Prereqs done: scene map at
   `experiments/clip_pretraining/embedding_feature_correlation/DespicableMe/vjepa2_scenes_map.csv`,
   recipe artifact at `movie_annotation/output/despicable_me/vjepa2_recipe.npz`.
5. **Multi-movie training** (`data.task=[ThePresent, DespicableMe]`) — does
   two movies improve generalization, or do they interfere? Run in progress
   (job 19712508) once the windowing-skip bug fix landed.
6. **Stronger memory bank / MoCo queue** — the multi-view-redundancy bound
   only tightens with more diverse negatives. Adding ~16k queued V-JEPA-2
   negatives is the next-most-principled upgrade.
7. **Cross-modal latent prediction (EEG-JEPA-against-V-JEPA-2 latents)** —
   replaces contrastive with direct regression of V-JEPA-2 latents from
   masked EEG. Avoids the log-K-bits ceiling of pure InfoNCE.

---

## §6. Reproducibility

```bash
# 1. One-time: download REVE-base from HF and convert key naming.
PYTHONPATH=. uv run --group eeg python \
    experiments/clip_pretraining/scene_clip_from_checkpoint/prepare_reve_checkpoint.py \
    --output reve_base_eet_init.pth.tar

# 2. Train the new winning recipe (lr=1e-4 at 300 epochs).
#    Delta job 19712256, wandb zd928rf0.
PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain \
    --fname=config/clip_pretrain_from_reve.yaml \
    --meta.encoder_init_from=/path/to/reve_base_eet_init.pth.tar \
    --loss.target_kind=per_window \
    --optim.lr=1e-4 \
    --optim.epochs=300 \
    --logging.wandb_group=clip_reve_warmstart_lr1e4_pw300
# config supplies: loss.proj_dim=512, vision_passthrough=false,
#                  optim.optimizer=adamw, weight_decay=0.05,
#                  warmup_epochs=10, lr_min=1e-7
# (The config still has optim.lr=1e-5 baked in for backward compat with the
#  earlier "winner"; the CLI override --optim.lr=1e-4 is what makes the new
#  recipe.)

# 3. Bootstrap-CI'd probe (the headline number).
PYTHONPATH=. uv run --group eeg python \
    eb_jepa/evaluation/clip_probe/probe_traintest.py --device cuda \
    --checkpoint /path/to/lr1e4_ep299/latest.pth.tar \
    --config config/clip_pretrain_from_reve.yaml \
    --bootstrap 2000 --seed 42 \
    --output probe_results/probe_tt_warmstart_lr1e4_ep299_boot.json

# Random baseline (substitute --random-baseline for --checkpoint).
```

Acceptance gate: mean Pearson r ≈ +0.238, mean Δr vs random ≈ +0.198,
all 12 per-feature CIs strictly above matched random. Any meaningfully
lower value indicates a recipe-config bug (e.g., wrong proj_dim, wrong
target_kind, wrong LR, encoder accidentally frozen).
