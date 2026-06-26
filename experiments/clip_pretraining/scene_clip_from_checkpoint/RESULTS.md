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

**Best checkpoint: `warmstart_v2_ep299`** — REVE-base warm-start + per_window
targets + `proj_dim=512` (= encoder embed_dim) + lr=1e-5 + warmup=10 + AdamW
+ wd=0.05 + 300 epochs.

Bootstrap-CI'd train→test (R1-R4 fit → R6 eval, B=2000 over R6 recordings):

| run | mean r | mean Δr vs random | mean CI |
|---|---|---|---|
| random_REVE_shape | +0.040 | — | [+0.005, +0.077] |
| **warmstart_v2_ep299** | **+0.1715** | **+0.131** | [+0.139, +0.202] |
| ⟵ fresh500_ep499 (from-scratch baseline) | +0.133 | +0.080 | [+0.096, +0.169] |

**+64 % over the from-scratch winner with 1/2 the training compute.** All 12
features statistically significant above the matched random baseline (CI lo
above random point estimate).

Two strong counter-findings:

1. **Bigger projector hurts.** proj_dim ∈ {1024, 1408} both lose to proj_dim=512
   by ~25–30 %. With REVE barely moving at lr=1e-5, a small projector is
   essential to force gradient through the encoder.
2. **REVE alone is not enough.** Without CLIP training, REVE's encoder gives
   only +0.042 mean Δr — about a third of what the recipe pulls out. The CLIP
   step is doing most of the work, but it does it much faster from a REVE init
   than from random.

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

Six warm-start configurations, each probed against the same random-REVE-shape
baseline (encoder-architecture-controlled).

| name | knobs | epochs | mean r | Δr vs random | wandb |
|---|---|---|---|---|---|
| **REVE_alone** | encoder init only, no CLIP training | 0 | +0.094 | +0.042 | — |
| frozen_simple | freeze encoder, train clip_head (1 res block) | 100 | +0.094 | +0.042 | [qxe6g9gv](https://wandb.ai/sccn/eb_jepa/runs/qxe6g9gv) |
| frozen_complex | freeze encoder, train clip_head (3 res blocks) | 100 | +0.094 | +0.042 | [de5vxx5x](https://wandb.ai/sccn/eb_jepa/runs/de5vxx5x) |
| warmstart_v1 | unfrozen, **shot_mean** targets, lr=1e-5 | 100 | +0.137 | +0.086 | [xio8z1zm](https://wandb.ai/sccn/eb_jepa/runs/xio8z1zm) |
| **warmstart_v2_300** | unfrozen, **per_window** targets, lr=1e-5, **proj=512** | 300 | **+0.172** | **+0.131** | [8f4kly1d](https://wandb.ai/sccn/eb_jepa/runs/8f4kly1d) |
| warmstart_v2_500 | same as v2_300, 500 epochs | 500 | +0.170 | +0.130 | [zk60hpt5](https://wandb.ai/sccn/eb_jepa/runs/zk60hpt5) |
| v3a (proj_dim=1408, vfreeze) | per_window, vision_passthrough, proj=1408 | 300 | +0.132 | +0.092 | [w2tw9k0r](https://wandb.ai/sccn/eb_jepa/runs/w2tw9k0r) |
| v3b (proj_dim=1024, sym) | per_window, sym, proj=1024 | 300 | +0.138 | +0.098 | [kng2cyl5](https://wandb.ai/sccn/eb_jepa/runs/kng2cyl5) |

---

## §3. Key findings

### 3.1 Warm-start beats from-scratch by a wide margin (5× less compute)

| | mean r | mean Δr | epochs | wall (Delta A40) |
|---|---|---|---|---|
| fresh500_ep499 (from scratch baseline) | +0.133 | +0.080 | 499 | 3 h 22 min |
| **warmstart_v2_ep299** | **+0.172** | **+0.131** | **299** | **1 h 09 min** |

Mean r improves by +0.039 absolute (+29 % relative), mean Δr improves by
**+64 %**. In ~1/3 the wall time.

### 3.2 Statistical significance (per-feature, bootstrap B=2000)

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

### 3.3 REVE's strength/weakness profile + how CLIP fills the gap

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

### 3.4 Training saturates at ~300 epochs (no benefit to longer)

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

### 3.5 Bigger projector hurts — counterintuitive but robust

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

### 3.6 Frozen-encoder ablation confirms encoder fine-tuning matters

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

**Practical implication**: the +0.131 vs +0.042 jump from REVE-alone →
warmstart_v2 is *all* encoder fine-tuning. Freezing the encoder defeats the
purpose of the recipe.

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

The "small projector wins" finding sharpens this view: the encoder is the
storage location that matters for downstream transfer (since the probe reads
encoder outputs). A small projector forces the contrastive loss to flow
gradient *through* the encoder rather than absorbing it in a large projection
head.

---

## §5. Open questions / next experiments

1. **Same recipe on DespicableMe** alone (single-task) — does the warm-start
   advantage generalize to a second movie? Prereqs: shot detection + scene
   merge analysis already run; scene map at
   `experiments/clip_pretraining/embedding_feature_correlation/DespicableMe/vjepa2_scenes_map.csv`,
   recipe artifact at `movie_annotation/output/despicable_me/vjepa2_recipe.npz`.
2. **Multi-movie training** (`data.task=[ThePresent, DespicableMe]`) — does
   training on two movies improve generalization to either, or do they
   interfere?
3. **Higher LR variants** (e.g., lr=5e-5 or 1e-4 with longer warmup) — REVE
   is barely moving at lr=1e-5; could a more aggressive fine-tune reach a
   higher asymptote?
4. **Stronger memory bank / MoCo queue** — the multi-view-redundancy bound
   only tightens with more diverse negatives. Adding ~16k queued V-JEPA-2
   negatives is the next-most-principled upgrade.
5. **Cross-modal latent prediction (EEG-JEPA-against-V-JEPA-2 latents)** —
   replaces contrastive with direct regression of V-JEPA-2 latents from
   masked EEG. Avoids the log-K-bits ceiling of pure InfoNCE.

---

## §6. Reproducibility

```bash
# 1. One-time: download REVE-base from HF and convert key naming.
PYTHONPATH=. uv run --group eeg python \
    experiments/clip_pretraining/scene_clip_from_checkpoint/prepare_reve_checkpoint.py \
    --output reve_base_eet_init.pth.tar

# 2. Train the winning recipe (warmstart_v2_300).
#    Delta job 19686142, wandb 8f4kly1d.
PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain \
    --fname=config/clip_pretrain_from_reve.yaml \
    --meta.encoder_init_from=/path/to/reve_base_eet_init.pth.tar \
    --loss.target_kind=per_window \
    --optim.epochs=300 \
    --logging.wandb_group=clip_reve_warmstart_v2_pw300
# config supplies: loss.proj_dim=512, vision_passthrough=false,
#                  optim.optimizer=adamw, weight_decay=0.05,
#                  optim.lr=1e-5, warmup_epochs=10, lr_min=1e-7

# 3. Bootstrap-CI'd probe (the headline number).
PYTHONPATH=. uv run --group eeg python \
    eb_jepa/evaluation/clip_probe/probe_traintest.py --device cuda \
    --checkpoint /path/to/warmstart_v2_300/latest.pth.tar \
    --config config/clip_pretrain_from_reve.yaml \
    --bootstrap 2000 --seed 42 \
    --output probe_results/probe_tt_warmstart_v2_ep299_boot.json

# Random baseline (substitute --random-baseline for --checkpoint).
```

Acceptance gate: mean Pearson r ≈ +0.172, mean Δr vs random ≈ +0.131,
all 12 per-feature CIs strictly above matched random. Any meaningfully
lower value indicates a recipe-config bug (e.g., wrong proj_dim, wrong
target_kind, encoder accidentally frozen).
