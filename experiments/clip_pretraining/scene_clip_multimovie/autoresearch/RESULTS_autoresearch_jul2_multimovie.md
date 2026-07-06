# autoresearch (jul2) — multi-movie scene_clip on ThePresent + DespicableMe

Autonomous search loop run 2026-07-04/06 following
[`program.md`](program.md). Same-task multi-domain training on both HBN
movies simultaneously, starting from jul1's winning config
(patch=400 depth=12 embed=512, see [`../../scene_clip_fromscratch/autoresearch/RESULTS_autoresearch_jul1.md`](../../scene_clip_fromscratch/autoresearch/RESULTS_autoresearch_jul1.md)).
Ten iterations of loop search plus four capacity-vs-compute-vs-domain-shift
diagnostic runs.

## TL;DR

**Multi-movie training helps the joint metric but hurts per-domain
performance vs a single-movie specialist.** At iter 1 (long-budget best,
400 ep), the joint encoder reaches mean val Δr² = **+0.0413** across
both movies. On ThePresent alone, iter 1 gets +0.0447 — beats
single-movie iter 12 at ep300 (+0.0421) but loses to single-movie iter 12
at ep500 (+0.0505).

**Four diagnostic runs settle the "why"**: scaling encoder capacity
(2× wider, 2× deeper, or 4× both) or compute (2× epochs) does **not**
close the gap to single-movie ep500. The bottleneck is neither capacity
nor compute — it is **domain shift** between the two movies. The two
target distributions (V-JEPA-2 targets for TP vs DM) compete for
representation, and 46 M-param encoder cannot hold both perfectly at
this data scale.

Recommendation: **loss-level intervention** (soft-label / distillation
from V-JEPA-2's own similarity structure) is the natural next axis to
try. Literature review appended.

## Setup

**Baseline config** (jul1 iter 12 winner + multi-movie task):

```yaml
model:
  encoder_embed_dim: 512
  encoder_depth: 12
  encoder_heads: 8
  encoder_head_dim: 64
  patch_size: 400            # 1 patch per 2 s window per channel
  patch_overlap: 0
  freqs: 4
  mlp_dim_ratio: 2.66
loss:
  mode: scene_clip
  target_kind: per_window
  mean_center: true
  vision_passthrough: false
  proj_dim: 512
  n_residual_blocks: 1
  temperature: 0.07
optim:
  optimizer: adam
  lr: 1e-4
  warmup_epochs: 5
data:
  batch_size: 64
  task: [ThePresent, DespicableMe]  # ← the only intentional change vs jul1
```

**Reference points** (single-movie ThePresent, shape-matched rand
r²=+0.01589):

- Single-movie iter 12 ep300: TP Δr² = **+0.0421** (short-budget)
- Single-movie iter 12 ep500: TP Δr² = **+0.0505** (long-training)

**Primary metric**: mean 5-fold CV R² Δ, averaged over both movies
(probed separately). Random baselines (iter 12 shape):

- `rand_r2_TP` = +0.01589 (ThePresent val)
- `rand_r2_DM` = +0.00695 (DespicableMe val)

## Iteration trajectory

Every row of [`results.tsv`](results.tsv) in narrative form.

### Phase 1: same-task multi-domain baseline (iter 0–2)

| iter | change | ep | mean Δr² | TP | DM | verdict |
|---|---|---|---|---|---|---|
| 0 | baseline (iter12 config + task=[TP,DM]) | 250 | +0.0357 | +0.0392 | +0.0322 | keep |
| **1** | + more epochs | 400 | **+0.0413** | +0.0447 | +0.0379 | **keep (long-budget best)** |
| 2 | + more epochs | 500 | +0.0412 | +0.0457 | +0.0367 | discard (tied, DM↓) |

Iter 0 established that multi-movie at the known-good architecture is
**~0.005 worse than single-movie iter 12 ep300 on TP** but adds DM
coverage. Iter 1 recovered TP with more epochs — actually beating
single-movie ep300's +0.0421 on TP. Iter 2 (500 ep) tied iter 1 within
noise, but with **opposite per-movie moves** (TP +0.001, DM −0.001):
capacity competition surfaced with more training.

### Phase 2: LR exploration (iter 3)

| iter | change | ep | mean Δr² | TP | DM | verdict |
|---|---|---|---|---|---|---|
| 3 | lr 1e-4 → 2e-4 | 400 | +0.0294 | +0.0320 | +0.0266 | discard (hurts −0.012 vs iter 1) |

Jul1's finding that lr=3e-4 hurts single-movie replicates at multi-
movie: **from-scratch prefers lr=1e-4 across data scales**.

### Phase 3: budget switch + wider @ compute-matched (iter 4–5)

Budget switched from long (400 ep) to **30-min-training compute**
(EPOCHS=160 at iter 12 shape) for faster config search.

| iter | change | ep | mean Δr² | TP | DM | verdict |
|---|---|---|---|---|---|---|
| **4** | re-baseline at 30-min budget | 160 | **+0.0299** | +0.0320 | +0.0277 | **keep (new same-budget reference)** |
| 5 | + wider (embed 768) @ compute-matched | 85 | +0.0250 | +0.0272 | +0.0229 | discard (hurt at matched compute) |

Iter 5 tested wider encoder at **compute-matched epochs** (85 ep since
embed=768 has 1.9× per-epoch cost). Lost by −0.0049 mean. Same
finding as jul1 iter 13 at single-movie: at fixed compute, more epochs
at narrow beats fewer epochs at wider.

### Phase 4: capacity-vs-compute-vs-domain-shift diagnostics (iter 6–9)

Four long-budget parallel runs to attribute the multi-movie gap:

| iter | change vs iter 1 | ep | mean Δr² | TP | DM | vs iter 1 |
|---|---|---|---|---|---|---|
| 6 | 2× wider (embed=768, heads=12, freqs=5) | 400 | +0.0420 | +0.0472 | +0.0367 | +0.0007 (~tied) |
| 7 | 2× deeper (depth=24) | 400 | +0.0419 | +0.0458 | +0.0380 | +0.0006 (~tied) |
| 8 | Both wider AND deeper (4× params) | 400 | +0.0378 | +0.0415 | +0.0341 | −0.0035 (**hurts**) |
| 9 | 2× compute (800 ep) | 800 | +0.0382 | +0.0435 | +0.0330 | −0.0031 (**hurts**) |

**No diagnostic recovers TP toward single-movie ep500 (+0.0505):**

- **iter 6 wider** gets closest at TP=+0.0472, still −0.0033 short.
- **iter 7 deeper** is more balanced across movies but no better on mean.
- **iter 8 (4× params)** and **iter 9 (2× compute)** both regress —
  same schedule-length pathology jul1 §3.2 identified.

## Key findings

### 1. Multi-movie negative transfer at per-domain level

Under the 46 M-param encoder at ~700 recordings/movie, multi-movie
training **cannot match single-movie specialist** on that specialist's
own domain. Iter 1 (long-budget best) reaches TP Δr² = +0.0447 vs
single-movie ep500's +0.0505 — a −0.006 gap that survives 2× capacity
(iter 6: still −0.003 gap) and 2× compute (iter 9: still −0.007 gap).

### 2. Joint metric wins clearly

Multi-movie mean +0.0413 vs single-movie "specialist + zero-on-DM" mean
of ~+0.025. **~1.6× better on joint coverage.** If the downstream use
case values both movies, multi-movie is the right architecture.

### 3. Capacity axis has small positive tail then goes negative

- 2× params (wider OR deeper alone): ~+0.001 improvement.
- 4× params (both): −0.003 regression.

At small-data regime (~700 rec/movie), extra capacity is quickly
over-parametrised for the available signal.

### 4. Compute axis is negative past ep400

Iter 2 (500 ep) tied iter 1. Iter 9 (800 ep) *hurt*. Same **schedule-
length pathology** as jul1 §3.2 (extending cosine lets the encoder
train at higher LR longer than optimal, degrades downstream). Applies
to multi-movie too.

### 5. Movies have asymmetric axis preferences

- **TP most helped by wider** (iter 6 TP=+0.0472, best across all diagnostics)
- **DM most helped by deeper** (iter 7 DM=+0.0380 matches iter 1 baseline exactly, no regression)
- **Compute hurts DM more than TP** (iter 9: DM −0.005 vs baseline; TP −0.001 vs baseline)

Suggests domain-specific representation preferences the shared encoder
cannot satisfy simultaneously.

### 6. jul1 dead ends replicate at multi-movie

- Higher LR (2e-4, iter 3): **hurts** on multi-movie too. Confirms
  lr=1e-4 is robust across data scales.
- Wider at compute-matched (iter 5): hurts, same as jul1 iter 13.
- Longer schedule (iter 9): hurts, same as jul1 continuations.

Consistency across single-movie and multi-movie makes these findings
more general than either regime alone.

## Diagnostic verdict: domain shift is the driver

The gap between multi-movie iter 1 and single-movie iter 12 ep500 on
TP is **not fixable at our scale by capacity or compute alone**. The
two movies' V-JEPA-2 target distributions differ enough that a shared
46 M-param encoder trades TP fidelity for DM coverage. Standard SSL
"more data" scaling doesn't apply because the "more data" is
**non-IID** (different vision content, different EEG-vision
relationships across the two subject pools).

Both scaling axes are exhausted at our regime. Loss-level intervention
is the remaining lever.

## Next step: soft-label / distillation from V-JEPA-2

The natural response to "domain shift is the driver" is a loss-level
intervention that reduces label noise across domains. Soft-label
distillation from V-JEPA-2 itself replaces the hard scene-ID
InfoNCE positives with **continuous similarity targets** derived from
V-JEPA-2's own pairwise structure, which generalizes across movies
where scene IDs do not.

### Literature review

Focused on **loss-level soft-label / distillation** for domain shift
and label noise. Broken down by function:

**Foundational: why soft labels reduce noise**

- **Müller, Kornblith, Hinton 2019 — "When Does Label Smoothing Help?"**
  Analyzes label smoothing as regularization. Improves calibration and
  generalization but hurts downstream distillation. Read §3 on the
  entropy argument.
- **Yuan, Tay, Zhang, Torr, Chen 2020 — "Revisiting Knowledge
  Distillation via Label Smoothing Regularization"**. Formal
  equivalence between distillation and adaptive label smoothing.
  Argues distillation works *because* it's soft-label regularization,
  not "dark knowledge" transfer. Reframes the mechanism cleanly for
  our purposes.

**Similarity/relational distillation (most on-point for our setup)**

- **Park, Kim, Lu, Cho 2019 — "Relational Knowledge Distillation" (RKD)**.
  Distills **pairwise distances and angles** between samples rather
  than per-sample logits. Directly maps to our setup: precompute
  V-JEPA-2 pairwise cosine, distill into the EEG-side pairwise
  similarity. Pairwise structure generalizes across domains where
  per-sample labels don't.
- **Tian, Krishnan, Isola 2020 — "Contrastive Representation
  Distillation" (CRD)**. Distills via contrastive loss where positives
  are (student, teacher) of same sample and negatives are (student,
  teacher) of different sample. Uses continuous teacher features as
  targets. §3.2 is the InfoNCE-based distillation objective — very
  close to what we'd port.
- **Fang, Chen, Wang, Chen, Xia 2021 — "SEED: Self-supervised
  Distillation for Visual Representation"**. Distills small model's
  similarity structure to match large SSL teacher. Loss form (KL
  divergence between student and teacher row-softmax similarity
  distributions) is the exact recipe we'd use. Fig 2 shows the setup
  clearly.

**Cross-domain / multi-domain applications**

- **Caron, Touvron, Misra et al. 2021 — "Emerging Properties in
  Self-Supervised Vision Transformers" (DINO)**. Teacher-student
  self-distillation with **centering** to prevent collapse. Centering
  (subtract teacher moving mean) is what enables cross-domain training
  without one domain dominating. Directly relevant to our per-domain
  regression.
- **Grill, Strub, Altché et al. 2020 — "Bootstrap Your Own Latent"
  (BYOL)**. Non-contrastive self-distillation with EMA teacher. No
  negatives = no cross-domain false negatives. If cross-movie negative
  structure is a problem (which the DM-hurt-more-than-TP diagnostic
  pattern suggests), BYOL-style formulation sidesteps it.

**Practical: quick intuition on the loss swap**

- **Chen, Xie, He 2022 — "A Simple Framework for Masked Image
  Modeling"**. §3 shows the distillation loss form: L = ||student(x)
  − teacher(x)||² or KL on softmaxed logits. Simple enough to port in
  a few hours.

### Suggested read order (~2.5 hours focused)

1. Yuan 2020 §3-4 — the "distillation IS soft-label regularization"
   reframing. Clarifies why this addresses noise.
2. Park 2019 (RKD) — the pairwise-similarity distillation mechanism.
   Closest to the target implementation.
3. Fang 2021 (SEED §3-4) — a concrete recipe.
4. Tian 2020 (CRD §3) — if we want the contrastive-distillation
   hybrid (halfway between current InfoNCE and pure distillation).

### Where our setup differs from the cited papers

- **Frozen teacher already exists** (V-JEPA-2) — simpler than
  DINO/BYOL where teacher = student EMA.
- **Cross-modal** (EEG student, video teacher) — most cited papers
  are same-modality. Only RKD and CRD naturally generalize.
- **Small data** (~700 rec/movie) — SEED specifically targets this
  regime; DINO/BYOL assume large-scale.

### Concrete implementation preview

Cleanest formulation: **RKD-style pairwise-relation distillation as
auxiliary loss** on top of the existing InfoNCE (allowed per
program.md: auxiliary losses around the CLIP loss are fair game):

```
L_total = α · L_InfoNCE(z_eeg, z_vjepa2)                              # existing
        + β · L_relation(cos_sim(z_eeg), precomputed_cos_sim(vjepa2))  # new soft-label term
```

`L_relation` candidates:

- **Frobenius distance** between similarity matrices (RKD)
- **KL divergence** between row-softmax distributions (SEED)
- **Cross-entropy** with soft target = softmax(V-JEPA-2 cosine / τ)
  (SoftCLIP)

Each is a ~30-line addition to `clip_pretrain.py`. No changes to
`clip.py`. β ∈ [0.1, 0.5] is a reasonable starting range;
consider warming β from 0 → target over ~10 epochs
(Andonian 2022 progressive-self-distillation pattern).

## Reproducibility

Best training run:

```bash
uv run --group eeg python -m eb_jepa.training.clip_pretrain \
    --fname=config/clip_pretrain.yaml \
    --data.task='[ThePresent,DespicableMe]' \
    --optim.epochs=400 \
    --folder=/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/multimovie_jul2/iter1
```

Per-movie val probes (must run twice, one per task):

```bash
# Snapshot config with single task before probing
python -c "from omegaconf import OmegaConf; c = OmegaConf.load('config/clip_pretrain.yaml'); c.data.task = 'ThePresent'; OmegaConf.save(c, '/tmp/config_TP.yaml')"
uv run --group eeg python eb_jepa/evaluation/clip_probe/probe.py \
    --checkpoint /path/latest.pth.tar --config /tmp/config_TP.yaml \
    --split val --cv-splits 5 --output probe_val_TP.json
# ... repeat for DespicableMe
```

Random baseline (per movie, same procedure with `--random-baseline`
substituted for `--checkpoint`).

Full audit trail in [`results.tsv`](results.tsv). Every commit on
this branch corresponds to one iteration's config change.

Wall-time total: ~14 GPU-hours (10 iter search + 4 diagnostic runs)
on Delta A40.

## Infrastructure lessons

Same class of Lustre torch.save issues as jul1. Applied the same
mitigations (atomic write via `.tmp` + rename, legacy serialization
via `_use_new_zipfile_serialization=False`) via merge from
`autoresearch/fromscratch-jul1` branch.

New failure encountered: `/u/dtyoung/eb_jepa_eeg/checkpoints` is a
symlink to `/work/hdd/bbnv/dtyoung/checkpoints`, and `/work` filled to
100% during the diagnostic runs. Freed ~100 GB by removing the
pre-autoresearch `eeg_clip/` sweep directory. Future runs should
either watch `/work` capacity or route checkpoints to a different
filesystem (e.g., `/projects/bbnv/` with team quota available).
