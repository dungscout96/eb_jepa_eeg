# autoresearch (jul1) — from-scratch scene_clip on ThePresent

Autonomous search loop run 2026-07-01/02 following
[`program.md`](program.md). Random-init encoder trained under the
`scene_clip` recipe on HBN R1-R4 → R5 val. 18 short-budget iterations
+ one long-training scale-up + two continuation attempts + full
downstream probe evals.

## TL;DR

**Winning config (iter 12)**: `patch_size=400`, `encoder_depth=12`,
`encoder_embed_dim=512`, `encoder_heads=8 head_dim=64`,
`mlp_dim_ratio=2.66`, `proj_dim=512`, `lr=1e-4`, `warmup=5`,
`Adam`, `batch_size=64`, `per_window`, `mean_center`, 500 epochs.

| metric | value | vs prior best |
|---|---|---|
| val 5-fold CV R² Δ (search metric) | **+0.04744** at ep500 | +0.0202 over fresh500_ep499 (+0.0272), **74 % improvement** |
| val Pearson r (probe_traintest, B=2000) | **+0.256** at ep500 | matches / slightly beats REVE warmstart on absolute r |
| val Pearson Δr vs iter 12 random init (r=+0.130) | **+0.126** | 58 % of REVE-warmstart lr=3e-4's +0.218 |
| wall time (train + probe) | ~50 min | vs REVE-warmstart's 1 h 09 min at ep300 (which needed pretrained encoder) |

**Key insight**: **patch coarsening was the dominant lever** (+0.024 val CV
Δ from patch=50 → 400 alone), followed by depth (+0.005 from depth
8 → 12). Every other axis at iter 12's shape saturated within noise.
Extended training past ep500 hurt (schedule-length pathology, §3.2
of the from_checkpoint sibling doc, applies here too).

## Winning config full spec

Best checkpoint: `/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/jul1/iter12_long_ep500_v4/latest.pth.tar`

```yaml
model:
  encoder_embed_dim: 512
  encoder_depth: 12
  encoder_heads: 8
  encoder_head_dim: 64
  patch_size: 400        # each 2s EEG window collapses to 1 patch per channel
  patch_overlap: 0
  freqs: 4               # FourierEmb4D min: freqs^4 >= embed_dim/2 = 256
  mlp_dim_ratio: 2.66

loss:
  mode: scene_clip
  mean_center: true
  target_kind: per_window
  temporal_buffer_s: 2.0
  vision_passthrough: false
  proj_dim: 512
  temperature: 0.07
  drop_proj: 0.5
  n_residual_blocks: 1

optim:
  optimizer: adam
  lr: 1.0e-4
  lr_min: 1.0e-6
  warmup_epochs: 5
  epochs: 500

data:
  batch_size: 64
  window_size_seconds: 2
  norm_mode: per_recording
```

## Iteration trajectory

Every row of [`results.tsv`](results.tsv) in narrative form.

### Phase 1: getting on the fresh500 trajectory (iter 0-3)

| iter | change | val ΔR² | verdict |
|---|---|---|---|
| 0 | baseline: config default (`shot_mean` → `per_window`, lr=3e-4, ep120) | +0.0032 | keep, but well below the program.md +0.015-0.020 expectation |
| 1 | **lr 3e-4 → 1e-4** (fresh500's winning LR at depth=4) | **+0.00965** | keep; matches fresh500 ep99 = +0.0099 exactly |
| 2 | depth 4 → 8 (keeping embed=1024) | crash | CUDA OOM at ep0 on A40, needs reshape |
| 3 | **shallow REVE-shape** (depth=8 embed=512 heads=8×64 proj=512) | **+0.01093** | keep, +0.001 over iter1 |

Iter 1 established that lr=1e-4 puts us on the fresh500 trajectory. Iter 3
tested "does deeper-and-thinner beat wider-and-shallower at fresh500 LR?"
— margin was small (+0.001) but positive.

### Phase 2: knob search at iter 3 shape (iter 4-8) — all misses except one

| iter | change | val ΔR² | verdict |
|---|---|---|---|
| 4 | lr 1e-4 → 3e-4 at REVE-shape | +0.00544 | **discard** (hurts) |
| 5 | channel_dropout p=0.1 | +0.00769 | **discard** |
| 6 | aux VC loss (std=cov=0.1) on encoder tokens via hook | +0.00927 | **discard** |
| 7 | warmup_epochs 5 → 10 | +0.00827 | **discard** |
| 8 | continuation from iter3 ep99 at lr=5e-5, +70 ep | +0.01353 | keep — but this is compute scaling, not a new config |

Four consecutive discards established the pattern: **at the ~70-100 epoch
budget, any regularization or LR increase hurts** because the model is
undertrained, not overfit. Iter 8 (continuation) confirmed iter 3 was
compute-bound: it kept climbing given more training. Off-mission for
config search under user's framing, but a useful sanity signal.

### Phase 3: patch coarsening breakthrough (iter 9-12)

User steering here shifted the search toward architecture axes that
rank-preserve across training lengths (not regularization, which
confounds with schedule length).

| iter | change | val ΔR² | gain | wall |
|---|---|---|---|---|
| 9 | patch_size 50 → 100 (tokens/window 12→6) | **+0.01443** | +0.0035 over iter3 | 29 min |
| 10 | patch_size 100 → 200 (2 tokens/window) | **+0.02353** | +0.0091 over iter9 | 18 min |
| 11 | patch_size 200 → 400 (1 token/window = CLIP alignment granularity) | **+0.03445** | +0.0109 over iter10 | 24 min |
| 12 | depth 8 → 12 at patch=400 | **+0.03902** | +0.0046 over iter11 | 25 min |

The `patch=400` win was the single largest jump of the whole search
(+0.011 in one iteration). Mechanism: `patch_size=400` matches the
2-second window exactly, so the encoder builds one token per channel
per window — the natural granularity for CLIP alignment against
per-window V-JEPA-2 targets. The transformer then operates on only 129
tokens (129 channels × 1 patch), which is very cheap and lets us fit
300 epochs inside the wall.

`motion_energy` r nearly doubled from iter 9 → 12 (0.032 → 0.080),
consistent with the causal story: coarser patches capture more
temporal context per token, which matches the receptive field motion
features need.

Depth 8 → 12 helped further at this shape (+0.005) — the deeper stack
processes the small 129-token sequence with more capacity per token.

### Phase 4: saturation confirmation at iter 12 shape (iter 13-18) — 6 dead ends

| iter | change vs iter 12 | val ΔR² | Δ vs iter 12 | verdict |
|---|---|---|---|---|
| 13 | embed 512 → 768 (heads 12, freqs 5) | +0.03877 | −0.00025 | discard, width saturated |
| 14 | depth 12 → 22 (REVE-full) | +0.03911 | +0.00009 | discard, depth saturated at 12 |
| 15 | mlp_dim_ratio 2.66 → 4 | +0.03845 | −0.00057 | discard, FFN saturated |
| 16 | n_residual_blocks 1 → 3 | +0.03634 | −0.00268 | discard, projection saturated |
| 17 | AdamW + wd=0.05 | +0.03901 | −0.00001 | discard, negligible at 300 ep |
| 18 | batch_size 64 → 128 | +0.03465 | −0.00437 | discard, confirms smaller batch wins |

**Six consecutive plateau-confirming results.** Iter 12 is a robust
local optimum on capacity/optimization axes.

### Phase 5: long-training scale-up (ep300 → ep500 → cont)

| run | epochs | val ΔR² | Δ | verdict |
|---|---|---|---|---|
| iter12 (short-budget best) | 300 | +0.03902 | — | baseline |
| iter12_long_ep500 (single cosine) | 500 | **+0.04744** | +0.0084 gain over ep300 | keep — new absolute best |
| cont_lr5e5 (continuation from ep500, lr=5e-5) | 500 + 500 | +0.04420 | −0.0032 | discard |
| cont_lr1e4 (SGDR-style, lr=1e-4 restart) | 500 + 500 | +0.04632 | −0.0011 | discard |

Extending single cosine 300 → 500 helped by +0.008 val CV R². But
BOTH continuation strategies (halving LR OR keeping same peak LR)
hurt vs ep500 base. This is the **schedule-length pathology**
program.md §3.2 warned about (originally observed for warmstart at
lr=1e-4, but the pattern generalizes): contrastive loss keeps
decreasing (from 2.58 at ep99 to 1.30 at ep499) but downstream
Pearson r has saturated — the AUC ≠ R² decoupling in effect.

**Ceiling for iter 12 config**: val CV R² Δ = +0.047, val Pearson
Δr = +0.126. Cannot be pushed further with more training under this
recipe.

## Downstream (probe_traintest) results

Ran `probe_traintest.py --eval-split val --bootstrap 2000` — the
SSL-literature protocol (fit RidgeCV on all R1-R4 embeddings, evaluate
on R5 val, per-recording bootstrap CI).

### iter 12 ep500 (trained) vs matched random-init baseline

| feature | random r | trained r | Δr |
|---|---|---|---|
| **motion_energy** | +0.096 | **+0.318** | **+0.222** |
| luminance_mean | +0.186 | +0.295 | +0.110 |
| contrast_rms | +0.157 | +0.278 | +0.121 |
| entropy | +0.192 | +0.305 | +0.113 |
| saturation_mean | +0.185 | +0.296 | +0.112 |
| edge_density | +0.100 | +0.219 | +0.118 |
| **depth_mean** | +0.078 | **+0.231** | **+0.153** |
| n_faces | +0.104 | +0.226 | +0.122 |
| face_area_frac | +0.110 | +0.207 | +0.097 |
| scene_natural_score | +0.134 | +0.248 | +0.114 |
| position_in_movie | +0.143 | +0.251 | +0.108 |
| **narrative_event_score** | +0.075 | **+0.198** | **+0.123** |
| **MEAN** | **+0.130** | **+0.256** | **+0.126** |

Random baseline r=+0.130 was surprising — much higher than warmstart's
random-REVE-shape (r≈+0.040). Explanation: iter 12's `patch_size=400`
means the encoder starts with `Linear(400 → 512)` as a random
projection of the full 2-s window per channel, which by itself
preserves substantial scalar-feature signal even at initialization.
This means iter 12 gets a "free" high random baseline as part of its
architectural prior — but consequently the Δr number is smaller than
a comparably-trained deeper architecture with more tokens.

### Comparison to key benchmarks

| operating point | val mean r | val mean Δr |
|---|---|---|
| REVE-warmstart lr=1e-5 (original recipe) | ~0.156 | +0.116 |
| REVE-warmstart lr=1e-4 @ ep300 | — | +0.198 |
| **REVE-warmstart lr=3e-4 @ ep300 (aspirational bar)** | ~0.34 | **+0.218 [+0.197, +0.238]** |
| fresh500_ep499 (prior from-scratch baseline) | +0.13 | +0.080 (test split, not directly comparable) |
| **iter 12 ep500 (this loop, from-scratch)** | **+0.256** | **+0.126** |

**Where we land**:
- On absolute Pearson r, iter 12 reaches **75 % of warmstart lr=3e-4**.
- On Δr, iter 12 reaches **58 % of warmstart lr=3e-4**.
- Motion_energy specifically closes to **95 %** of warmstart's motion Δr
  (+0.222 vs warmstart's ~+0.232) — the highest-level feature where
  warmstart was previously strongest.
- Every feature is above matched random baseline with narrow CIs; no inversions.

The warmstart absolute-r advantage isn't fully closed, but iter 12
gets there in ~50 min of GPU compute from random init vs warmstart's
1 h 09 min on top of REVE's already-pretrained 50 M-param encoder.

## Insights from the search

### What worked

1. **Patch coarsening dominates.** patch 50 → 400 gave +0.024 val CV
   Δ alone. Every axis (depth, width, FFN, projection, optimizer,
   batch size) has smaller effect than this one architectural choice
   when tested against the winning shape. **The natural granularity
   for CLIP alignment is the alignment granularity itself** —
   `patch_size=window_size` means one token per channel per window,
   matching V-JEPA-2's per-window target.
2. **Depth up to 12 helps** at the coarse-patch shape. Beyond that
   (depth 22, embed 768, mlp_ratio 4) saturates. The 129-token
   sequence caps how much depth can add.
3. **lr=1e-4 is the right base LR for from-scratch** across every
   architecture we tested. lr=3e-4 hurt at depth=4 shape (iter 4) and
   at depth=8 shape (implicit — not explicitly re-tested since iter 4
   established the pattern strongly).

### What didn't work (dead ends confirmed)

- **Higher LR** (iter 4): 3e-4 hurts vs 1e-4. Consistent with fresh500 sweep.
- **Regularization** at short budget (iter 5 channel dropout, iter 6
  aux VC): both hurt. Model is undertrained, not overfit, in the
  100-500 ep regime. May reverse at longer training but we couldn't test.
- **Longer warmup** (iter 7): warmup 5 → 10 hurt slightly. Warmup=5
  suffices for from-scratch at this shape.
- **Wider encoder** (iter 13): embed 512 → 768 tied within noise. Simplicity
  favors 512.
- **Deeper encoder** past 12 (iter 14): depth 22 tied. 12 is the sweet spot.
- **Higher FFN ratio** (iter 15): mlp_ratio 4 slightly worse.
- **Complex projection head** (iter 16): n_residual_blocks 3 hurts.
  Consistent with from_checkpoint's finding at lr=1e-5 (bigger
  projector always hurts).
- **AdamW + wd=0.05** (iter 17): identical to Adam at 300 ep — wd
  effect is negligible at this training length.
- **Larger batch** (iter 18): bs=128 hurts. Fewer optimizer steps per
  epoch hurts in the compute-bound regime.
- **Extended schedule past ep500** (cont_lr5e5, cont_lr1e4): both
  continuation strategies hurt. Schedule-length pathology (§3.2)
  applies here too.

### The core lesson: architecture > everything else at this scale

At medium params (~46 M) on small data (~700 recordings), the
architectural inductive bias (patch=window) does most of the work.
Regularization, optimizer choice, and LR fine-tuning all have secondary
or negligible effects. Once patch=400 + depth=12 was found, no other
axis moved the needle.

## Infrastructure lessons

### Lustre / torch.save fragility

Delta's `/work/hdd/bbnv` and `/u/dtyoung` filesystems both hit two
distinct classes of failure during long-training runs:

1. **RPC transport errors** on `/u/dtyoung` (`Cannot send after transport
   endpoint shutdown`) — transient during heavy write bursts. Cleared
   by themselves after minutes.
2. **torch.save zipfile position mismatch** (`unexpected pos X vs Y`)
   — Lustre write cache inconsistency during the newer zipfile
   serialization format's seek-based writes.

**Mitigations applied**:
- **Atomic writes**: save to `.tmp`, rename atomically. Catches
  in-progress failures cleanly. Added in `training_utils.py`.
- **Legacy serialization**: `torch.save(...,
  _use_new_zipfile_serialization=False)` — sequential-write codepath
  more robust to Lustre partial-flush semantics.
- **`save_every=50` at long runs**: 10 saves for 500-ep run instead of
  50. Fewer chances to hit the bug.
- **Config snapshotting**: `cp config/clip_pretrain.yaml
  {exp_dir}/config.yaml` at job start, then both training and probe
  read from the snapshot. Immunizes against mid-flight repo syncs
  (iter 8 got hit by exactly this and required a probe follow-up).

### `/u/dtyoung` quota

Long-training runs write ~469 MB per checkpoint. At `save_every=10`
across 500 epochs = 50 saves = 23 GB per iter dir. Home quota is 102 GB
soft. Failed ep500 attempts (with `save_every=10` writes to `/work/hdd`
that was over team-quota, then a retry on `/u/dtyoung` that ran into
`disk quota exceeded`) — resolved by cleanup + `save_every=50`.

Post-cleanup: only iter 12's checkpoints remain (~14 GB on `/work/hdd`
+ ~1.5 GB on `/u/dtyoung`). All other iter checkpoints deleted for space
recovery.

## What we didn't try (open questions)

Ordered by likelihood of moving beyond iter 12's ceiling:

1. **Memory bank / MoCo queue** (program idea #5). Multi-view-redundancy
   theory says more diverse negatives tighten the bound. With bs=64
   the effective negative pool is small; queued negatives could raise
   the asymptote.
2. **Soft V-JEPA-2 similarity targets (SoftCLIP / DistillCLIP)**
   (design notes §6 option 4 + program idea #7). Replace binary
   same/different scene labels with the continuous V-JEPA-2 cosine
   matrix. Sidesteps the label-noise floor.
3. **Attention-pooling window aggregation** instead of mean-pool over
   channels × patches. Learned queries could preserve more per-window
   signal.
4. **Multi-movie training** — see `../scene_clip_multimovie/`. 2× data
   should shift the "compute-bound vs overfit" balance; some jul1
   dead ends (aug, VC, longer schedules) may reverse there.
5. **Cross-modal latent prediction (EEG-JEPA-against-V-JEPA-2 latents)**
   — biggest architectural change; program idea #6. Direct regression
   instead of contrastive avoids the log-K-bits ceiling of InfoNCE.

## Reproducibility

Best training run:

```bash
uv run --group eeg python experiments/clip_pretraining/scene_clip_fromscratch/autoresearch/_submit_long.py --submit
# submits 500-ep training + probe val on Delta A40. ~50 min wall.
```

Downstream probe_traintest (val split, B=2000):

```bash
uv run --group eeg python experiments/clip_pretraining/scene_clip_fromscratch/autoresearch/_probe_traintest_ep500.py submit
```

Random baseline for Δr computation:

```bash
uv run --group eeg python experiments/clip_pretraining/scene_clip_fromscratch/autoresearch/_probe_traintest_rand.py submit
```

Full audit trail in [`results.tsv`](results.tsv). Every commit on
this branch corresponds to one iteration's config change.

Wall-time total across all iterations: ~14 GPU-hours on Delta A40.
