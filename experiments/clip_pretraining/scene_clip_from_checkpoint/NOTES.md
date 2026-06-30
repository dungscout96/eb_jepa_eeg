# scene_clip from checkpoint — Experiment Plan

Sibling to [`../scene_clip_fromscratch/`](../scene_clip_fromscratch/). Tests
whether **warm-starting from a pretrained EEG foundation model** (REVE-base
from braindecode) accelerates and/or improves the `scene_clip` recipe vs
training the encoder from random init.

## Hypothesis

REVE-base is a 22-layer EEG transformer (~50 M params) pretrained on broad
multi-site EEG data by braindecode (`brain-bzh/reve-base` on HuggingFace).
Warm-starting CLIP from REVE's representations should:

1. **Accelerate convergence** — the encoder already knows useful EEG features
   from large-scale pretraining; the contrastive loss only needs to nudge them
   into V-JEPA-2 alignment, not learn EEG representations from scratch.
2. **Raise the asymptote on high-level features** — REVE's representations
   are richer than what 500 epochs of CLIP-from-random can build. Features
   that scene_clip_fromscratch struggled with (motion, depth, narrative
   — Δ Pearson r ≈ +0.08 each on test) may climb further.
3. **Improve generalization across HBN releases** — REVE's training data
   spans many subject pools; its features should be more subject-invariant
   than EET trained only on HBN R1-R4.

## Architecture compatibility

Confirmed by inspection (REVE.__init__ signature matches
EEGEncoderTokens.__init__; state_dicts overlap 1:1):

- REVE-base: `embed_dim=512, depth=22, heads=8, head_dim=64,
  mlp_dim_ratio=2.66, freqs=4, patch_size=200, patch_overlap=20`.
- `EEGEncoderTokens(...)` with the same kwargs produces a state_dict that
  matches REVE's encoder-side keys **65 / 67 keys**.
- Two key mismatches:
  - `to_patch_embedding.{weight,bias}` (EET) vs
    `to_patch_embedding.0.{weight,bias}` (REVE) — trivial rename.
  - REVE has an extra `final_layer.*` classification head (132 k → n_outputs)
    — drop it for warm-start.

## Implementation plan

### 1. Offline adapter script (no trainer code change)

`prepare_reve_checkpoint.py` — one-time download + rename + save as a
`.pth.tar` with `model_state_dict` keys prefixed `encoder.` (matching what
`load_encoder_weights` already strips). Output:
`/projects/bbnv/dtyoung/checkpoints/reve_base_eet_init.pth.tar`.

Pseudocode:

```python
reve = REVE.from_pretrained("brain-bzh/reve-base", ...)
sd = reve.state_dict()
# Drop classification head
sd = {k: v for k, v in sd.items() if not k.startswith("final_layer.")}
# Rename patch embedding
sd = {k.replace("to_patch_embedding.0.", "to_patch_embedding."): v
      for k, v in sd.items()}
# Prefix with encoder. so load_encoder_weights picks it up
sd = {f"encoder.{k}": v for k, v in sd.items()}
torch.save({"model_state_dict": sd, "epoch": -1, "step": 0}, out_path)
```

Use the existing `--meta.encoder_init_from=<path>` flag (already supported by
the trainer). No changes to `clip_pretrain.py` or `builder.py`.

### 2. Config changes (vs scene_clip_fromscratch)

| field | scene_clip_fromscratch | scene_clip_from_checkpoint |
|---|---|---|
| `model.encoder_embed_dim` | 1024 | **512** |
| `model.encoder_depth` | 4 | **22** |
| `model.encoder_heads` | 4 | **8** |
| `model.encoder_head_dim` | 16 | **64** |
| `model.patch_size` | 50 | **200** |
| `model.patch_overlap` | 20 | **20** (same) |
| `model.freqs` | 6 | **4** |
| `model.mlp_dim_ratio` | 2.66 | **2.66** (same) |
| `loss.proj_dim` | 1024 | **512** (match new embed_dim) |
| `data.window_size_seconds` | 2 | **2** initially; consider 4 |
| `meta.encoder_init_from` | null | **path to reve_base_eet_init.pth.tar** |

Other recipe knobs (target_kind=per_window, mean_center=true,
temporal_buffer_s=2.0, scene-ID labels) stay identical to fresh500.

### 3. Hyperparameter starting points

REVE-base is large and pretrained, so:

- **Lower LR than scene_clip_fromscratch**: start `lr=1e-5` (1/10 of
  fresh500's lr=1e-4). Fine-tuning a pretrained encoder typically wants
  10-100× smaller LR than training from scratch.
- **Longer warmup**: `warmup_epochs=10` (vs 5) — let the optimizer adapt
  before reaching peak LR with these initial gradients.
- **Fewer epochs**: warm-start usually converges faster. Start at
  `epochs=100`; extend if the curve hasn't plateaued.
- **Optimizer**: `AdamW + weight_decay=0.05` (already supported via
  cfg.optim.optimizer=adamw, cfg.optim.weight_decay).

### 4. Compute estimate

REVE-base is ~50 M params (depth=22, embed=512) vs the fresh500 encoder's
~35 M (depth=4, embed=1024). Per-step forward+backward roughly **3-4× slower**
because depth dominates. At bs=64, expect ~1 minute per epoch vs fresh500's
~24 s/epoch. 100 epochs ≈ 1.5–2 h wall on Delta A40.

## Open questions

1. **window_size_seconds=2 vs 4**: at `patch_size=200` (1 s of EEG at 200 Hz),
   a 2 s window has only **2 patches** per recording — very few tokens for
   the encoder to attend over. Bumping `window_size_seconds=4` gives 4
   patches and more attention context, matching how REVE was trained.
   Trade-off: V-JEPA-2 mean-pooling already happens over the EEG window, so
   the longer window averages out finer-time-scale visual info.
2. **Match REVE's normalization?** REVE was trained on its own normalization
   pipeline. We use per-recording z-scoring. Should be compatible since
   z-scoring is the standard EEG preprocessing, but worth a sanity check
   (e.g., embed a few clips with random-init EET vs REVE-init EET; the
   REVE-init outputs should look structured, not collapsed).
3. **Freeze early layers?** Could freeze the bottom N layers and only
   fine-tune the top few + clip_head. Reduces compute and prevents catastrophic
   forgetting of REVE's features. Not in v1 — train all layers initially.

## Comparison endpoints

After the first run finishes, probe on:

- **CV-on-val** (clip_probe): direct comparison to scene_clip_fromscratch's
  fresh500 (Δ R² = +0.0272 was the best). If REVE warm-start beats this, the
  hypothesis holds.
- **ImageNet-style train→test** (probe_traintest): the headline. fresh500
  hit Δ Pearson r = +0.080 on test. REVE-warm-start success = exceeds that.

The random baseline on test (probe_random_depth4_test) is unchanged across
both experiments (encoder-architecture-controlled per-folder, but here the
architecture differs — we'll need a fresh random baseline for the REVE shape).

## Order of operations

1. Write & run `prepare_reve_checkpoint.py` locally → upload .pth.tar to Delta.
2. Add `config/clip_pretrain_from_reve.yaml` with the REVE-shape overrides
   (or just pass them as CLI flags for the first run; promote to yaml after
   the first run validates).
3. Submit first warm-start job: 100 ep, lr=1e-5, warmup=10, AdamW.
4. Probe ep99 latest checkpoint with both `probe.py --split val` and
   `probe_traintest.py`.
5. Compare against scene_clip_fromscratch baselines.
6. Iterate based on results.
