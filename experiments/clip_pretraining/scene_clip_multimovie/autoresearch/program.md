# autoresearch: multi-movie CLIP pretraining

Have the LLM autonomously improve EEG↔V-JEPA-2 alignment when training
on **two HBN movies simultaneously** (*ThePresent* + *DespicableMe*),
starting from a random-initialized encoder. The specific challenge:

> Does training on 2 movies together produce a better encoder than
> single-movie training? And if so, does the winning single-movie
> recipe (see `../../scene_clip_fromscratch/autoresearch/`) transfer,
> or does multi-movie prefer a different config?

**Reference baselines (single-movie from-scratch)**:

- `iter12_ep500` (jul1 autoresearch winner) — patch_size=400,
  encoder_depth=12, embed_dim=512, heads=8, head_dim=64, proj_dim=512,
  lr=1e-4, 500 epochs. On ThePresent val: **CV R² Δ = +0.04744**,
  probe_traintest Pearson r = +0.256 (**Δr = +0.126** vs random init
  r=+0.130). Wall 40 min on Delta A40.
- `warmstart_lr3e4_ep299` (single-movie ThePresent REVE warmstart) —
  probe_traintest val Pearson **Δr = +0.218** [+0.197, +0.238].
  Aspirational bar for from-scratch to close.

**Aspirational target for this loop**: **beat iter12_ep500 on
ThePresent val** using the multi-movie training signal, and ideally
close more of the gap to the +0.218 warmstart Δr. If the loop instead
finds "multi-movie hurts or is neutral", that's also an answer worth
having.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag** (e.g. `jul2`). The branch
   `autoresearch/multimovie-<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/multimovie-<tag>`
   from `main`.
3. **Read the in-scope files** for full context:
   - [`README.md`](../../../../README.md) — repo layout; CLIP section.
   - [`../NOTES.md`](../NOTES.md) — hypothesis + v1 plan + known
     minor caveats (temporal-buffer leak across movies).
   - [`../../scene_clip_fromscratch/autoresearch/program.md`](../../scene_clip_fromscratch/autoresearch/program.md)
     — the single-movie autoresearch program (this is the sibling
     loop; same shape).
   - [`../../scene_clip_fromscratch/autoresearch/results.tsv`](../../scene_clip_fromscratch/autoresearch/results.tsv)
     — the full single-movie search history. Read every row: iter 12
     is the winner; iterations 4/5/6/7/13/14/15/16/17/18 record
     confirmed dead ends at that shape.
   - [`../../scene_clip_fromscratch/RESULTS.md`](../../scene_clip_fromscratch/RESULTS.md)
     — pre-autoresearch single-movie sweep results.
   - [`../../scene_clip_from_checkpoint/RESULTS.md`](../../scene_clip_from_checkpoint/RESULTS.md)
     — the REVE-warmstart sibling. §3 known pathologies you MUST NOT repeat.
   - [`../../embedding_feature_correlation/clip_design_observations_vjepa2.md`](../../embedding_feature_correlation/clip_design_observations_vjepa2.md)
     — §9 bottom-line recipe.
   - [`eb_jepa/clip.py`](../../../../eb_jepa/clip.py) — CLIP objective.
     **Read-only.**
   - [`eb_jepa/evaluation/clip_probe/probe_traintest.py`](../../../../eb_jepa/evaluation/clip_probe/probe_traintest.py),
     [`eb_jepa/evaluation/clip_probe/probe.py`](../../../../eb_jepa/evaluation/clip_probe/probe.py)
     — eval harness. **Read-only.**
   - [`eb_jepa/datasets/hbn.py`](../../../../eb_jepa/datasets/hbn.py)
     — check §"Scene-ID namespacing" (NOTES §"What's new") to
     understand multi-movie plumbing. **Read-only** (do not modify
     dataset/loading code).
   - [`eb_jepa/architectures.py`](../../../../eb_jepa/architectures.py),
     [`eb_jepa/losses.py`](../../../../eb_jepa/losses.py),
     [`eb_jepa/training/clip_pretrain.py`](../../../../eb_jepa/training/clip_pretrain.py),
     [`config/clip_pretrain.yaml`](../../../../config/clip_pretrain.yaml)
     — all **editable**.
4. **Verify cluster / data**: use the `neurolab` skill to verify
   Delta has a live env, preprocessed HBN data (all releases R1-R6
   contain both tasks), and *DespicableMe* artifacts
   (`movie_annotation/output/despicable_me/vjepa2_recipe.npz`,
   `vjepa2_embeddings.npz`, and
   `experiments/clip_pretraining/embedding_feature_correlation/DespicableMe/vjepa2_scenes_map.csv`).
5. **Initialize `results.tsv`** with the header row.
6. **Confirm and go.**

Once confirmed, kick off the experimentation loop.

## Experimentation

Each experiment runs one training + one probe eval on a single GPU.

**Fixed budget: ~50 min training + ~10 min probe eval per iteration**
(60-min wall). Multi-movie has ~2× the data per epoch vs single-movie,
so per-epoch is roughly doubled. At iter 12's shape (patch=400
depth=12 embed=512), single-movie ran ~5 s/ep for 500 epochs in 40
min; expect multi-movie ~10 s/ep, so ~300 epochs fit inside 50 min
of training. Aim for **~15 iterations/day** on one A40.

Launch template (Delta A40, single job):

```bash
# 1. Train (~50 min training). Wall is 60 min.
PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain \
    --fname=config/clip_pretrain.yaml \
    --data.task='[ThePresent,DespicableMe]' \
    --optim.epochs=<budget-scaled-int> \
    --folder=<absolute-checkpoint-dir> \
    --logging.wandb_group=auto_<tag>_<iter>_<hash>

# 2. Probe val R² separately on each movie (single-movie probe.py).
#    R5 val for each task, 5-fold GroupKFold, no bootstrap.
PYTHONPATH=. uv run --group eeg python \
    eb_jepa/evaluation/clip_probe/probe.py \
    --checkpoint <ckpt>/latest.pth.tar \
    --config <ckpt>/config.yaml \
    --split val --cv-splits 5 \
    --task ThePresent \
    --output probe_val_<iter>_ThePresent.json

PYTHONPATH=. uv run --group eeg python \
    eb_jepa/evaluation/clip_probe/probe.py \
    --checkpoint <ckpt>/latest.pth.tar \
    --config <ckpt>/config.yaml \
    --split val --cv-splits 5 \
    --task DespicableMe \
    --output probe_val_<iter>_DespicableMe.json
```

(If probe.py doesn't yet support `--task` for multi-movie configs,
check for CLI/API changes needed. Probing per-movie separately is
important so we can attribute gains.)

**Recommended baseline configuration (iter 0)**: start from single-
movie iter12's config (jul1 winner) + `data.task=[ThePresent,
DespicableMe]`. This tests whether multi-movie helps at the known-
good architecture. If iter 0 already beats iter12_ep500 on ThePresent
val, positive transfer confirmed. If not, structural change is
needed.

**What you CAN modify**:
- Encoder architecture (depth, width, patch, positional encoding,
  attention pattern, mlp_dim_ratio) — but note iter 12's shape is a
  strong prior; deviations should be motivated.
- Projector head shape / normalization.
- Optimizer choice + weight decay.
- LR schedule (warmup, cosine, WSD, warm restarts).
- Batch size, gradient accumulation.
- **Positive/negative construction** — including the multi-movie
  temporal-buffer leak: currently the same `|Δt|<buffer_s` mask
  applies cross-movie. Fixing this by adding a `same_movie` guard is
  a specific idea worth trying (NOTES §"Known minor caveat"), but
  requires a careful edit to `SceneCLIPPretrain`'s exclusion mask.
- **Task weighting**: if DespicableMe is over/under-represented per
  epoch, does explicit re-balancing help?
- Regularization (auxiliary VC, augmentation) — but note the
  single-movie loop confirmed these HURT at 100-ep budget; only try
  at multi-movie's larger training budget with theory.

**What you CANNOT modify**:
- `eb_jepa/clip.py` — the InfoNCE objective stays fixed. The
  exclusion mask inside `SceneCLIPPretrain` is off-limits *for the
  InfoNCE formula itself*; however, if you want to add a same-movie
  guard, the correct place is to pass a `movie_ids` tensor from the
  dataset through the training loop into the mask computation. That
  edit lives on the boundary — flag it explicitly if you do it.
- `eb_jepa/evaluation/` — the probe is the ground truth metric.
- `eb_jepa/datasets/hbn.py` data loading is fixed. Data augmentation
  applied upstream of the encoder in the training loop is fine.
- `pyproject.toml` — no new dependencies.

**VRAM is a soft constraint.** Some increase is acceptable for
meaningful gains. Do not blow up dramatically (>60 GB on A40).

**Simplicity criterion**: All else being equal, simpler is better.

**First run**: baseline = iter 12 config + `data.task=[ThePresent,
DespicableMe]`. Don't modify anything else. The first row of
`results.tsv` is the multi-movie iter 12 result.

## The primary metric: probe val ΔR², averaged across movies

- **Metric**: mean 5-fold CV R² Δ vs random-init baseline, computed
  per movie, then averaged. Two probe calls per iteration (one per
  movie), then average `(ThePresent_ΔR² + DespicableMe_ΔR²) / 2`.
- **Why average?** Multi-movie's promise is a joint encoder that
  does well on both. The right selection metric is the mean, not the
  max or one movie alone.
- **Random baseline caveat**: iter 12's random baseline was measured
  at r=+0.130 on ThePresent because patch=400 gives an unusually
  strong random-init prior. Same must be measured for DespicableMe
  (probably similar since architecture is identical). Cache both
  values, subtract per-movie.
- **Selection rule**: higher mean Δr² is better. Iterations strictly
  above the current best are kept. Ties/worse are discarded.

### Do NOT select on
- Train/val loss.
- `clip_scene_auc` (decouples from probe R² under long schedules;
  see [scene_clip_from_checkpoint/RESULTS.md §3.2](../../scene_clip_from_checkpoint/RESULTS.md)).
- Only-one-movie's probe R² (would bias against joint representation).
- Test set — never touch during the loop.

## Output format

`results.tsv` (tab-separated, headers on first line):

```
commit	val_delta_r2_mean	val_delta_r2_TP	val_delta_r2_DM	val_mean_r2_TP	val_mean_r2_DM	memory_gb	epochs	status	description
```

- `commit`: 7-char short hash.
- `val_delta_r2_mean`: `(ΔR²_TP + ΔR²_DM) / 2`. Primary metric.
- `val_delta_r2_TP`: ThePresent-only Δ.
- `val_delta_r2_DM`: DespicableMe-only Δ.
- `val_mean_r2_TP`, `val_mean_r2_DM`: raw mean 5-fold-CV R² per
  movie (provenance).
- `memory_gb`, `epochs`, `status` (`keep`|`discard`|`crash`),
  `description` (concise, no commas).

Do not commit `results.tsv`. Per-branch scratchpad.

## The experiment loop

LOOP FOREVER:

1. Read git state: current branch/commit, current best mean ΔR².
2. Pick an experimental change; edit files directly.
3. `git add -A && git commit -m "iter <N>: <short description>"`.
4. Submit training job. Redirect output to `run.log` — do NOT tee,
   do NOT flood context.
5. When training completes, run **BOTH** probe evals (per movie).
6. Extract per-movie `val_mean_r2`; compute Δ vs matching per-movie
   random baseline; compute mean.
7. Append row to `results.tsv`.
8. If `val_delta_r2_mean` strictly improved: keep, advance branch.
9. If tie/worse: `git reset --hard HEAD^`.
10. Go to 1.

**Timeout**: each iteration ≤ 60 min wall. If exceeded, kill, log
`crash`, revert.

**Crashes**: OOM / import errors — fix and re-run if trivial. If the
idea is broken, log crash and move on. Do not spend >15 min debugging
a single failing iteration.

**NEVER STOP**: once experimentation begins, do NOT pause to ask
"should I keep going?" The human is asleep or away and expects you
to continue *indefinitely* until manually stopped.

## Known baselines (as of 2026-07-02)

| operating point | val CV mean ΔR² | notes |
|---|---|---|
| random-init encoder (iter 12 shape) | 0.000 | anchor per movie; must be measured |
| `iter12_ep500` (single-movie ThePresent, from-scratch) | +0.04744 on TP | number to beat on TP; DM not measured |
| `warmstart_lr3e4_ep299` (single-movie ThePresent, REVE) | — (val Δr = +0.218 Pearson) | aspirational; not directly comparable to CV R² |

Multi-movie is expected to either **help** (positive transfer,
higher mean ΔR²) or **hurt** (interference, lower per-movie ΔR²).
The loop decides which by running.

## Known dead ends (from single-movie jul1 loop; do not re-run
without new theory)

- **Wider encoder (embed=768) at iter12 shape** — tied with iter 12
  (iter 13 discard). Simplicity favors embed=512.
- **Deeper encoder past 12** (depth=22) at iter12 shape — tied
  (iter 14 discard). Depth saturates at 12.
- **mlp_dim_ratio 2.66 → 4** at iter12 shape — slightly worse
  (iter 15 discard).
- **Complex projection head (n_residual_blocks=3)** at iter12 shape
  — worse (iter 16 discard). Small projection head wins.
- **AdamW + weight_decay=0.05** at iter12 shape — identical to Adam
  at 300 ep (iter 17). Effect too small to matter at these training
  lengths.
- **Larger batch (128)** at iter12 shape — worse (iter 18 discard).
  Smaller batch wins in the compute-bound regime.
- **channel_dropout p=0.1** — hurt (iter 5). Any regularization at
  the ~70-300 ep budget hurt (undertrained not overfit).
- **auxiliary VC loss** (std_coeff=cov_coeff=0.1 on encoder tokens)
  — hurt (iter 6). Same pattern.
- **Higher LR (lr=3e-4)** at iter12 shape — hurt (iter 4). From-
  scratch prefers lr=1e-4 across shapes; single-movie evidence,
  worth re-testing under multi-movie only with new theory.
- **Extending cosine schedule past cosine floor** — hurt in
  continuations (iter12 cont_lr5e5 and cont_lr1e4). Schedule-length
  pathology from §3.2 of from_checkpoint RESULTS applies here too.

Note: some of the above may reverse under multi-movie because more
data changes the "compute-bound vs overfit" balance. In particular
**channel_dropout, aux VC, and longer schedules** are worth
re-testing with theory once you're confident of the baseline.

## Ideas worth trying (multi-movie specific)

Ordered by expected impact × difficulty.

1. **Baseline first** (iter 0) — confirm iter 12 config on multi-
   movie: does it beat single-movie iter 12 on TP val, and by how
   much on DM val (which has no prior baseline)?
2. **Fix cross-movie temporal-buffer leak** (NOTES.md §caveat) —
   pass `movie_ids` through the training loop into
   SceneCLIPPretrain's exclusion mask, add `same_movie` guard.
   Estimated to fix ~1-2% of mask entries in a balanced batch.
   Small-but-principled correctness fix.
3. **Task weighting** — currently the dataset concatenates recordings
   from both tasks; DespicableMe is shorter (2:50 vs 3:23) so per-
   epoch it contributes ~90% of ThePresent's window count. Test
   whether up/downweighting one task helps.
4. **Longer training at multi-movie's larger effective dataset** —
   single-movie iter 12 saturated at ep500 because ~700 recordings
   isn't enough to sustain gains. Multi-movie has 2× that. Extending
   to ep500+ may keep gaining where single-movie plateaued.
5. **Aux VC / channel dropout with small p** — single-movie found
   these hurt at short budgets. Multi-movie's larger data may make
   them help; re-test with p=0.05 (not 0.1) and small coeffs (0.05).
6. **Memory bank / MoCo queue** — program-idea #5 from single-movie.
   With multi-movie, negatives naturally include cross-movie hard
   negatives; queue expands this further.
7. **Cross-modal latent prediction** (V-JEPA-2 latent regression
   instead of contrastive) — biggest code change; program idea #6.
   Multi-movie provides more training signal for regression.
8. **Higher-LR + shorter-schedule combo** — program idea #7. Multi-
   movie has more data per epoch so gradient signal is less noisy;
   higher LR may work here where it hurt single-movie.
