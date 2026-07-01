# autoresearch: from-scratch CLIP pretraining

Have the LLM autonomously improve EEG↔V-JEPA-2 alignment starting from a
**randomly-initialized** encoder. The specific challenge:

> Can a from-scratch encoder trained on ~700 HBN recordings beat the
> REVE-warmstart operating point, which brings ~50 M params of pretrained
> broad-EEG knowledge to the table?

**Aspirational target**: `warmstart_lr3e4_ep299` — REVE-base warm-start,
`per_window` targets, `proj_dim=512`, `lr=3e-4`, 300 epochs.
probe_traintest val Δr = **+0.218** (Pearson r, B=2000). Note: not
directly comparable to the in-loop CV-on-val R² we use here; the two
protocols move together but the numbers live on different scales.

**Existing from-scratch reference (the number to beat)**: `fresh500_ep499`
— random init, `per_window`, `lr=1e-4`, 499 epochs. **CV-on-val
mean R² = +0.0462, Δ vs random = +0.0272**. Test Pearson Δr =
+0.080 (context only; test is off-limits in the loop).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag** (e.g. `jul1`). The branch
   `autoresearch/fromscratch-<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/fromscratch-<tag>`
   from `main`.
3. **Read the in-scope files** for full context. The repo is large;
   these are the ones you actually need:
   - [`README.md`](../../../../README.md) — repository context. Focus on
     the CLIP pretraining section.
   - [`../RESULTS.md`](../RESULTS.md) — current from-scratch results.
     This is the number you are trying to beat.
   - [`../../scene_clip_from_checkpoint/RESULTS.md`](../../scene_clip_from_checkpoint/RESULTS.md)
     — the REVE-warmstart operating point (the aspirational target
     +0.218 val Δr). Read §3 carefully — it contains **the known
     pathologies you must not repeat**.
   - [`../../embedding_feature_correlation/clip_design_observations_vjepa2.md`](../../embedding_feature_correlation/clip_design_observations_vjepa2.md)
     — the §9 bottom-line recipe design; explains *why* scene_clip
     works and its known failure modes.
   - [`eb_jepa/clip.py`](../../../../eb_jepa/clip.py) — CLIP objective.
     **Read-only.** The InfoNCE loss stays fixed.
   - [`eb_jepa/evaluation/clip_probe/probe_traintest.py`](../../../../eb_jepa/evaluation/clip_probe/probe_traintest.py)
     — the eval harness. **Read-only.**
   - [`eb_jepa/architectures.py`](../../../../eb_jepa/architectures.py)
     — model architecture. **Editable.**
   - [`eb_jepa/losses.py`](../../../../eb_jepa/losses.py) —
     auxiliary regularizers. **Editable.**
   - [`eb_jepa/training/clip_pretrain.py`](../../../../eb_jepa/training/clip_pretrain.py)
     — training loop, optimizer, scheduler. **Editable.**
   - [`config/clip_pretrain.yaml`](../../../../config/clip_pretrain.yaml)
     — defaults. **Editable via CLI overrides or in-file changes.**
4. **Verify cluster / data**: use the `neurolab` skill to verify Delta
   (or whichever cluster the user picks) has a live env, preprocessed
   HBN data, and a clean git state.
5. **Initialize `results.tsv`** with the header row. The baseline is
   recorded after the first (unmodified) run.
6. **Confirm and go.**

Once confirmed, kick off the experimentation loop.

## Experimentation

Each experiment runs one training + one probe eval on a single GPU.

**Fixed budget: ~30 min training + ~5 min probe eval per iteration**
(~130 epochs at 14 s/epoch on Delta A40). Aim for **~20
iterations/day** per GPU.

Why not 5 min like Karpathy's nanoGPT autoresearch? Karpathy's setup
is from-scratch LM on data-rich text where train loss IS the eval
metric — 5 min is enough to see the trajectory shape. Ours is
contrastive alignment on a small dataset with a *separate* downstream
probe. Learning curves are much flatter and at ~20 epochs we're
still in warmup + noise-dominated territory. 30 min (~130 epochs) is
the empirical floor where LR / architecture effects start to
resolve; beyond that, the schedule-length pathology (see below) sets
in past ~300 epochs.

Launch template (Delta A40, single job):

```bash
# 1. Train (should complete in ~30 min at ~14 s/epoch × ~130 epochs).
#    Adjust --optim.epochs if per-epoch cost changes (bigger model /
#    batch). The wall-clock is the budget, not the epoch count.
PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain \
    --fname=config/clip_pretrain.yaml \
    --optim.epochs=130 \
    --logging.wandb_group=auto_<tag>_<iter>_<hash>

# 2. Probe val R² (the metric — always on val, never touch test).
#    CV-by-recording, 5-fold GroupKFold, no bootstrap.
PYTHONPATH=. uv run --group eeg python \
    eb_jepa/evaluation/clip_probe/probe.py \
    --checkpoint /path/to/latest.pth.tar \
    --config config/clip_pretrain.yaml \
    --split val \
    --cv-splits 5 \
    --output probe_val.json
```

**What you CAN modify**:
- Encoder architecture: depth, width, attention pattern, patchification,
  spatial encoding, dropout, norm placement.
- Projector head: shape, depth, activation, normalization.
- Optimizer: choice (AdamW, Lion, etc.), betas, weight decay, gradient
  clipping.
- LR schedule: warmup shape, decay curve, cosine vs linear vs constant.
- Regularization: auxiliary losses (VC, SIGReg, contrastive-consistency),
  EMA target networks, data augmentation.
- Batch size, gradient accumulation.
- Positive/negative construction *inside the scene_clip recipe*:
  temporal buffer, target kind (`per_window` vs `shot_mean`), mean-
  centering toggle. These live in the config and dataset; you can
  toggle them.

**What you CANNOT modify**:
- [`eb_jepa/clip.py`](../../../../eb_jepa/clip.py) — the InfoNCE
  objective stays fixed. Auxiliary losses added *around* the CLIP loss
  are fair game; the loss formula itself is not.
- [`eb_jepa/evaluation/`](../../../../eb_jepa/evaluation/) — the probe
  is the ground truth metric. Do not modify.
- [`eb_jepa/datasets/hbn.py`](../../../../eb_jepa/datasets/hbn.py) data
  loading + preprocessing — treat as fixed. Augmentation added
  *upstream of the encoder in the training loop* is fine.
- `pyproject.toml` — no new dependencies. Use what's already installed.

**VRAM is a soft constraint.** Some increase is acceptable for
meaningful val Δr gains. Do not blow up dramatically (>60 GB on A40).

**Simplicity criterion**: All else being equal, simpler is better. A
+0.001 Δr improvement that adds 20 lines of hacky code is not worth
keeping. A +0.001 Δr improvement from *removing* code is a great
outcome — always keep simplifications. Prefer changes that also
explain-through to related runs.

**First run**: baseline. Do not modify anything. Just run the config
as-is on `main`, so the first row of `results.tsv` is your reference.

## The primary metric: probe val ΔR²

- **Metric**: mean 5-fold CV R² on R5 (val) minus the matched
  random-init baseline of the same encoder architecture. Averaged over
  the 12 scalar features in
  [`SCALAR_FEATURES_DEFAULT`](../../../../eb_jepa/evaluation/clip_probe/probe.py).
- **Protocol**: `probe.py --split val --cv-splits 5` — 5-fold
  GroupKFold-by-recording on R5 (no cross-split leakage), RidgeCV
  per feature. No bootstrap. Deterministic. Fast (~5 min for a 50 M-
  param encoder — dominated by encoding all val windows).
- **Why not probe_traintest?** In-loop budget is tight and the extra
  train-set embedding pass adds ~5 min without changing the *ranking*
  signal at iteration scale. Save probe_traintest + bootstrap for the
  final write-up of any promising configuration.
- **Random baseline**: run the probe once with `--random-baseline` on
  the initial (untrained) encoder architecture. Cache the number as
  `rand_r2`; every iteration reports `mean_r2 - rand_r2`.
- **Selection rule**: higher is better. Iterations with ΔR² strictly
  above the current best are kept; ties or worse are discarded and the
  branch is reset.

### Do NOT select on

- **`val/clip_scene_auc`** — decouples from probe R² under long
  schedules (see [scene_clip_from_checkpoint/RESULTS.md §3.2](../../scene_clip_from_checkpoint/RESULTS.md)).
  The 1000-ep run peaked AUC=0.86 at ep700 but had probe val Δr =
  +0.115, versus its 300-ep sibling at +0.198. Picking on AUC would
  select the *worse* checkpoint. Log AUC as a sanity signal, ignore
  it for selection.
- **`train/clip_top1_e2v`** — under the multi-positive scene_clip
  loss, same-scene neighbors compete with the exact-diagonal target,
  so top-1 is downward-biased by design. Also decoupled from probe R².
- **Test set** — never probe on test during the loop. It's the final
  report, not a selection oracle. Every touch inflates the risk of
  test-set overfitting via selection.

## Output format

The training script logs to W&B (`train/*` and `val/*` metrics) and
saves checkpoints per `save_every`. `probe.py` writes a JSON keyed by
feature with `r2_mean` (5-fold CV mean) + `r2_std` per feature.

Extract the metric like:

```bash
uv run python -c "
import json, statistics as st
with open('probe_val.json') as f: d = json.load(f)
mean_r2 = st.mean(v['r2_mean'] for v in d['features'].values())
print(f'val_mean_r2: {mean_r2:.6f}')
"
```

Also log peak VRAM from the training log (grep `peak_vram_mb` or
`torch.cuda.max_memory_allocated`).

## Logging results

Append to `results.tsv` (tab-separated, headers on the first line):

```
commit	val_delta_r2	val_mean_r2	memory_gb	epochs	status	description
```

- `commit`: 7-char short hash of the iteration's git commit
- `val_delta_r2`: `mean_r2 - rand_r2`; e.g. `+0.02720`. Use `0.000000` for crashes.
- `val_mean_r2`: raw mean 5-fold-CV R² on val (before subtracting rand). Useful for provenance.
- `memory_gb`: peak GPU memory rounded to .1f. Use `0.0` for crashes.
- `epochs`: actual epochs completed (may be < config target if run OOM'd or was cut).
- `status`: `keep`, `discard`, or `crash`.
- `description`: short text of what this iteration tried. Concise, no commas.

Example:

```
commit	val_delta_r2	val_mean_r2	memory_gb	epochs	status	description
a1b2c3d	+0.02720	+0.0462	14.2	130	keep	baseline: fresh500-style config at 130 ep
b2c3d4e	+0.04100	+0.0602	14.8	130	keep	depth 4 -> 8 encoder
c3d4e5f	+0.03800	+0.0572	14.2	130	discard	AdamW -> Lion
d4e5f6g	+0.00000	0.0	0.0	0	crash	encoder width 2048 (OOM)
```

Do not commit `results.tsv`. It's a per-branch scratchpad.

## The experiment loop

LOOP FOREVER:

1. Read git state: current branch/commit, current best val ΔR².
2. Pick an experimental change; hack it into the editable files
   directly.
3. `git add -A && git commit -m "iter <N>: <short description>"`.
4. Submit / run the training job. Redirect all output to `run.log` —
   do NOT tee, do NOT let training output flood your context.
5. When training completes, run the probe:
   `PYTHONPATH=. uv run --group eeg python eb_jepa/evaluation/clip_probe/probe.py --split val --cv-splits 5 --output probe_val.json --checkpoint /path/to/latest.pth.tar --config config/clip_pretrain.yaml`
6. Extract `val_mean_r2` from the JSON; compute `val_delta_r2 =
   val_mean_r2 - rand_r2`.
7. Append the row to `results.tsv`.
8. If `val_delta_r2` strictly improved: keep the commit, advance the
   branch.
9. If equal or worse: `git reset --hard HEAD^` to revert to the
   pre-iteration commit.
10. Go to 1.

**Timeout**: each iteration ≤ 45 min wall (30 min train + 5 min
probe + 10 min slack). If it exceeds 45 min, kill it, log `crash`,
revert.

**Crashes**: if OOM, missing import, typo — fix it and re-run only if
the fix is obviously small (< 5 min effort). If the idea itself is
broken, log `crash` and move on. Do not spend > 15 min debugging a
single failing iteration.

**Rewind sparingly**: the loop advances on the branch by design. If
you get stuck in a local optimum, you may `git reset` to an earlier
commit — but do this at most once per session, and log why.

**NEVER STOP**: once experimentation begins, do NOT pause to ask
"should I keep going?" or "is this a good stopping point?" The human
is asleep or away and expects you to continue *indefinitely* until
manually stopped. If you run out of ideas, think harder: re-read the
in-scope docs, look at what worked in the REVE-warmstart runs and ask
"can from-scratch match this by other means?", try combining
previous near-misses, try more radical architectural swaps. The loop
runs until the human interrupts you.

## Known baselines (as of 2026-07-01)

| operating point | val CV mean R² | val ΔR² | notes |
|---|---|---|---|
| random-init encoder (matched arch, no training) | +0.019 | 0.000 | anchor; the `rand_r2` you subtract |
| `fresh500_ep499` (from-scratch, lr=1e-4, 499 ep) | +0.046 | **+0.027** | the number to beat |
| `warmstart_lr3e4_ep299` (REVE warm-start, lr=3e-4, 300 ep) | — | — | measured only on probe_traintest (Pearson r Δr = +0.218). Aspirational only; not directly comparable to CV R². |

Under the 30-min budget the baseline gets fewer epochs than fresh500
(~130 vs 499), so the first-iteration ΔR² will start lower than
+0.027 — expect ~+0.015 to +0.020 for a straight 130-ep run of the
current config. Beating +0.027 is the near-term goal; **closing 50 %
of the gap to REVE-warmstart** is the moral-win threshold.

## Known dead ends (do not re-run without a new theory)

From [scene_clip_from_checkpoint/RESULTS.md](../../scene_clip_from_checkpoint/RESULTS.md)
and [scene_clip_fromscratch/RESULTS.md](../RESULTS.md):

- **Extending schedule past the cosine floor** — 1000 ep at lr=1e-4
  landed at val Δr +0.11 vs the 300-ep sibling's +0.198. Same effect
  probably applies to lr=3e-4. Under the 30-min budget this mostly
  isn't reachable anyway, but if you're tempted to trade batch size /
  model width for more epochs, remember that "more epochs" past the
  cosine minimum hurts, not helps.
- **`lr=1e-5`** — under-tunes the encoder. The old "1/10 from-scratch
  LR" prior does not apply here.
- **`proj_dim=1024` or `1408` at lr=1e-5** — the ablation showed
  bigger projectors hurt when the encoder was barely moving. Was
  never re-tested at higher LR; if you try bigger projectors, use
  lr ≥ 1e-4.
- **Selecting on `clip_scene_auc`** — it decouples from probe R² and
  will mislead you (§3.2).
- **`target_kind=shot_mean`** at from-scratch scale — mildly worse
  than `per_window` in the existing sweeps; unlikely to be the
  breakthrough.
- **Freezing the encoder** — probing frozen-encoder outputs is a
  constant; the recipe requires the encoder to move.

## Ideas worth trying (starting-point suggestions)

Ordered by expected impact × difficulty. Don't take this list as
prescriptive — synthesize your own hypotheses from re-reading the
docs. But if you're bootstrapping:

1. **Deeper / wider from-scratch encoder** — REVE has depth=22, embed=512.
   Fresh500 was depth=4, embed=1024. Trying depth=8/12/22 at proj=512
   with lr=3e-4 explicitly answers "is the from-scratch deficit about
   architecture capacity or about pretraining?"
2. **Pretraining objective inside the loop** — hybrid loss:
   scene_clip InfoNCE + auxiliary VC/SIGReg regularizer on encoder
   tokens (borrowed from `losses.py`). Might act as a lightweight
   masked-pretraining substitute.
3. **EEG augmentation** — channel dropout, temporal jitter, phase
   scrambling — none currently applied. Standard CV / speech CLIP
   uses heavy augmentation on the anchor side.
4. **Data efficiency at higher LR** — fresh500 was lr=1e-4. Given
   that REVE-warmstart wins at lr=3e-4, fresh may also want higher
   LR (maybe 1e-3 with more warmup + more weight decay).
5. **Memory bank / MoCo queue** — expands negatives beyond batch
   size. Multi-view-redundancy bound only tightens with more
   diverse negatives.
6. **Larger batch / gradient accumulation** — CLIP is famously
   batch-size-sensitive. bs=64 is small. Try bs=256 via gradient
   accumulation.
7. **Warmup + higher peak LR + shorter total schedule** — §3.2 shows
   the schedule length itself is a regularizer. A 200-ep cosine at
   lr=5e-4 might beat the current 300-ep lr=3e-4 recipe.
