# autoresearch: from-scratch CLIP pretraining

Have the LLM autonomously improve EEG↔V-JEPA-2 alignment starting from a
**randomly-initialized** encoder. The specific challenge:

> Can a from-scratch encoder trained on ~700 HBN recordings beat the
> REVE-warmstart operating point, which brings ~50 M params of pretrained
> broad-EEG knowledge to the table?

**Baseline to match**: `warmstart_lr3e4_ep299` — REVE-base warm-start,
`per_window` targets, `proj_dim=512`, `lr=3e-4`, 300 epochs. Val Δr =
**+0.218** (probe_traintest on R5, B=2000 by recording).

**Existing from-scratch reference**: `fresh500_ep499` — random init,
`per_window`, `lr=1e-4`, 499 epochs. Val Δr ≈ **+0.027** (CV-on-val R²
scale) / test Δr = **+0.080**. That's the number to *beat*.

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

**Fixed budget: ~75 min training + ~10 min probe eval per iteration**
(see §"Budget rationale" below for why not 5 min like Karpathy). Aim
for **~10 iterations/day** per GPU.

Launch template (Delta A40, single job):

```bash
# 1. Train (should complete in <75 min at 14 s/epoch × 300 epochs).
PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain \
    --fname=config/clip_pretrain.yaml \
    --optim.epochs=300 \
    --logging.wandb_group=auto_<tag>_<iter>_<hash>

# 2. Probe val Δr (the metric — always on val, never touch test).
PYTHONPATH=. uv run --group eeg python \
    eb_jepa/evaluation/clip_probe/probe_traintest.py \
    --checkpoint /path/to/latest.pth.tar \
    --config config/clip_pretrain.yaml \
    --eval-split val \
    --bootstrap 0 \
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

## The primary metric: probe val Δr

- **Metric**: mean Pearson r on R5 (val) minus the matched
  random-init baseline of the same encoder architecture. Averaged over
  the 12 scalar features in
  [`SCALAR_FEATURES_DEFAULT`](../../../../eb_jepa/evaluation/clip_probe/probe.py).
- **Protocol**: `probe_traintest.py --eval-split val --bootstrap 0`
  — fits Ridge on R1–R4 embeddings, evaluates on R5 embeddings. No
  bootstrap during the loop (saves time; use `--bootstrap 2000` only
  when writing up a final result). Deterministic.
- **Random baseline**: run the probe once with `--random-baseline` on
  the initial (untrained) encoder architecture. Cache this number as
  `rand_r`; every iteration reports `mean_r - rand_r`.
- **Selection rule**: higher is better. Iterations with val Δr strictly
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
saves checkpoints per `save_every`. The probe writes a JSON with the
per-feature Pearson r's, R², and (if bootstrap>0) 95 % CIs.

You extract the metric like:

```bash
uv run python -c "
import json, statistics as st
with open('probe_val.json') as f: d = json.load(f)
mean_r = st.mean(v['pearson_r'] for v in d['features'].values())
print(f'val_mean_r: {mean_r:.6f}')
"
```

Also log peak VRAM from the training log (grep `peak_vram_mb` or
`torch.cuda.max_memory_allocated`).

## Logging results

Append to `results.tsv` (tab-separated, headers on the first line):

```
commit	val_delta_r	val_mean_r	memory_gb	epochs	status	description
```

- `commit`: 7-char short hash of the iteration's git commit
- `val_delta_r`: `mean_r - rand_r`; e.g. `+0.02720`. Use `0.000000` for crashes.
- `val_mean_r`: raw mean Pearson r on val (before subtracting rand). Useful for provenance.
- `memory_gb`: peak GPU memory rounded to .1f. Use `0.0` for crashes.
- `epochs`: actual epochs completed (may be < config target if run OOM'd or was cut).
- `status`: `keep`, `discard`, or `crash`.
- `description`: short text of what this iteration tried. Concise, no commas.

Example:

```
commit	val_delta_r	val_mean_r	memory_gb	epochs	status	description
a1b2c3d	+0.02720	+0.0462	14.2	499	keep	baseline: fresh500_ep499 config
b2c3d4e	+0.04100	+0.0602	14.8	300	keep	depth 4 -> 8 encoder
c3d4e5f	+0.03800	+0.0572	14.2	300	discard	AdamW -> Lion
d4e5f6g	+0.00000	0.0	0.0	0	crash	encoder width 2048 (OOM)
```

Do not commit `results.tsv`. It's a per-branch scratchpad.

## The experiment loop

LOOP FOREVER:

1. Read git state: current branch/commit, current best val Δr.
2. Pick an experimental change; hack it into the editable files
   directly.
3. `git add -A && git commit -m "iter <N>: <short description>"`.
4. Submit / run the training job. Redirect all output to `run.log` —
   do NOT tee, do NOT let training output flood your context.
5. When training completes, run the probe:
   `PYTHONPATH=. uv run --group eeg python eb_jepa/evaluation/clip_probe/probe_traintest.py ... --eval-split val --bootstrap 0 --output probe_val.json`
6. Extract `val_mean_r` from the JSON; compute `val_delta_r =
   val_mean_r - rand_r`.
7. Append the row to `results.tsv`.
8. If `val_delta_r` strictly improved: keep the commit, advance the
   branch.
9. If equal or worse: `git reset --hard HEAD^` to revert to the
   pre-iteration commit.
10. Go to 1.

**Timeout**: each iteration ≤ 100 min wall (75 min train + 10 min
probe + 15 min slack). If it exceeds 100 min, kill it, log `crash`,
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

## Budget rationale

Why not 5 min like Karpathy's nanoGPT autoresearch? Because our
learning dynamics are qualitatively different:

- nanoGPT is **from-scratch language modeling on data-rich text**
  (millions of new tokens per iteration; steep clean loss curves;
  train loss IS the eval metric). 5 min is enough to see the
  trajectory shape.
- Ours is **contrastive alignment on a small dataset** with a
  **downstream probe metric** requiring separate embedding + Ridge
  fit. Learning curves are much flatter, small differences matter,
  and we've directly observed that:
  - At ~50 epochs the fresh500 recipe is at val R² ≈ +0.026 — barely
    above chance. Signal is dominated by noise.
  - ~130 epochs (~30 min) is the empirical floor where LR effects
    start resolving.
  - ~300 epochs (~75 min) is where the recipe reaches its plateau.
  - Longer than that *hurts* (see §3.2 in the from-checkpoint doc).
- **Probe eval isn't cheap**: embedding R1–R4 + R5 takes ~10 min for
  a 50 M-param encoder.

If you want faster iterations, drop budget to ~30 min (≈130 epochs) as
a *screening* pass — you'll catch large-effect changes (bigger LR,
different loss, encoder swap) but miss ±0.02 Δr differences. If
you're in "confirm a promising direction" mode, use the full 75 min.

## Known baselines (as of 2026-07-01)

| operating point | val Δr | notes |
|---|---|---|
| random-init encoder (matched arch, no training) | ~0.00 | anchor; the "rand_r" you subtract |
| `fresh500_ep499` (from-scratch, lr=1e-4, 499 ep) | ~+0.03 CV, +0.08 test | the number to beat |
| `warmstart_lr3e4_ep299` (REVE warm-start, lr=3e-4, 300 ep) | **+0.218** | the aspirational target |

To beat REVE-warmstart from scratch you need to close a ≈+0.19 val Δr
gap — 7× the current from-scratch result. That's very hard; a "moral
win" here is closing 50 % of the gap (val Δr ≈ +0.12).

## Known dead ends (do not re-run without a new theory)

From [scene_clip_from_checkpoint/RESULTS.md](../../scene_clip_from_checkpoint/RESULTS.md)
and [scene_clip_fromscratch/RESULTS.md](../RESULTS.md):

- **Extending schedule beyond 300 epochs at lr=1e-4** — 1000 ep is
  materially worse than 300 ep. Same effect probably applies to lr=3e-4.
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
