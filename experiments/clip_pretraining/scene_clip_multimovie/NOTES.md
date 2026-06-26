# scene_clip multi-movie — Experiment Plan

Sibling to [`../scene_clip_fromscratch/`](../scene_clip_fromscratch/) and
[`../scene_clip_from_checkpoint/`](../scene_clip_from_checkpoint/). Tests
whether **training the `scene_clip` recipe on two HBN movies simultaneously**
(*ThePresent* + *DespicableMe*) yields better encoders than single-movie
training, holding all other recipe knobs at their winning values from the
single-movie sweep.

## Hypothesis

Three possible outcomes — the run will tell us which:

1. **Multi-movie helps** (positive transfer): more diverse vision targets +
   subject pools → encoder learns a more general EEG→vision representation
   → higher Δr on both ThePresent and DespicableMe probes than single-movie
   training on either alone.
2. **Multi-movie hurts** (negative transfer / interference): the two movies
   share a teacher (V-JEPA-2) but the EEG distribution shifts between subject
   pools / recording sessions. Forcing one encoder to align to both could
   degrade per-movie alignment.
3. **Multi-movie is neutral**: per-movie probes look similar to single-movie
   baselines.

Prior on outcomes: contrastive learning *generally* benefits from larger,
more diverse training sets, so #1 is the default expectation. But the
EEG-side variance from a second subject pool (R1-R4 watching DespicableMe is
the same release set as R1-R4 watching ThePresent, but the per-recording
EEG is different) could either help (regularization) or hurt (signal
dilution).

## What's new vs single-movie

Three pieces of plumbing that exist already and need to behave correctly
under multi-movie:

1. **Per-task artifacts**: each movie has its own
   `vjepa2_recipe.npz` (global_mean / shot_means / scene_id_per_shot) and
   scene_map CSV. The dataset loads both at init time
   (`_vjepa2_recipes[task]`). Already tested via single-task code paths.
2. **Scene-ID namespacing**: scene IDs are namespaced as
   `task_idx * 100_000 + local_scene_id` so positives from different movies
   never collide. ThePresent → 0..35, DespicableMe → 100_000..100_028. Code
   in [`hbn.py`](../../../eb_jepa/datasets/hbn.py) `_SCENE_NAMESPACE`.
3. **Per-task embedding replacement**: shot-mean replacement and global mean
   centering happen per-recording using each recording's task lookup, so a
   ThePresent recording gets ThePresent's shot_means and a DespicableMe
   recording gets DespicableMe's. Already correct.

## Known minor caveat

The SceneCLIPPretrain exclusion mask uses `|Δt|<temporal_buffer_s`
**regardless of movie**. So a ThePresent window at t=5s and a DespicableMe
window at t=5s would be erroneously excluded from negatives (both are
cross-scene but get masked due to identical `t_start`). Estimated impact:
~1-2% of pairs in a balanced batch. Small enough to accept for v1; can fix
later by also passing `movie_ids` and adding `same_movie` to the exclusion
mask.

## v1 config (this run, job 19711753)

Inherits the winning single-movie recipe from `scene_clip_from_checkpoint`:

| field | value | note |
|---|---|---|
| `meta.encoder_init_from` | REVE-base | the winning warm-start |
| `data.task` | **[ThePresent, DespicableMe]** | ← the only intentional change |
| `loss.target_kind` | per_window | recipe winner |
| `loss.proj_dim` | 512 | matches REVE embed_dim (recipe winner) |
| `loss.vision_passthrough` | false | symmetric projection |
| `loss.mean_center` | true | recipe default |
| `loss.temporal_buffer_s` | 2.0 | recipe default |
| `optim.optimizer` | adamw | recipe winner |
| `optim.weight_decay` | 0.05 | recipe winner |
| `optim.lr` | 1e-5 | recipe winner |
| `optim.warmup_epochs` | 10 | recipe winner |
| `optim.epochs` | 300 | saturation point per single-movie sweep |
| `data.window_size_seconds` | 2 | recipe default |
| `data.batch_size` | 64 | recipe default |
| wandb | `clip_reve_multimovie_v1_pw300` | |

Wall ETA: ~2.5 h (2× the single-movie 1:09 since dataset has ~2× recordings).

## Comparison endpoints

After training, probe with two protocols on each movie separately:

- **probe_traintest on R6/ThePresent** (CLIP-style train→test, Pearson r + bootstrap CI):
  compare against `scene_clip_from_checkpoint/probe_tt_warmstart_v2_ep299_boot`
  (single-movie ThePresent baseline, mean Δr = +0.131).
- **probe_traintest on R6/DespicableMe**: no single-movie DespicableMe
  baseline yet — would need either a single-movie DespicableMe warm-start run
  for comparison, or accept comparing against the random_REVE_shape baseline
  (which is movie-independent on the random encoder side, but the test set
  is different).
- **CV-on-val** for quick reads on both movies separately.

The interesting per-feature view: the high-level features (motion, depth,
faces, narrative) that REVE was *weak* on are where the warm-start recipe
adds the most signal. Multi-movie should show whether two movies' worth of
visual diversity makes that signal grow further or saturate.

## Open questions / future variants

1. **Multi-movie with `from_scratch`**: would training from random init on
   2 movies make up the ground that REVE warm-start gained on 1 movie?
2. **Asymmetric movie weights**: what if DespicableMe (170 s, smaller) is
   over-represented per epoch — does balancing help/hurt?
3. **Fix the cross-movie temporal-buffer leak**: add `movie_ids` to the
   loss exclusion mask so cross-movie pairs are never excluded for `|Δt|<2s`.
   Required if multi-movie becomes the primary recipe.
4. **More movies**: HBN has additional movie tasks (RestingState etc., but
   most aren't movies). DespicableMe was the second-most-complete movie
   dataset.
