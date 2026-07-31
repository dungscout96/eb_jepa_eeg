# Demo — EEG → video retrieval

A static page showing what §3.10 of
[`../experiments/clip_pretraining/scene_clip_from_checkpoint/RESULTS.md`](../experiments/clip_pretraining/scene_clip_from_checkpoint/RESULTS.md)
reports as a table: given 2 s of EEG from someone watching *The Present*, the
model ranks candidate moments of the film, and the correct scene lands in the
top-10 **53.2 %** of the time on the held-out test split (1.9× chance).

Two stages. The heavy one runs on Delta (checkpoints and the HBN FIFs live
there); the page is built locally against `data/movies/The_Present.mp4`.

```
Delta                                  laptop
─────                                  ──────
export_retrieval_npz.py  ──scp──▶  build_demo.py ──▶ index.html + assets/
   ↓ validate                            ↑
retrieval.py --from-npz            The_Present.mp4
```

## 1. Export on Delta

Interactive allocation usually beats the batch queue:

```bash
srun --account=bbnv-delta-gpu --partition=gpuA40x4-interactive \
     --nodes=1 --gpus-per-node=1 --tasks=1 \
     --tasks-per-node=16 --cpus-per-task=1 --mem=40g --pty bash
cd /u/dtyoung/eb_jepa_eeg
python demo/_submit_export.py interactive   # prints the exact commands to paste
```

Or as a batch job: `uv run --group eeg python demo/_submit_export.py submit`.

Either way this runs, per split:

1. `export_retrieval_npz.py` — projects every EEG window of the split and its
   paired V-JEPA-2 target into the shared CLIP space, and ships raw EEG for 120
   showcase windows. ~25 MB per split.
2. `retrieval.py --from-npz` — the *unmodified* metric script, fed the export.
   Its JSON must reproduce RESULTS.md §3.10.

The export deliberately writes the `z_shared` / `is_vision` layout that
`retrieval.py --from-npz` already reads, so step 2 needs no new code and the
demo cannot drift away from the published metric.

Copy both files back:

```bash
mkdir -p demo/data
scp dtyoung@delta:/work/hdd/bbnv/dtyoung/eb_jepa/demo_export/demo_{test,val}.npz demo/data/
```

## 2. Build the page locally

```bash
PYTHONPATH=. .venv/bin/python demo/build_demo.py \
  --npz demo/data/demo_test.npz \
  --reference experiments/clip_pretraining/scene_clip_from_checkpoint/probe_results/retrieval_warmstart_lr3e4_retrain_jul22_ep299_test.json
```

`--reference` is not optional in spirit: `build_demo.py` recomputes Top-K with
the same helpers `retrieval.py` uses and **refuses to write the page** if the
result disagrees with the committed JSON. Without it you get a page whose
numbers nobody has checked.

Writes `demo/index.html` plus `demo/assets/` (one JPEG per candidate shown, one
GIF per correct answer — roughly 5 MB). Open `index.html` directly; it has no
external dependencies.

Useful flags: `--n-rows`, `--n-misses`, `--no-gifs`, `--out-dir`.

## 3. The scrubbing timeline (`timeline.html`)

A second page: walk one participant's recording end to end and watch what the
model retrieves at each 2 s step, at **time** granularity (101 candidates, one
per 2 s of film — the hardest level, and the only one where every position has
exactly one right answer).

It needs every window of the featured recordings, which the default stratified
showcase does not ship, so re-run the export with `--showcase-recordings`:

```bash
# on Delta
python demo/export_retrieval_npz.py --config $CK/config.yaml \
  --checkpoint $CK/latest.pth.tar --split test \
  --showcase-recordings 3,84,103 --output $OUT/demo_test_timeline.npz

# locally
PYTHONPATH=. .venv/bin/python demo/build_timeline.py \
  --npz demo/data/demo_test_timeline.npz
```

**Three participants, deliberately.** Per-participant Top-1 on test runs from
0.000 to 0.178 and the spread is real, not noise: split-half reliability
r = 0.81, and the best participant sits ~6 SD above a homogeneous-population
null. But the *selected* best is still inflated by winner's curse — recording 3's
own two halves give 0.216 and 0.140 — so the page ships the strongest, a typical
(median) and a weakest participant, and labels the percentile of each. Pick them
with the snippet in `build_timeline.py`'s docstring; **exclude dead recordings
first** (see below).

The EEG is embedded as base64 int8 at 50 Hz (~80 KB per participant), so the
whole page including the canvas rendering is ~0.4 MB plus the shared frame
stills. Scrubbing is an array lookup; nothing is computed in the browser.

## What's here

| file | role |
|---|---|
| `export_retrieval_npz.py` | cluster side: checkpoint → shared-space embeddings + showcase EEG |
| `_submit_export.py` | neurolab job / interactive command printer for the above |
| `build_demo.py` | local: recompute retrieval, gate on the committed numbers, pick rows, drive assets + render |
| `movie_assets.py` | pool entry → movie time span → still / GIF from the mp4 |
| `render.py` | EEG trace SVGs and the page itself |

Nearly all the model-side work is borrowed rather than reimplemented:
`build_dataset` / `load_encoder_state` from
[`../eb_jepa/evaluation/clip_probe/probe.py`](../eb_jepa/evaluation/clip_probe/probe.py),
the per-recording encode loop (`embed_windows`, `load_clip_head_state`) from
[`../experiments/clip_pretraining/cs_aligner/plot_modality_gap.py`](../experiments/clip_pretraining/cs_aligner/plot_modality_gap.py),
and the pool builders and Top-K math from
[`../eb_jepa/evaluation/clip_probe/retrieval.py`](../eb_jepa/evaluation/clip_probe/retrieval.py).

## Status

Both exports have been run (2026-07-30, Delta `gpuA40x4-interactive`, ~3 min for
test / ~5 min for val) and land in `demo/data/`. Both reproduce §3.10 exactly:

| split | M | scene e→v Top-1 / 5 / 10 | published |
|---|---:|---|---|
| test | 10,908 | 0.1492 / 0.3838 / 0.5324 | 0.149 / 0.384 / 0.532 |
| val | 29,593 | 0.2169 / 0.5257 / 0.6748 | 0.217 / 0.526 / 0.675 |

`index.html` is currently built from the **test** split.

### One thing the aggregate numbers hide

At the scene level, the model names **scene 0 — the 10 s black title card — as
its top-1 for 40.6 % of all test windows**, though scene 0 is the correct answer
only 5.0 % of the time (uniform: 2.9 %). A large share of first guesses collapses
onto one attractor. This doesn't change the Top-K figures, but it is not visible
in them; `build_demo.py` measures it (`top1_concentration`) and the page states
it. Blank scenes are excluded from being *asked about* in the showcase rows, but
they stay in the candidate pool, so you can see the behaviour.

## Provenance

Best checkpoint per RESULTS.md §3.10 (Delta job 20400451):

```
/work/hdd/bbnv/dtyoung/eb_jepa/scene_clip_from_checkpoint/jul22_warmstart_lr3e4_ep299/
    latest.pth.tar
    config.yaml          # REVE shape — use this, not config/clip_pretrain.yaml
```

The page headlines the **test** split. Val is ~15 points stronger; export both
and rebuild with `--npz demo/data/demo_val.npz` to see it.

## 4. The subject-averaged timeline (`timeline_group.html`)

The same scrub, but the query at each moment is the **average** of K
participants' embeddings instead of one person's. The candidate pool is
untouched, so chance stays `K/101` and every K is directly comparable — you
switch K and watch both the trace settle and the guesses improve. This is
`snr_scaling` §2.6 as an object you can drag.

Retrieval needs no cluster work (`demo_test.npz` already holds `z_eeg` for every
recording). Only the displayed *waveform* does — the subject-averaged raw EEG at
each K:

```bash
# on Delta — CHANS and ORDER are printed by build_timeline_group.py's picker
PYTHONPATH=. uv run --group eeg python export_avg.py \
  $CK/config.yaml test "<chans>" "<nested order>" 1,2,4,8,16,32,64,106 \
  $OUT/avg_eeg_test.npz

# locally
PYTHONPATH=. .venv/bin/python demo/build_timeline_group.py \
  --npz demo/data/demo_test.npz --avg-npz demo/data/avg_eeg_test.npz \
  --amp demo/data/amp_test.json
```

Time-level, test split, dead recordings excluded (106 live participants):

| K | Top-1 | Top-5 | Top-10 |
|---:|---|---|---|
| 1 | 0.055 ± 0.046 | 0.186 ± 0.128 | 0.292 ± 0.169 |
| 4 | 0.094 ± 0.038 | 0.299 ± 0.090 | 0.460 ± 0.118 |
| 16 | 0.115 ± 0.036 | 0.380 ± 0.074 | 0.575 ± 0.073 |
| 32 | 0.132 ± 0.027 | 0.413 ± 0.046 | 0.619 ± 0.045 |
| 106 | 0.119 | 0.436 | 0.634 |

Mean ± sd over 40 random participant sets per K. **Top-10 more than doubles;
Top-1 flattens around K ≈ 32** — past that the remaining errors are not the kind
more brains fix. The waveform tells the same story: mean |z| falls 0.36 (K=1) to
0.081 (K=106), a factor of ~4.4 against the 1/√106 ≈ 1/10.3 you would see for
pure noise, implying a single-trial reliability of roughly 0.06 — the range
CorrCA measures in `snr_scaling` §2.

**Two disclosures.** With only 101 queries, a single set of K participants
carries ~3 points of binomial noise at Top-1, which is why the page reports the
40-draw mean beside each tab's own number. And the nested participant order is
*chosen*: it starts from the median-accuracy participant, and the ordering of
the rest was picked (seed 6 of 40 tried) to **minimise** deviation from those
40-draw means. That selects for representativeness, not for a flattering result
— an arbitrary seed can put an unlucky participant at K=1 and show chance-level
accuracy there, which misrepresents K=1 just as badly in the other direction.

## Dead recordings — check before featuring a participant

Building the timeline surfaced this: **2 of 108 test recordings and 5 of 293 val
recordings carry no usable EEG at all.** Every channel sits below 1 % of the
global median channel sd; `51-raw.fif` (test index 50) peaks at sd = 0.000713
with 43 channels bitwise constant. They are not hard participants, they are dead
files, and the first draft of this demo labelled one of them "weakest
participant" — which would have been wrong.

Effect on the published numbers is small but nonzero; dropping them gives:

| split | scene e→v Top-1/5/10 (all) | (dead dropped) |
|---|---|---|
| test | 0.1492 / 0.3838 / 0.5324 | 0.1501 / 0.3859 / 0.5348 |
| val | 0.2169 / 0.5257 / 0.6748 | 0.2190 / 0.5308 / 0.6804 |

Census in `demo/data/amp_{test,val}.json` (`rel_amp`, `frac_dead` per
recording); dead means `rel_amp < 0.01`. Filter on it before choosing
`--showcase-recordings`.

One trap worth recording: the first census normalised each recording by the
median of *its own* positive channel sds. For a dead recording that median is
itself ~0, so the threshold collapsed and the count came out at 43 instead of
"everything". Reference the **across-recording** median, never the
within-recording one.

## Things that will bite

- **`--t-bucket-s 0.5`.** The `retrieval.py` CLI defaults to 0.1. Every
  published number used 0.5; at 0.1 the time pool is ~406 entries instead of
  101 and nothing lines up. `build_demo.py` hardcodes 0.5.
- **Dataset construction must match `retrieval.py`, not training.** The export
  calls `probe.build_dataset` — the same call `retrieval.py` makes — which
  leaves the V-JEPA-2 targets raw (no `recipe_mode`, so no mean-centering).
  That differs from what scene-CLIP trained against, but it is how the published
  §3.10 numbers were computed, and matching it is what lets the gate assert
  against the committed JSON. Switching to `plot_modality_gap.build_dataset`
  (which does mean-center) changes every number and fails the gate.
- **`ffmpeg`.** There is no system ffmpeg on this machine and the shell exports
  `IMAGEIO_FFMPEG_EXE` pointing at a path that no longer exists.
  `movie_assets.py` falls back to the binary `imageio_ffmpeg` ships.
- **`data/movies/` is gitignored.** The mp4 has to be present locally to build;
  the generated `assets/` are small and can be committed.
- **This is EEG→video only.** The reverse direction is much weaker on this
  checkpoint (Top-1 ≈ 2× chance, Top-10 *below* chance). The page says so; keep
  it that way.
