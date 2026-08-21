# RESULTS_epoch_curve — does the checkpoint-selection rule matter?

Every number here is generated from the raw JSONs in `raw_results/` by
`src/write_results_epoch_curve.py`. Re-run it after any change to the curve.

## Verdict

**The rule barely matters for anything the paper claims, and the one place it
does matter, it overturns a claim.**

1. The two selectors name the same checkpoint on only **3 of 28** cells
   (median disagreement 75 epochs, max 250) -- and the score barely
   responds. The probe-selected checkpoint beats fixed epoch 325 by a mean of
   **+0.0036 r**, at most +0.0120, and at S>=1000 at most +0.0013.
   A flat optimum is the whole picture: the epoch is weakly determined, which is
   exactly why two reasonable rules disagree about it while agreeing about the result.

2. **RETRACTS the high-S budget reading.** The appendix notes that
   8 of 28 cells put the smoothed AUC argmax at epoch 399 -- the last
   epoch searched -- and reads that as the 4400-step budget being close to
   binding at large S. Under the probe selector only **2 of 28** cells
   put their optimum at the last saved checkpoint, and only **1**
   (`e04_s1863_a101_nd`) is among the AUC's 8.
   The rest land well inside the budget. The ceiling was mostly a property of
   the selection metric, not of the training runs.

## What this is not

**Not a protocol change.** The paper still reports at fixed epoch 325. This is a
robustness check, and the already-measured protocol spread (mean |delta| 0.0019 r,
max 0.0064 over 40 probe cells, `RESULTS_model_scaling.md` sections 1-4 vs 5-8)
bounds what any selector could be worth.

**The gain column is an upper bound, not an estimate.** `r@prb - r@325` cannot be
negative: epoch 325 is one of the 15 candidates the argmax ranges over, so the
winner is at least as good as it by construction. Read it as bounding what
selection could buy. The unbiased number is the test-split comparison already in
`RESULTS_model_scaling.md`.

**e04 ONLY, and not by convenience.** Selection reads the val split (R5), which is
disjoint from every e04 cohort because e04 pretrains on [R1-R4, R7-R10]. The
`e05_addval` and `e05_fromscratch` arms have R5 *in* their pretraining pool, so
there this would measure training-set fit. Those arms have no held-out selection
data at all and stay pinned to a fixed epoch -- the same constraint
`config/config_probe_TP_headfit_R5.yaml` documents for the zero-overlap head fit.

## Method

- **Selection set.** 80 recordings drawn from the val split
  (R5, 293 available), seed 42: 60 fit the ridge head,
  20 score it, split BY RECORDING so the scored half is unseen subjects.
  8080 windows total. R6 (test) is never read.
- **Ridge alpha fixed at 3162** across every cell and epoch, so the curve
  moves because the encoder moved, not because the head's regularisation was
  re-tuned under it. See the calibration section below.
- **Every saved checkpoint**: 15 per cell (epoch 25..375), 28 cells, 420 in total.
- **Two readouts from one forward pass**: the 12-feature ridge r, and time/shot/scene
  retrieval through the trained CLIP head. Retrieval fits nothing and is never
  selected on, so it is the check that probe selection is not merely selecting the probe.

### Why it is affordable

The EEG windows, the regression targets and the V-JEPA-2 vectors do not depend on
the checkpoint -- only the forward pass does. Reading them once and re-encoding
only that is what makes a 420-checkpoint curve a few GPU-hours instead of a few
hundred.

| | per checkpoint | 28 cells x 15 epochs |
|---|---|---|
| `probe_traintest.py` per checkpoint | ~199k windows, 1971 FIF reads, ~40 min | ~280 GPU-hours |
| this selector | ~8k windows, 0 FIF reads, ~30 s | **~3.5 GPU-hours** |

Measured: 7 Delta jobs of 4 cells each, 30:46--31:37 elapsed, 420/420 checkpoints,
no NaN. One-time cache build 33 s / 1.67 GB per job.

### The alpha, and a trap

`RidgeCV` on this fit picks alpha in **1000--10000** -- all 36 feature-draws,
across epochs 100, 200, 375 of `e04_s701_a101_nd_d22` -- with the epoch-to-epoch
drift inside that band rather than across it. The median, 10^3.5, is what the sweep uses.

The trap: the `probe_traintest.py` artifacts record alphas of ~30--100, and reusing
those is wrong by 1.5 orders of magnitude. Those fits see 188k rows; this one sees
60 recordings' worth. The ridge optimum scales with n, so the small head-fit
pool that makes the selector cheap is exactly what makes it need a stronger prior.

### Does a 60-recording head fit rank cells like the real thing?

It has to, or the selector is measuring its own small pool. Against the published
full-pipeline numbers (1863-recording head, R6 test split) at the same epoch 325,
across all 28 cells: **Pearson r = 0.9875, Spearman = 0.9814**.
Absolute values differ -- a 60-recording head scores lower in general -- but the
ordering, which is all a selector uses, is preserved.

## 1. Three selectors, every cell

`auc` = the incumbent smoothed-AUC choice; `prb` = probe argmax; `retr` = scene-retrieval
argmax. `r@*` are on the 20 scoring recordings, NOT the reported test number.

| S | draw | auc | prb | retr | \|prb-auc\| | r@prb | r@auc | r@325 | prb-325 |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 11 | 75 | 150 | 25 | 75 | 0.1187 | 0.1170 | 0.1139 | +0.0049 |
| 10 | 22 | 350 | 100 | 150 | 250 | 0.1331 | 0.1299 | 0.1304 | +0.0027 |
| 10 | 33 | 200 | 150 | 150 | 50 | 0.1394 | 0.1380 | 0.1353 | +0.0041 |
| 20 | 11 | 300 | 100 | 125 | 200 | 0.1521 | 0.1420 | 0.1420 | +0.0101 |
| 20 | 22 | 225 | 175 | 125 | 50 | 0.1489 | 0.1459 | 0.1445 | +0.0044 |
| 20 | 33 | 300 | 150 | 125 | 150 | 0.1537 | 0.1527 | 0.1526 | +0.0011 |
| 50 | 11 | 175 | 275 | 200 | 100 | 0.1739 | 0.1717 | 0.1717 | +0.0022 |
| 50 | 22 | 375 | 200 | 250 | 175 | 0.1700 | 0.1670 | 0.1684 | +0.0016 |
| 50 | 33 | 300 | 275 | 25 | 25 | 0.1624 | 0.1600 | 0.1619 | +0.0004 |
| 100 | 11 | 375 | 125 | 50 | 250 | 0.1946 | 0.1918 | 0.1921 | +0.0024 |
| 100 | 22 | 375 | 225 | 100 | 150 | 0.2037 | 0.2007 | 0.2003 | +0.0034 |
| 100 | 33 | 200 | 75 | 225 | 125 | 0.1985 | 0.1888 | 0.1924 | +0.0061 |
| 200 | 11 | 200 | 275 | 100 | 75 | 0.2489 | 0.2395 | 0.2466 | +0.0023 |
| 200 | 22 | 175 | 250 | 100 | 75 | 0.2339 | 0.2253 | 0.2255 | +0.0084 |
| 200 | 33 | 225 | 225 | 25 | 0 | 0.2255 | 0.2255 | 0.2236 | +0.0019 |
| 400 | 11 | 225 | 225 | 225 | 0 | 0.2577 | 0.2577 | 0.2457 | +0.0120 |
| 400 | 22 | 225 | 125 | 225 | 100 | 0.2733 | 0.2678 | 0.2630 | +0.0103 |
| 400 | 33 | 250 | 225 | 275 | 25 | 0.2757 | 0.2733 | 0.2663 | +0.0094 |
| 701 | 11 | 200 | 250 | 225 | 50 | 0.3050 | 0.3023 | 0.3018 | +0.0032 |
| 701 | 22 | 250 | 225 | 200 | 25 | 0.2920 | 0.2880 | 0.2898 | +0.0021 |
| 701 | 33 | 375 | 250 | 250 | 125 | 0.2853 | 0.2806 | 0.2806 | +0.0047 |
| 1000 | 11 | 375 | 250 | 275 | 125 | 0.3182 | 0.3176 | 0.3178 | +0.0003 |
| 1000 | 22 | 200 | 300 | 325 | 100 | 0.3254 | 0.3128 | 0.3253 | +0.0002 |
| 1000 | 33 | 225 | 375 | 275 | 150 | 0.3270 | 0.3178 | 0.3258 | +0.0013 |
| 1400 | 11 | 375 | 325 | 325 | 50 | 0.3238 | 0.3231 | 0.3238 | +0.0000 |
| 1400 | 22 | 350 | 300 | 350 | 50 | 0.3158 | 0.3138 | 0.3148 | +0.0010 |
| 1400 | 33 | 375 | 325 | 300 | 50 | 0.3240 | 0.3219 | 0.3240 | +0.0000 |
| 1863 | -- | 375 | 375 | 275 | 0 | 0.2740 | 0.2740 | 0.2731 | +0.0010 |

## 2. The budget ceiling does not replicate

| cell | S | AUC argmax | probe argmax | last saved |
|---|---|---|---|---|
| `e04_s50_a101_nd_d22` | 50 | 399 | 200 | 375 |
| `e04_s100_a101_nd_d11` | 100 | 399 | 125 | 375 |
| `e04_s100_a101_nd_d22` | 100 | 399 | 225 | 375 |
| `e04_s701_a101_nd_d33` | 701 | 399 | 250 | 375 |
| `e04_s1000_a101_nd_d11` | 1000 | 399 | 250 | 375 |
| `e04_s1400_a101_nd_d11` | 1400 | 399 | 325 | 375 |
| `e04_s1400_a101_nd_d33` | 1400 | 399 | 325 | 375 |
| `e04_s1863_a101_nd` | 1863 | 399 | 375 | 375 |

All 8 cells whose smoothed AUC argmax pinned at 399. Under the probe curve
7 of them move well inside the budget (epochs 125--325).

The conservative reading the appendix draws from this -- *if anything is
under-trained it is the high-S end, which flattens the measured subject slope
rather than inflating it* -- should therefore be claimed only for the cells where
BOTH selectors agree the curve is still climbing at the end: `e04_s1863_a101_nd`.

## 3. Probe vs retrieval, and why 4/28 is not the story

The probe and scene-retrieval argmaxes coincide on 4 of 28 cells
(median |delta| 50 epochs, max 250). Taken flat that is
unreassuring, but it is not one number:

- At **S<=200** (15 cells) scene retrieval sits near its floor, so its argmax is
  chasing noise -- several cells put it at epoch 25, where the probe is still
  climbing steeply. Median |delta| there is 75 epochs.
- At **S>200** (13 cells), where retrieval is well clear of chance, median
  |delta| is 25 epochs.

So it is a cross-check at high S and noise at low S. Do not quote the headline
agreement rate as if it were a single measurement.

## 4. Per-S summary

| S | n | probe epochs | mean r@prb | mean r@325 | mean gain |
|---|---|---|---|---|---|
| 10 | 3 | 100, 150, 150 | 0.1304 | 0.1265 | +0.0039 |
| 20 | 3 | 100, 150, 175 | 0.1516 | 0.1464 | +0.0052 |
| 50 | 3 | 200, 275, 275 | 0.1688 | 0.1673 | +0.0014 |
| 100 | 3 | 75, 125, 225 | 0.1989 | 0.1949 | +0.0040 |
| 200 | 3 | 225, 250, 275 | 0.2361 | 0.2319 | +0.0042 |
| 400 | 3 | 125, 225, 225 | 0.2689 | 0.2583 | +0.0106 |
| 701 | 3 | 225, 250, 250 | 0.2941 | 0.2908 | +0.0033 |
| 1000 | 3 | 250, 300, 375 | 0.3235 | 0.3230 | +0.0006 |
| 1400 | 3 | 300, 325, 325 | 0.3212 | 0.3209 | +0.0003 |
| 1863 | 1 | 375 | 0.2740 | 0.2731 | +0.0010 |

The largest per-S gain is at **S=400**, the same place the full-pipeline
protocol comparison puts its largest difference (`RESULTS_model_scaling.md`:
+0.0064 at S=400 on the test split). Two different selectors, two different
splits, two different head-fit pools -- and the same cell group is the sensitive
one, which is a stronger statement than either measurement alone.

## Provenance

| what | where |
|---|---|
| selector | `eb_jepa/evaluation/clip_probe/epoch_curve.py` |
| submitter | `submit/_submit_epoch_curve.py` (`dry`/`submit`/`smoke`/`merge`/`verify`) |
| comparison | `src/analyse_epoch_curve.py` (writes `e04_selection_probe.json`) |
| this file | `src/write_results_epoch_curve.py` |
| curve | `raw_results/e04_epoch_curve.json` (28 cells x 15 epochs) |
| alpha calibration | `raw_results/e04_epochcurve_alpha_calibration.json` |
| probe selection | `e04_selection_probe.json`, same shape as `e04_selection.json` |
| tests | `tests/unit/test_epoch_curve.py` |

`e04_selection_probe.json` is drop-in for the submitters that read
`e04_selection.json`. It records `snap_distance: 0` for every cell, because the
probe curve is only ever evaluated AT saved checkpoints -- the snapping step that
the AUC selector needs does not exist here.
