# RESULTS_epoch_curve — does the checkpoint-selection rule matter?

Every number here is generated from the raw JSONs in `raw_results/` by
`src/write_results_epoch_curve.py`. Re-run it after any change to the curve.

## Verdict

Three arms scanned, 1260 checkpoints. **Whether the selection rule matters
depends on the initialisation, which is itself the finding.**

0. **The 4400-step budget was binding for the from-scratch arm above S=1000,
   and not at all below S=400.** Measured against the same arm trained twice
   as long: the cost is exactly 0.0000 r up to S=400 and +0.012 to +0.016
   above S=1000. Section 0b. An earlier reading of the 400-epoch curves as
   'flat at the top, so converged' was wrong -- they were truncated.

0a. **From scratch, the optimal epoch tracks cohort size; warm-started it
   barely moves.** The from-scratch arms peak near epoch 25 at S=10-20 and at
   the last saved checkpoint by S=1400, so a fixed 325 costs them
   ~0.011-0.014 r at S<=200. The warm-started arm costs ~0.004 over the same
   range. Section 0 below, and it is the reason points 1-2 are e04-only
   statements rather than statements about the recipe.

1. **On e04 (warm start)** the two selectors name the same checkpoint on only
   **3 of 28** cells
   (median disagreement 75 epochs, max 250) -- and the score barely
   responds. The probe-selected checkpoint beats fixed epoch 325 by a mean of
   **+0.0036 r**, at most +0.0120, and at S>=1000 at most +0.0013.
   A flat optimum is the whole picture: the epoch is weakly determined, which is
   exactly why two reasonable rules disagree about it while agreeing about the result.

2. **RETRACTS the high-S budget reading, on e04.** The appendix notes that
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

**This file is the e04 arm. It is not the only scannable arm.** Selection reads
the val split (R5), so the method works for any arm whose pretraining pool
excludes R5 -- which is a fact about the pool, not about the arm's name:

| arm | init | depth | pool | scannable |
|---|---|---|---|---|
| `e04_reve_scaling` | warm (REVE) | 22 | 1863 | yes — **this file** |
| `e05_random_scaling` (`_nd`, `_ep800`) | from scratch | 22 | 1863 | yes — not yet run |
| `e03_scaling` | from scratch | 12 | 1863 | yes — not yet run |
| `e05_addval` | warm | 22 | 2156 | no, R5 in train |
| `e05_fromscratch` | from scratch | 22 | 2156 | no, R5 in train |

Note the last two rows: `e05_fromscratch` and `e05_random_scaling` are both
from-scratch depth-22 arms and land on opposite sides of the line, because they
differ in pool rather than in initialisation. For the 2156-pool arms the only
unseen split is R6 -- the reported test split -- so they have no held-out
selection data at all, the same constraint
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

## 0. Across arms: the optimal epoch depends on S only when training from scratch

The e04 result below -- epoch weakly determined, score barely affected -- is
**a property of the warm start, not of the recipe.** Scanning the two
from-scratch arms on the identical selection set gives a different shape:
their optimum moves monotonically with cohort size, from epoch ~25 at S=10
to the last saved checkpoint at S=1400.

| S | e04 (warm st, d22) | e05rand (from sc, d22) | e03 (from sc, d12) |
|---|---|---|---|
| 10 | 100/150/150 (+0.0039) | 25/25/75 (+0.0089) | 25/50/150 (+0.0108) |
| 20 | 100/150/175 (+0.0052) | 25/25/25 (+0.0112) | 25/25/200 (+0.0055) |
| 50 | 200/275/275 (+0.0014) | 50/50/125 (+0.0111) | 50/250/300 (+0.0075) |
| 100 | 75/125/225 (+0.0040) | 75/100/125 (+0.0180) | 75/75/75 (+0.0167) |
| 200 | 225/250/275 (+0.0042) | 150/175/250 (+0.0183) | 100/125/150 (+0.0161) |
| 400 | 125/225/225 (+0.0106) | 250/300/350 (+0.0027) | 225/250/275 (+0.0064) |
| 701 | 225/250/250 (+0.0033) | 350/375/375 (+0.0008) | 350/375/375 (+0.0018) |
| 1000 | 250/300/375 (+0.0006) | 350/375/375 (+0.0013) | 325/375/375 (+0.0013) |
| 1400 | 300/325/325 (+0.0003) | 375/375/375 (+0.0016) | 350/375/375 (+0.0017) |
| 1863 | 375 (+0.0010) | 325 (+0.0000) | 375 (+0.0005) |

Each cell is the probe-selected epochs of the 3 draws, then the mean gain
over fixed epoch 325 on the selection set. Summarised:

| arm | median optimum, S<=200 | median optimum, S>=701 | mean gain S<=200 | mean gain S>=701 |
|---|---|---|---|---|
| `e04` (warm start (REVE), d22) | 175 | 300 | +0.0037 | +0.0014 |
| `e05rand` (from scratch, d22) | 75 | 375 | +0.0135 | +0.0011 |
| `e03` (from scratch, d12) | 75 | 375 | +0.0113 | +0.0015 |

**Warm-starting compresses the epoch dependence.** From scratch the optimum
travels 75 -> 375 across the ladder and a fixed 325 costs ~0.011--0.014 r at
S<=200; warm-started it travels 175 -> 300 and costs ~0.004. Whatever REVE
pretraining supplies, one thing it supplies is insensitivity to when you stop.

**Is the early optimum real, or the winner's curse?** These gains cannot be
negative, so a noisy curve manufactures a positive one. The evidence that it
is real is the CONCENTRATION of the argmax across independent draws: under
noise the three draws of a cell would scatter, and instead the from-scratch
arm puts all three at epoch 25 at S=20 and all three at 375 at S=1400. A
winner's curse does not reproduce across seeds.

### What this does to the initialisation claim

The paper reports the warm start as worth a draw-separated $+0.016$ to
$+0.031\,r$ at $S=10$--$20$, with both arms evaluated at a fixed epoch 325.
But a fixed 325 is not neutral between them: it sits near the warm arm's
optimum and far past the from-scratch arm's. The part of that gap which is
protocol rather than initialisation is the DIFFERENCE of the two arms'
gains, not the from-scratch arm's gain alone:

| S | warm gains | scratch gains | differential |
|---|---|---|---|
| 10 | +0.0039 | +0.0089 | +0.0050 |
| 20 | +0.0052 | +0.0112 | +0.0060 |
| 50 | +0.0014 | +0.0111 | +0.0096 |
| 100 | +0.0040 | +0.0180 | +0.0140 |
| 200 | +0.0042 | +0.0183 | +0.0141 |
| 400 | +0.0106 | +0.0027 | -0.0078 |
| 701 | +0.0033 | +0.0008 | -0.0025 |
| 1000 | +0.0006 | +0.0013 | +0.0007 |
| 1400 | +0.0003 | +0.0016 | +0.0013 |
| 1863 | +0.0010 | +0.0000 | -0.0010 |

At $S=10$--$20$ the differential is +0.0055, i.e.\ per-cell selection
would close at most 34 % of the low end of the claimed gap
and 18 % of the high end. **The initialisation result
survives, and is overstated at small cohorts.** It peaks at $S=100$--$200$,
where the differential reaches +0.0141.

Three things keep this a caveat rather than a correction. The arms measured
here draw from the 1863 pool while the paper's initialisation comparison
uses the 2156-pool arms; the gains are measured on the R5 selection set
with a 60-recording head, not on R6 with 1863; and both are upper bounds,
for the reason above. Settling it needs the 2156-pool arms re-evaluated at
per-cell epochs, which their pool makes impossible without retraining.

## 0b. What the step budget cost, measured rather than inferred

The same from-scratch depth-22 arm exists at **double the step budget**
(800 epochs x 703 = 8800 steps against 4400), 31 checkpoints per cell out to
epoch 775. That turns the budget question from an inference about where an
argmax lands into a subtraction, because a curve truncated at 400 and a curve
converged by 400 are indistinguishable from inside 400.

| S | probe argmax (800 ep) | best reachable <=400 | best <=800 | budget cost |
|---|---|---|---|---|
| 10 | 25/25/50 | 0.1105 | 0.1105 | +0.0000 |
| 20 | 25/25/25 | 0.1237 | 0.1237 | +0.0000 |
| 50 | 50/150/375 | 0.1390 | 0.1390 | +0.0000 |
| 100 | 75/100/400 | 0.1624 | 0.1624 | +0.0000 |
| 200 | 200/250/400 | 0.1905 | 0.1905 | +0.0000 |
| 400 | 300/350/400 | 0.2232 | 0.2232 | +0.0000 |
| 701 | 425/450/675 | 0.2448 | 0.2462 | +0.0014 |
| 1000 | 575/575/750 | 0.2432 | 0.2549 | +0.0117 |
| 1400 | 600/725/750 | 0.2455 | 0.2575 | +0.0120 |
| 1863 | 775 | 0.2514 | 0.2669 | +0.0156 |

**The 4400-step budget cost this arm +0.0000 r at S<=400 and +0.0131 at
S>=1000.** Probe argmax at the last saved checkpoint falls from 7 of 28 in the
400-epoch arm to 1 of 28 here -- the holdout being the full-pool
S=1863 cell, which wants more than 8800 steps.

**The optimum grows with cohort size**: epoch 25 at S=10--20, 775 at S=1863.
Bigger cohorts need more steps. That is a scaling statement in its own right,
and one the 400-epoch arm could not have produced -- inside that budget its
high-S curves look flat at the end because they stop, not because they settle.

**Consequence for the subject-scaling slope.** The from-scratch arm's high-S
points sit ~0.012 r low, so its measured slope is understated and the
initialisation gap at high S is overstated by roughly that much. Note this is
the same direction as the small-cohort effect in section 0, reached by a
different mechanism: there the fixed epoch is too late for the cell, here the
budget ends before the cell is done. The from-scratch arm is disadvantaged by
the protocol at BOTH ends of the ladder, for unrelated reasons.

## 1. Three selectors, every cell (e04)

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
