# RESULTS_cbramod — E1.2: an HBN-free warm start on the subject-scaling axis

Generated 2026-09-22 by `src/write_results_cbramod.py` from `e12cb_nested.json` and `e12cbr_nested.json`. Re-run the script after any change to the sweep; do not edit numbers by hand.

Two arms of CBraMod (TUEG-pretrained, 4.9M params, 200-d; never saw HBN) on E0.4's split — pool R1–R4 + R7–R10, val R5, test R6 — at fixed epoch 325. `e12cb` warm-starts from the released weights; `e12cbr` is the same shape, recipe, cohorts and seed from random init. Δr² is against an UNTRAINED encoder of the CBraMod shape; it is not on the same axis as the REVE arms' Δr² and must not be plotted with them. Design and decision: [`PLAN_cbramod.md`](PLAN_cbramod.md).

## 1. Level — warm / random Δr² at matched S (12-feature CV probe, val)

Null mean r² (untrained CBraMod shape): warm-arm file 0.013419, random-arm file 0.013419 — these must be identical (same measurement, copied under both prefixes).

| S | warm Δr² | random Δr² | warm / random |
|---:|---|---|---:|
| 50 | 0.00027 ± 0.00167 (n=3) | -0.00168 ± 0.00072 (n=3) | -0.16× |
| 200 | 0.00992 ± 0.00159 (n=3) | 0.00114 ± 0.00016 (n=3) | 8.68× |
| 701 | 0.01374 ± 0.00142 (n=3) | 0.00338 ± 0.00042 (n=3) | 4.07× |
| 1400 | 0.01376 ± 0.00062 (n=3) | 0.00362 ± 0.00006 (n=3) | 3.80× |
| 1863 | 0.01271 (n=1) | 0.00364 (n=1) | 3.50× |

Ratio across the axis: min -0.16×, max 8.68×, median 3.80× over 5 S values. E0.4 vs E0.5 (REVE warm vs random, depth 22) measured ~1.7× at every S.

## 2. Slope — subject-scaling exponents per arm (Δr²)

| arm | fitted d log Δr² / d log S | pooled between-draw sd |
|---|---:|---:|
| warm | 0.973 | 0.00139 |
| random | — | 0.00042 |

Local exponents between adjacent S (and the 701 → 1863 step the claim turns on):

| step | warm exponent | warm Δ in sd | random exponent | random Δ in sd |
|---|---:|---:|---:|---:|
| 50 → 200 | 2.587 | 6.9 | — | 6.7 |
| 200 → 701 | 0.260 | 2.8 | 0.864 | 5.3 |
| 701 → 1400 | 0.002 | 0.0 | 0.100 | 0.6 |
| 701 → 1863 | -0.080 | -0.7 | 0.076 | 0.6 |
| 1400 → 1863 | -0.277 | -0.8 | 0.017 | 0.0 |

For reference, E0.4 (REVE warm, depth 22) had a 701 → 1863 local exponent of +0.303, its random-init twin E0.5 +0.265, and depth-12 from scratch +0.049.

## 3. Train → test Pearson r (head fit on the 1863-recording pool, evaluated on R6)

Ceilings on file: val 0.313, test 0.172. Per the house rule, r / ceiling is quoted against the VAL ceiling only; the test ceiling is a known underestimate (RESULTS.md §2.5).

| S | warm r (test) | warm r / val ceiling | random r (test) | random r / val ceiling |
|---:|---|---:|---|---:|
| 50 | 0.0896 ± 0.0025 (n=3) | 0.286 | 0.0801 ± 0.0005 (n=3) | 0.256 |
| 200 | 0.1195 ± 0.0034 (n=3) | 0.381 | 0.0926 ± 0.0013 (n=3) | 0.296 |
| 701 | 0.1327 ± 0.0024 (n=3) | 0.424 | 0.1024 ± 0.0018 (n=3) | 0.327 |
| 1400 | 0.1296 ± 0.0008 (n=3) | 0.413 | 0.1049 ± 0.0007 (n=3) | 0.335 |
| 1863 | 0.1308 (n=1) | 0.417 | 0.1042 (n=1) | 0.333 |

- warm: fitted d log r / d log S = 0.100
- random: fitted d log r / d log S = 0.075

## 4. Scene-level e→v retrieval on TEST (top-1 / 5 / 10)

Null (untrained CBraMod shape): top1 —, top5 —, top10 —

| S | warm top-1 | warm top-5 | warm top-10 | random top-1 | random top-5 | random top-10 |
|---:|---|---|---|---|---|---|
| 50 | 0.0675 ± 0.0172 (n=3) | 0.2179 ± 0.0210 (n=3) | 0.3567 ± 0.0214 (n=3) | 0.0497 ± 0.0194 (n=3) | 0.2035 ± 0.0364 (n=3) | 0.3531 ± 0.0193 (n=3) |
| 200 | 0.0715 ± 0.0092 (n=3) | 0.2543 ± 0.0144 (n=3) | 0.3960 ± 0.0082 (n=3) | 0.0663 ± 0.0046 (n=3) | 0.2308 ± 0.0034 (n=3) | 0.3699 ± 0.0076 (n=3) |
| 701 | 0.0835 ± 0.0145 (n=3) | 0.2708 ± 0.0223 (n=3) | 0.4113 ± 0.0239 (n=3) | 0.0752 ± 0.0059 (n=3) | 0.2669 ± 0.0143 (n=3) | 0.4002 ± 0.0163 (n=3) |
| 1400 | 0.0763 ± 0.0107 (n=3) | 0.2675 ± 0.0215 (n=3) | 0.4003 ± 0.0163 (n=3) | 0.0660 ± 0.0049 (n=3) | 0.2512 ± 0.0171 (n=3) | 0.3914 ± 0.0193 (n=3) |
| 1863 | 0.0958 (n=1) | 0.2645 (n=1) | 0.4041 (n=1) | 0.0617 (n=1) | 0.2478 (n=1) | 0.3849 (n=1) |

## 5. Coverage

Cells aggregated: warm 13, random 13. Missing readouts: 26.

- {'slug': 'e12cb_s50_a101_nd_d11', 'readout': 'tt_val'}
- {'slug': 'e12cb_s50_a101_nd_d22', 'readout': 'tt_val'}
- {'slug': 'e12cb_s50_a101_nd_d33', 'readout': 'tt_val'}
- {'slug': 'e12cb_s200_a101_nd_d11', 'readout': 'tt_val'}
- {'slug': 'e12cb_s200_a101_nd_d22', 'readout': 'tt_val'}
- {'slug': 'e12cb_s200_a101_nd_d33', 'readout': 'tt_val'}
- {'slug': 'e12cb_s701_a101_nd_d11', 'readout': 'tt_val'}
- {'slug': 'e12cb_s701_a101_nd_d22', 'readout': 'tt_val'}
- {'slug': 'e12cb_s701_a101_nd_d33', 'readout': 'tt_val'}
- {'slug': 'e12cb_s1400_a101_nd_d11', 'readout': 'tt_val'}
- {'slug': 'e12cb_s1400_a101_nd_d22', 'readout': 'tt_val'}
- {'slug': 'e12cb_s1400_a101_nd_d33', 'readout': 'tt_val'}
- {'slug': 'e12cb_s1863_a101_nd', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s50_a101_nd_d11', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s50_a101_nd_d22', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s50_a101_nd_d33', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s200_a101_nd_d11', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s200_a101_nd_d22', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s200_a101_nd_d33', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s701_a101_nd_d11', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s701_a101_nd_d22', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s701_a101_nd_d33', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s1400_a101_nd_d11', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s1400_a101_nd_d22', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s1400_a101_nd_d33', 'readout': 'tt_val'}
- {'slug': 'e12cbr_s1863_a101_nd', 'readout': 'tt_val'}
