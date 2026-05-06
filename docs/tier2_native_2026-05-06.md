# Tier 2 supervised end-to-end native heads — 2026-05-06

4 architectures × 5 seeds; trained from scratch on the 4 stim-feature
regression task with raw 129-ch EEG, per-recording z-norm, nw=2 ws=4.
Per-clip predictions come from the trained regression head directly
(no Ridge probe). Schema and bootstrap match Tier 4/6
(`movie_reg_preds [n_rec=108, n_passes=20, n_features=4]`).

## L3 (5-seed mean ± 1σ of bootstrap means, B=2000 recording-level resamples)

| Method | lum | cont | pos | narr |
|---|---|---|---|---|
| **Cine-JEPA + Ridge (ours, ref)** | 0.226 ± 0.018 ✓ | 0.223 ± 0.014 ✓ | 0.226 ± 0.016 ✓ | 0.155 ± 0.023 ✓ |
| Tier 2 Deep4Net (native)          | 0.191 ± 0.037 ✓ | 0.142 ± 0.056 ✓ | 0.160 ± 0.021 ✓ | 0.024 ± 0.032 ns |
| Tier 2 ShallowFBCSPNet (native)   | 0.111 ± 0.023 ✓ | 0.057 ± 0.069 ns | 0.042 ± 0.082 ns | 0.027 ± 0.017 ✓ |
| **Tier 2 EEGNetv4 (native)**      | **0.311** ± 0.026 ✓ | 0.228 ± 0.018 ✓ | **0.273** ± 0.022 ✓ | **0.177** ± 0.018 ✓ |
| **Tier 2 EEGNeX (native)**        | **0.334** ± 0.010 ✓ | **0.277** ± 0.013 ✓ | **0.310** ± 0.014 ✓ | **0.178** ± 0.040 ✓ |

✓ = bootstrap mean significantly above 0 (1-sample two-sided t-test on 5 seeds, p<0.05). ns = not significant.

## Per-seed L1 detail (Pearson r over 2160 (rec, pass) flat clips)

```
model    seed       lum         cont        pos         narr
deep4    42         0.2179      0.1517      0.1580      0.0144
deep4    123        0.1722      0.1181      0.1485     -0.0229
deep4    456        0.1509      0.1052      0.1571      0.0528
deep4    789        0.2401      0.2347      0.1948      0.0555
deep4    2025       0.1744      0.0986      0.1410      0.0209
shallow  42         0.1157      0.1404      0.1005      0.0248
shallow  123        0.1340      0.0930     -0.1017      0.0503
shallow  456        0.0798      0.0673      0.0718      0.0364
shallow  789        0.0954      0.0250      0.0566      0.0061
shallow  2025       0.1272     -0.0397      0.0805      0.0158
eegnet   42         0.2784      0.2220      0.2568      0.1603
eegnet   123        0.3206      0.2570      0.2785      0.2042
eegnet   456        0.2948      0.2332      0.3028      0.1597
eegnet   789        0.3428      0.2150      0.2463      0.1766
eegnet   2025       0.3208      0.2189      0.2817      0.1836
eegnex   42         0.3260      0.2887      0.3034      0.1778
eegnex   123        0.3432      0.2861      0.3131      0.1916
eegnex   456        0.3249      0.2595      0.3318      0.1151
eegnex   789        0.3463      0.2840      0.3065      0.2282
eegnex   2025       0.3300      0.2686      0.2956      0.1765
```

## Notes

- **EEGNeX and EEGNetv4 (supervised end-to-end) outperform Cine-JEPA on every
  stim metric.** EEGNeX is the column winner on all 4. This is a real finding:
  a small supervised CNN trained directly on the regression task beats the
  SSL encoder + Ridge probe protocol.
- **Deep4Net native (no Ridge) drops dramatically on narrative** (0.024 vs
  0.144 with Ridge probe on backbone). The Ridge probe was salvaging signal
  the trained head wasn't using directly. This pattern is unique to Deep4Net
  in our suite — EEGNet/EEGNeX/Shallow all use their native heads efficiently.
- **Shallow is at the bottom** (lum 0.110, narr 0.027) — expected; ShallowFBCSPNet
  is a 2-layer conv designed for motor-imagery BCI, not stimulus decoding.
- All numbers are per-clip flat Pearson r over 2160 (rec × pass) pairs.
  Bootstrap = recording-level resampling (108 recs, B=2000) per seed; L3 is
  the 5-seed mean ± std of those bootstrap means (matches Tier 4/6 protocol).
- Subject features (age, sex) and movie_id heads were NOT trained — native
  Tier 2 architecture has only the 4-feature stim regression head.

## Per-seed L1 narrative (model-native head, raw Pearson r)

```
deep4    [+0.014, -0.023, +0.053, +0.056, +0.021]
shallow  [+0.025, +0.050, +0.036, +0.006, +0.016]
eegnet   [+0.160, +0.204, +0.160, +0.177, +0.184]
eegnex   [+0.178, +0.192, +0.115, +0.228, +0.177]
```

## Source artifacts

```
predictions/tier2_native/{model}_seed{S}/test_seed{S+2}.npz   # native head preds
scripts/extract_tier2_native.py                                # train + extract
scripts/extract_tier2_native.sbatch                            # SLURM wrapper
```

Job IDs: 18090411–31 (4 models × 5 seeds = 20).
