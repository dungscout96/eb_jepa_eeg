# Annotation Validation Report — The Present

Validates all 24 per-frame visual features.
Primary features (Issue #5): `contrast_rms`, `luminance_mean`, `entropy`, `scene_natural_score`.
Additional: all remaining features from the annotation pipeline.

---

## 1. Code Review

| Feature | Method | Library | Model-free? |
|---------|--------|---------|------------|
| `luminance_mean` | `mean(gray)/255` | OpenCV+NumPy | Yes |
| `contrast_rms` | `std(gray)/255` (ddof=0) | OpenCV+NumPy | Yes |
| `color_{r,g,b}_mean` | channel mean/255 | OpenCV+NumPy | Yes |
| `saturation_mean` | HSV S-channel mean/255 | OpenCV | Yes |
| `edge_density` | Canny edge pixel fraction | OpenCV | Yes |
| `spatial_freq_energy` | FFT high-freq power ratio | NumPy | Yes |
| `entropy` | Shannon entropy of 256-bin gray histogram | SciPy | Yes |
| `motion_energy` | Farneback optical flow mean magnitude | OpenCV | Yes |
| `scene_cut` | frame-diff threshold on luminance | NumPy | Yes |
| `n_faces`, `face_area_frac` | RetinaFace detector | insightface | No (CNN) |
| `depth_mean/std/range` | MiDaS monocular depth | timm/MiDaS | No (CNN) |
| `n_objects`, `object_categories` | YOLOv8 detection | ultralytics | No (CNN) |
| `scene_category`, `scene_category_score` | CLIP softmax (15 categories) | CLIP ViT-B/32 | No |
| `scene_natural_score` | CLIP cosine diff (natural vs urban) | CLIP ViT-B/32 | No |
| `scene_open_score` | CLIP cosine diff (open vs enclosed) | CLIP ViT-B/32 | No |

---

## 2. Recomputation Check (All 11 Model-free Features)

Re-extracted all 11 model-free features (9 low-level + motion_energy + scene_cut) and compared against stored CSV values.

### 2a. Low-level features (9 features)

Sampled ~30 frames (every 163th), recomputed via `extract_lowlevel()`.

| Feature | Frames | Max Abs Error | Mean Error | Error Std | Pearson r | Note | Result |
|---------|--------|--------------|------------|-----------|-----------|------|--------|
| `luminance_mean` | 30 | 3.91e-03 | 4.82e-04 | 9.06e-04 | 0.999994 | cross-platform codec offset | PASS |
| `contrast_rms` | 30 | 1.02e-03 | 1.66e-04 | 1.89e-04 | 0.999995 | cross-platform codec offset | PASS |
| `entropy` | 30 | 1.38e-01 | 9.03e-02 | 3.39e-02 | 0.999901 | cross-platform codec offset | PASS |
| `color_r_mean` | 30 | 3.92e-03 | 3.17e-03 | 2.74e-04 | 1.000000 | cross-platform codec offset | PASS |
| `color_g_mean` | 30 | 1.73e-03 | 1.01e-03 | 4.41e-04 | 0.999998 | cross-platform codec offset | PASS |
| `color_b_mean` | 30 | 3.92e-03 | 1.15e-03 | 8.02e-04 | 0.999992 | cross-platform codec offset | PASS |
| `saturation_mean` | 30 | 1.81e-02 | 4.06e-03 | 3.17e-03 | 0.999858 | cross-platform codec offset | PASS |
| `edge_density` | 30 | 1.32e-03 | 2.93e-04 | 3.03e-04 | 0.999973 | cross-platform codec offset | PASS |
| `spatial_freq_energy` | 30 | 3.51e-03 | 1.75e-04 | 6.88e-04 | 0.999598 | cross-platform codec offset | PASS |

> Pass criteria: Pearson r > 0.999. Consistent mean offset = cross-platform codec difference, not formula error.

### 2b. Motion features (motion_energy + scene_cut)

Sampled ~15 consecutive frame pairs, recomputed via `extract_motion()`.

| Feature | Frames | Max Abs Error | Mean Error | Error Std | Pearson r | Note | Result |
|---------|--------|--------------|------------|-----------|-----------|------|--------|
| `motion_energy` | 15 | 1.07e-01 | 2.40e-02 | 3.83e-02 | 0.999928 | cross-platform codec offset | PASS |

**`scene_cut`** boolean agreement: 15/15 (100.0%) — PASS

> scene_cut pass criteria: > 95% agreement.

---

## 3. Distribution Checks

### Continuous features

| Feature | Min | Max | Mean | Std | NaN | In range? | Variation? | Result |
|---------|-----|-----|------|-----|-----|-----------|-----------|--------|
| `luminance_mean` | 0.0046 | 0.9216 | 0.4816 | 0.1603 | 0 | Yes | Yes | PASS |
| `contrast_rms` | 0.0135 | 0.3311 | 0.2437 | 0.0575 | 0 | Yes | Yes | PASS |
| `entropy` | 0.0482 | 7.7797 | 7.1678 | 1.2686 | 0 | Yes | Yes | PASS |
| `color_r_mean` | 0.0046 | 0.9368 | 0.5300 | 0.1699 | 0 | Yes | Yes | PASS |
| `color_g_mean` | 0.0007 | 0.9204 | 0.4697 | 0.1587 | 0 | Yes | Yes | PASS |
| `color_b_mean` | 0.0086 | 0.8876 | 0.4169 | 0.1493 | 0 | Yes | Yes | PASS |
| `saturation_mean` | 0.0675 | 0.9967 | 0.3132 | 0.1409 | 0 | Yes | Yes | PASS |
| `edge_density` | 0.0008 | 0.2305 | 0.0572 | 0.0423 | 0 | Yes | Yes | PASS |
| `spatial_freq_energy` | 0.0001 | 0.0373 | 0.0021 | 0.0045 | 0 | Yes | Yes | PASS |
| `motion_energy` | 0.0000 | 21.0006 | 1.2200 | 2.4037 | 0 | Yes | Yes | PASS |
| `face_area_frac` | 0.0000 | 0.6258 | 0.0518 | 0.0917 | 0 | Yes | Yes | PASS |
| `depth_mean` | 5.5648 | 27.5316 | 15.7259 | 2.7316 | 0 | Yes | Yes | PASS |
| `depth_std` | 2.3282 | 27.5307 | 9.6777 | 3.8978 | 0 | Yes | Yes | PASS |
| `depth_range` | 11.9523 | 82.2905 | 33.7578 | 11.1513 | 0 | Yes | Yes | PASS |
| `scene_natural_score` | -0.0651 | 0.0445 | -0.0090 | 0.0151 | 0 | Yes | Yes | PASS |
| `scene_open_score` | -0.0804 | 0.0234 | -0.0301 | 0.0193 | 0 | Yes | Yes | PASS |
| `scene_category_score` | 0.0675 | 0.0718 | 0.0699 | 0.0007 | 0 | Yes | Yes | PASS |

### Count features

**`n_faces`**: {0: 2913, 1: 1936, 2: 28}  NaN=0  — PASS

**`n_objects`**: min=0 max=31 mean=3.13 std=2.74  NaN=0  — PASS

**`scene_cut`**: 18 cuts, 4859 non-cuts  NaN=0  — PASS

**`object_categories`** — top 15 detected categories:

| Category | Total count |
|----------|------------|
| person | 3964 |
| potted plant | 2331 |
| chair | 1304 |
| dog | 1053 |
| couch | 973 |
| cell phone | 702 |
| book | 678 |
| sports ball | 437 |
| vase | 392 |
| dining table | 384 |
| tv | 365 |
| teddy bear | 341 |
| remote | 311 |
| tie | 271 |
| clock | 267 |

Total unique categories: 55  NaN=0  — PASS

**`scene_category`** (CLIP 15-way): 10 categories assigned, NaN=0  — PASS

| Category | Frame count |
|----------|------------|
| fantasy or animated world | 4029 |
| indoor room | 249 |
| kitchen | 187 |
| stage or theater | 132 |
| living room | 102 |
| bedroom | 91 |
| bathroom | 49 |
| office or workspace | 30 |
| vehicle interior | 7 |
| outdoor street | 1 |

---

## 4. Low-level Features — Visual Inspection

### `luminance_mean`

### Q1 dark

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0046 | [Q1_dark_frame0000_score0.0046.png](output/The_Present/validation_frames/luminance_mean/Q1_dark_frame0000_score0.0046.png) |
| 90 | 3.8 | 0.0314 | [Q1_dark_frame0090_score0.0314.png](output/The_Present/validation_frames/luminance_mean/Q1_dark_frame0090_score0.0314.png) |
| 327 | 13.6 | 0.1828 | [Q1_dark_frame0327_score0.1828.png](output/The_Present/validation_frames/luminance_mean/Q1_dark_frame0327_score0.1828.png) |
| 451 | 18.8 | 0.2431 | [Q1_dark_frame0451_score0.2431.png](output/The_Present/validation_frames/luminance_mean/Q1_dark_frame0451_score0.2431.png) |
| 563 | 23.5 | 0.2912 | [Q1_dark_frame0563_score0.2912.png](output/The_Present/validation_frames/luminance_mean/Q1_dark_frame0563_score0.2912.png) |
| 766 | 31.9 | 0.3185 | [Q1_dark_frame0766_score0.3185.png](output/The_Present/validation_frames/luminance_mean/Q1_dark_frame0766_score0.3185.png) |
| 191 | 8.0 | 0.3813 | [Q1_dark_frame0191_score0.3813.png](output/The_Present/validation_frames/luminance_mean/Q1_dark_frame0191_score0.3813.png) |
| 3250 | 135.4 | 0.4022 | [Q1_dark_frame3250_score0.4022.png](output/The_Present/validation_frames/luminance_mean/Q1_dark_frame3250_score0.4022.png) |

### Q2 mid dark

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2674 | 111.4 | 0.4325 | [Q2_mid_dark_frame2674_score0.4325.png](output/The_Present/validation_frames/luminance_mean/Q2_mid_dark_frame2674_score0.4325.png) |
| 2726 | 113.6 | 0.4438 | [Q2_mid_dark_frame2726_score0.4438.png](output/The_Present/validation_frames/luminance_mean/Q2_mid_dark_frame2726_score0.4438.png) |
| 2426 | 101.1 | 0.4539 | [Q2_mid_dark_frame2426_score0.4539.png](output/The_Present/validation_frames/luminance_mean/Q2_mid_dark_frame2426_score0.4539.png) |
| 1829 | 76.2 | 0.4637 | [Q2_mid_dark_frame1829_score0.4637.png](output/The_Present/validation_frames/luminance_mean/Q2_mid_dark_frame1829_score0.4637.png) |
| 1246 | 51.9 | 0.4753 | [Q2_mid_dark_frame1246_score0.4753.png](output/The_Present/validation_frames/luminance_mean/Q2_mid_dark_frame1246_score0.4753.png) |
| 1157 | 48.2 | 0.4846 | [Q2_mid_dark_frame1157_score0.4846.png](output/The_Present/validation_frames/luminance_mean/Q2_mid_dark_frame1157_score0.4846.png) |
| 1894 | 78.9 | 0.4944 | [Q2_mid_dark_frame1894_score0.4944.png](output/The_Present/validation_frames/luminance_mean/Q2_mid_dark_frame1894_score0.4944.png) |
| 1353 | 56.4 | 0.5064 | [Q2_mid_dark_frame1353_score0.5064.png](output/The_Present/validation_frames/luminance_mean/Q2_mid_dark_frame1353_score0.5064.png) |

### Q3 mid bright

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2934 | 122.3 | 0.5130 | [Q3_mid_bright_frame2934_score0.5130.png](output/The_Present/validation_frames/luminance_mean/Q3_mid_bright_frame2934_score0.5130.png) |
| 2851 | 118.8 | 0.5190 | [Q3_mid_bright_frame2851_score0.5190.png](output/The_Present/validation_frames/luminance_mean/Q3_mid_bright_frame2851_score0.5190.png) |
| 2966 | 123.6 | 0.5258 | [Q3_mid_bright_frame2966_score0.5258.png](output/The_Present/validation_frames/luminance_mean/Q3_mid_bright_frame2966_score0.5258.png) |
| 4259 | 177.5 | 0.5354 | [Q3_mid_bright_frame4259_score0.5354.png](output/The_Present/validation_frames/luminance_mean/Q3_mid_bright_frame4259_score0.5354.png) |
| 4154 | 173.1 | 0.5414 | [Q3_mid_bright_frame4154_score0.5414.png](output/The_Present/validation_frames/luminance_mean/Q3_mid_bright_frame4154_score0.5414.png) |
| 3893 | 162.2 | 0.5507 | [Q3_mid_bright_frame3893_score0.5507.png](output/The_Present/validation_frames/luminance_mean/Q3_mid_bright_frame3893_score0.5507.png) |
| 3739 | 155.8 | 0.5598 | [Q3_mid_bright_frame3739_score0.5598.png](output/The_Present/validation_frames/luminance_mean/Q3_mid_bright_frame3739_score0.5598.png) |
| 4501 | 187.6 | 0.5724 | [Q3_mid_bright_frame4501_score0.5724.png](output/The_Present/validation_frames/luminance_mean/Q3_mid_bright_frame4501_score0.5724.png) |

### Q4 bright

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 4636 | 193.2 | 0.5765 | [Q4_bright_frame4636_score0.5765.png](output/The_Present/validation_frames/luminance_mean/Q4_bright_frame4636_score0.5765.png) |
| 911 | 38.0 | 0.5792 | [Q4_bright_frame0911_score0.5792.png](output/The_Present/validation_frames/luminance_mean/Q4_bright_frame0911_score0.5792.png) |
| 4721 | 196.7 | 0.5821 | [Q4_bright_frame4721_score0.5821.png](output/The_Present/validation_frames/luminance_mean/Q4_bright_frame4721_score0.5821.png) |
| 1060 | 44.2 | 0.5864 | [Q4_bright_frame1060_score0.5864.png](output/The_Present/validation_frames/luminance_mean/Q4_bright_frame1060_score0.5864.png) |
| 2217 | 92.4 | 0.6171 | [Q4_bright_frame2217_score0.6171.png](output/The_Present/validation_frames/luminance_mean/Q4_bright_frame2217_score0.6171.png) |
| 2453 | 102.2 | 0.6515 | [Q4_bright_frame2453_score0.6515.png](output/The_Present/validation_frames/luminance_mean/Q4_bright_frame2453_score0.6515.png) |
| 3817 | 159.1 | 0.6916 | [Q4_bright_frame3817_score0.6916.png](output/The_Present/validation_frames/luminance_mean/Q4_bright_frame3817_score0.6916.png) |
| 3452 | 143.9 | 0.7548 | [Q4_bright_frame3452_score0.7548.png](output/The_Present/validation_frames/luminance_mean/Q4_bright_frame3452_score0.7548.png) |

### `contrast_rms`

### Q1 low contrast

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0135 | [Q1_low_contrast_frame0000_score0.0135.png](output/The_Present/validation_frames/contrast_rms/Q1_low_contrast_frame0000_score0.0135.png) |
| 104 | 4.3 | 0.1224 | [Q1_low_contrast_frame0104_score0.1224.png](output/The_Present/validation_frames/contrast_rms/Q1_low_contrast_frame0104_score0.1224.png) |
| 441 | 18.4 | 0.1542 | [Q1_low_contrast_frame0441_score0.1542.png](output/The_Present/validation_frames/contrast_rms/Q1_low_contrast_frame0441_score0.1542.png) |
| 3977 | 165.7 | 0.1625 | [Q1_low_contrast_frame3977_score0.1625.png](output/The_Present/validation_frames/contrast_rms/Q1_low_contrast_frame3977_score0.1625.png) |
| 186 | 7.8 | 0.1699 | [Q1_low_contrast_frame0186_score0.1699.png](output/The_Present/validation_frames/contrast_rms/Q1_low_contrast_frame0186_score0.1699.png) |
| 970 | 40.4 | 0.1810 | [Q1_low_contrast_frame0970_score0.1810.png](output/The_Present/validation_frames/contrast_rms/Q1_low_contrast_frame0970_score0.1810.png) |
| 313 | 13.0 | 0.1926 | [Q1_low_contrast_frame0313_score0.1926.png](output/The_Present/validation_frames/contrast_rms/Q1_low_contrast_frame0313_score0.1926.png) |
| 2117 | 88.2 | 0.2068 | [Q1_low_contrast_frame2117_score0.2068.png](output/The_Present/validation_frames/contrast_rms/Q1_low_contrast_frame2117_score0.2068.png) |

### Q2 mid low

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1494 | 62.3 | 0.2158 | [Q2_mid_low_frame1494_score0.2158.png](output/The_Present/validation_frames/contrast_rms/Q2_mid_low_frame1494_score0.2158.png) |
| 2657 | 110.7 | 0.2227 | [Q2_mid_low_frame2657_score0.2227.png](output/The_Present/validation_frames/contrast_rms/Q2_mid_low_frame2657_score0.2227.png) |
| 588 | 24.5 | 0.2334 | [Q2_mid_low_frame0588_score0.2334.png](output/The_Present/validation_frames/contrast_rms/Q2_mid_low_frame0588_score0.2334.png) |
| 2817 | 117.4 | 0.2417 | [Q2_mid_low_frame2817_score0.2417.png](output/The_Present/validation_frames/contrast_rms/Q2_mid_low_frame2817_score0.2417.png) |
| 3248 | 135.4 | 0.2459 | [Q2_mid_low_frame3248_score0.2459.png](output/The_Present/validation_frames/contrast_rms/Q2_mid_low_frame3248_score0.2459.png) |
| 3044 | 126.9 | 0.2493 | [Q2_mid_low_frame3044_score0.2493.png](output/The_Present/validation_frames/contrast_rms/Q2_mid_low_frame3044_score0.2493.png) |
| 3385 | 141.1 | 0.2550 | [Q2_mid_low_frame3385_score0.2550.png](output/The_Present/validation_frames/contrast_rms/Q2_mid_low_frame3385_score0.2550.png) |
| 2397 | 99.9 | 0.2583 | [Q2_mid_low_frame2397_score0.2583.png](output/The_Present/validation_frames/contrast_rms/Q2_mid_low_frame2397_score0.2583.png) |

### Q3 mid high

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3158 | 131.6 | 0.2612 | [Q3_mid_high_frame3158_score0.2612.png](output/The_Present/validation_frames/contrast_rms/Q3_mid_high_frame3158_score0.2612.png) |
| 2620 | 109.2 | 0.2638 | [Q3_mid_high_frame2620_score0.2638.png](output/The_Present/validation_frames/contrast_rms/Q3_mid_high_frame2620_score0.2638.png) |
| 2227 | 92.8 | 0.2663 | [Q3_mid_high_frame2227_score0.2663.png](output/The_Present/validation_frames/contrast_rms/Q3_mid_high_frame2227_score0.2663.png) |
| 1109 | 46.2 | 0.2694 | [Q3_mid_high_frame1109_score0.2694.png](output/The_Present/validation_frames/contrast_rms/Q3_mid_high_frame1109_score0.2694.png) |
| 1746 | 72.8 | 0.2717 | [Q3_mid_high_frame1746_score0.2717.png](output/The_Present/validation_frames/contrast_rms/Q3_mid_high_frame1746_score0.2717.png) |
| 3324 | 138.5 | 0.2742 | [Q3_mid_high_frame3324_score0.2742.png](output/The_Present/validation_frames/contrast_rms/Q3_mid_high_frame3324_score0.2742.png) |
| 910 | 37.9 | 0.2765 | [Q3_mid_high_frame0910_score0.2765.png](output/The_Present/validation_frames/contrast_rms/Q3_mid_high_frame0910_score0.2765.png) |
| 1221 | 50.9 | 0.2795 | [Q3_mid_high_frame1221_score0.2795.png](output/The_Present/validation_frames/contrast_rms/Q3_mid_high_frame1221_score0.2795.png) |

### Q4 high contrast

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2587 | 107.8 | 0.2820 | [Q4_high_contrast_frame2587_score0.2820.png](output/The_Present/validation_frames/contrast_rms/Q4_high_contrast_frame2587_score0.2820.png) |
| 2448 | 102.0 | 0.2848 | [Q4_high_contrast_frame2448_score0.2848.png](output/The_Present/validation_frames/contrast_rms/Q4_high_contrast_frame2448_score0.2848.png) |
| 2161 | 90.1 | 0.2877 | [Q4_high_contrast_frame2161_score0.2877.png](output/The_Present/validation_frames/contrast_rms/Q4_high_contrast_frame2161_score0.2877.png) |
| 4419 | 184.2 | 0.2961 | [Q4_high_contrast_frame4419_score0.2961.png](output/The_Present/validation_frames/contrast_rms/Q4_high_contrast_frame4419_score0.2961.png) |
| 4792 | 199.7 | 0.2973 | [Q4_high_contrast_frame4792_score0.2973.png](output/The_Present/validation_frames/contrast_rms/Q4_high_contrast_frame4792_score0.2973.png) |
| 4701 | 195.9 | 0.2977 | [Q4_high_contrast_frame4701_score0.2977.png](output/The_Present/validation_frames/contrast_rms/Q4_high_contrast_frame4701_score0.2977.png) |
| 835 | 34.8 | 0.3025 | [Q4_high_contrast_frame0835_score0.3025.png](output/The_Present/validation_frames/contrast_rms/Q4_high_contrast_frame0835_score0.3025.png) |
| 4279 | 178.3 | 0.3144 | [Q4_high_contrast_frame4279_score0.3144.png](output/The_Present/validation_frames/contrast_rms/Q4_high_contrast_frame4279_score0.3144.png) |

### `entropy`

### Q1 low entropy

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0482 | [Q1_low_entropy_frame0000_score0.0482.png](output/The_Present/validation_frames/entropy/Q1_low_entropy_frame0000_score0.0482.png) |
| 90 | 3.8 | 1.3540 | [Q1_low_entropy_frame0090_score1.3540.png](output/The_Present/validation_frames/entropy/Q1_low_entropy_frame0090_score1.3540.png) |
| 323 | 13.5 | 6.4621 | [Q1_low_entropy_frame0323_score6.4621.png](output/The_Present/validation_frames/entropy/Q1_low_entropy_frame0323_score6.4621.png) |
| 483 | 20.1 | 6.7756 | [Q1_low_entropy_frame0483_score6.7756.png](output/The_Present/validation_frames/entropy/Q1_low_entropy_frame0483_score6.7756.png) |
| 457 | 19.0 | 6.9270 | [Q1_low_entropy_frame0457_score6.9270.png](output/The_Present/validation_frames/entropy/Q1_low_entropy_frame0457_score6.9270.png) |
| 3417 | 142.4 | 6.9964 | [Q1_low_entropy_frame3417_score6.9964.png](output/The_Present/validation_frames/entropy/Q1_low_entropy_frame3417_score6.9964.png) |
| 3482 | 145.1 | 7.0648 | [Q1_low_entropy_frame3482_score7.0648.png](output/The_Present/validation_frames/entropy/Q1_low_entropy_frame3482_score7.0648.png) |
| 4100 | 170.9 | 7.1484 | [Q1_low_entropy_frame4100_score7.1484.png](output/The_Present/validation_frames/entropy/Q1_low_entropy_frame4100_score7.1484.png) |

### Q2 mid low

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2257 | 94.1 | 7.2266 | [Q2_mid_low_frame2257_score7.2266.png](output/The_Present/validation_frames/entropy/Q2_mid_low_frame2257_score7.2266.png) |
| 584 | 24.3 | 7.3008 | [Q2_mid_low_frame0584_score7.3008.png](output/The_Present/validation_frames/entropy/Q2_mid_low_frame0584_score7.3008.png) |
| 3603 | 150.2 | 7.4018 | [Q2_mid_low_frame3603_score7.4018.png](output/The_Present/validation_frames/entropy/Q2_mid_low_frame3603_score7.4018.png) |
| 3790 | 157.9 | 7.4419 | [Q2_mid_low_frame3790_score7.4419.png](output/The_Present/validation_frames/entropy/Q2_mid_low_frame3790_score7.4419.png) |
| 3169 | 132.1 | 7.4786 | [Q2_mid_low_frame3169_score7.4786.png](output/The_Present/validation_frames/entropy/Q2_mid_low_frame3169_score7.4786.png) |
| 1819 | 75.8 | 7.4934 | [Q2_mid_low_frame1819_score7.4934.png](output/The_Present/validation_frames/entropy/Q2_mid_low_frame1819_score7.4934.png) |
| 1497 | 62.4 | 7.5054 | [Q2_mid_low_frame1497_score7.5054.png](output/The_Present/validation_frames/entropy/Q2_mid_low_frame1497_score7.5054.png) |
| 2892 | 120.5 | 7.5143 | [Q2_mid_low_frame2892_score7.5143.png](output/The_Present/validation_frames/entropy/Q2_mid_low_frame2892_score7.5143.png) |

### Q3 mid high

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2889 | 120.4 | 7.5223 | [Q3_mid_high_frame2889_score7.5223.png](output/The_Present/validation_frames/entropy/Q3_mid_high_frame2889_score7.5223.png) |
| 894 | 37.3 | 7.5351 | [Q3_mid_high_frame0894_score7.5351.png](output/The_Present/validation_frames/entropy/Q3_mid_high_frame0894_score7.5351.png) |
| 4809 | 200.4 | 7.5452 | [Q3_mid_high_frame4809_score7.5452.png](output/The_Present/validation_frames/entropy/Q3_mid_high_frame4809_score7.5452.png) |
| 4722 | 196.8 | 7.5497 | [Q3_mid_high_frame4722_score7.5497.png](output/The_Present/validation_frames/entropy/Q3_mid_high_frame4722_score7.5497.png) |
| 2655 | 110.6 | 7.5549 | [Q3_mid_high_frame2655_score7.5549.png](output/The_Present/validation_frames/entropy/Q3_mid_high_frame2655_score7.5549.png) |
| 4447 | 185.3 | 7.5608 | [Q3_mid_high_frame4447_score7.5608.png](output/The_Present/validation_frames/entropy/Q3_mid_high_frame4447_score7.5608.png) |
| 4585 | 191.1 | 7.5696 | [Q3_mid_high_frame4585_score7.5696.png](output/The_Present/validation_frames/entropy/Q3_mid_high_frame4585_score7.5696.png) |
| 2461 | 102.6 | 7.5822 | [Q3_mid_high_frame2461_score7.5822.png](output/The_Present/validation_frames/entropy/Q3_mid_high_frame2461_score7.5822.png) |

### Q4 high entropy

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3028 | 126.2 | 7.5951 | [Q4_high_entropy_frame3028_score7.5951.png](output/The_Present/validation_frames/entropy/Q4_high_entropy_frame3028_score7.5951.png) |
| 2385 | 99.4 | 7.6172 | [Q4_high_entropy_frame2385_score7.6172.png](output/The_Present/validation_frames/entropy/Q4_high_entropy_frame2385_score7.6172.png) |
| 1292 | 53.8 | 7.6342 | [Q4_high_entropy_frame1292_score7.6342.png](output/The_Present/validation_frames/entropy/Q4_high_entropy_frame1292_score7.6342.png) |
| 2715 | 113.1 | 7.6493 | [Q4_high_entropy_frame2715_score7.6493.png](output/The_Present/validation_frames/entropy/Q4_high_entropy_frame2715_score7.6493.png) |
| 2360 | 98.4 | 7.6638 | [Q4_high_entropy_frame2360_score7.6638.png](output/The_Present/validation_frames/entropy/Q4_high_entropy_frame2360_score7.6638.png) |
| 1325 | 55.2 | 7.7009 | [Q4_high_entropy_frame1325_score7.7009.png](output/The_Present/validation_frames/entropy/Q4_high_entropy_frame1325_score7.7009.png) |
| 1983 | 82.6 | 7.7317 | [Q4_high_entropy_frame1983_score7.7317.png](output/The_Present/validation_frames/entropy/Q4_high_entropy_frame1983_score7.7317.png) |
| 2970 | 123.8 | 7.7560 | [Q4_high_entropy_frame2970_score7.7560.png](output/The_Present/validation_frames/entropy/Q4_high_entropy_frame2970_score7.7560.png) |

---

## 5. Color Features — Visual Inspection

### `color_r_mean`

### Q1 low red

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0046 | [Q1_low_red_frame0000_score0.0046.png](output/The_Present/validation_frames/color_r_mean/Q1_low_red_frame0000_score0.0046.png) |
| 135 | 5.6 | 0.0325 | [Q1_low_red_frame0135_score0.0325.png](output/The_Present/validation_frames/color_r_mean/Q1_low_red_frame0135_score0.0325.png) |
| 400 | 16.7 | 0.2017 | [Q1_low_red_frame0400_score0.2017.png](output/The_Present/validation_frames/color_r_mean/Q1_low_red_frame0400_score0.2017.png) |
| 633 | 26.4 | 0.2763 | [Q1_low_red_frame0633_score0.2763.png](output/The_Present/validation_frames/color_r_mean/Q1_low_red_frame0633_score0.2763.png) |
| 983 | 41.0 | 0.3301 | [Q1_low_red_frame0983_score0.3301.png](output/The_Present/validation_frames/color_r_mean/Q1_low_red_frame0983_score0.3301.png) |
| 699 | 29.1 | 0.3600 | [Q1_low_red_frame0699_score0.3600.png](output/The_Present/validation_frames/color_r_mean/Q1_low_red_frame0699_score0.3600.png) |
| 2914 | 121.4 | 0.4355 | [Q1_low_red_frame2914_score0.4355.png](output/The_Present/validation_frames/color_r_mean/Q1_low_red_frame2914_score0.4355.png) |
| 3261 | 135.9 | 0.4536 | [Q1_low_red_frame3261_score0.4536.png](output/The_Present/validation_frames/color_r_mean/Q1_low_red_frame3261_score0.4536.png) |

### Q2 mid low red

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1949 | 81.2 | 0.4788 | [Q2_mid_low_red_frame1949_score0.4788.png](output/The_Present/validation_frames/color_r_mean/Q2_mid_low_red_frame1949_score0.4788.png) |
| 2718 | 113.3 | 0.4915 | [Q2_mid_low_red_frame2718_score0.4915.png](output/The_Present/validation_frames/color_r_mean/Q2_mid_low_red_frame2718_score0.4915.png) |
| 2518 | 104.9 | 0.5028 | [Q2_mid_low_red_frame2518_score0.5028.png](output/The_Present/validation_frames/color_r_mean/Q2_mid_low_red_frame2518_score0.5028.png) |
| 2361 | 98.4 | 0.5097 | [Q2_mid_low_red_frame2361_score0.5097.png](output/The_Present/validation_frames/color_r_mean/Q2_mid_low_red_frame2361_score0.5097.png) |
| 2576 | 107.4 | 0.5207 | [Q2_mid_low_red_frame2576_score0.5207.png](output/The_Present/validation_frames/color_r_mean/Q2_mid_low_red_frame2576_score0.5207.png) |
| 4231 | 176.3 | 0.5450 | [Q2_mid_low_red_frame4231_score0.5450.png](output/The_Present/validation_frames/color_r_mean/Q2_mid_low_red_frame4231_score0.5450.png) |
| 3514 | 146.4 | 0.5549 | [Q2_mid_low_red_frame3514_score0.5549.png](output/The_Present/validation_frames/color_r_mean/Q2_mid_low_red_frame3514_score0.5549.png) |
| 3127 | 130.3 | 0.5681 | [Q2_mid_low_red_frame3127_score0.5681.png](output/The_Present/validation_frames/color_r_mean/Q2_mid_low_red_frame3127_score0.5681.png) |

### Q3 mid high red

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 4173 | 173.9 | 0.5730 | [Q3_mid_high_red_frame4173_score0.5730.png](output/The_Present/validation_frames/color_r_mean/Q3_mid_high_red_frame4173_score0.5730.png) |
| 3084 | 128.5 | 0.5768 | [Q3_mid_high_red_frame3084_score0.5768.png](output/The_Present/validation_frames/color_r_mean/Q3_mid_high_red_frame3084_score0.5768.png) |
| 2027 | 84.5 | 0.5818 | [Q3_mid_high_red_frame2027_score0.5818.png](output/The_Present/validation_frames/color_r_mean/Q3_mid_high_red_frame2027_score0.5818.png) |
| 3543 | 147.7 | 0.5879 | [Q3_mid_high_red_frame3543_score0.5879.png](output/The_Present/validation_frames/color_r_mean/Q3_mid_high_red_frame3543_score0.5879.png) |
| 2845 | 118.6 | 0.5988 | [Q3_mid_high_red_frame2845_score0.5988.png](output/The_Present/validation_frames/color_r_mean/Q3_mid_high_red_frame2845_score0.5988.png) |
| 1537 | 64.1 | 0.6059 | [Q3_mid_high_red_frame1537_score0.6059.png](output/The_Present/validation_frames/color_r_mean/Q3_mid_high_red_frame1537_score0.6059.png) |
| 4426 | 184.5 | 0.6176 | [Q3_mid_high_red_frame4426_score0.6176.png](output/The_Present/validation_frames/color_r_mean/Q3_mid_high_red_frame4426_score0.6176.png) |
| 4562 | 190.1 | 0.6191 | [Q3_mid_high_red_frame4562_score0.6191.png](output/The_Present/validation_frames/color_r_mean/Q3_mid_high_red_frame4562_score0.6191.png) |

### Q4 high red

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 4782 | 199.3 | 0.6225 | [Q4_high_red_frame4782_score0.6225.png](output/The_Present/validation_frames/color_r_mean/Q4_high_red_frame4782_score0.6225.png) |
| 4858 | 202.5 | 0.6244 | [Q4_high_red_frame4858_score0.6244.png](output/The_Present/validation_frames/color_r_mean/Q4_high_red_frame4858_score0.6244.png) |
| 3548 | 147.9 | 0.6345 | [Q4_high_red_frame3548_score0.6345.png](output/The_Present/validation_frames/color_r_mean/Q4_high_red_frame3548_score0.6345.png) |
| 3697 | 154.1 | 0.6483 | [Q4_high_red_frame3697_score0.6483.png](output/The_Present/validation_frames/color_r_mean/Q4_high_red_frame3697_score0.6483.png) |
| 2474 | 103.1 | 0.6641 | [Q4_high_red_frame2474_score0.6641.png](output/The_Present/validation_frames/color_r_mean/Q4_high_red_frame2474_score0.6641.png) |
| 2449 | 102.1 | 0.7028 | [Q4_high_red_frame2449_score0.7028.png](output/The_Present/validation_frames/color_r_mean/Q4_high_red_frame2449_score0.7028.png) |
| 3823 | 159.3 | 0.7461 | [Q4_high_red_frame3823_score0.7461.png](output/The_Present/validation_frames/color_r_mean/Q4_high_red_frame3823_score0.7461.png) |
| 4072 | 169.7 | 0.8027 | [Q4_high_red_frame4072_score0.8027.png](output/The_Present/validation_frames/color_r_mean/Q4_high_red_frame4072_score0.8027.png) |

### `color_g_mean`

### Q1 low green

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0007 | [Q1_low_green_frame0000_score0.0007.png](output/The_Present/validation_frames/color_g_mean/Q1_low_green_frame0000_score0.0007.png) |
| 90 | 3.8 | 0.0277 | [Q1_low_green_frame0090_score0.0277.png](output/The_Present/validation_frames/color_g_mean/Q1_low_green_frame0090_score0.0277.png) |
| 500 | 20.8 | 0.1793 | [Q1_low_green_frame0500_score0.1793.png](output/The_Present/validation_frames/color_g_mean/Q1_low_green_frame0500_score0.1793.png) |
| 594 | 24.8 | 0.2334 | [Q1_low_green_frame0594_score0.2334.png](output/The_Present/validation_frames/color_g_mean/Q1_low_green_frame0594_score0.2334.png) |
| 1028 | 42.8 | 0.2839 | [Q1_low_green_frame1028_score0.2839.png](output/The_Present/validation_frames/color_g_mean/Q1_low_green_frame1028_score0.2839.png) |
| 794 | 33.1 | 0.3064 | [Q1_low_green_frame0794_score0.3064.png](output/The_Present/validation_frames/color_g_mean/Q1_low_green_frame0794_score0.3064.png) |
| 2541 | 105.9 | 0.3560 | [Q1_low_green_frame2541_score0.3560.png](output/The_Present/validation_frames/color_g_mean/Q1_low_green_frame2541_score0.3560.png) |
| 2880 | 120.0 | 0.3909 | [Q1_low_green_frame2880_score0.3909.png](output/The_Present/validation_frames/color_g_mean/Q1_low_green_frame2880_score0.3909.png) |

### Q2 mid low green

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1872 | 78.0 | 0.4229 | [Q2_mid_low_green_frame1872_score0.4229.png](output/The_Present/validation_frames/color_g_mean/Q2_mid_low_green_frame1872_score0.4229.png) |
| 1487 | 62.0 | 0.4311 | [Q2_mid_low_green_frame1487_score0.4311.png](output/The_Present/validation_frames/color_g_mean/Q2_mid_low_green_frame1487_score0.4311.png) |
| 2523 | 105.1 | 0.4414 | [Q2_mid_low_green_frame2523_score0.4414.png](output/The_Present/validation_frames/color_g_mean/Q2_mid_low_green_frame2523_score0.4414.png) |
| 2352 | 98.0 | 0.4505 | [Q2_mid_low_green_frame2352_score0.4505.png](output/The_Present/validation_frames/color_g_mean/Q2_mid_low_green_frame2352_score0.4505.png) |
| 3512 | 146.4 | 0.4617 | [Q2_mid_low_green_frame3512_score0.4617.png](output/The_Present/validation_frames/color_g_mean/Q2_mid_low_green_frame3512_score0.4617.png) |
| 3337 | 139.1 | 0.4738 | [Q2_mid_low_green_frame3337_score0.4738.png](output/The_Present/validation_frames/color_g_mean/Q2_mid_low_green_frame3337_score0.4738.png) |
| 1151 | 48.0 | 0.4810 | [Q2_mid_low_green_frame1151_score0.4810.png](output/The_Present/validation_frames/color_g_mean/Q2_mid_low_green_frame1151_score0.4810.png) |
| 3081 | 128.4 | 0.4884 | [Q2_mid_low_green_frame3081_score0.4884.png](output/The_Present/validation_frames/color_g_mean/Q2_mid_low_green_frame3081_score0.4884.png) |

### Q3 mid high green

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1966 | 81.9 | 0.4979 | [Q3_mid_high_green_frame1966_score0.4979.png](output/The_Present/validation_frames/color_g_mean/Q3_mid_high_green_frame1966_score0.4979.png) |
| 4224 | 176.0 | 0.5046 | [Q3_mid_high_green_frame4224_score0.5046.png](output/The_Present/validation_frames/color_g_mean/Q3_mid_high_green_frame4224_score0.5046.png) |
| 4237 | 176.6 | 0.5120 | [Q3_mid_high_green_frame4237_score0.5120.png](output/The_Present/validation_frames/color_g_mean/Q3_mid_high_green_frame4237_score0.5120.png) |
| 4182 | 174.3 | 0.5232 | [Q3_mid_high_green_frame4182_score0.5232.png](output/The_Present/validation_frames/color_g_mean/Q3_mid_high_green_frame4182_score0.5232.png) |
| 3520 | 146.7 | 0.5303 | [Q3_mid_high_green_frame3520_score0.5303.png](output/The_Present/validation_frames/color_g_mean/Q3_mid_high_green_frame3520_score0.5303.png) |
| 3879 | 161.7 | 0.5376 | [Q3_mid_high_green_frame3879_score0.5376.png](output/The_Present/validation_frames/color_g_mean/Q3_mid_high_green_frame3879_score0.5376.png) |
| 1201 | 50.1 | 0.5434 | [Q3_mid_high_green_frame1201_score0.5434.png](output/The_Present/validation_frames/color_g_mean/Q3_mid_high_green_frame1201_score0.5434.png) |
| 1190 | 49.6 | 0.5616 | [Q3_mid_high_green_frame1190_score0.5616.png](output/The_Present/validation_frames/color_g_mean/Q3_mid_high_green_frame1190_score0.5616.png) |

### Q4 high green

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 4595 | 191.5 | 0.5645 | [Q4_high_green_frame4595_score0.5645.png](output/The_Present/validation_frames/color_g_mean/Q4_high_green_frame4595_score0.5645.png) |
| 846 | 35.3 | 0.5669 | [Q4_high_green_frame0846_score0.5669.png](output/The_Present/validation_frames/color_g_mean/Q4_high_green_frame0846_score0.5669.png) |
| 4821 | 200.9 | 0.5703 | [Q4_high_green_frame4821_score0.5703.png](output/The_Present/validation_frames/color_g_mean/Q4_high_green_frame4821_score0.5703.png) |
| 1062 | 44.3 | 0.5762 | [Q4_high_green_frame1062_score0.5762.png](output/The_Present/validation_frames/color_g_mean/Q4_high_green_frame1062_score0.5762.png) |
| 2211 | 92.1 | 0.6115 | [Q4_high_green_frame2211_score0.6115.png](output/The_Present/validation_frames/color_g_mean/Q4_high_green_frame2211_score0.6115.png) |
| 2446 | 101.9 | 0.6388 | [Q4_high_green_frame2446_score0.6388.png](output/The_Present/validation_frames/color_g_mean/Q4_high_green_frame2446_score0.6388.png) |
| 3812 | 158.9 | 0.6771 | [Q4_high_green_frame3812_score0.6771.png](output/The_Present/validation_frames/color_g_mean/Q4_high_green_frame3812_score0.6771.png) |
| 3437 | 143.2 | 0.7425 | [Q4_high_green_frame3437_score0.7425.png](output/The_Present/validation_frames/color_g_mean/Q4_high_green_frame3437_score0.7425.png) |

### `color_b_mean`

### Q1 low blue

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0086 | [Q1_low_blue_frame0000_score0.0086.png](output/The_Present/validation_frames/color_b_mean/Q1_low_blue_frame0000_score0.0086.png) |
| 148 | 6.2 | 0.0336 | [Q1_low_blue_frame0148_score0.0336.png](output/The_Present/validation_frames/color_b_mean/Q1_low_blue_frame0148_score0.0336.png) |
| 420 | 17.5 | 0.1585 | [Q1_low_blue_frame0420_score0.1585.png](output/The_Present/validation_frames/color_b_mean/Q1_low_blue_frame0420_score0.1585.png) |
| 415 | 17.3 | 0.1978 | [Q1_low_blue_frame0415_score0.1978.png](output/The_Present/validation_frames/color_b_mean/Q1_low_blue_frame0415_score0.1978.png) |
| 965 | 40.2 | 0.2265 | [Q1_low_blue_frame0965_score0.2265.png](output/The_Present/validation_frames/color_b_mean/Q1_low_blue_frame0965_score0.2265.png) |
| 745 | 31.0 | 0.2480 | [Q1_low_blue_frame0745_score0.2480.png](output/The_Present/validation_frames/color_b_mean/Q1_low_blue_frame0745_score0.2480.png) |
| 526 | 21.9 | 0.3084 | [Q1_low_blue_frame0526_score0.3084.png](output/The_Present/validation_frames/color_b_mean/Q1_low_blue_frame0526_score0.3084.png) |
| 2873 | 119.7 | 0.3324 | [Q1_low_blue_frame2873_score0.3324.png](output/The_Present/validation_frames/color_b_mean/Q1_low_blue_frame2873_score0.3324.png) |

### Q2 mid low blue

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1268 | 52.8 | 0.3552 | [Q2_mid_low_blue_frame1268_score0.3552.png](output/The_Present/validation_frames/color_b_mean/Q2_mid_low_blue_frame1268_score0.3552.png) |
| 2514 | 104.8 | 0.3689 | [Q2_mid_low_blue_frame2514_score0.3689.png](output/The_Present/validation_frames/color_b_mean/Q2_mid_low_blue_frame2514_score0.3689.png) |
| 1821 | 75.9 | 0.3746 | [Q2_mid_low_blue_frame1821_score0.3746.png](output/The_Present/validation_frames/color_b_mean/Q2_mid_low_blue_frame1821_score0.3746.png) |
| 3942 | 164.3 | 0.3801 | [Q2_mid_low_blue_frame3942_score0.3801.png](output/The_Present/validation_frames/color_b_mean/Q2_mid_low_blue_frame3942_score0.3801.png) |
| 1847 | 77.0 | 0.3895 | [Q2_mid_low_blue_frame1847_score0.3895.png](output/The_Present/validation_frames/color_b_mean/Q2_mid_low_blue_frame1847_score0.3895.png) |
| 2583 | 107.6 | 0.4151 | [Q2_mid_low_blue_frame2583_score0.4151.png](output/The_Present/validation_frames/color_b_mean/Q2_mid_low_blue_frame2583_score0.4151.png) |
| 3018 | 125.8 | 0.4223 | [Q2_mid_low_blue_frame3018_score0.4223.png](output/The_Present/validation_frames/color_b_mean/Q2_mid_low_blue_frame3018_score0.4223.png) |
| 2604 | 108.5 | 0.4317 | [Q2_mid_low_blue_frame2604_score0.4317.png](output/The_Present/validation_frames/color_b_mean/Q2_mid_low_blue_frame2604_score0.4317.png) |

### Q3 mid high blue

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1988 | 82.9 | 0.4367 | [Q3_mid_high_blue_frame1988_score0.4367.png](output/The_Present/validation_frames/color_b_mean/Q3_mid_high_blue_frame1988_score0.4367.png) |
| 1140 | 47.5 | 0.4425 | [Q3_mid_high_blue_frame1140_score0.4425.png](output/The_Present/validation_frames/color_b_mean/Q3_mid_high_blue_frame1140_score0.4425.png) |
| 2641 | 110.1 | 0.4508 | [Q3_mid_high_blue_frame2641_score0.4508.png](output/The_Present/validation_frames/color_b_mean/Q3_mid_high_blue_frame2641_score0.4508.png) |
| 2838 | 118.3 | 0.4596 | [Q3_mid_high_blue_frame2838_score0.4596.png](output/The_Present/validation_frames/color_b_mean/Q3_mid_high_blue_frame2838_score0.4596.png) |
| 1645 | 68.6 | 0.4758 | [Q3_mid_high_blue_frame1645_score0.4758.png](output/The_Present/validation_frames/color_b_mean/Q3_mid_high_blue_frame1645_score0.4758.png) |
| 4238 | 176.6 | 0.4861 | [Q3_mid_high_blue_frame4238_score0.4861.png](output/The_Present/validation_frames/color_b_mean/Q3_mid_high_blue_frame4238_score0.4861.png) |
| 4245 | 176.9 | 0.4919 | [Q3_mid_high_blue_frame4245_score0.4919.png](output/The_Present/validation_frames/color_b_mean/Q3_mid_high_blue_frame4245_score0.4919.png) |
| 1666 | 69.4 | 0.4987 | [Q3_mid_high_blue_frame1666_score0.4987.png](output/The_Present/validation_frames/color_b_mean/Q3_mid_high_blue_frame1666_score0.4987.png) |

### Q4 high blue

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 4357 | 181.6 | 0.5214 | [Q4_high_blue_frame4357_score0.5214.png](output/The_Present/validation_frames/color_b_mean/Q4_high_blue_frame4357_score0.5214.png) |
| 4472 | 186.4 | 0.5293 | [Q4_high_blue_frame4472_score0.5293.png](output/The_Present/validation_frames/color_b_mean/Q4_high_blue_frame4472_score0.5293.png) |
| 4546 | 189.5 | 0.5317 | [Q4_high_blue_frame4546_score0.5317.png](output/The_Present/validation_frames/color_b_mean/Q4_high_blue_frame4546_score0.5317.png) |
| 4724 | 196.9 | 0.5361 | [Q4_high_blue_frame4724_score0.5361.png](output/The_Present/validation_frames/color_b_mean/Q4_high_blue_frame4724_score0.5361.png) |
| 4849 | 202.1 | 0.5378 | [Q4_high_blue_frame4849_score0.5378.png](output/The_Present/validation_frames/color_b_mean/Q4_high_blue_frame4849_score0.5378.png) |
| 2458 | 102.4 | 0.5890 | [Q4_high_blue_frame2458_score0.5890.png](output/The_Present/validation_frames/color_b_mean/Q4_high_blue_frame2458_score0.5890.png) |
| 3821 | 159.2 | 0.6191 | [Q4_high_blue_frame3821_score0.6191.png](output/The_Present/validation_frames/color_b_mean/Q4_high_blue_frame3821_score0.6191.png) |
| 3414 | 142.3 | 0.6872 | [Q4_high_blue_frame3414_score0.6872.png](output/The_Present/validation_frames/color_b_mean/Q4_high_blue_frame3414_score0.6872.png) |

### `saturation_mean`

### Q1 desaturated

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 864 | 36.0 | 0.0675 | [Q1_desaturated_frame0864_score0.0675.png](output/The_Present/validation_frames/saturation_mean/Q1_desaturated_frame0864_score0.0675.png) |
| 3987 | 166.2 | 0.1622 | [Q1_desaturated_frame3987_score0.1622.png](output/The_Present/validation_frames/saturation_mean/Q1_desaturated_frame3987_score0.1622.png) |
| 2095 | 87.3 | 0.1771 | [Q1_desaturated_frame2095_score0.1771.png](output/The_Present/validation_frames/saturation_mean/Q1_desaturated_frame2095_score0.1771.png) |
| 4674 | 194.8 | 0.1971 | [Q1_desaturated_frame4674_score0.1971.png](output/The_Present/validation_frames/saturation_mean/Q1_desaturated_frame4674_score0.1971.png) |
| 4645 | 193.6 | 0.1978 | [Q1_desaturated_frame4645_score0.1978.png](output/The_Present/validation_frames/saturation_mean/Q1_desaturated_frame4645_score0.1978.png) |
| 4595 | 191.5 | 0.2004 | [Q1_desaturated_frame4595_score0.2004.png](output/The_Present/validation_frames/saturation_mean/Q1_desaturated_frame4595_score0.2004.png) |
| 4502 | 187.6 | 0.2043 | [Q1_desaturated_frame4502_score0.2043.png](output/The_Present/validation_frames/saturation_mean/Q1_desaturated_frame4502_score0.2043.png) |
| 2822 | 117.6 | 0.2200 | [Q1_desaturated_frame2822_score0.2200.png](output/The_Present/validation_frames/saturation_mean/Q1_desaturated_frame2822_score0.2200.png) |

### Q2 mid low sat

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 4182 | 174.3 | 0.2314 | [Q2_mid_low_sat_frame4182_score0.2314.png](output/The_Present/validation_frames/saturation_mean/Q2_mid_low_sat_frame4182_score0.2314.png) |
| 2294 | 95.6 | 0.2504 | [Q2_mid_low_sat_frame2294_score0.2504.png](output/The_Present/validation_frames/saturation_mean/Q2_mid_low_sat_frame2294_score0.2504.png) |
| 1060 | 44.2 | 0.2634 | [Q2_mid_low_sat_frame1060_score0.2634.png](output/The_Present/validation_frames/saturation_mean/Q2_mid_low_sat_frame1060_score0.2634.png) |
| 3895 | 162.3 | 0.2737 | [Q2_mid_low_sat_frame3895_score0.2737.png](output/The_Present/validation_frames/saturation_mean/Q2_mid_low_sat_frame3895_score0.2737.png) |
| 3630 | 151.3 | 0.2810 | [Q2_mid_low_sat_frame3630_score0.2810.png](output/The_Present/validation_frames/saturation_mean/Q2_mid_low_sat_frame3630_score0.2810.png) |
| 2968 | 123.7 | 0.2874 | [Q2_mid_low_sat_frame2968_score0.2874.png](output/The_Present/validation_frames/saturation_mean/Q2_mid_low_sat_frame2968_score0.2874.png) |
| 2994 | 124.8 | 0.2906 | [Q2_mid_low_sat_frame2994_score0.2906.png](output/The_Present/validation_frames/saturation_mean/Q2_mid_low_sat_frame2994_score0.2906.png) |
| 1968 | 82.0 | 0.2944 | [Q2_mid_low_sat_frame1968_score0.2944.png](output/The_Present/validation_frames/saturation_mean/Q2_mid_low_sat_frame1968_score0.2944.png) |

### Q3 mid high sat

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3922 | 163.5 | 0.3047 | [Q3_mid_high_sat_frame3922_score0.3047.png](output/The_Present/validation_frames/saturation_mean/Q3_mid_high_sat_frame3922_score0.3047.png) |
| 3039 | 126.7 | 0.3118 | [Q3_mid_high_sat_frame3039_score0.3118.png](output/The_Present/validation_frames/saturation_mean/Q3_mid_high_sat_frame3039_score0.3118.png) |
| 3136 | 130.7 | 0.3158 | [Q3_mid_high_sat_frame3136_score0.3158.png](output/The_Present/validation_frames/saturation_mean/Q3_mid_high_sat_frame3136_score0.3158.png) |
| 3058 | 127.4 | 0.3188 | [Q3_mid_high_sat_frame3058_score0.3188.png](output/The_Present/validation_frames/saturation_mean/Q3_mid_high_sat_frame3058_score0.3188.png) |
| 1848 | 77.0 | 0.3225 | [Q3_mid_high_sat_frame1848_score0.3225.png](output/The_Present/validation_frames/saturation_mean/Q3_mid_high_sat_frame1848_score0.3225.png) |
| 3746 | 156.1 | 0.3275 | [Q3_mid_high_sat_frame3746_score0.3275.png](output/The_Present/validation_frames/saturation_mean/Q3_mid_high_sat_frame3746_score0.3275.png) |
| 3248 | 135.4 | 0.3329 | [Q3_mid_high_sat_frame3248_score0.3329.png](output/The_Present/validation_frames/saturation_mean/Q3_mid_high_sat_frame3248_score0.3329.png) |
| 1121 | 46.7 | 0.3388 | [Q3_mid_high_sat_frame1121_score0.3388.png](output/The_Present/validation_frames/saturation_mean/Q3_mid_high_sat_frame1121_score0.3388.png) |

### Q4 vivid

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1550 | 64.6 | 0.3426 | [Q4_vivid_frame1550_score0.3426.png](output/The_Present/validation_frames/saturation_mean/Q4_vivid_frame1550_score0.3426.png) |
| 1840 | 76.7 | 0.3519 | [Q4_vivid_frame1840_score0.3519.png](output/The_Present/validation_frames/saturation_mean/Q4_vivid_frame1840_score0.3519.png) |
| 1614 | 67.3 | 0.3608 | [Q4_vivid_frame1614_score0.3608.png](output/The_Present/validation_frames/saturation_mean/Q4_vivid_frame1614_score0.3608.png) |
| 1260 | 52.5 | 0.3663 | [Q4_vivid_frame1260_score0.3663.png](output/The_Present/validation_frames/saturation_mean/Q4_vivid_frame1260_score0.3663.png) |
| 3524 | 146.9 | 0.3812 | [Q4_vivid_frame3524_score0.3812.png](output/The_Present/validation_frames/saturation_mean/Q4_vivid_frame3524_score0.3812.png) |
| 798 | 33.3 | 0.3968 | [Q4_vivid_frame0798_score0.3968.png](output/The_Present/validation_frames/saturation_mean/Q4_vivid_frame0798_score0.3968.png) |
| 1358 | 56.6 | 0.4290 | [Q4_vivid_frame1358_score0.4290.png](output/The_Present/validation_frames/saturation_mean/Q4_vivid_frame1358_score0.4290.png) |
| 92 | 3.8 | 0.9067 | [Q4_vivid_frame0092_score0.9067.png](output/The_Present/validation_frames/saturation_mean/Q4_vivid_frame0092_score0.9067.png) |

---

## 6. Texture Features — Visual Inspection

### `edge_density`

### Q1 smooth

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 150 | 6.3 | 0.0008 | [Q1_smooth_frame0150_score0.0008.png](output/The_Present/validation_frames/edge_density/Q1_smooth_frame0150_score0.0008.png) |
| 89 | 3.7 | 0.0081 | [Q1_smooth_frame0089_score0.0081.png](output/The_Present/validation_frames/edge_density/Q1_smooth_frame0089_score0.0081.png) |
| 2776 | 115.7 | 0.0166 | [Q1_smooth_frame2776_score0.0166.png](output/The_Present/validation_frames/edge_density/Q1_smooth_frame2776_score0.0166.png) |
| 228 | 9.5 | 0.0228 | [Q1_smooth_frame0228_score0.0228.png](output/The_Present/validation_frames/edge_density/Q1_smooth_frame0228_score0.0228.png) |
| 196 | 8.2 | 0.0251 | [Q1_smooth_frame0196_score0.0251.png](output/The_Present/validation_frames/edge_density/Q1_smooth_frame0196_score0.0251.png) |
| 187 | 7.8 | 0.0274 | [Q1_smooth_frame0187_score0.0274.png](output/The_Present/validation_frames/edge_density/Q1_smooth_frame0187_score0.0274.png) |
| 1212 | 50.5 | 0.0302 | [Q1_smooth_frame1212_score0.0302.png](output/The_Present/validation_frames/edge_density/Q1_smooth_frame1212_score0.0302.png) |
| 2422 | 100.9 | 0.0317 | [Q1_smooth_frame2422_score0.0317.png](output/The_Present/validation_frames/edge_density/Q1_smooth_frame2422_score0.0317.png) |

### Q2 mid low edge

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 900 | 37.5 | 0.0332 | [Q2_mid_low_edge_frame0900_score0.0332.png](output/The_Present/validation_frames/edge_density/Q2_mid_low_edge_frame0900_score0.0332.png) |
| 3139 | 130.8 | 0.0341 | [Q2_mid_low_edge_frame3139_score0.0341.png](output/The_Present/validation_frames/edge_density/Q2_mid_low_edge_frame3139_score0.0341.png) |
| 3184 | 132.7 | 0.0353 | [Q2_mid_low_edge_frame3184_score0.0353.png](output/The_Present/validation_frames/edge_density/Q2_mid_low_edge_frame3184_score0.0353.png) |
| 3747 | 156.2 | 0.0364 | [Q2_mid_low_edge_frame3747_score0.0364.png](output/The_Present/validation_frames/edge_density/Q2_mid_low_edge_frame3747_score0.0364.png) |
| 2193 | 91.4 | 0.0381 | [Q2_mid_low_edge_frame2193_score0.0381.png](output/The_Present/validation_frames/edge_density/Q2_mid_low_edge_frame2193_score0.0381.png) |
| 527 | 22.0 | 0.0410 | [Q2_mid_low_edge_frame0527_score0.0410.png](output/The_Present/validation_frames/edge_density/Q2_mid_low_edge_frame0527_score0.0410.png) |
| 1902 | 79.3 | 0.0441 | [Q2_mid_low_edge_frame1902_score0.0441.png](output/The_Present/validation_frames/edge_density/Q2_mid_low_edge_frame1902_score0.0441.png) |
| 3561 | 148.4 | 0.0461 | [Q2_mid_low_edge_frame3561_score0.0461.png](output/The_Present/validation_frames/edge_density/Q2_mid_low_edge_frame3561_score0.0461.png) |

### Q3 mid high edge

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 408 | 17.0 | 0.0483 | [Q3_mid_high_edge_frame0408_score0.0483.png](output/The_Present/validation_frames/edge_density/Q3_mid_high_edge_frame0408_score0.0483.png) |
| 1249 | 52.1 | 0.0502 | [Q3_mid_high_edge_frame1249_score0.0502.png](output/The_Present/validation_frames/edge_density/Q3_mid_high_edge_frame1249_score0.0502.png) |
| 734 | 30.6 | 0.0516 | [Q3_mid_high_edge_frame0734_score0.0516.png](output/The_Present/validation_frames/edge_density/Q3_mid_high_edge_frame0734_score0.0516.png) |
| 1604 | 66.8 | 0.0527 | [Q3_mid_high_edge_frame1604_score0.0527.png](output/The_Present/validation_frames/edge_density/Q3_mid_high_edge_frame1604_score0.0527.png) |
| 2031 | 84.6 | 0.0542 | [Q3_mid_high_edge_frame2031_score0.0542.png](output/The_Present/validation_frames/edge_density/Q3_mid_high_edge_frame2031_score0.0542.png) |
| 4200 | 175.0 | 0.0558 | [Q3_mid_high_edge_frame4200_score0.0558.png](output/The_Present/validation_frames/edge_density/Q3_mid_high_edge_frame4200_score0.0558.png) |
| 4285 | 178.6 | 0.0578 | [Q3_mid_high_edge_frame4285_score0.0578.png](output/The_Present/validation_frames/edge_density/Q3_mid_high_edge_frame4285_score0.0578.png) |
| 1534 | 63.9 | 0.0609 | [Q3_mid_high_edge_frame1534_score0.0609.png](output/The_Present/validation_frames/edge_density/Q3_mid_high_edge_frame1534_score0.0609.png) |

### Q4 edgy

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2580 | 107.5 | 0.0627 | [Q4_edgy_frame2580_score0.0627.png](output/The_Present/validation_frames/edge_density/Q4_edgy_frame2580_score0.0627.png) |
| 4814 | 200.6 | 0.0692 | [Q4_edgy_frame4814_score0.0692.png](output/The_Present/validation_frames/edge_density/Q4_edgy_frame4814_score0.0692.png) |
| 4700 | 195.9 | 0.0715 | [Q4_edgy_frame4700_score0.0715.png](output/The_Present/validation_frames/edge_density/Q4_edgy_frame4700_score0.0715.png) |
| 4574 | 190.6 | 0.0732 | [Q4_edgy_frame4574_score0.0732.png](output/The_Present/validation_frames/edge_density/Q4_edgy_frame4574_score0.0732.png) |
| 4482 | 186.8 | 0.0755 | [Q4_edgy_frame4482_score0.0755.png](output/The_Present/validation_frames/edge_density/Q4_edgy_frame4482_score0.0755.png) |
| 2934 | 122.3 | 0.1229 | [Q4_edgy_frame2934_score0.1229.png](output/The_Present/validation_frames/edge_density/Q4_edgy_frame2934_score0.1229.png) |
| 3703 | 154.3 | 0.1411 | [Q4_edgy_frame3703_score0.1411.png](output/The_Present/validation_frames/edge_density/Q4_edgy_frame3703_score0.1411.png) |
| 3978 | 165.8 | 0.1899 | [Q4_edgy_frame3978_score0.1899.png](output/The_Present/validation_frames/edge_density/Q4_edgy_frame3978_score0.1899.png) |

### `spatial_freq_energy`

### Q1 low freq

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 861 | 35.9 | 0.0001 | [Q1_low_freq_frame0861_score0.0001.png](output/The_Present/validation_frames/spatial_freq_energy/Q1_low_freq_frame0861_score0.0001.png) |
| 3581 | 149.2 | 0.0005 | [Q1_low_freq_frame3581_score0.0005.png](output/The_Present/validation_frames/spatial_freq_energy/Q1_low_freq_frame3581_score0.0005.png) |
| 951 | 39.6 | 0.0005 | [Q1_low_freq_frame0951_score0.0005.png](output/The_Present/validation_frames/spatial_freq_energy/Q1_low_freq_frame0951_score0.0005.png) |
| 1064 | 44.3 | 0.0006 | [Q1_low_freq_frame1064_score0.0006.png](output/The_Present/validation_frames/spatial_freq_energy/Q1_low_freq_frame1064_score0.0006.png) |
| 2843 | 118.5 | 0.0006 | [Q1_low_freq_frame2843_score0.0006.png](output/The_Present/validation_frames/spatial_freq_energy/Q1_low_freq_frame2843_score0.0006.png) |
| 1682 | 70.1 | 0.0007 | [Q1_low_freq_frame1682_score0.0007.png](output/The_Present/validation_frames/spatial_freq_energy/Q1_low_freq_frame1682_score0.0007.png) |
| 4006 | 167.0 | 0.0007 | [Q1_low_freq_frame4006_score0.0007.png](output/The_Present/validation_frames/spatial_freq_energy/Q1_low_freq_frame4006_score0.0007.png) |
| 3397 | 141.6 | 0.0007 | [Q1_low_freq_frame3397_score0.0007.png](output/The_Present/validation_frames/spatial_freq_energy/Q1_low_freq_frame3397_score0.0007.png) |

### Q2 mid low freq

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3554 | 148.1 | 0.0007 | [Q2_mid_low_freq_frame3554_score0.0007.png](output/The_Present/validation_frames/spatial_freq_energy/Q2_mid_low_freq_frame3554_score0.0007.png) |
| 1243 | 51.8 | 0.0008 | [Q2_mid_low_freq_frame1243_score0.0008.png](output/The_Present/validation_frames/spatial_freq_energy/Q2_mid_low_freq_frame1243_score0.0008.png) |
| 1769 | 73.7 | 0.0008 | [Q2_mid_low_freq_frame1769_score0.0008.png](output/The_Present/validation_frames/spatial_freq_energy/Q2_mid_low_freq_frame1769_score0.0008.png) |
| 1349 | 56.2 | 0.0008 | [Q2_mid_low_freq_frame1349_score0.0008.png](output/The_Present/validation_frames/spatial_freq_energy/Q2_mid_low_freq_frame1349_score0.0008.png) |
| 532 | 22.2 | 0.0009 | [Q2_mid_low_freq_frame0532_score0.0009.png](output/The_Present/validation_frames/spatial_freq_energy/Q2_mid_low_freq_frame0532_score0.0009.png) |
| 3633 | 151.4 | 0.0009 | [Q2_mid_low_freq_frame3633_score0.0009.png](output/The_Present/validation_frames/spatial_freq_energy/Q2_mid_low_freq_frame3633_score0.0009.png) |
| 1829 | 76.2 | 0.0010 | [Q2_mid_low_freq_frame1829_score0.0010.png](output/The_Present/validation_frames/spatial_freq_energy/Q2_mid_low_freq_frame1829_score0.0010.png) |
| 3639 | 151.7 | 0.0010 | [Q2_mid_low_freq_frame3639_score0.0010.png](output/The_Present/validation_frames/spatial_freq_energy/Q2_mid_low_freq_frame3639_score0.0010.png) |

### Q3 mid high freq

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 761 | 31.7 | 0.0010 | [Q3_mid_high_freq_frame0761_score0.0010.png](output/The_Present/validation_frames/spatial_freq_energy/Q3_mid_high_freq_frame0761_score0.0010.png) |
| 2525 | 105.2 | 0.0011 | [Q3_mid_high_freq_frame2525_score0.0011.png](output/The_Present/validation_frames/spatial_freq_energy/Q3_mid_high_freq_frame2525_score0.0011.png) |
| 2671 | 111.3 | 0.0012 | [Q3_mid_high_freq_frame2671_score0.0012.png](output/The_Present/validation_frames/spatial_freq_energy/Q3_mid_high_freq_frame2671_score0.0012.png) |
| 589 | 24.5 | 0.0013 | [Q3_mid_high_freq_frame0589_score0.0013.png](output/The_Present/validation_frames/spatial_freq_energy/Q3_mid_high_freq_frame0589_score0.0013.png) |
| 4274 | 178.1 | 0.0015 | [Q3_mid_high_freq_frame4274_score0.0015.png](output/The_Present/validation_frames/spatial_freq_energy/Q3_mid_high_freq_frame4274_score0.0015.png) |
| 4293 | 178.9 | 0.0016 | [Q3_mid_high_freq_frame4293_score0.0016.png](output/The_Present/validation_frames/spatial_freq_energy/Q3_mid_high_freq_frame4293_score0.0016.png) |
| 4152 | 173.0 | 0.0017 | [Q3_mid_high_freq_frame4152_score0.0017.png](output/The_Present/validation_frames/spatial_freq_energy/Q3_mid_high_freq_frame4152_score0.0017.png) |
| 3005 | 125.2 | 0.0018 | [Q3_mid_high_freq_frame3005_score0.0018.png](output/The_Present/validation_frames/spatial_freq_energy/Q3_mid_high_freq_frame3005_score0.0018.png) |

### Q4 high freq

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2470 | 102.9 | 0.0018 | [Q4_high_freq_frame2470_score0.0018.png](output/The_Present/validation_frames/spatial_freq_energy/Q4_high_freq_frame2470_score0.0018.png) |
| 4435 | 184.8 | 0.0019 | [Q4_high_freq_frame4435_score0.0019.png](output/The_Present/validation_frames/spatial_freq_energy/Q4_high_freq_frame4435_score0.0019.png) |
| 4762 | 198.5 | 0.0019 | [Q4_high_freq_frame4762_score0.0019.png](output/The_Present/validation_frames/spatial_freq_energy/Q4_high_freq_frame4762_score0.0019.png) |
| 4506 | 187.8 | 0.0020 | [Q4_high_freq_frame4506_score0.0020.png](output/The_Present/validation_frames/spatial_freq_energy/Q4_high_freq_frame4506_score0.0020.png) |
| 2583 | 107.6 | 0.0020 | [Q4_high_freq_frame2583_score0.0020.png](output/The_Present/validation_frames/spatial_freq_energy/Q4_high_freq_frame2583_score0.0020.png) |
| 1006 | 41.9 | 0.0025 | [Q4_high_freq_frame1006_score0.0025.png](output/The_Present/validation_frames/spatial_freq_energy/Q4_high_freq_frame1006_score0.0025.png) |
| 491 | 20.5 | 0.0052 | [Q4_high_freq_frame0491_score0.0052.png](output/The_Present/validation_frames/spatial_freq_energy/Q4_high_freq_frame0491_score0.0052.png) |
| 148 | 6.2 | 0.0092 | [Q4_high_freq_frame0148_score0.0092.png](output/The_Present/validation_frames/spatial_freq_energy/Q4_high_freq_frame0148_score0.0092.png) |

---

## 7. Motion Features — Visual Inspection

### `motion_energy`

### Q1 static

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0000 | [Q1_static_frame0000_score0.0000.png](output/The_Present/validation_frames/motion_energy/Q1_static_frame0000_score0.0000.png) |
| 117 | 4.9 | 0.0051 | [Q1_static_frame0117_score0.0051.png](output/The_Present/validation_frames/motion_energy/Q1_static_frame0117_score0.0051.png) |
| 4851 | 202.2 | 0.0225 | [Q1_static_frame4851_score0.0225.png](output/The_Present/validation_frames/motion_energy/Q1_static_frame4851_score0.0225.png) |
| 861 | 35.9 | 0.0347 | [Q1_static_frame0861_score0.0347.png](output/The_Present/validation_frames/motion_energy/Q1_static_frame0861_score0.0347.png) |
| 4416 | 184.0 | 0.0507 | [Q1_static_frame4416_score0.0507.png](output/The_Present/validation_frames/motion_energy/Q1_static_frame4416_score0.0507.png) |
| 3707 | 154.5 | 0.0645 | [Q1_static_frame3707_score0.0645.png](output/The_Present/validation_frames/motion_energy/Q1_static_frame3707_score0.0645.png) |
| 3951 | 164.7 | 0.0800 | [Q1_static_frame3951_score0.0800.png](output/The_Present/validation_frames/motion_energy/Q1_static_frame3951_score0.0800.png) |
| 3720 | 155.0 | 0.0954 | [Q1_static_frame3720_score0.0954.png](output/The_Present/validation_frames/motion_energy/Q1_static_frame3720_score0.0954.png) |

### Q2 slow motion

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1121 | 46.7 | 0.1110 | [Q2_slow_motion_frame1121_score0.1110.png](output/The_Present/validation_frames/motion_energy/Q2_slow_motion_frame1121_score0.1110.png) |
| 2915 | 121.5 | 0.1287 | [Q2_slow_motion_frame2915_score0.1287.png](output/The_Present/validation_frames/motion_energy/Q2_slow_motion_frame2915_score0.1287.png) |
| 3714 | 154.8 | 0.1474 | [Q2_slow_motion_frame3714_score0.1474.png](output/The_Present/validation_frames/motion_energy/Q2_slow_motion_frame3714_score0.1474.png) |
| 1255 | 52.3 | 0.1683 | [Q2_slow_motion_frame1255_score0.1683.png](output/The_Present/validation_frames/motion_energy/Q2_slow_motion_frame1255_score0.1683.png) |
| 468 | 19.5 | 0.1917 | [Q2_slow_motion_frame0468_score0.1917.png](output/The_Present/validation_frames/motion_energy/Q2_slow_motion_frame0468_score0.1917.png) |
| 2664 | 111.0 | 0.2170 | [Q2_slow_motion_frame2664_score0.2170.png](output/The_Present/validation_frames/motion_energy/Q2_slow_motion_frame2664_score0.2170.png) |
| 903 | 37.6 | 0.2437 | [Q2_slow_motion_frame0903_score0.2437.png](output/The_Present/validation_frames/motion_energy/Q2_slow_motion_frame0903_score0.2437.png) |
| 3330 | 138.8 | 0.2693 | [Q2_slow_motion_frame3330_score0.2693.png](output/The_Present/validation_frames/motion_energy/Q2_slow_motion_frame3330_score0.2693.png) |

### Q3 mid motion

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 514 | 21.4 | 0.3057 | [Q3_mid_motion_frame0514_score0.3057.png](output/The_Present/validation_frames/motion_energy/Q3_mid_motion_frame0514_score0.3057.png) |
| 3392 | 141.4 | 0.3406 | [Q3_mid_motion_frame3392_score0.3406.png](output/The_Present/validation_frames/motion_energy/Q3_mid_motion_frame3392_score0.3406.png) |
| 1774 | 73.9 | 0.3866 | [Q3_mid_motion_frame1774_score0.3866.png](output/The_Present/validation_frames/motion_energy/Q3_mid_motion_frame1774_score0.3866.png) |
| 1007 | 42.0 | 0.4568 | [Q3_mid_motion_frame1007_score0.4568.png](output/The_Present/validation_frames/motion_energy/Q3_mid_motion_frame1007_score0.4568.png) |
| 2531 | 105.5 | 0.5261 | [Q3_mid_motion_frame2531_score0.5261.png](output/The_Present/validation_frames/motion_energy/Q3_mid_motion_frame2531_score0.5261.png) |
| 3809 | 158.7 | 0.6358 | [Q3_mid_motion_frame3809_score0.6358.png](output/The_Present/validation_frames/motion_energy/Q3_mid_motion_frame3809_score0.6358.png) |
| 1194 | 49.8 | 0.7719 | [Q3_mid_motion_frame1194_score0.7719.png](output/The_Present/validation_frames/motion_energy/Q3_mid_motion_frame1194_score0.7719.png) |
| 254 | 10.6 | 0.9374 | [Q3_mid_motion_frame0254_score0.9374.png](output/The_Present/validation_frames/motion_energy/Q3_mid_motion_frame0254_score0.9374.png) |

### Q4 fast motion

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1929 | 80.4 | 1.1320 | [Q4_fast_motion_frame1929_score1.1320.png](output/The_Present/validation_frames/motion_energy/Q4_fast_motion_frame1929_score1.1320.png) |
| 2691 | 112.1 | 1.3755 | [Q4_fast_motion_frame2691_score1.3755.png](output/The_Present/validation_frames/motion_energy/Q4_fast_motion_frame2691_score1.3755.png) |
| 3131 | 130.5 | 1.6935 | [Q4_fast_motion_frame3131_score1.6935.png](output/The_Present/validation_frames/motion_energy/Q4_fast_motion_frame3131_score1.6935.png) |
| 1516 | 63.2 | 2.0657 | [Q4_fast_motion_frame1516_score2.0657.png](output/The_Present/validation_frames/motion_energy/Q4_fast_motion_frame1516_score2.0657.png) |
| 2155 | 89.8 | 2.5827 | [Q4_fast_motion_frame2155_score2.5827.png](output/The_Present/validation_frames/motion_energy/Q4_fast_motion_frame2155_score2.5827.png) |
| 2784 | 116.0 | 3.3573 | [Q4_fast_motion_frame2784_score3.3573.png](output/The_Present/validation_frames/motion_energy/Q4_fast_motion_frame2784_score3.3573.png) |
| 2601 | 108.4 | 4.9249 | [Q4_fast_motion_frame2601_score4.9249.png](output/The_Present/validation_frames/motion_energy/Q4_fast_motion_frame2601_score4.9249.png) |
| 685 | 28.5 | 8.3026 | [Q4_fast_motion_frame0685_score8.3026.png](output/The_Present/validation_frames/motion_energy/Q4_fast_motion_frame0685_score8.3026.png) |

### `scene_cut`

3-frame context (before / cut / after) for each detected cut.

| Cut frame | Offset | Frame idx | Timestamp (s) | Image |
|-----------|--------|-----------|--------------|-------|
| 174 | before | 173 | 7.2 | [cut0174_before_frame0173.png](output/The_Present/validation_frames/scene_cut/cut0174_before_frame0173.png) |
| 174 | cut | 174 | 7.3 | [cut0174_cut_frame0174.png](output/The_Present/validation_frames/scene_cut/cut0174_cut_frame0174.png) |
| 174 | after | 175 | 7.3 | [cut0174_after_frame0175.png](output/The_Present/validation_frames/scene_cut/cut0174_after_frame0175.png) |
| 259 | before | 258 | 10.8 | [cut0259_before_frame0258.png](output/The_Present/validation_frames/scene_cut/cut0259_before_frame0258.png) |
| 259 | cut | 259 | 10.8 | [cut0259_cut_frame0259.png](output/The_Present/validation_frames/scene_cut/cut0259_cut_frame0259.png) |
| 259 | after | 260 | 10.8 | [cut0259_after_frame0260.png](output/The_Present/validation_frames/scene_cut/cut0259_after_frame0260.png) |
| 605 | before | 604 | 25.2 | [cut0605_before_frame0604.png](output/The_Present/validation_frames/scene_cut/cut0605_before_frame0604.png) |
| 605 | cut | 605 | 25.2 | [cut0605_cut_frame0605.png](output/The_Present/validation_frames/scene_cut/cut0605_cut_frame0605.png) |
| 605 | after | 606 | 25.3 | [cut0605_after_frame0606.png](output/The_Present/validation_frames/scene_cut/cut0605_after_frame0606.png) |
| 792 | before | 791 | 33.0 | [cut0792_before_frame0791.png](output/The_Present/validation_frames/scene_cut/cut0792_before_frame0791.png) |
| 792 | cut | 792 | 33.0 | [cut0792_cut_frame0792.png](output/The_Present/validation_frames/scene_cut/cut0792_cut_frame0792.png) |
| 792 | after | 793 | 33.0 | [cut0792_after_frame0793.png](output/The_Present/validation_frames/scene_cut/cut0792_after_frame0793.png) |
| 881 | before | 880 | 36.7 | [cut0881_before_frame0880.png](output/The_Present/validation_frames/scene_cut/cut0881_before_frame0880.png) |
| 881 | cut | 881 | 36.7 | [cut0881_cut_frame0881.png](output/The_Present/validation_frames/scene_cut/cut0881_cut_frame0881.png) |
| 881 | after | 882 | 36.8 | [cut0881_after_frame0882.png](output/The_Present/validation_frames/scene_cut/cut0881_after_frame0882.png) |
| 957 | before | 956 | 39.8 | [cut0957_before_frame0956.png](output/The_Present/validation_frames/scene_cut/cut0957_before_frame0956.png) |
| 957 | cut | 957 | 39.9 | [cut0957_cut_frame0957.png](output/The_Present/validation_frames/scene_cut/cut0957_cut_frame0957.png) |
| 957 | after | 958 | 39.9 | [cut0957_after_frame0958.png](output/The_Present/validation_frames/scene_cut/cut0957_after_frame0958.png) |
| 1535 | before | 1534 | 63.9 | [cut1535_before_frame1534.png](output/The_Present/validation_frames/scene_cut/cut1535_before_frame1534.png) |
| 1535 | cut | 1535 | 64.0 | [cut1535_cut_frame1535.png](output/The_Present/validation_frames/scene_cut/cut1535_cut_frame1535.png) |
| 1535 | after | 1536 | 64.0 | [cut1535_after_frame1536.png](output/The_Present/validation_frames/scene_cut/cut1535_after_frame1536.png) |
| 2192 | before | 2191 | 91.3 | [cut2192_before_frame2191.png](output/The_Present/validation_frames/scene_cut/cut2192_before_frame2191.png) |
| 2192 | cut | 2192 | 91.4 | [cut2192_cut_frame2192.png](output/The_Present/validation_frames/scene_cut/cut2192_cut_frame2192.png) |
| 2192 | after | 2193 | 91.4 | [cut2192_after_frame2193.png](output/The_Present/validation_frames/scene_cut/cut2192_after_frame2193.png) |
| 2506 | before | 2505 | 104.4 | [cut2506_before_frame2505.png](output/The_Present/validation_frames/scene_cut/cut2506_before_frame2505.png) |
| 2506 | cut | 2506 | 104.4 | [cut2506_cut_frame2506.png](output/The_Present/validation_frames/scene_cut/cut2506_cut_frame2506.png) |
| 2506 | after | 2507 | 104.5 | [cut2506_after_frame2507.png](output/The_Present/validation_frames/scene_cut/cut2506_after_frame2507.png) |
| 2519 | before | 2518 | 104.9 | [cut2519_before_frame2518.png](output/The_Present/validation_frames/scene_cut/cut2519_before_frame2518.png) |
| 2519 | cut | 2519 | 105.0 | [cut2519_cut_frame2519.png](output/The_Present/validation_frames/scene_cut/cut2519_cut_frame2519.png) |
| 2519 | after | 2520 | 105.0 | [cut2519_after_frame2520.png](output/The_Present/validation_frames/scene_cut/cut2519_after_frame2520.png) |
| 2787 | before | 2786 | 116.1 | [cut2787_before_frame2786.png](output/The_Present/validation_frames/scene_cut/cut2787_before_frame2786.png) |
| 2787 | cut | 2787 | 116.1 | [cut2787_cut_frame2787.png](output/The_Present/validation_frames/scene_cut/cut2787_cut_frame2787.png) |
| 2787 | after | 2788 | 116.2 | [cut2787_after_frame2788.png](output/The_Present/validation_frames/scene_cut/cut2787_after_frame2788.png) |
| 2811 | before | 2810 | 117.1 | [cut2811_before_frame2810.png](output/The_Present/validation_frames/scene_cut/cut2811_before_frame2810.png) |
| 2811 | cut | 2811 | 117.1 | [cut2811_cut_frame2811.png](output/The_Present/validation_frames/scene_cut/cut2811_cut_frame2811.png) |
| 2811 | after | 2812 | 117.2 | [cut2811_after_frame2812.png](output/The_Present/validation_frames/scene_cut/cut2811_after_frame2812.png) |
| 2865 | before | 2864 | 119.4 | [cut2865_before_frame2864.png](output/The_Present/validation_frames/scene_cut/cut2865_before_frame2864.png) |
| 2865 | cut | 2865 | 119.4 | [cut2865_cut_frame2865.png](output/The_Present/validation_frames/scene_cut/cut2865_cut_frame2865.png) |
| 2865 | after | 2866 | 119.4 | [cut2865_after_frame2866.png](output/The_Present/validation_frames/scene_cut/cut2865_after_frame2866.png) |
| 3278 | before | 3277 | 136.6 | [cut3278_before_frame3277.png](output/The_Present/validation_frames/scene_cut/cut3278_before_frame3277.png) |
| 3278 | cut | 3278 | 136.6 | [cut3278_cut_frame3278.png](output/The_Present/validation_frames/scene_cut/cut3278_cut_frame3278.png) |
| 3278 | after | 3279 | 136.7 | [cut3278_after_frame3279.png](output/The_Present/validation_frames/scene_cut/cut3278_after_frame3279.png) |
| 3405 | before | 3404 | 141.9 | [cut3405_before_frame3404.png](output/The_Present/validation_frames/scene_cut/cut3405_before_frame3404.png) |
| 3405 | cut | 3405 | 141.9 | [cut3405_cut_frame3405.png](output/The_Present/validation_frames/scene_cut/cut3405_cut_frame3405.png) |
| 3405 | after | 3406 | 141.9 | [cut3405_after_frame3406.png](output/The_Present/validation_frames/scene_cut/cut3405_after_frame3406.png) |
| 3621 | before | 3620 | 150.9 | [cut3621_before_frame3620.png](output/The_Present/validation_frames/scene_cut/cut3621_before_frame3620.png) |
| 3621 | cut | 3621 | 150.9 | [cut3621_cut_frame3621.png](output/The_Present/validation_frames/scene_cut/cut3621_cut_frame3621.png) |
| 3621 | after | 3622 | 150.9 | [cut3621_after_frame3622.png](output/The_Present/validation_frames/scene_cut/cut3621_after_frame3622.png) |
| 3855 | before | 3854 | 160.6 | [cut3855_before_frame3854.png](output/The_Present/validation_frames/scene_cut/cut3855_before_frame3854.png) |
| 3855 | cut | 3855 | 160.7 | [cut3855_cut_frame3855.png](output/The_Present/validation_frames/scene_cut/cut3855_cut_frame3855.png) |
| 3855 | after | 3856 | 160.7 | [cut3855_after_frame3856.png](output/The_Present/validation_frames/scene_cut/cut3855_after_frame3856.png) |
| 3966 | before | 3965 | 165.2 | [cut3966_before_frame3965.png](output/The_Present/validation_frames/scene_cut/cut3966_before_frame3965.png) |
| 3966 | cut | 3966 | 165.3 | [cut3966_cut_frame3966.png](output/The_Present/validation_frames/scene_cut/cut3966_cut_frame3966.png) |
| 3966 | after | 3967 | 165.3 | [cut3966_after_frame3967.png](output/The_Present/validation_frames/scene_cut/cut3966_after_frame3967.png) |

---

## 8. Face Features — Visual Inspection

### `n_faces` (count bins)

#### n_faces = 0

| Frame idx | Timestamp (s) | face_area_frac | Image |
|-----------|--------------|----------------|-------|
| 0 | 0.0 | 0.000 | [nfaces0_frame0000_frac0.000.png](output/The_Present/validation_frames/n_faces/nfaces0_frame0000_frac0.000.png) |
| 1766 | 73.6 | 0.000 | [nfaces0_frame1766_frac0.000.png](output/The_Present/validation_frames/n_faces/nfaces0_frame1766_frac0.000.png) |
| 2944 | 122.7 | 0.000 | [nfaces0_frame2944_frac0.000.png](output/The_Present/validation_frames/n_faces/nfaces0_frame2944_frac0.000.png) |
| 4130 | 172.1 | 0.000 | [nfaces0_frame4130_frac0.000.png](output/The_Present/validation_frames/n_faces/nfaces0_frame4130_frac0.000.png) |

#### n_faces = 1

| Frame idx | Timestamp (s) | face_area_frac | Image |
|-----------|--------------|----------------|-------|
| 174 | 7.3 | 0.481 | [nfaces1_frame0174_frac0.481.png](output/The_Present/validation_frames/n_faces/nfaces1_frame0174_frac0.481.png) |
| 821 | 34.2 | 0.068 | [nfaces1_frame0821_frac0.068.png](output/The_Present/validation_frames/n_faces/nfaces1_frame0821_frac0.068.png) |
| 1681 | 70.1 | 0.090 | [nfaces1_frame1681_frac0.090.png](output/The_Present/validation_frames/n_faces/nfaces1_frame1681_frac0.090.png) |
| 2922 | 121.8 | 0.135 | [nfaces1_frame2922_frac0.135.png](output/The_Present/validation_frames/n_faces/nfaces1_frame2922_frac0.135.png) |

#### n_faces = 2

| Frame idx | Timestamp (s) | face_area_frac | Image |
|-----------|--------------|----------------|-------|
| 408 | 17.0 | 0.071 | [nfaces2_frame0408_frac0.071.png](output/The_Present/validation_frames/n_faces/nfaces2_frame0408_frac0.071.png) |
| 587 | 24.5 | 0.162 | [nfaces2_frame0587_frac0.162.png](output/The_Present/validation_frames/n_faces/nfaces2_frame0587_frac0.162.png) |
| 594 | 24.8 | 0.168 | [nfaces2_frame0594_frac0.168.png](output/The_Present/validation_frames/n_faces/nfaces2_frame0594_frac0.168.png) |
| 601 | 25.0 | 0.171 | [nfaces2_frame0601_frac0.171.png](output/The_Present/validation_frames/n_faces/nfaces2_frame0601_frac0.171.png) |

### `face_area_frac` (quartiles)

### Q1 no face

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0000 | [Q1_no_face_frame0000_score0.0000.png](output/The_Present/validation_frames/face_area_frac/Q1_no_face_frame0000_score0.0000.png) |
| 3548 | 147.9 | 0.0000 | [Q1_no_face_frame3548_score0.0000.png](output/The_Present/validation_frames/face_area_frac/Q1_no_face_frame3548_score0.0000.png) |
| 4131 | 172.2 | 0.0000 | [Q1_no_face_frame4131_score0.0000.png](output/The_Present/validation_frames/face_area_frac/Q1_no_face_frame4131_score0.0000.png) |
| 4511 | 188.0 | 0.0000 | [Q1_no_face_frame4511_score0.0000.png](output/The_Present/validation_frames/face_area_frac/Q1_no_face_frame4511_score0.0000.png) |
| 2944 | 122.7 | 0.0000 | [Q1_no_face_frame2944_score0.0000.png](output/The_Present/validation_frames/face_area_frac/Q1_no_face_frame2944_score0.0000.png) |
| 972 | 40.5 | 0.0000 | [Q1_no_face_frame0972_score0.0000.png](output/The_Present/validation_frames/face_area_frac/Q1_no_face_frame0972_score0.0000.png) |
| 1765 | 73.6 | 0.0000 | [Q1_no_face_frame1765_score0.0000.png](output/The_Present/validation_frames/face_area_frac/Q1_no_face_frame1765_score0.0000.png) |
| 2263 | 94.3 | 0.0000 | [Q1_no_face_frame2263_score0.0000.png](output/The_Present/validation_frames/face_area_frac/Q1_no_face_frame2263_score0.0000.png) |

### Q2 small face

_No frames saved._

### Q3 mid face

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 997 | 41.6 | 0.0080 | [Q3_mid_face_frame0997_score0.0080.png](output/The_Present/validation_frames/face_area_frac/Q3_mid_face_frame0997_score0.0080.png) |
| 4169 | 173.7 | 0.0290 | [Q3_mid_face_frame4169_score0.0290.png](output/The_Present/validation_frames/face_area_frac/Q3_mid_face_frame4169_score0.0290.png) |
| 1519 | 63.3 | 0.0465 | [Q3_mid_face_frame1519_score0.0465.png](output/The_Present/validation_frames/face_area_frac/Q3_mid_face_frame1519_score0.0465.png) |
| 3732 | 155.5 | 0.0565 | [Q3_mid_face_frame3732_score0.0565.png](output/The_Present/validation_frames/face_area_frac/Q3_mid_face_frame3732_score0.0565.png) |
| 434 | 18.1 | 0.0639 | [Q3_mid_face_frame0434_score0.0639.png](output/The_Present/validation_frames/face_area_frac/Q3_mid_face_frame0434_score0.0639.png) |
| 829 | 34.5 | 0.0691 | [Q3_mid_face_frame0829_score0.0691.png](output/The_Present/validation_frames/face_area_frac/Q3_mid_face_frame0829_score0.0691.png) |
| 722 | 30.1 | 0.0759 | [Q3_mid_face_frame0722_score0.0759.png](output/The_Present/validation_frames/face_area_frac/Q3_mid_face_frame0722_score0.0759.png) |
| 1183 | 49.3 | 0.0849 | [Q3_mid_face_frame1183_score0.0849.png](output/The_Present/validation_frames/face_area_frac/Q3_mid_face_frame1183_score0.0849.png) |

### Q4 large face

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3437 | 143.2 | 0.0939 | [Q4_large_face_frame3437_score0.0939.png](output/The_Present/validation_frames/face_area_frac/Q4_large_face_frame3437_score0.0939.png) |
| 2188 | 91.2 | 0.1022 | [Q4_large_face_frame2188_score0.1022.png](output/The_Present/validation_frames/face_area_frac/Q4_large_face_frame2188_score0.1022.png) |
| 3448 | 143.7 | 0.1092 | [Q4_large_face_frame3448_score0.1092.png](output/The_Present/validation_frames/face_area_frac/Q4_large_face_frame3448_score0.1092.png) |
| 2726 | 113.6 | 0.1175 | [Q4_large_face_frame2726_score0.1175.png](output/The_Present/validation_frames/face_area_frac/Q4_large_face_frame2726_score0.1175.png) |
| 2788 | 116.2 | 0.1360 | [Q4_large_face_frame2788_score0.1360.png](output/The_Present/validation_frames/face_area_frac/Q4_large_face_frame2788_score0.1360.png) |
| 2404 | 100.2 | 0.1464 | [Q4_large_face_frame2404_score0.1464.png](output/The_Present/validation_frames/face_area_frac/Q4_large_face_frame2404_score0.1464.png) |
| 3181 | 132.6 | 0.1583 | [Q4_large_face_frame3181_score0.1583.png](output/The_Present/validation_frames/face_area_frac/Q4_large_face_frame3181_score0.1583.png) |
| 3381 | 140.9 | 0.2100 | [Q4_large_face_frame3381_score0.2100.png](output/The_Present/validation_frames/face_area_frac/Q4_large_face_frame3381_score0.2100.png) |

---

## 9. Depth Features — Visual Inspection

> Depth values are in model-relative units from MiDaS, not real-world metres.

### `depth_mean`

### Q1 near

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 88 | 3.7 | 5.5648 | [Q1_near_frame0088_score5.5648.png](output/The_Present/validation_frames/depth_mean/Q1_near_frame0088_score5.5648.png) |
| 4289 | 178.7 | 10.6234 | [Q1_near_frame4289_score10.6234.png](output/The_Present/validation_frames/depth_mean/Q1_near_frame4289_score10.6234.png) |
| 1963 | 81.8 | 11.6442 | [Q1_near_frame1963_score11.6442.png](output/The_Present/validation_frames/depth_mean/Q1_near_frame1963_score11.6442.png) |
| 4148 | 172.9 | 12.2275 | [Q1_near_frame4148_score12.2275.png](output/The_Present/validation_frames/depth_mean/Q1_near_frame4148_score12.2275.png) |
| 2461 | 102.6 | 12.9158 | [Q1_near_frame2461_score12.9158.png](output/The_Present/validation_frames/depth_mean/Q1_near_frame2461_score12.9158.png) |
| 4382 | 182.6 | 13.4884 | [Q1_near_frame4382_score13.4884.png](output/The_Present/validation_frames/depth_mean/Q1_near_frame4382_score13.4884.png) |
| 4600 | 191.7 | 13.7744 | [Q1_near_frame4600_score13.7744.png](output/The_Present/validation_frames/depth_mean/Q1_near_frame4600_score13.7744.png) |
| 2556 | 106.5 | 13.9572 | [Q1_near_frame2556_score13.9572.png](output/The_Present/validation_frames/depth_mean/Q1_near_frame2556_score13.9572.png) |

### Q2 mid near

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2263 | 94.3 | 14.1173 | [Q2_mid_near_frame2263_score14.1173.png](output/The_Present/validation_frames/depth_mean/Q2_mid_near_frame2263_score14.1173.png) |
| 288 | 12.0 | 14.3129 | [Q2_mid_near_frame0288_score14.3129.png](output/The_Present/validation_frames/depth_mean/Q2_mid_near_frame0288_score14.3129.png) |
| 4708 | 196.2 | 14.4694 | [Q2_mid_near_frame4708_score14.4694.png](output/The_Present/validation_frames/depth_mean/Q2_mid_near_frame4708_score14.4694.png) |
| 1736 | 72.3 | 14.6138 | [Q2_mid_near_frame1736_score14.6138.png](output/The_Present/validation_frames/depth_mean/Q2_mid_near_frame1736_score14.6138.png) |
| 4628 | 192.9 | 14.8043 | [Q2_mid_near_frame4628_score14.8043.png](output/The_Present/validation_frames/depth_mean/Q2_mid_near_frame4628_score14.8043.png) |
| 3405 | 141.9 | 14.9864 | [Q2_mid_near_frame3405_score14.9864.png](output/The_Present/validation_frames/depth_mean/Q2_mid_near_frame3405_score14.9864.png) |
| 879 | 36.6 | 15.1510 | [Q2_mid_near_frame0879_score15.1510.png](output/The_Present/validation_frames/depth_mean/Q2_mid_near_frame0879_score15.1510.png) |
| 1421 | 59.2 | 15.3021 | [Q2_mid_near_frame1421_score15.3021.png](output/The_Present/validation_frames/depth_mean/Q2_mid_near_frame1421_score15.3021.png) |

### Q3 mid far

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 461 | 19.2 | 15.4679 | [Q3_mid_far_frame0461_score15.4679.png](output/The_Present/validation_frames/depth_mean/Q3_mid_far_frame0461_score15.4679.png) |
| 1099 | 45.8 | 15.6346 | [Q3_mid_far_frame1099_score15.6346.png](output/The_Present/validation_frames/depth_mean/Q3_mid_far_frame1099_score15.6346.png) |
| 3997 | 166.6 | 15.8528 | [Q3_mid_far_frame3997_score15.8528.png](output/The_Present/validation_frames/depth_mean/Q3_mid_far_frame3997_score15.8528.png) |
| 4085 | 170.2 | 16.1272 | [Q3_mid_far_frame4085_score16.1272.png](output/The_Present/validation_frames/depth_mean/Q3_mid_far_frame4085_score16.1272.png) |
| 920 | 38.3 | 16.4573 | [Q3_mid_far_frame0920_score16.4573.png](output/The_Present/validation_frames/depth_mean/Q3_mid_far_frame0920_score16.4573.png) |
| 2674 | 111.4 | 16.7579 | [Q3_mid_far_frame2674_score16.7579.png](output/The_Present/validation_frames/depth_mean/Q3_mid_far_frame2674_score16.7579.png) |
| 390 | 16.3 | 17.1015 | [Q3_mid_far_frame0390_score17.1015.png](output/The_Present/validation_frames/depth_mean/Q3_mid_far_frame0390_score17.1015.png) |
| 2 | 0.1 | 17.3829 | [Q3_mid_far_frame0002_score17.3829.png](output/The_Present/validation_frames/depth_mean/Q3_mid_far_frame0002_score17.3829.png) |

### Q4 far

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3246 | 135.3 | 17.6447 | [Q4_far_frame3246_score17.6447.png](output/The_Present/validation_frames/depth_mean/Q4_far_frame3246_score17.6447.png) |
| 3718 | 154.9 | 17.9813 | [Q4_far_frame3718_score17.9813.png](output/The_Present/validation_frames/depth_mean/Q4_far_frame3718_score17.9813.png) |
| 784 | 32.7 | 18.2569 | [Q4_far_frame0784_score18.2569.png](output/The_Present/validation_frames/depth_mean/Q4_far_frame0784_score18.2569.png) |
| 2382 | 99.3 | 18.4702 | [Q4_far_frame2382_score18.4702.png](output/The_Present/validation_frames/depth_mean/Q4_far_frame2382_score18.4702.png) |
| 998 | 41.6 | 18.7013 | [Q4_far_frame0998_score18.7013.png](output/The_Present/validation_frames/depth_mean/Q4_far_frame0998_score18.7013.png) |
| 2868 | 119.5 | 19.0624 | [Q4_far_frame2868_score19.0624.png](output/The_Present/validation_frames/depth_mean/Q4_far_frame2868_score19.0624.png) |
| 607 | 25.3 | 19.6751 | [Q4_far_frame0607_score19.6751.png](output/The_Present/validation_frames/depth_mean/Q4_far_frame0607_score19.6751.png) |
| 2539 | 105.8 | 20.8015 | [Q4_far_frame2539_score20.8015.png](output/The_Present/validation_frames/depth_mean/Q4_far_frame2539_score20.8015.png) |

### `depth_std`

### Q1 flat depth

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 8 | 0.3 | 2.3282 | [Q1_flat_depth_frame0008_score2.3282.png](output/The_Present/validation_frames/depth_std/Q1_flat_depth_frame0008_score2.3282.png) |
| 4379 | 182.5 | 3.6594 | [Q1_flat_depth_frame4379_score3.6594.png](output/The_Present/validation_frames/depth_std/Q1_flat_depth_frame4379_score3.6594.png) |
| 4219 | 175.8 | 4.7167 | [Q1_flat_depth_frame4219_score4.7167.png](output/The_Present/validation_frames/depth_std/Q1_flat_depth_frame4219_score4.7167.png) |
| 4636 | 193.2 | 5.2097 | [Q1_flat_depth_frame4636_score5.2097.png](output/The_Present/validation_frames/depth_std/Q1_flat_depth_frame4636_score5.2097.png) |
| 4777 | 199.1 | 5.4251 | [Q1_flat_depth_frame4777_score5.4251.png](output/The_Present/validation_frames/depth_std/Q1_flat_depth_frame4777_score5.4251.png) |
| 4776 | 199.0 | 5.6105 | [Q1_flat_depth_frame4776_score5.6105.png](output/The_Present/validation_frames/depth_std/Q1_flat_depth_frame4776_score5.6105.png) |
| 2459 | 102.5 | 5.9292 | [Q1_flat_depth_frame2459_score5.9292.png](output/The_Present/validation_frames/depth_std/Q1_flat_depth_frame2459_score5.9292.png) |
| 2607 | 108.6 | 6.5278 | [Q1_flat_depth_frame2607_score6.5278.png](output/The_Present/validation_frames/depth_std/Q1_flat_depth_frame2607_score6.5278.png) |

### Q2 mid low depth

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3991 | 166.3 | 7.0515 | [Q2_mid_low_depth_frame3991_score7.0515.png](output/The_Present/validation_frames/depth_std/Q2_mid_low_depth_frame3991_score7.0515.png) |
| 457 | 19.0 | 7.3392 | [Q2_mid_low_depth_frame0457_score7.3392.png](output/The_Present/validation_frames/depth_std/Q2_mid_low_depth_frame0457_score7.3392.png) |
| 1638 | 68.3 | 7.7965 | [Q2_mid_low_depth_frame1638_score7.7965.png](output/The_Present/validation_frames/depth_std/Q2_mid_low_depth_frame1638_score7.7965.png) |
| 610 | 25.4 | 8.3697 | [Q2_mid_low_depth_frame0610_score8.3697.png](output/The_Present/validation_frames/depth_std/Q2_mid_low_depth_frame0610_score8.3697.png) |
| 3501 | 145.9 | 8.7753 | [Q2_mid_low_depth_frame3501_score8.7753.png](output/The_Present/validation_frames/depth_std/Q2_mid_low_depth_frame3501_score8.7753.png) |
| 3252 | 135.5 | 9.0732 | [Q2_mid_low_depth_frame3252_score9.0732.png](output/The_Present/validation_frames/depth_std/Q2_mid_low_depth_frame3252_score9.0732.png) |
| 3518 | 146.6 | 9.2464 | [Q2_mid_low_depth_frame3518_score9.2464.png](output/The_Present/validation_frames/depth_std/Q2_mid_low_depth_frame3518_score9.2464.png) |
| 2327 | 97.0 | 9.4536 | [Q2_mid_low_depth_frame2327_score9.4536.png](output/The_Present/validation_frames/depth_std/Q2_mid_low_depth_frame2327_score9.4536.png) |

### Q3 mid high depth

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1529 | 63.7 | 9.5809 | [Q3_mid_high_depth_frame1529_score9.5809.png](output/The_Present/validation_frames/depth_std/Q3_mid_high_depth_frame1529_score9.5809.png) |
| 1813 | 75.6 | 9.7953 | [Q3_mid_high_depth_frame1813_score9.7953.png](output/The_Present/validation_frames/depth_std/Q3_mid_high_depth_frame1813_score9.7953.png) |
| 1922 | 80.1 | 10.0224 | [Q3_mid_high_depth_frame1922_score10.0224.png](output/The_Present/validation_frames/depth_std/Q3_mid_high_depth_frame1922_score10.0224.png) |
| 2675 | 111.5 | 10.1883 | [Q3_mid_high_depth_frame2675_score10.1883.png](output/The_Present/validation_frames/depth_std/Q3_mid_high_depth_frame2675_score10.1883.png) |
| 1576 | 65.7 | 10.3621 | [Q3_mid_high_depth_frame1576_score10.3621.png](output/The_Present/validation_frames/depth_std/Q3_mid_high_depth_frame1576_score10.3621.png) |
| 2818 | 117.4 | 10.5365 | [Q3_mid_high_depth_frame2818_score10.5365.png](output/The_Present/validation_frames/depth_std/Q3_mid_high_depth_frame2818_score10.5365.png) |
| 2540 | 105.9 | 10.7777 | [Q3_mid_high_depth_frame2540_score10.7777.png](output/The_Present/validation_frames/depth_std/Q3_mid_high_depth_frame2540_score10.7777.png) |
| 3127 | 130.3 | 11.0291 | [Q3_mid_high_depth_frame3127_score11.0291.png](output/The_Present/validation_frames/depth_std/Q3_mid_high_depth_frame3127_score11.0291.png) |

### Q4 varied depth

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1266 | 52.8 | 11.2966 | [Q4_varied_depth_frame1266_score11.2966.png](output/The_Present/validation_frames/depth_std/Q4_varied_depth_frame1266_score11.2966.png) |
| 819 | 34.1 | 11.4956 | [Q4_varied_depth_frame0819_score11.4956.png](output/The_Present/validation_frames/depth_std/Q4_varied_depth_frame0819_score11.4956.png) |
| 1186 | 49.4 | 11.8697 | [Q4_varied_depth_frame1186_score11.8697.png](output/The_Present/validation_frames/depth_std/Q4_varied_depth_frame1186_score11.8697.png) |
| 3949 | 164.6 | 12.3174 | [Q4_varied_depth_frame3949_score12.3174.png](output/The_Present/validation_frames/depth_std/Q4_varied_depth_frame3949_score12.3174.png) |
| 910 | 37.9 | 12.7625 | [Q4_varied_depth_frame0910_score12.7625.png](output/The_Present/validation_frames/depth_std/Q4_varied_depth_frame0910_score12.7625.png) |
| 2299 | 95.8 | 14.6281 | [Q4_varied_depth_frame2299_score14.6281.png](output/The_Present/validation_frames/depth_std/Q4_varied_depth_frame2299_score14.6281.png) |
| 1169 | 48.7 | 16.3775 | [Q4_varied_depth_frame1169_score16.3775.png](output/The_Present/validation_frames/depth_std/Q4_varied_depth_frame1169_score16.3775.png) |
| 3739 | 155.8 | 17.3652 | [Q4_varied_depth_frame3739_score17.3652.png](output/The_Present/validation_frames/depth_std/Q4_varied_depth_frame3739_score17.3652.png) |

### `depth_range`

### Q1 narrow depth

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 4329 | 180.4 | 11.9523 | [Q1_narrow_depth_frame4329_score11.9523.png](output/The_Present/validation_frames/depth_range/Q1_narrow_depth_frame4329_score11.9523.png) |
| 4378 | 182.5 | 16.5759 | [Q1_narrow_depth_frame4378_score16.5759.png](output/The_Present/validation_frames/depth_range/Q1_narrow_depth_frame4378_score16.5759.png) |
| 2464 | 102.7 | 19.7235 | [Q1_narrow_depth_frame2464_score19.7235.png](output/The_Present/validation_frames/depth_range/Q1_narrow_depth_frame2464_score19.7235.png) |
| 3870 | 161.3 | 23.3363 | [Q1_narrow_depth_frame3870_score23.3363.png](output/The_Present/validation_frames/depth_range/Q1_narrow_depth_frame3870_score23.3363.png) |
| 4636 | 193.2 | 24.3506 | [Q1_narrow_depth_frame4636_score24.3506.png](output/The_Present/validation_frames/depth_range/Q1_narrow_depth_frame4636_score24.3506.png) |
| 673 | 28.0 | 25.1491 | [Q1_narrow_depth_frame0673_score25.1491.png](output/The_Present/validation_frames/depth_range/Q1_narrow_depth_frame0673_score25.1491.png) |
| 4492 | 187.2 | 25.7054 | [Q1_narrow_depth_frame4492_score25.7054.png](output/The_Present/validation_frames/depth_range/Q1_narrow_depth_frame4492_score25.7054.png) |
| 4660 | 194.2 | 26.2201 | [Q1_narrow_depth_frame4660_score26.2201.png](output/The_Present/validation_frames/depth_range/Q1_narrow_depth_frame4660_score26.2201.png) |

### Q2 mid narrow

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1526 | 63.6 | 26.8380 | [Q2_mid_narrow_frame1526_score26.8380.png](output/The_Present/validation_frames/depth_range/Q2_mid_narrow_frame1526_score26.8380.png) |
| 138 | 5.8 | 27.3501 | [Q2_mid_narrow_frame0138_score27.3501.png](output/The_Present/validation_frames/depth_range/Q2_mid_narrow_frame0138_score27.3501.png) |
| 1479 | 61.6 | 27.8355 | [Q2_mid_narrow_frame1479_score27.8355.png](output/The_Present/validation_frames/depth_range/Q2_mid_narrow_frame1479_score27.8355.png) |
| 1651 | 68.8 | 28.3886 | [Q2_mid_narrow_frame1651_score28.3886.png](output/The_Present/validation_frames/depth_range/Q2_mid_narrow_frame1651_score28.3886.png) |
| 2697 | 112.4 | 28.9716 | [Q2_mid_narrow_frame2697_score28.9716.png](output/The_Present/validation_frames/depth_range/Q2_mid_narrow_frame2697_score28.9716.png) |
| 3402 | 141.8 | 29.3553 | [Q2_mid_narrow_frame3402_score29.3553.png](output/The_Present/validation_frames/depth_range/Q2_mid_narrow_frame3402_score29.3553.png) |
| 2335 | 97.3 | 29.8635 | [Q2_mid_narrow_frame2335_score29.8635.png](output/The_Present/validation_frames/depth_range/Q2_mid_narrow_frame2335_score29.8635.png) |
| 228 | 9.5 | 30.5293 | [Q2_mid_narrow_frame0228_score30.5293.png](output/The_Present/validation_frames/depth_range/Q2_mid_narrow_frame0228_score30.5293.png) |

### Q3 mid wide

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 209 | 8.7 | 31.1751 | [Q3_mid_wide_frame0209_score31.1751.png](output/The_Present/validation_frames/depth_range/Q3_mid_wide_frame0209_score31.1751.png) |
| 773 | 32.2 | 32.1195 | [Q3_mid_wide_frame0773_score32.1195.png](output/The_Present/validation_frames/depth_range/Q3_mid_wide_frame0773_score32.1195.png) |
| 1916 | 79.8 | 32.7587 | [Q3_mid_wide_frame1916_score32.7587.png](output/The_Present/validation_frames/depth_range/Q3_mid_wide_frame1916_score32.7587.png) |
| 2143 | 89.3 | 33.4317 | [Q3_mid_wide_frame2143_score33.4317.png](output/The_Present/validation_frames/depth_range/Q3_mid_wide_frame2143_score33.4317.png) |
| 751 | 31.3 | 34.3967 | [Q3_mid_wide_frame0751_score34.3967.png](output/The_Present/validation_frames/depth_range/Q3_mid_wide_frame0751_score34.3967.png) |
| 2169 | 90.4 | 35.5462 | [Q3_mid_wide_frame2169_score35.5462.png](output/The_Present/validation_frames/depth_range/Q3_mid_wide_frame2169_score35.5462.png) |
| 2835 | 118.1 | 37.0482 | [Q3_mid_wide_frame2835_score37.0482.png](output/The_Present/validation_frames/depth_range/Q3_mid_wide_frame2835_score37.0482.png) |
| 1191 | 49.6 | 38.1213 | [Q3_mid_wide_frame1191_score38.1213.png](output/The_Present/validation_frames/depth_range/Q3_mid_wide_frame1191_score38.1213.png) |

### Q4 wide depth

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3611 | 150.5 | 39.0794 | [Q4_wide_depth_frame3611_score39.0794.png](output/The_Present/validation_frames/depth_range/Q4_wide_depth_frame3611_score39.0794.png) |
| 3139 | 130.8 | 40.2133 | [Q4_wide_depth_frame3139_score40.2133.png](output/The_Present/validation_frames/depth_range/Q4_wide_depth_frame3139_score40.2133.png) |
| 1765 | 73.6 | 41.2605 | [Q4_wide_depth_frame1765_score41.2605.png](output/The_Present/validation_frames/depth_range/Q4_wide_depth_frame1765_score41.2605.png) |
| 374 | 15.6 | 42.5706 | [Q4_wide_depth_frame0374_score42.5706.png](output/The_Present/validation_frames/depth_range/Q4_wide_depth_frame0374_score42.5706.png) |
| 665 | 27.7 | 45.2053 | [Q4_wide_depth_frame0665_score45.2053.png](output/The_Present/validation_frames/depth_range/Q4_wide_depth_frame0665_score45.2053.png) |
| 2582 | 107.6 | 50.1415 | [Q4_wide_depth_frame2582_score50.1415.png](output/The_Present/validation_frames/depth_range/Q4_wide_depth_frame2582_score50.1415.png) |
| 3665 | 152.7 | 51.9437 | [Q4_wide_depth_frame3665_score51.9437.png](output/The_Present/validation_frames/depth_range/Q4_wide_depth_frame3665_score51.9437.png) |
| 3314 | 138.1 | 58.0801 | [Q4_wide_depth_frame3314_score58.0801.png](output/The_Present/validation_frames/depth_range/Q4_wide_depth_frame3314_score58.0801.png) |

---

## 10. Object Features — Visual Inspection

### `n_objects` (count bins)

#### 0 to 1

| Frame idx | Timestamp (s) | n_objects | Image |
|-----------|--------------|-----------|-------|
| 0 | 0.0 | 0 | [0_to_1_frame0000_n0.png](output/The_Present/validation_frames/n_objects/0_to_1_frame0000_n0.png) |
| 17 | 0.7 | 0 | [0_to_1_frame0017_n0.png](output/The_Present/validation_frames/n_objects/0_to_1_frame0017_n0.png) |
| 4791 | 199.7 | 1 | [0_to_1_frame4791_n1.png](output/The_Present/validation_frames/n_objects/0_to_1_frame4791_n1.png) |
| 871 | 36.3 | 1 | [0_to_1_frame0871_n1.png](output/The_Present/validation_frames/n_objects/0_to_1_frame0871_n1.png) |

#### 2 to 4

| Frame idx | Timestamp (s) | n_objects | Image |
|-----------|--------------|-----------|-------|
| 174 | 7.3 | 2 | [2_to_4_frame0174_n2.png](output/The_Present/validation_frames/n_objects/2_to_4_frame0174_n2.png) |
| 4377 | 182.4 | 2 | [2_to_4_frame4377_n2.png](output/The_Present/validation_frames/n_objects/2_to_4_frame4377_n2.png) |
| 4732 | 197.2 | 3 | [2_to_4_frame4732_n3.png](output/The_Present/validation_frames/n_objects/2_to_4_frame4732_n3.png) |
| 1329 | 55.4 | 4 | [2_to_4_frame1329_n4.png](output/The_Present/validation_frames/n_objects/2_to_4_frame1329_n4.png) |

#### 5 to 9

| Frame idx | Timestamp (s) | n_objects | Image |
|-----------|--------------|-----------|-------|
| 1370 | 57.1 | 5 | [5_to_9_frame1370_n5.png](output/The_Present/validation_frames/n_objects/5_to_9_frame1370_n5.png) |
| 3342 | 139.3 | 5 | [5_to_9_frame3342_n5.png](output/The_Present/validation_frames/n_objects/5_to_9_frame3342_n5.png) |
| 1515 | 63.1 | 6 | [5_to_9_frame1515_n6.png](output/The_Present/validation_frames/n_objects/5_to_9_frame1515_n6.png) |
| 1398 | 58.3 | 7 | [5_to_9_frame1398_n7.png](output/The_Present/validation_frames/n_objects/5_to_9_frame1398_n7.png) |

#### 10 plus

| Frame idx | Timestamp (s) | n_objects | Image |
|-----------|--------------|-----------|-------|
| 957 | 39.9 | 10 | [10_plus_frame0957_n10.png](output/The_Present/validation_frames/n_objects/10_plus_frame0957_n10.png) |
| 2569 | 107.1 | 10 | [10_plus_frame2569_n10.png](output/The_Present/validation_frames/n_objects/10_plus_frame2569_n10.png) |
| 993 | 41.4 | 10 | [10_plus_frame0993_n10.png](output/The_Present/validation_frames/n_objects/10_plus_frame0993_n10.png) |
| 965 | 40.2 | 12 | [10_plus_frame0965_n12.png](output/The_Present/validation_frames/n_objects/10_plus_frame0965_n12.png) |

### `object_categories` (top 10 categories)

#### person

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 174 | 7.3 | 2 | [person_frame0174_count2.png](output/The_Present/validation_frames/object_categories/person_frame0174_count2.png) |
| 1258 | 52.4 | 1 | [person_frame1258_count1.png](output/The_Present/validation_frames/object_categories/person_frame1258_count1.png) |
| 3363 | 140.2 | 1 | [person_frame3363_count1.png](output/The_Present/validation_frames/object_categories/person_frame3363_count1.png) |

#### potted plant

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 268 | 11.2 | 1 | [potted_plant_frame0268_count1.png](output/The_Present/validation_frames/object_categories/potted_plant_frame0268_count1.png) |
| 1117 | 46.6 | 1 | [potted_plant_frame1117_count1.png](output/The_Present/validation_frames/object_categories/potted_plant_frame1117_count1.png) |
| 2875 | 119.8 | 1 | [potted_plant_frame2875_count1.png](output/The_Present/validation_frames/object_categories/potted_plant_frame2875_count1.png) |

#### chair

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 259 | 10.8 | 1 | [chair_frame0259_count1.png](output/The_Present/validation_frames/object_categories/chair_frame0259_count1.png) |
| 987 | 41.1 | 1 | [chair_frame0987_count1.png](output/The_Present/validation_frames/object_categories/chair_frame0987_count1.png) |
| 2806 | 116.9 | 1 | [chair_frame2806_count1.png](output/The_Present/validation_frames/object_categories/chair_frame2806_count1.png) |

#### dog

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 1344 | 56.0 | 1 | [dog_frame1344_count1.png](output/The_Present/validation_frames/object_categories/dog_frame1344_count1.png) |
| 3076 | 128.2 | 1 | [dog_frame3076_count1.png](output/The_Present/validation_frames/object_categories/dog_frame3076_count1.png) |
| 3980 | 165.9 | 1 | [dog_frame3980_count1.png](output/The_Present/validation_frames/object_categories/dog_frame3980_count1.png) |

#### couch

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 416 | 17.3 | 1 | [couch_frame0416_count1.png](output/The_Present/validation_frames/object_categories/couch_frame0416_count1.png) |
| 1827 | 76.1 | 2 | [couch_frame1827_count2.png](output/The_Present/validation_frames/object_categories/couch_frame1827_count2.png) |
| 2881 | 120.1 | 1 | [couch_frame2881_count1.png](output/The_Present/validation_frames/object_categories/couch_frame2881_count1.png) |

#### cell phone

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 259 | 10.8 | 1 | [cell_phone_frame0259_count1.png](output/The_Present/validation_frames/object_categories/cell_phone_frame0259_count1.png) |
| 580 | 24.2 | 1 | [cell_phone_frame0580_count1.png](output/The_Present/validation_frames/object_categories/cell_phone_frame0580_count1.png) |
| 1942 | 80.9 | 1 | [cell_phone_frame1942_count1.png](output/The_Present/validation_frames/object_categories/cell_phone_frame1942_count1.png) |

#### book

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 845 | 35.2 | 1 | [book_frame0845_count1.png](output/The_Present/validation_frames/object_categories/book_frame0845_count1.png) |
| 1275 | 53.1 | 1 | [book_frame1275_count1.png](output/The_Present/validation_frames/object_categories/book_frame1275_count1.png) |
| 1586 | 66.1 | 9 | [book_frame1586_count9.png](output/The_Present/validation_frames/object_categories/book_frame1586_count9.png) |

#### sports ball

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 2122 | 88.4 | 1 | [sports_ball_frame2122_count1.png](output/The_Present/validation_frames/object_categories/sports_ball_frame2122_count1.png) |
| 2563 | 106.8 | 1 | [sports_ball_frame2563_count1.png](output/The_Present/validation_frames/object_categories/sports_ball_frame2563_count1.png) |
| 3678 | 153.3 | 1 | [sports_ball_frame3678_count1.png](output/The_Present/validation_frames/object_categories/sports_ball_frame3678_count1.png) |

#### vase

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 504 | 21.0 | 1 | [vase_frame0504_count1.png](output/The_Present/validation_frames/object_categories/vase_frame0504_count1.png) |
| 4178 | 174.1 | 1 | [vase_frame4178_count1.png](output/The_Present/validation_frames/object_categories/vase_frame4178_count1.png) |
| 4512 | 188.0 | 1 | [vase_frame4512_count1.png](output/The_Present/validation_frames/object_categories/vase_frame4512_count1.png) |

#### dining table

| Frame idx | Timestamp (s) | Count | Image |
|-----------|--------------|-------|-------|
| 384 | 16.0 | 1 | [dining_table_frame0384_count1.png](output/The_Present/validation_frames/object_categories/dining_table_frame0384_count1.png) |
| 491 | 20.5 | 1 | [dining_table_frame0491_count1.png](output/The_Present/validation_frames/object_categories/dining_table_frame0491_count1.png) |
| 1055 | 44.0 | 1 | [dining_table_frame1055_count1.png](output/The_Present/validation_frames/object_categories/dining_table_frame1055_count1.png) |

---

## 11. CLIP Scene Features — Visual Inspection

### `scene_natural_score`

### Q1 low natural

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3052 | 127.2 | -0.0651 | [Q1_low_natural_frame3052_score-0.0651.png](output/The_Present/validation_frames/scene_natural_score/Q1_low_natural_frame3052_score-0.0651.png) |
| 2167 | 90.3 | -0.0378 | [Q1_low_natural_frame2167_score-0.0378.png](output/The_Present/validation_frames/scene_natural_score/Q1_low_natural_frame2167_score-0.0378.png) |
| 1792 | 74.7 | -0.0304 | [Q1_low_natural_frame1792_score-0.0304.png](output/The_Present/validation_frames/scene_natural_score/Q1_low_natural_frame1792_score-0.0304.png) |
| 4366 | 182.0 | -0.0265 | [Q1_low_natural_frame4366_score-0.0265.png](output/The_Present/validation_frames/scene_natural_score/Q1_low_natural_frame4366_score-0.0265.png) |
| 4485 | 186.9 | -0.0241 | [Q1_low_natural_frame4485_score-0.0241.png](output/The_Present/validation_frames/scene_natural_score/Q1_low_natural_frame4485_score-0.0241.png) |
| 518 | 21.6 | -0.0225 | [Q1_low_natural_frame0518_score-0.0225.png](output/The_Present/validation_frames/scene_natural_score/Q1_low_natural_frame0518_score-0.0225.png) |
| 4681 | 195.1 | -0.0210 | [Q1_low_natural_frame4681_score-0.0210.png](output/The_Present/validation_frames/scene_natural_score/Q1_low_natural_frame4681_score-0.0210.png) |
| 4764 | 198.5 | -0.0196 | [Q1_low_natural_frame4764_score-0.0196.png](output/The_Present/validation_frames/scene_natural_score/Q1_low_natural_frame4764_score-0.0196.png) |

### Q2 lower mid

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1698 | 70.8 | -0.0181 | [Q2_lower_mid_frame1698_score-0.0181.png](output/The_Present/validation_frames/scene_natural_score/Q2_lower_mid_frame1698_score-0.0181.png) |
| 4232 | 176.4 | -0.0171 | [Q2_lower_mid_frame4232_score-0.0171.png](output/The_Present/validation_frames/scene_natural_score/Q2_lower_mid_frame4232_score-0.0171.png) |
| 1340 | 55.8 | -0.0160 | [Q2_lower_mid_frame1340_score-0.0160.png](output/The_Present/validation_frames/scene_natural_score/Q2_lower_mid_frame1340_score-0.0160.png) |
| 4400 | 183.4 | -0.0148 | [Q2_lower_mid_frame4400_score-0.0148.png](output/The_Present/validation_frames/scene_natural_score/Q2_lower_mid_frame4400_score-0.0148.png) |
| 4271 | 178.0 | -0.0135 | [Q2_lower_mid_frame4271_score-0.0135.png](output/The_Present/validation_frames/scene_natural_score/Q2_lower_mid_frame4271_score-0.0135.png) |
| 4524 | 188.5 | -0.0123 | [Q2_lower_mid_frame4524_score-0.0123.png](output/The_Present/validation_frames/scene_natural_score/Q2_lower_mid_frame4524_score-0.0123.png) |
| 1992 | 83.0 | -0.0113 | [Q2_lower_mid_frame1992_score-0.0113.png](output/The_Present/validation_frames/scene_natural_score/Q2_lower_mid_frame1992_score-0.0113.png) |
| 4270 | 178.0 | -0.0103 | [Q2_lower_mid_frame4270_score-0.0103.png](output/The_Present/validation_frames/scene_natural_score/Q2_lower_mid_frame4270_score-0.0103.png) |

### Q3 upper mid

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1186 | 49.4 | -0.0093 | [Q3_upper_mid_frame1186_score-0.0093.png](output/The_Present/validation_frames/scene_natural_score/Q3_upper_mid_frame1186_score-0.0093.png) |
| 3762 | 156.8 | -0.0082 | [Q3_upper_mid_frame3762_score-0.0082.png](output/The_Present/validation_frames/scene_natural_score/Q3_upper_mid_frame3762_score-0.0082.png) |
| 106 | 4.4 | -0.0072 | [Q3_upper_mid_frame0106_score-0.0072.png](output/The_Present/validation_frames/scene_natural_score/Q3_upper_mid_frame0106_score-0.0072.png) |
| 2806 | 116.9 | -0.0062 | [Q3_upper_mid_frame2806_score-0.0062.png](output/The_Present/validation_frames/scene_natural_score/Q3_upper_mid_frame2806_score-0.0062.png) |
| 2216 | 92.4 | -0.0054 | [Q3_upper_mid_frame2216_score-0.0054.png](output/The_Present/validation_frames/scene_natural_score/Q3_upper_mid_frame2216_score-0.0054.png) |
| 837 | 34.9 | -0.0044 | [Q3_upper_mid_frame0837_score-0.0044.png](output/The_Present/validation_frames/scene_natural_score/Q3_upper_mid_frame0837_score-0.0044.png) |
| 1976 | 82.4 | -0.0034 | [Q3_upper_mid_frame1976_score-0.0034.png](output/The_Present/validation_frames/scene_natural_score/Q3_upper_mid_frame1976_score-0.0034.png) |
| 3883 | 161.8 | -0.0020 | [Q3_upper_mid_frame3883_score-0.0020.png](output/The_Present/validation_frames/scene_natural_score/Q3_upper_mid_frame3883_score-0.0020.png) |

### Q4 high natural

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 2258 | 94.1 | -0.0007 | [Q4_high_natural_frame2258_score-0.0007.png](output/The_Present/validation_frames/scene_natural_score/Q4_high_natural_frame2258_score-0.0007.png) |
| 2727 | 113.6 | 0.0009 | [Q4_high_natural_frame2727_score0.0009.png](output/The_Present/validation_frames/scene_natural_score/Q4_high_natural_frame2727_score0.0009.png) |
| 1829 | 76.2 | 0.0029 | [Q4_high_natural_frame1829_score0.0029.png](output/The_Present/validation_frames/scene_natural_score/Q4_high_natural_frame1829_score0.0029.png) |
| 2630 | 109.6 | 0.0055 | [Q4_high_natural_frame2630_score0.0055.png](output/The_Present/validation_frames/scene_natural_score/Q4_high_natural_frame2630_score0.0055.png) |
| 3280 | 136.7 | 0.0091 | [Q4_high_natural_frame3280_score0.0091.png](output/The_Present/validation_frames/scene_natural_score/Q4_high_natural_frame3280_score0.0091.png) |
| 3695 | 154.0 | 0.0122 | [Q4_high_natural_frame3695_score0.0122.png](output/The_Present/validation_frames/scene_natural_score/Q4_high_natural_frame3695_score0.0122.png) |
| 1551 | 64.6 | 0.0161 | [Q4_high_natural_frame1551_score0.0161.png](output/The_Present/validation_frames/scene_natural_score/Q4_high_natural_frame1551_score0.0161.png) |
| 3782 | 157.6 | 0.0213 | [Q4_high_natural_frame3782_score0.0213.png](output/The_Present/validation_frames/scene_natural_score/Q4_high_natural_frame3782_score0.0213.png) |

### `scene_open_score`

### Q1 enclosed

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 3000 | 125.0 | -0.0804 | [Q1_enclosed_frame3000_score-0.0804.png](output/The_Present/validation_frames/scene_open_score/Q1_enclosed_frame3000_score-0.0804.png) |
| 1172 | 48.8 | -0.0667 | [Q1_enclosed_frame1172_score-0.0667.png](output/The_Present/validation_frames/scene_open_score/Q1_enclosed_frame1172_score-0.0667.png) |
| 1265 | 52.7 | -0.0623 | [Q1_enclosed_frame1265_score-0.0623.png](output/The_Present/validation_frames/scene_open_score/Q1_enclosed_frame1265_score-0.0623.png) |
| 3359 | 140.0 | -0.0590 | [Q1_enclosed_frame3359_score-0.0590.png](output/The_Present/validation_frames/scene_open_score/Q1_enclosed_frame3359_score-0.0590.png) |
| 2192 | 91.4 | -0.0552 | [Q1_enclosed_frame2192_score-0.0552.png](output/The_Present/validation_frames/scene_open_score/Q1_enclosed_frame2192_score-0.0552.png) |
| 3965 | 165.2 | -0.0521 | [Q1_enclosed_frame3965_score-0.0521.png](output/The_Present/validation_frames/scene_open_score/Q1_enclosed_frame3965_score-0.0521.png) |
| 1814 | 75.6 | -0.0494 | [Q1_enclosed_frame1814_score-0.0494.png](output/The_Present/validation_frames/scene_open_score/Q1_enclosed_frame1814_score-0.0494.png) |
| 1246 | 51.9 | -0.0470 | [Q1_enclosed_frame1246_score-0.0470.png](output/The_Present/validation_frames/scene_open_score/Q1_enclosed_frame1246_score-0.0470.png) |

### Q2 mid enclosed

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1865 | 77.7 | -0.0440 | [Q2_mid_enclosed_frame1865_score-0.0440.png](output/The_Present/validation_frames/scene_open_score/Q2_mid_enclosed_frame1865_score-0.0440.png) |
| 4404 | 183.5 | -0.0415 | [Q2_mid_enclosed_frame4404_score-0.0415.png](output/The_Present/validation_frames/scene_open_score/Q2_mid_enclosed_frame4404_score-0.0415.png) |
| 4696 | 195.7 | -0.0393 | [Q2_mid_enclosed_frame4696_score-0.0393.png](output/The_Present/validation_frames/scene_open_score/Q2_mid_enclosed_frame4696_score-0.0393.png) |
| 164 | 6.8 | -0.0374 | [Q2_mid_enclosed_frame0164_score-0.0374.png](output/The_Present/validation_frames/scene_open_score/Q2_mid_enclosed_frame0164_score-0.0374.png) |
| 2600 | 108.4 | -0.0357 | [Q2_mid_enclosed_frame2600_score-0.0357.png](output/The_Present/validation_frames/scene_open_score/Q2_mid_enclosed_frame2600_score-0.0357.png) |
| 1863 | 77.6 | -0.0341 | [Q2_mid_enclosed_frame1863_score-0.0341.png](output/The_Present/validation_frames/scene_open_score/Q2_mid_enclosed_frame1863_score-0.0341.png) |
| 1690 | 70.4 | -0.0320 | [Q2_mid_enclosed_frame1690_score-0.0320.png](output/The_Present/validation_frames/scene_open_score/Q2_mid_enclosed_frame1690_score-0.0320.png) |
| 4145 | 172.7 | -0.0301 | [Q2_mid_enclosed_frame4145_score-0.0301.png](output/The_Present/validation_frames/scene_open_score/Q2_mid_enclosed_frame4145_score-0.0301.png) |

### Q3 mid open

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 768 | 32.0 | -0.0284 | [Q3_mid_open_frame0768_score-0.0284.png](output/The_Present/validation_frames/scene_open_score/Q3_mid_open_frame0768_score-0.0284.png) |
| 4177 | 174.1 | -0.0270 | [Q3_mid_open_frame4177_score-0.0270.png](output/The_Present/validation_frames/scene_open_score/Q3_mid_open_frame4177_score-0.0270.png) |
| 4031 | 168.0 | -0.0255 | [Q3_mid_open_frame4031_score-0.0255.png](output/The_Present/validation_frames/scene_open_score/Q3_mid_open_frame4031_score-0.0255.png) |
| 610 | 25.4 | -0.0236 | [Q3_mid_open_frame0610_score-0.0236.png](output/The_Present/validation_frames/scene_open_score/Q3_mid_open_frame0610_score-0.0236.png) |
| 4449 | 185.4 | -0.0220 | [Q3_mid_open_frame4449_score-0.0220.png](output/The_Present/validation_frames/scene_open_score/Q3_mid_open_frame4449_score-0.0220.png) |
| 652 | 27.2 | -0.0203 | [Q3_mid_open_frame0652_score-0.0203.png](output/The_Present/validation_frames/scene_open_score/Q3_mid_open_frame0652_score-0.0203.png) |
| 3494 | 145.6 | -0.0186 | [Q3_mid_open_frame3494_score-0.0186.png](output/The_Present/validation_frames/scene_open_score/Q3_mid_open_frame3494_score-0.0186.png) |
| 916 | 38.2 | -0.0169 | [Q3_mid_open_frame0916_score-0.0169.png](output/The_Present/validation_frames/scene_open_score/Q3_mid_open_frame0916_score-0.0169.png) |

### Q4 open

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1098 | 45.8 | -0.0147 | [Q4_open_frame1098_score-0.0147.png](output/The_Present/validation_frames/scene_open_score/Q4_open_frame1098_score-0.0147.png) |
| 2510 | 104.6 | -0.0124 | [Q4_open_frame2510_score-0.0124.png](output/The_Present/validation_frames/scene_open_score/Q4_open_frame2510_score-0.0124.png) |
| 1763 | 73.5 | -0.0107 | [Q4_open_frame1763_score-0.0107.png](output/The_Present/validation_frames/scene_open_score/Q4_open_frame1763_score-0.0107.png) |
| 4109 | 171.2 | -0.0089 | [Q4_open_frame4109_score-0.0089.png](output/The_Present/validation_frames/scene_open_score/Q4_open_frame4109_score-0.0089.png) |
| 3193 | 133.1 | -0.0066 | [Q4_open_frame3193_score-0.0066.png](output/The_Present/validation_frames/scene_open_score/Q4_open_frame3193_score-0.0066.png) |
| 237 | 9.9 | -0.0048 | [Q4_open_frame0237_score-0.0048.png](output/The_Present/validation_frames/scene_open_score/Q4_open_frame0237_score-0.0048.png) |
| 2377 | 99.1 | -0.0025 | [Q4_open_frame2377_score-0.0025.png](output/The_Present/validation_frames/scene_open_score/Q4_open_frame2377_score-0.0025.png) |
| 1946 | 81.1 | 0.0001 | [Q4_open_frame1946_score0.0001.png](output/The_Present/validation_frames/scene_open_score/Q4_open_frame1946_score0.0001.png) |

### `scene_category_score`

### Q1 low conf

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 0 | 0.0 | 0.0675 | [Q1_low_conf_frame0000_score0.0675.png](output/The_Present/validation_frames/scene_category_score/Q1_low_conf_frame0000_score0.0675.png) |
| 48 | 2.0 | 0.0684 | [Q1_low_conf_frame0048_score0.0684.png](output/The_Present/validation_frames/scene_category_score/Q1_low_conf_frame0048_score0.0684.png) |
| 3047 | 127.0 | 0.0689 | [Q1_low_conf_frame3047_score0.0689.png](output/The_Present/validation_frames/scene_category_score/Q1_low_conf_frame3047_score0.0689.png) |
| 3071 | 128.0 | 0.0691 | [Q1_low_conf_frame3071_score0.0691.png](output/The_Present/validation_frames/scene_category_score/Q1_low_conf_frame3071_score0.0691.png) |
| 4421 | 184.2 | 0.0692 | [Q1_low_conf_frame4421_score0.0692.png](output/The_Present/validation_frames/scene_category_score/Q1_low_conf_frame4421_score0.0692.png) |
| 281 | 11.7 | 0.0693 | [Q1_low_conf_frame0281_score0.0693.png](output/The_Present/validation_frames/scene_category_score/Q1_low_conf_frame0281_score0.0693.png) |
| 3359 | 140.0 | 0.0694 | [Q1_low_conf_frame3359_score0.0694.png](output/The_Present/validation_frames/scene_category_score/Q1_low_conf_frame3359_score0.0694.png) |
| 1334 | 55.6 | 0.0694 | [Q1_low_conf_frame1334_score0.0694.png](output/The_Present/validation_frames/scene_category_score/Q1_low_conf_frame1334_score0.0694.png) |

### Q2 mid low conf

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1929 | 80.4 | 0.0695 | [Q2_mid_low_conf_frame1929_score0.0695.png](output/The_Present/validation_frames/scene_category_score/Q2_mid_low_conf_frame1929_score0.0695.png) |
| 641 | 26.7 | 0.0696 | [Q2_mid_low_conf_frame0641_score0.0696.png](output/The_Present/validation_frames/scene_category_score/Q2_mid_low_conf_frame0641_score0.0696.png) |
| 3268 | 136.2 | 0.0696 | [Q2_mid_low_conf_frame3268_score0.0696.png](output/The_Present/validation_frames/scene_category_score/Q2_mid_low_conf_frame3268_score0.0696.png) |
| 1844 | 76.8 | 0.0697 | [Q2_mid_low_conf_frame1844_score0.0697.png](output/The_Present/validation_frames/scene_category_score/Q2_mid_low_conf_frame1844_score0.0697.png) |
| 1345 | 56.1 | 0.0697 | [Q2_mid_low_conf_frame1345_score0.0697.png](output/The_Present/validation_frames/scene_category_score/Q2_mid_low_conf_frame1345_score0.0697.png) |
| 2020 | 84.2 | 0.0697 | [Q2_mid_low_conf_frame2020_score0.0697.png](output/The_Present/validation_frames/scene_category_score/Q2_mid_low_conf_frame2020_score0.0697.png) |
| 3351 | 139.7 | 0.0698 | [Q2_mid_low_conf_frame3351_score0.0698.png](output/The_Present/validation_frames/scene_category_score/Q2_mid_low_conf_frame3351_score0.0698.png) |
| 1757 | 73.2 | 0.0698 | [Q2_mid_low_conf_frame1757_score0.0698.png](output/The_Present/validation_frames/scene_category_score/Q2_mid_low_conf_frame1757_score0.0698.png) |

### Q3 mid high conf

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 4099 | 170.8 | 0.0699 | [Q3_mid_high_conf_frame4099_score0.0699.png](output/The_Present/validation_frames/scene_category_score/Q3_mid_high_conf_frame4099_score0.0699.png) |
| 1644 | 68.5 | 0.0699 | [Q3_mid_high_conf_frame1644_score0.0699.png](output/The_Present/validation_frames/scene_category_score/Q3_mid_high_conf_frame1644_score0.0699.png) |
| 2396 | 99.9 | 0.0700 | [Q3_mid_high_conf_frame2396_score0.0700.png](output/The_Present/validation_frames/scene_category_score/Q3_mid_high_conf_frame2396_score0.0700.png) |
| 1959 | 81.6 | 0.0700 | [Q3_mid_high_conf_frame1959_score0.0700.png](output/The_Present/validation_frames/scene_category_score/Q3_mid_high_conf_frame1959_score0.0700.png) |
| 1806 | 75.3 | 0.0701 | [Q3_mid_high_conf_frame1806_score0.0701.png](output/The_Present/validation_frames/scene_category_score/Q3_mid_high_conf_frame1806_score0.0701.png) |
| 754 | 31.4 | 0.0701 | [Q3_mid_high_conf_frame0754_score0.0701.png](output/The_Present/validation_frames/scene_category_score/Q3_mid_high_conf_frame0754_score0.0701.png) |
| 1116 | 46.5 | 0.0702 | [Q3_mid_high_conf_frame1116_score0.0702.png](output/The_Present/validation_frames/scene_category_score/Q3_mid_high_conf_frame1116_score0.0702.png) |
| 485 | 20.2 | 0.0702 | [Q3_mid_high_conf_frame0485_score0.0702.png](output/The_Present/validation_frames/scene_category_score/Q3_mid_high_conf_frame0485_score0.0702.png) |

### Q4 high conf

| Frame idx | Timestamp (s) | Score | Image |
|-----------|--------------|-------|-------|
| 1934 | 80.6 | 0.0703 | [Q4_high_conf_frame1934_score0.0703.png](output/The_Present/validation_frames/scene_category_score/Q4_high_conf_frame1934_score0.0703.png) |
| 1089 | 45.4 | 0.0704 | [Q4_high_conf_frame1089_score0.0704.png](output/The_Present/validation_frames/scene_category_score/Q4_high_conf_frame1089_score0.0704.png) |
| 826 | 34.4 | 0.0705 | [Q4_high_conf_frame0826_score0.0705.png](output/The_Present/validation_frames/scene_category_score/Q4_high_conf_frame0826_score0.0705.png) |
| 926 | 38.6 | 0.0705 | [Q4_high_conf_frame0926_score0.0705.png](output/The_Present/validation_frames/scene_category_score/Q4_high_conf_frame0926_score0.0705.png) |
| 906 | 37.8 | 0.0707 | [Q4_high_conf_frame0906_score0.0707.png](output/The_Present/validation_frames/scene_category_score/Q4_high_conf_frame0906_score0.0707.png) |
| 3115 | 129.8 | 0.0708 | [Q4_high_conf_frame3115_score0.0708.png](output/The_Present/validation_frames/scene_category_score/Q4_high_conf_frame3115_score0.0708.png) |
| 226 | 9.4 | 0.0710 | [Q4_high_conf_frame0226_score0.0710.png](output/The_Present/validation_frames/scene_category_score/Q4_high_conf_frame0226_score0.0710.png) |
| 3624 | 151.0 | 0.0712 | [Q4_high_conf_frame3624_score0.0712.png](output/The_Present/validation_frames/scene_category_score/Q4_high_conf_frame3624_score0.0712.png) |

### `scene_category`

#### bathroom

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 0 | 0.0 | 0.068 | [bathroom_frame0000_conf0.068.png](output/The_Present/validation_frames/scene_category/bathroom_frame0000_conf0.068.png) |
| 31 | 1.3 | 0.068 | [bathroom_frame0031_conf0.068.png](output/The_Present/validation_frames/scene_category/bathroom_frame0031_conf0.068.png) |
| 3654 | 152.3 | 0.070 | [bathroom_frame3654_conf0.070.png](output/The_Present/validation_frames/scene_category/bathroom_frame3654_conf0.070.png) |

#### bedroom

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 270 | 11.3 | 0.069 | [bedroom_frame0270_conf0.069.png](output/The_Present/validation_frames/scene_category/bedroom_frame0270_conf0.069.png) |
| 315 | 13.1 | 0.069 | [bedroom_frame0315_conf0.069.png](output/The_Present/validation_frames/scene_category/bedroom_frame0315_conf0.069.png) |
| 348 | 14.5 | 0.070 | [bedroom_frame0348_conf0.070.png](output/The_Present/validation_frames/scene_category/bedroom_frame0348_conf0.070.png) |

#### fantasy or animated world

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 1 | 0.0 | 0.068 | [fantasy_or_animated_world_frame0001_conf0.068.png](output/The_Present/validation_frames/scene_category/fantasy_or_animated_world_frame0001_conf0.068.png) |
| 1693 | 70.6 | 0.070 | [fantasy_or_animated_world_frame1693_conf0.070.png](output/The_Present/validation_frames/scene_category/fantasy_or_animated_world_frame1693_conf0.070.png) |
| 3272 | 136.4 | 0.070 | [fantasy_or_animated_world_frame3272_conf0.070.png](output/The_Present/validation_frames/scene_category/fantasy_or_animated_world_frame3272_conf0.070.png) |

#### indoor room

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 301 | 12.5 | 0.069 | [indoor_room_frame0301_conf0.069.png](output/The_Present/validation_frames/scene_category/indoor_room_frame0301_conf0.069.png) |
| 4710 | 196.3 | 0.069 | [indoor_room_frame4710_conf0.069.png](output/The_Present/validation_frames/scene_category/indoor_room_frame4710_conf0.069.png) |
| 4794 | 199.8 | 0.070 | [indoor_room_frame4794_conf0.070.png](output/The_Present/validation_frames/scene_category/indoor_room_frame4794_conf0.070.png) |

#### kitchen

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 259 | 10.8 | 0.069 | [kitchen_frame0259_conf0.069.png](output/The_Present/validation_frames/scene_category/kitchen_frame0259_conf0.069.png) |
| 2082 | 86.8 | 0.070 | [kitchen_frame2082_conf0.070.png](output/The_Present/validation_frames/scene_category/kitchen_frame2082_conf0.070.png) |
| 2209 | 92.1 | 0.070 | [kitchen_frame2209_conf0.070.png](output/The_Present/validation_frames/scene_category/kitchen_frame2209_conf0.070.png) |

#### living room

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 1859 | 77.5 | 0.070 | [living_room_frame1859_conf0.070.png](output/The_Present/validation_frames/scene_category/living_room_frame1859_conf0.070.png) |
| 2933 | 122.2 | 0.069 | [living_room_frame2933_conf0.069.png](output/The_Present/validation_frames/scene_category/living_room_frame2933_conf0.069.png) |
| 2977 | 124.1 | 0.070 | [living_room_frame2977_conf0.070.png](output/The_Present/validation_frames/scene_category/living_room_frame2977_conf0.070.png) |

#### office or workspace

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 668 | 27.8 | 0.069 | [office_or_workspace_frame0668_conf0.069.png](output/The_Present/validation_frames/scene_category/office_or_workspace_frame0668_conf0.069.png) |
| 839 | 35.0 | 0.068 | [office_or_workspace_frame0839_conf0.068.png](output/The_Present/validation_frames/scene_category/office_or_workspace_frame0839_conf0.068.png) |
| 3011 | 125.5 | 0.069 | [office_or_workspace_frame3011_conf0.069.png](output/The_Present/validation_frames/scene_category/office_or_workspace_frame3011_conf0.069.png) |

#### outdoor street

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 2494 | 103.9 | 0.069 | [outdoor_street_frame2494_conf0.069.png](output/The_Present/validation_frames/scene_category/outdoor_street_frame2494_conf0.069.png) |

#### stage or theater

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 37 | 1.5 | 0.068 | [stage_or_theater_frame0037_conf0.068.png](output/The_Present/validation_frames/scene_category/stage_or_theater_frame0037_conf0.068.png) |
| 82 | 3.4 | 0.068 | [stage_or_theater_frame0082_conf0.068.png](output/The_Present/validation_frames/scene_category/stage_or_theater_frame0082_conf0.068.png) |
| 126 | 5.3 | 0.068 | [stage_or_theater_frame0126_conf0.068.png](output/The_Present/validation_frames/scene_category/stage_or_theater_frame0126_conf0.068.png) |

#### vehicle interior

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 677 | 28.2 | 0.069 | [vehicle_interior_frame0677_conf0.069.png](output/The_Present/validation_frames/scene_category/vehicle_interior_frame0677_conf0.069.png) |
| 685 | 28.5 | 0.071 | [vehicle_interior_frame0685_conf0.071.png](output/The_Present/validation_frames/scene_category/vehicle_interior_frame0685_conf0.071.png) |
| 2129 | 88.7 | 0.070 | [vehicle_interior_frame2129_conf0.070.png](output/The_Present/validation_frames/scene_category/vehicle_interior_frame2129_conf0.070.png) |

---

## 12. Visual Inspection Observations

### Low-level features

**`luminance_mean`** — PASS. Q1 = black fade-in + title card + dark blind-filtered room. Q4 = sunlit carpet overhead shots and open-door daylight. Monotonic, correct direction.

**`contrast_rms`** — PASS. Q1 = pure-black frame + face close-ups (uniform skin). Q4 = window-blind stripes (maximum regular alternation) + harsh sunlit hallway. Physically correct.

**`entropy`** — PASS (skewed distribution). Q1 anchored by rare pure-black (0.05 bits) and title card (1.35 bits). 75% of frames occupy only 0.63 bits of range (7.15–7.78). Feature discriminates title/fade frames well; limited for normal content.

### Color features

**`color_r/g/b_mean`** — WEAK as semantic color labels, but numerically correct as channel means. Quartiles mostly track warm-vs-cool illumination and overall brightness; `Q4_high_green` and `Q4_high_blue` are often bright neutral doorway shots rather than genuinely green- or blue-dominant scenes. Use these as raw channel-intensity features, not as human-interpretable color names.

**`saturation_mean`** — WEAK. Mid/high-saturation frames often look plausible, but dark/title-card frames can score as maximally vivid. Example: frame 92 is almost black (`luminance_mean=0.0315`) yet has `saturation_mean=0.9067`, so HSV saturation becomes unstable on near-black inputs.

### Texture features

**`edge_density`** — PASS. Q1 contains black/title frames and smooth face close-ups; Q4 contains blind slats, room geometry, rug texture, and hand+ball shots with visibly denser contours. Direction is correct.

**`spatial_freq_energy`** — WEAK. It does rank title text, blind slats, and other sharp repetitive patterns above flatter shots, but the dynamic range is tiny (0.0001–0.037) and the quartiles are visually mixed. For this animated short it is technically consistent, but not very discriminative.

### Motion features

**`motion_energy`** — PASS. Q1 is dominated by static black/title frames and held poses; Q4 contains dog-play, hand/ball interaction, and larger expression/body changes. Frame 0 = 0.0 is also correct by construction.

**`scene_cut`** — FAIL. Several reported cuts are not edits at all, just adjacent motion frames. Examples: cut 2506 (frames 2505/2506/2507), cut 2811, and cut 3405 all show the same shot with small motion. The luminance-diff threshold is too weak to distinguish shot changes from within-shot motion/lighting change.

### Face features

**`n_faces`** — WEAK. The sampled 0/1/2-face examples are mostly sensible, but the detector is contaminated by animal-face false positives on this film. Across the CSV, 206 frames that also contain a detected `dog` have `n_faces > 0`, so the count is not reliably 'human faces only'.

**`face_area_frac`** — FAIL as a human-face-size metric. High-area examples include dog close-ups (frames 2188, 3437, 3448), so the largest values are not consistently measuring human face prominence.

### Depth features

**`depth_mean/std/range`** — WEAK overall. `depth_std` and `depth_range` broadly track flatter vs more layered shots, but `depth_mean` is not visually aligned with the `near`/`far` labels in the report: Q1 includes wide doorway/room views while Q4 includes close face/dog close-ups. The metric is likely inverse-depth-like or otherwise model-relative, so the current `near`/`far` interpretation is backwards or at least ambiguous.

### Object features

**`n_objects`** — WEAK. Counts rise on cluttered kitchen/living-room shots as expected, but they inherit the detector's domain-mismatch errors on animation, so the absolute counts should be treated as rough complexity signals rather than trusted object cardinality.

**`object_categories`** — WEAK/NOISY. Some tags are correct (`dog`, `person`, `potted plant`), but sampled false positives are obvious: `book` on window blinds (frame 845), `cell phone` on a game controller (frame 1942), and `teddy bear` on dog frames (e.g. frame 3437). Good enough for rough tags, not for clean semantics.

### CLIP scene features

**`scene_natural_score`** — FAIL for this film. See cross-check notes below.

**`scene_open_score`** — FAIL for this film. Range [-0.080, +0.023], all negative (all interior).

**`scene_category_score`** — FAIL. Constant ≈ 1/15 = 0.067 across all frames (uniform softmax).

**`scene_category`** — FAIL. 10 of 15 categories appear including 'bathroom' and 'vehicle interior' for a living-room film. Assignments are noise due to near-uniform confidence.

### CLIP cross-check: code and index alignment confirmed correct

- `frame_idx` is sequential 0–4876, no gaps. Low-level features (luminance, contrast) confirmed against known visual frames — index alignment is correct.
- Top `scene_natural_score` frames (3650, 2814) are the film's outdoor patio/garden ending scenes — direction is correct at the extremes.
- Pure-black frames (0–5) get spurious positive score (~+0.02) due to L2-normalizing a near-zero CLIP embedding (floating-point noise amplification). This is a degenerate input issue, not a code bug.
- All CLIP features fail due to content mismatch, not implementation error.

### Wrong or weak metrics for this film

- **Failing**: `scene_cut`, `face_area_frac`, `scene_natural_score`, `scene_open_score`, `scene_category_score`, `scene_category`.
- **Weak / noisy**: `color_r_mean`, `color_g_mean`, `color_b_mean`, `saturation_mean`, `spatial_freq_energy`, `n_faces`, `depth_mean`, `depth_std`, `depth_range`, `n_objects`, `object_categories`.

---

## 13. Overall Validation Summary

| Feature | Code | Recompute | Distribution | Visual | Overall |
|---------|------|-----------|-------------|--------|---------|
| `luminance_mean` | PASS | PASS | PASS | PASS | **PASS** |
| `contrast_rms` | PASS | PASS | PASS | PASS | **PASS** |
| `entropy` | PASS | PASS | PASS | PASS (skewed) | **PASS** |
| `color_r/g/b_mean` | PASS | PASS | PASS | WEAK (channel means, not semantic colors) | **WEAK** |
| `saturation_mean` | PASS | PASS | PASS | WEAK (dark-frame artifact) | **WEAK** |
| `edge_density` | PASS | PASS | PASS | PASS | **PASS** |
| `spatial_freq_energy` | PASS | PASS | PASS (tiny range) | WEAK | **WEAK** |
| `motion_energy` | PASS | PASS | PASS | PASS | **PASS** |
| `scene_cut` | PASS | PASS | PASS (18 cuts) | FAIL | **FAIL** |
| `n_faces` | PASS | N/A (GPU) | PASS | WEAK (dog false positives) | **WEAK** |
| `face_area_frac` | PASS | N/A (GPU) | PASS | FAIL | **FAIL** |
| `depth_mean/std/range` | PASS | N/A (GPU) | PASS | WEAK (`depth_mean` direction ambiguous) | **WEAK** |
| `n_objects` | PASS | N/A (GPU) | PASS | WEAK | **WEAK** |
| `object_categories` | PASS | N/A (GPU) | PASS | WEAK | **WEAK** |
| `scene_natural_score` | PASS | N/A (GPU) | PASS (narrow) | FAIL | **FAIL (this film)** |
| `scene_open_score` | PASS | N/A (GPU) | PASS (narrow) | FAIL | **FAIL (this film)** |
| `scene_category_score` | PASS | N/A (GPU) | PASS (constant) | FAIL | **FAIL (this film)** |
| `scene_category` | PASS | N/A (GPU) | PASS | FAIL | **FAIL (this film)** |

> CLIP failures are content-mismatch (single-location animated short), not code errors.
> GPU-based features (depth, objects, faces, CLIP) cannot be recomputed without GPU + model re-run.

