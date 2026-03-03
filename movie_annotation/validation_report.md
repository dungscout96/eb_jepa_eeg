# Annotation Validation Report — The Present

Validates annotation features for Issue #5.
Primary features: `contrast_rms`, `luminance_mean`, `entropy`, `scene_natural_score`.
Additional features: `scene_open_score`, `scene_category`, `scene_category_score`.

---

## 1. Code Review

| Feature | Method | Library | Model-free? | Deterministic? |
|---------|--------|---------|------------|----------------|
| `luminance_mean` | `mean(grayscale) / 255.0` | OpenCV + NumPy | Yes | Yes |
| `contrast_rms` | `std(grayscale) / 255.0` (population std, ddof=0) | OpenCV + NumPy | Yes | Yes |
| `entropy` | Shannon entropy of 256-bin grayscale histogram (base 2) | SciPy | Yes | Yes |
| `scene_natural_score` | `cosine_sim(frame, 'natural scene') - cosine_sim(frame, 'urban scene')` | CLIP ViT-B/32 | No | Yes |
| `scene_open_score` | `cosine_sim(frame, 'open outdoor') - cosine_sim(frame, 'enclosed indoor')` | CLIP ViT-B/32 | No | Yes |
| `scene_category` | argmax softmax over 15 category text prompts | CLIP ViT-B/32 | No | Yes |
| `scene_category_score` | softmax probability of top category | CLIP ViT-B/32 | No | Yes |

---

## 2. Recomputation Check (Low-level Features)

Re-extracted `luminance_mean`, `contrast_rms`, `entropy` from the movie for
~30 evenly-spaced frames (every 163th frame) and compared
against stored CSV values.

| Feature | Frames Checked | Max Abs Error | Mean Error | Error Std | Pearson r | Note | Result |
|---------|---------------|--------------|------------|-----------|-----------|------|--------|
| `luminance_mean` | 30 | 3.91e-03 | 4.82e-04 | 9.06e-04 | 0.999994 | cross-platform codec offset | PASS |
| `contrast_rms` | 30 | 1.02e-03 | 1.66e-04 | 1.89e-04 | 0.999995 | cross-platform codec offset | PASS |
| `entropy` | 30 | 1.38e-01 | 9.03e-02 | 3.39e-02 | 0.999901 | cross-platform codec offset | PASS |

> **Pass criteria**: Pearson r > 0.999.
> A consistent mean offset with very low error std indicates cross-platform codec
> differences (e.g. H.264 limited vs full YUV range), **not** a formula error.

---

## 3. Distribution Checks

| Feature | Min | Max | Mean | Std | NaN count | In range? | Has variation? | Result |
|---------|-----|-----|------|-----|-----------|-----------|---------------|--------|
| `luminance_mean` | 0.0046 | 0.9216 | 0.4816 | 0.1603 | 0 | Yes | Yes | PASS |
| `contrast_rms` | 0.0135 | 0.3311 | 0.2437 | 0.0575 | 0 | Yes | Yes | PASS |
| `entropy` | 0.0482 | 7.7797 | 7.1678 | 1.2686 | 0 | Yes | Yes | PASS |
| `scene_natural_score` | -0.0651 | 0.0445 | -0.0090 | 0.0151 | 0 | Yes | Yes | PASS |
| `scene_open_score` | -0.0804 | 0.0234 | -0.0301 | 0.0193 | 0 | Yes | Yes | PASS |
| `scene_category_score` | 0.0675 | 0.0718 | 0.0699 | 0.0007 | 0 | Yes | Yes | PASS |

**`scene_category`** (categorical): 10 distinct categories, 0 NaN — PASS

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

## 4. `luminance_mean` — Visual Frame Inspection

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

---

## 5. `contrast_rms` — Visual Frame Inspection

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

---

## 6. `entropy` — Visual Frame Inspection

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

## 7. `scene_natural_score` — Visual Frame Inspection

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

---

## 8. `scene_open_score` — Visual Frame Inspection

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

---

## 9. `scene_category_score` — Visual Frame Inspection

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

---

## 10. `scene_category` — Visual Frame Inspection

### bathroom

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 0 | 0.0 | 0.068 | [bathroom_frame0000_conf0.068.png](output/The_Present/validation_frames/scene_category/bathroom_frame0000_conf0.068.png) |
| 31 | 1.3 | 0.068 | [bathroom_frame0031_conf0.068.png](output/The_Present/validation_frames/scene_category/bathroom_frame0031_conf0.068.png) |
| 3654 | 152.3 | 0.070 | [bathroom_frame3654_conf0.070.png](output/The_Present/validation_frames/scene_category/bathroom_frame3654_conf0.070.png) |

### bedroom

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 270 | 11.3 | 0.069 | [bedroom_frame0270_conf0.069.png](output/The_Present/validation_frames/scene_category/bedroom_frame0270_conf0.069.png) |
| 315 | 13.1 | 0.069 | [bedroom_frame0315_conf0.069.png](output/The_Present/validation_frames/scene_category/bedroom_frame0315_conf0.069.png) |
| 348 | 14.5 | 0.070 | [bedroom_frame0348_conf0.070.png](output/The_Present/validation_frames/scene_category/bedroom_frame0348_conf0.070.png) |

### fantasy or animated world

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 1 | 0.0 | 0.068 | [fantasy_or_animated_world_frame0001_conf0.068.png](output/The_Present/validation_frames/scene_category/fantasy_or_animated_world_frame0001_conf0.068.png) |
| 1693 | 70.6 | 0.070 | [fantasy_or_animated_world_frame1693_conf0.070.png](output/The_Present/validation_frames/scene_category/fantasy_or_animated_world_frame1693_conf0.070.png) |
| 3272 | 136.4 | 0.070 | [fantasy_or_animated_world_frame3272_conf0.070.png](output/The_Present/validation_frames/scene_category/fantasy_or_animated_world_frame3272_conf0.070.png) |

### indoor room

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 301 | 12.5 | 0.069 | [indoor_room_frame0301_conf0.069.png](output/The_Present/validation_frames/scene_category/indoor_room_frame0301_conf0.069.png) |
| 4710 | 196.3 | 0.069 | [indoor_room_frame4710_conf0.069.png](output/The_Present/validation_frames/scene_category/indoor_room_frame4710_conf0.069.png) |
| 4794 | 199.8 | 0.070 | [indoor_room_frame4794_conf0.070.png](output/The_Present/validation_frames/scene_category/indoor_room_frame4794_conf0.070.png) |

### kitchen

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 259 | 10.8 | 0.069 | [kitchen_frame0259_conf0.069.png](output/The_Present/validation_frames/scene_category/kitchen_frame0259_conf0.069.png) |
| 2082 | 86.8 | 0.070 | [kitchen_frame2082_conf0.070.png](output/The_Present/validation_frames/scene_category/kitchen_frame2082_conf0.070.png) |
| 2209 | 92.1 | 0.070 | [kitchen_frame2209_conf0.070.png](output/The_Present/validation_frames/scene_category/kitchen_frame2209_conf0.070.png) |

### living room

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 1859 | 77.5 | 0.070 | [living_room_frame1859_conf0.070.png](output/The_Present/validation_frames/scene_category/living_room_frame1859_conf0.070.png) |
| 2933 | 122.2 | 0.069 | [living_room_frame2933_conf0.069.png](output/The_Present/validation_frames/scene_category/living_room_frame2933_conf0.069.png) |
| 2977 | 124.1 | 0.070 | [living_room_frame2977_conf0.070.png](output/The_Present/validation_frames/scene_category/living_room_frame2977_conf0.070.png) |

### office or workspace

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 668 | 27.8 | 0.069 | [office_or_workspace_frame0668_conf0.069.png](output/The_Present/validation_frames/scene_category/office_or_workspace_frame0668_conf0.069.png) |
| 839 | 35.0 | 0.068 | [office_or_workspace_frame0839_conf0.068.png](output/The_Present/validation_frames/scene_category/office_or_workspace_frame0839_conf0.068.png) |
| 3011 | 125.5 | 0.069 | [office_or_workspace_frame3011_conf0.069.png](output/The_Present/validation_frames/scene_category/office_or_workspace_frame3011_conf0.069.png) |

### outdoor street

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 2494 | 103.9 | 0.069 | [outdoor_street_frame2494_conf0.069.png](output/The_Present/validation_frames/scene_category/outdoor_street_frame2494_conf0.069.png) |

### stage or theater

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 37 | 1.5 | 0.068 | [stage_or_theater_frame0037_conf0.068.png](output/The_Present/validation_frames/scene_category/stage_or_theater_frame0037_conf0.068.png) |
| 82 | 3.4 | 0.068 | [stage_or_theater_frame0082_conf0.068.png](output/The_Present/validation_frames/scene_category/stage_or_theater_frame0082_conf0.068.png) |
| 126 | 5.3 | 0.068 | [stage_or_theater_frame0126_conf0.068.png](output/The_Present/validation_frames/scene_category/stage_or_theater_frame0126_conf0.068.png) |

### vehicle interior

| Frame idx | Timestamp (s) | Conf | Image |
|-----------|--------------|------|-------|
| 677 | 28.2 | 0.069 | [vehicle_interior_frame0677_conf0.069.png](output/The_Present/validation_frames/scene_category/vehicle_interior_frame0677_conf0.069.png) |
| 685 | 28.5 | 0.071 | [vehicle_interior_frame0685_conf0.071.png](output/The_Present/validation_frames/scene_category/vehicle_interior_frame0685_conf0.071.png) |
| 2129 | 88.7 | 0.070 | [vehicle_interior_frame2129_conf0.070.png](output/The_Present/validation_frames/scene_category/vehicle_interior_frame2129_conf0.070.png) |

---

## 11. Visual Inspection Observations

### `luminance_mean` — PASS

The feature correctly tracks perceived brightness.

- **Q1 (dark, 0.005–0.40)**: Opens with the pure-black fade-in frame (score 0.005) and the Filmakademie title card. Remaining frames are dim indoor shots — the dark living room seen from ground level, moody close-ups of the boy under blinds-filtered sidelighting. Brightness increases monotonically through the quartile.
- **Q2–Q3 (0.43–0.57)**: Typical mid-tone interior shots — boy on sofa, hallway with closed door. Steady increase in average brightness.
- **Q4 (bright, 0.58–0.75)**: Bright sunlit carpet shots (overhead view of dog/red ball), hallway with open door flooding in daylight. The highest-scoring frame (0.755) is the dog on a cream-white tile floor — correctly identified as the brightest scene.

**Verdict**: Direction correct, monotonic, no anomalies. Feature is valid for use as an EEG label.

---

### `contrast_rms` — PASS

The feature correctly tracks pixel-intensity variance (spread of the grayscale histogram).

- **Q1 (low contrast, 0.014–0.207)**: Pure-black frame (score 0.014 — nearly zero std), title card (tiny bright text on black — low std), then smooth close-up faces filling the frame with uniform skin tones. Under-table/cabinet shots with flat muted surfaces also land here.
- **Q2–Q3 (0.216–0.280)**: Mixed interior scenes — wide shots with both light and shadow, sofa + background objects, moderate variation.
- **Q4 (high contrast, 0.282–0.314)**: The window-blinds frame (score 0.303) is the clearest validation — dense alternating bright/dark horizontal stripes maximise std. Hallway shots with harsh sunlight streaming through an open door (large bright patches against dark walls) also rank highly.

**Verdict**: Direction correct, physically interpretable extremes. Feature is valid for use as an EEG label.

---

### `entropy` — PASS (with skew caveat)

Shannon entropy of the 256-bin grayscale histogram correctly identifies pixel-value diversity.

- **Q1 (0.05–7.15)**: Scores span a huge range within this quartile. The lowest two frames are the pure-black fade-in (entropy ≈ 0.05 — single-bin histogram) and the title card (entropy ≈ 1.35 — near-empty histogram). These are correct. The remaining 6 Q1 frames score 6.5–7.15 — they look like normal but slightly lower-complexity scenes (dark room wide shots, simple face close-ups with smooth skin dominating).
- **Q2–Q4 (7.23–7.78)**: All look like typical movie frames with well-distributed grayscale values — wide shots with furniture, carpet, characters. The highest-entropy frames are room-wide shots with many simultaneous surfaces (cardboard box, carpet, sofa, wall, plants).

**Key caveat — heavily skewed distribution**: The full entropy range is [0.05, 7.78], but 75% of frames lie in [7.15, 7.78] — a spread of only 0.63 bits. The large Q1 score range (0.05 to 7.15) is driven by rare title/fade frames which are statistical outliers. For the 99%+ of normal movie frames, entropy varies by < 0.6 bits and has limited dynamic range.

**Verdict**: Feature is correctly computed. Useful for detecting scene-change / title frames (very low entropy) but offers limited discriminability across normal movie content.

---

### `scene_natural_score` — FAIL for this film

See Section 7 frame images. The entire film is interior. Score range is only 0.086 out of [-2, +2]. Frames with the open front door (most outdoor content) paradoxically score lowest ("most urban"). Beige carpet/dog close-ups score highest ("most natural") due to accidental texture correlation. No meaningful natural vs urban signal.

---

### `scene_open_score` — FAIL for this film

Distribution: min=-0.080, max=0.023, range=0.103. The film is almost entirely indoors, so the score hovers near zero throughout with no useful variation. Same root cause as `scene_natural_score`: the film lacks the content diversity this feature requires.

---

### `scene_category_score` — FAIL (uninformative constant)

Distribution: min=0.067, max=0.072, std=0.0007. With 15 categories, uniform softmax = 1/15 = 0.067. Every frame scores ≈0.070 — barely above uniform. CLIP assigns near-equal probability to all 15 scene categories for every frame of this animated short, meaning the "winning" category is effectively random. This feature carries no information for this film.

---

### `scene_category` — FAIL (noise labels)

10 of 15 categories appear (bathroom, bedroom, vehicle interior, stage or theater, etc.) for a film set entirely in a single living room/hallway. Since `scene_category_score` ≈ 1/15 for all frames, the assigned labels are noise. Categories such as "bathroom", "vehicle interior", and "stage or theater" appearing in a living-room film confirm the assignments are meaningless.

---

## 12. Overall Validation Summary

| Feature | Code Review | Recomputation | Distribution | Visual | Overall |
|---------|------------|--------------|-------------|--------|---------|
| `luminance_mean` | PASS | PASS | PASS | **PASS** | **PASS** |
| `contrast_rms` | PASS | PASS | PASS | **PASS** | **PASS** |
| `entropy` | PASS | PASS | PASS | **PASS (skewed)** | **PASS** |
| `scene_natural_score` | PASS | N/A (GPU) | PASS (narrow range) | **FAIL** | **FAIL for this film** |
| `scene_open_score` | PASS | N/A (GPU) | PASS (narrow range) | **FAIL** | **FAIL for this film** |
| `scene_category_score` | PASS | N/A (GPU) | PASS (constant ~1/15) | **FAIL** | **FAIL for this film** |
| `scene_category` | PASS | N/A (GPU) | PASS | **FAIL** | **FAIL for this film** |

**Summary**: The three low-level features (`luminance_mean`, `contrast_rms`, `entropy`) are correctly computed and suitable as EEG stimulus labels. All four CLIP-based features fail for "The Present" because the film is a single-location animated short without the scene diversity (natural landscapes, urban environments, varied room types) that CLIP's prompts require. The code is correct; the failure is a content-mismatch between the film and the feature design.

> CLIP-based feature recomputation requires GPU + model re-run.
> Code review confirms correct implementation for all CLIP features.

