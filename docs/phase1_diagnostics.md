# Phase 1 — Encoder Diagnostic Sweep Results

Anchor checkpoint: `phaseD_nw4ws2_baseline_s42` (single enc seed, 5 probe seeds per condition).  All metrics on the **test** split.

Conditions are coded `<layer>_<tower>_<routing>` where:
- `layer` ∈ {patch_embed, block0, final}
- `tower` ∈ {stu (student / context encoder), tea (EMA target encoder)}
- `routing` ∈ {mp (mean-pool, default), kc (--keep_channels)}
- `chN` = single-channel attribution (channel N, mean-pool)
- `prepred` = project final-layer tokens through predictor.input_proj (24-d)

## reg_narrative_event_score_corr

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_prepred` | +0.0332 ± 0.0165 | 5 |
| `final_stu_ch4` | +0.0307 ± 0.0332 | 5 |
| `block0_tea_kc` | +0.0303 ± 0.0546 | 5 |
| `patch_embed_stu_kc` | +0.0303 ± 0.0655 | 5 |
| `block0_stu_kc` | +0.0265 ± 0.0469 | 5 |
| `final_stu_kc` | +0.0262 ± 0.0349 | 5 |
| `final_tea_kc` | +0.0241 ± 0.0430 | 5 |
| `final_stu_ch3` | +0.0175 ± 0.0362 | 5 |
| `patch_embed_tea_kc` | +0.0168 ± 0.0599 | 5 |
| `block0_stu_mp` | +0.0003 ± 0.0359 | 5 |
| `final_tea_mp` | -0.0007 ± 0.0354 | 5 |
| `final_stu_mp` | -0.0020 ± 0.0169 | 5 |
| `block0_tea_mp` | -0.0087 ± 0.0385 | 5 |
| `final_stu_ch0` | -0.0105 ± 0.0493 | 5 |
| `final_stu_ch2` | -0.0109 ± 0.0711 | 5 |
| `final_stu_ch1` | -0.0118 ± 0.0415 | 5 |
| `patch_embed_stu_mp` | -0.0400 ± 0.0396 | 5 |
| `patch_embed_tea_mp` | -0.0402 ± 0.0376 | 5 |

## reg_position_in_movie_corr

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_mp` | +0.1898 ± 0.0625 | 5 |
| `final_stu_ch1` | +0.1776 ± 0.0652 | 5 |
| `block0_stu_mp` | +0.1698 ± 0.0736 | 5 |
| `final_stu_prepred` | +0.1584 ± 0.0759 | 5 |
| `final_tea_mp` | +0.1550 ± 0.0673 | 5 |
| `final_stu_ch3` | +0.1472 ± 0.1005 | 5 |
| `final_stu_ch4` | +0.1463 ± 0.0562 | 5 |
| `block0_tea_mp` | +0.1254 ± 0.0593 | 5 |
| `final_stu_ch2` | +0.1247 ± 0.0495 | 5 |
| `final_stu_kc` | +0.1243 ± 0.0615 | 5 |
| `block0_tea_kc` | +0.1241 ± 0.0524 | 5 |
| `block0_stu_kc` | +0.1146 ± 0.0490 | 5 |
| `final_tea_kc` | +0.1128 ± 0.0465 | 5 |
| `patch_embed_tea_kc` | +0.0982 ± 0.0366 | 5 |
| `patch_embed_stu_kc` | +0.0905 ± 0.0380 | 5 |
| `final_stu_ch0` | +0.0886 ± 0.0852 | 5 |
| `patch_embed_tea_mp` | +0.0403 ± 0.0216 | 5 |
| `patch_embed_stu_mp` | +0.0315 ± 0.0244 | 5 |

## reg_luminance_mean_corr

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_mp` | +0.1745 ± 0.0933 | 5 |
| `final_stu_prepred` | +0.1734 ± 0.0347 | 5 |
| `block0_stu_mp` | +0.1535 ± 0.0899 | 5 |
| `final_stu_ch1` | +0.1516 ± 0.0879 | 5 |
| `final_tea_mp` | +0.1434 ± 0.0857 | 5 |
| `final_stu_ch4` | +0.1421 ± 0.0644 | 5 |
| `block0_stu_kc` | +0.1343 ± 0.0521 | 5 |
| `block0_tea_kc` | +0.1296 ± 0.0402 | 5 |
| `final_stu_kc` | +0.1291 ± 0.0675 | 5 |
| `final_tea_kc` | +0.1280 ± 0.0503 | 5 |
| `block0_tea_mp` | +0.1204 ± 0.0694 | 5 |
| `final_stu_ch3` | +0.1191 ± 0.0869 | 5 |
| `patch_embed_tea_kc` | +0.1131 ± 0.0273 | 5 |
| `final_stu_ch2` | +0.1125 ± 0.0884 | 5 |
| `patch_embed_stu_kc` | +0.1118 ± 0.0253 | 5 |
| `final_stu_ch0` | +0.0933 ± 0.0917 | 5 |
| `patch_embed_stu_mp` | +0.0523 ± 0.0480 | 5 |
| `patch_embed_tea_mp` | +0.0498 ± 0.0513 | 5 |

## reg_contrast_rms_corr

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_prepred` | +0.1373 ± 0.0690 | 5 |
| `final_stu_ch1` | +0.1291 ± 0.0413 | 5 |
| `block0_stu_mp` | +0.1201 ± 0.0759 | 5 |
| `final_stu_mp` | +0.1150 ± 0.0761 | 5 |
| `final_tea_mp` | +0.1123 ± 0.0698 | 5 |
| `final_stu_ch4` | +0.0958 ± 0.0550 | 5 |
| `block0_tea_mp` | +0.0935 ± 0.0612 | 5 |
| `final_stu_ch2` | +0.0917 ± 0.0513 | 5 |
| `final_tea_kc` | +0.0830 ± 0.0294 | 5 |
| `final_stu_ch3` | +0.0823 ± 0.0858 | 5 |
| `block0_tea_kc` | +0.0793 ± 0.0336 | 5 |
| `block0_stu_kc` | +0.0773 ± 0.0349 | 5 |
| `patch_embed_tea_kc` | +0.0727 ± 0.0194 | 5 |
| `final_stu_kc` | +0.0719 ± 0.0197 | 5 |
| `patch_embed_stu_kc` | +0.0711 ± 0.0208 | 5 |
| `final_stu_ch0` | +0.0697 ± 0.1047 | 5 |
| `patch_embed_tea_mp` | +0.0302 ± 0.0237 | 5 |
| `patch_embed_stu_mp` | +0.0199 ± 0.0371 | 5 |

## cls_position_in_movie_auc

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_mp` | +0.5754 ± 0.0302 | 5 |
| `final_stu_prepred` | +0.5656 ± 0.0295 | 5 |
| `final_tea_mp` | +0.5617 ± 0.0332 | 5 |
| `block0_stu_mp` | +0.5613 ± 0.0384 | 5 |
| `final_stu_ch3` | +0.5558 ± 0.0463 | 5 |
| `final_stu_ch4` | +0.5520 ± 0.0238 | 5 |
| `block0_tea_mp` | +0.5500 ± 0.0154 | 5 |
| `final_stu_ch1` | +0.5493 ± 0.0267 | 5 |
| `block0_tea_kc` | +0.5489 ± 0.0288 | 5 |
| `block0_stu_kc` | +0.5471 ± 0.0397 | 5 |
| `final_tea_kc` | +0.5447 ± 0.0336 | 5 |
| `final_stu_kc` | +0.5442 ± 0.0442 | 5 |
| `patch_embed_tea_kc` | +0.5355 ± 0.0307 | 5 |
| `patch_embed_stu_kc` | +0.5346 ± 0.0305 | 5 |
| `final_stu_ch2` | +0.5318 ± 0.0268 | 5 |
| `final_stu_ch0` | +0.5238 ± 0.0329 | 5 |
| `patch_embed_tea_mp` | +0.5201 ± 0.0130 | 5 |
| `patch_embed_stu_mp` | +0.5158 ± 0.0140 | 5 |

## cls_narrative_event_score_auc

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_ch0` | +0.5371 ± 0.0266 | 5 |
| `final_stu_ch4` | +0.5366 ± 0.0309 | 5 |
| `final_stu_ch3` | +0.5348 ± 0.0137 | 5 |
| `final_stu_mp` | +0.5329 ± 0.0207 | 5 |
| `block0_stu_mp` | +0.5266 ± 0.0095 | 5 |
| `final_tea_mp` | +0.5228 ± 0.0083 | 5 |
| `block0_tea_kc` | +0.5164 ± 0.0321 | 5 |
| `block0_stu_kc` | +0.5151 ± 0.0309 | 5 |
| `final_stu_kc` | +0.5141 ± 0.0276 | 5 |
| `final_stu_prepred` | +0.5135 ± 0.0109 | 5 |
| `patch_embed_stu_kc` | +0.5122 ± 0.0244 | 5 |
| `final_tea_kc` | +0.5119 ± 0.0232 | 5 |
| `final_stu_ch2` | +0.5101 ± 0.0370 | 5 |
| `block0_tea_mp` | +0.5094 ± 0.0049 | 5 |
| `patch_embed_tea_kc` | +0.5073 ± 0.0174 | 5 |
| `patch_embed_stu_mp` | +0.5007 ± 0.0188 | 5 |
| `patch_embed_tea_mp` | +0.4990 ± 0.0188 | 5 |
| `final_stu_ch1` | +0.4959 ± 0.0218 | 5 |

## cls_luminance_mean_auc

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_mp` | +0.5564 ± 0.0277 | 5 |
| `final_stu_prepred` | +0.5555 ± 0.0348 | 5 |
| `block0_stu_mp` | +0.5486 ± 0.0203 | 5 |
| `final_tea_mp` | +0.5466 ± 0.0171 | 5 |
| `patch_embed_stu_kc` | +0.5463 ± 0.0242 | 5 |
| `patch_embed_tea_kc` | +0.5443 ± 0.0213 | 5 |
| `block0_tea_kc` | +0.5438 ± 0.0370 | 5 |
| `final_tea_kc` | +0.5412 ± 0.0296 | 5 |
| `block0_stu_kc` | +0.5381 ± 0.0358 | 5 |
| `final_stu_kc` | +0.5365 ± 0.0244 | 5 |
| `block0_tea_mp` | +0.5351 ± 0.0261 | 5 |
| `final_stu_ch3` | +0.5327 ± 0.0251 | 5 |
| `final_stu_ch1` | +0.5317 ± 0.0274 | 5 |
| `final_stu_ch2` | +0.5271 ± 0.0287 | 5 |
| `final_stu_ch4` | +0.5265 ± 0.0207 | 5 |
| `patch_embed_stu_mp` | +0.5140 ± 0.0230 | 5 |
| `final_stu_ch0` | +0.5131 ± 0.0271 | 5 |
| `patch_embed_tea_mp` | +0.5129 ± 0.0238 | 5 |

## cls_contrast_rms_auc

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_prepred` | +0.5736 ± 0.0173 | 5 |
| `final_stu_ch1` | +0.5710 ± 0.0220 | 5 |
| `final_tea_mp` | +0.5527 ± 0.0255 | 5 |
| `block0_tea_kc` | +0.5507 ± 0.0401 | 5 |
| `block0_stu_mp` | +0.5495 ± 0.0250 | 5 |
| `final_tea_kc` | +0.5485 ± 0.0372 | 5 |
| `final_stu_ch0` | +0.5477 ± 0.0307 | 5 |
| `final_stu_mp` | +0.5468 ± 0.0289 | 5 |
| `block0_stu_kc` | +0.5464 ± 0.0385 | 5 |
| `final_stu_kc` | +0.5445 ± 0.0377 | 5 |
| `final_stu_ch2` | +0.5437 ± 0.0195 | 5 |
| `block0_tea_mp` | +0.5377 ± 0.0225 | 5 |
| `patch_embed_stu_kc` | +0.5362 ± 0.0124 | 5 |
| `patch_embed_tea_kc` | +0.5350 ± 0.0207 | 5 |
| `final_stu_ch4` | +0.5258 ± 0.0206 | 5 |
| `final_stu_ch3` | +0.5254 ± 0.0294 | 5 |
| `patch_embed_stu_mp` | +0.5142 ± 0.0383 | 5 |
| `patch_embed_tea_mp` | +0.5087 ± 0.0328 | 5 |

## movie_id/top1_acc

| Condition | mean ± std | n |
|---|---|---|
| `block0_stu_kc` | +0.0778 ± 0.0319 | 5 |
| `block0_tea_kc` | +0.0704 ± 0.0191 | 5 |
| `patch_embed_stu_kc` | +0.0667 ± 0.0206 | 5 |
| `final_tea_kc` | +0.0667 ± 0.0214 | 5 |
| `final_stu_ch1` | +0.0648 ± 0.0203 | 5 |
| `final_stu_kc` | +0.0648 ± 0.0241 | 5 |
| `final_stu_mp` | +0.0630 ± 0.0136 | 5 |
| `patch_embed_tea_kc` | +0.0593 ± 0.0181 | 5 |
| `final_tea_mp` | +0.0537 ± 0.0214 | 5 |
| `final_stu_ch2` | +0.0519 ± 0.0246 | 5 |
| `final_stu_prepred` | +0.0519 ± 0.0284 | 5 |
| `block0_tea_mp` | +0.0500 ± 0.0150 | 5 |
| `final_stu_ch4` | +0.0481 ± 0.0214 | 5 |
| `block0_stu_mp` | +0.0481 ± 0.0170 | 5 |
| `final_stu_ch0` | +0.0463 ± 0.0155 | 5 |
| `final_stu_ch3` | +0.0444 ± 0.0230 | 5 |
| `patch_embed_stu_mp` | +0.0315 ± 0.0126 | 5 |
| `patch_embed_tea_mp` | +0.0315 ± 0.0111 | 5 |

## movie_id/top5_acc

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_kc` | +0.3111 ± 0.0364 | 5 |
| `block0_stu_kc` | +0.2926 ± 0.0474 | 5 |
| `block0_tea_kc` | +0.2889 ± 0.0386 | 5 |
| `final_tea_kc` | +0.2889 ± 0.0451 | 5 |
| `final_stu_ch1` | +0.2833 ± 0.0444 | 5 |
| `final_stu_ch4` | +0.2648 ± 0.0344 | 5 |
| `final_stu_prepred` | +0.2611 ± 0.0390 | 5 |
| `patch_embed_tea_kc` | +0.2556 ± 0.0478 | 5 |
| `patch_embed_stu_kc` | +0.2537 ± 0.0489 | 5 |
| `block0_tea_mp` | +0.2519 ± 0.0474 | 5 |
| `final_stu_ch2` | +0.2519 ± 0.0230 | 5 |
| `final_stu_ch0` | +0.2481 ± 0.0244 | 5 |
| `final_stu_ch3` | +0.2370 ± 0.0364 | 5 |
| `block0_stu_mp` | +0.2370 ± 0.0259 | 5 |
| `final_tea_mp` | +0.2333 ± 0.0328 | 5 |
| `final_stu_mp` | +0.2333 ± 0.0180 | 5 |
| `patch_embed_tea_mp` | +0.2259 ± 0.0172 | 5 |
| `patch_embed_stu_mp` | +0.2148 ± 0.0214 | 5 |

## subject/age_reg/corr

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_kc` | +0.4982 ± 0.0362 | 5 |
| `final_tea_kc` | +0.4759 ± 0.0227 | 5 |
| `block0_stu_kc` | +0.4712 ± 0.0164 | 5 |
| `block0_tea_kc` | +0.4194 ± 0.0164 | 5 |
| `final_stu_mp` | +0.3823 ± 0.0569 | 5 |
| `block0_stu_mp` | +0.3741 ± 0.0725 | 5 |
| `final_stu_ch3` | +0.3645 ± 0.0713 | 5 |
| `final_stu_ch4` | +0.3596 ± 0.0778 | 5 |
| `final_stu_ch2` | +0.3478 ± 0.0391 | 5 |
| `final_stu_prepred` | +0.3393 ± 0.0370 | 5 |
| `final_stu_ch0` | +0.3251 ± 0.0392 | 5 |
| `block0_tea_mp` | +0.3238 ± 0.0631 | 5 |
| `final_tea_mp` | +0.3121 ± 0.0731 | 5 |
| `final_stu_ch1` | +0.2794 ± 0.0644 | 5 |
| `patch_embed_tea_kc` | +0.0630 ± 0.0556 | 5 |
| `patch_embed_stu_kc` | +0.0307 ± 0.0679 | 5 |
| `patch_embed_tea_mp` | -0.0223 ± 0.1410 | 5 |
| `patch_embed_stu_mp` | -0.0314 ± 0.1354 | 5 |

## subject/age_cls/auc

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_kc` | +0.7249 ± 0.0106 | 5 |
| `final_stu_ch4` | +0.7046 ± 0.0151 | 5 |
| `block0_stu_kc` | +0.6985 ± 0.0100 | 5 |
| `final_tea_kc` | +0.6951 ± 0.0087 | 5 |
| `final_stu_ch3` | +0.6841 ± 0.0231 | 5 |
| `block0_tea_kc` | +0.6837 ± 0.0082 | 5 |
| `final_stu_mp` | +0.6801 ± 0.0109 | 5 |
| `block0_stu_mp` | +0.6698 ± 0.0202 | 5 |
| `final_stu_ch2` | +0.6671 ± 0.0029 | 5 |
| `final_stu_ch0` | +0.6598 ± 0.0088 | 5 |
| `final_tea_mp` | +0.6594 ± 0.0200 | 5 |
| `block0_tea_mp` | +0.6591 ± 0.0225 | 5 |
| `final_stu_ch1` | +0.6535 ± 0.0206 | 5 |
| `final_stu_prepred` | +0.6003 ± 0.0600 | 5 |
| `patch_embed_tea_kc` | +0.5453 ± 0.0345 | 5 |
| `patch_embed_stu_kc` | +0.5433 ± 0.0369 | 5 |
| `patch_embed_tea_mp` | +0.5332 ± 0.0826 | 5 |
| `patch_embed_stu_mp` | +0.5225 ± 0.0811 | 5 |

## subject/sex/auc

| Condition | mean ± std | n |
|---|---|---|
| `final_stu_kc` | +0.7100 ± 0.0093 | 5 |
| `block0_stu_kc` | +0.6802 ± 0.0180 | 5 |
| `final_tea_kc` | +0.6781 ± 0.0218 | 5 |
| `final_stu_ch2` | +0.6505 ± 0.0101 | 5 |
| `block0_tea_kc` | +0.6480 ± 0.0175 | 5 |
| `final_stu_ch4` | +0.6477 ± 0.0080 | 5 |
| `final_stu_mp` | +0.6235 ± 0.0046 | 5 |
| `block0_tea_mp` | +0.6167 ± 0.0111 | 5 |
| `final_tea_mp` | +0.6149 ± 0.0045 | 5 |
| `block0_stu_mp` | +0.6142 ± 0.0041 | 5 |
| `final_stu_ch0` | +0.6055 ± 0.0088 | 5 |
| `final_stu_ch3` | +0.5934 ± 0.0062 | 5 |
| `final_stu_prepred` | +0.5877 ± 0.0477 | 5 |
| `final_stu_ch1` | +0.5846 ± 0.0095 | 5 |
| `patch_embed_stu_kc` | +0.5487 ± 0.0163 | 5 |
| `patch_embed_tea_kc` | +0.5421 ± 0.0196 | 5 |
| `patch_embed_stu_mp` | +0.5060 ± 0.0617 | 5 |
| `patch_embed_tea_mp` | +0.5033 ± 0.0626 | 5 |

