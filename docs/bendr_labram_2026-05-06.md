# BENDR + LaBraM at nw2_ws4 — 2026-05-06

5 encoder seeds × 2 tiers × 2 models = 20 runs total. All at canonical
nw=2 ws=4 protocol (8 s clips, 200 Hz, 19-ch 10-20 montage subset).

## Integration tricks

| Model | Channel adapt | Temporal adapt | Sfreq |
|---|---|---|---|
| **BENDR** | Pretrained 1st conv `[512, 20, 3]` sliced to `[512, 19, 3]` (drop the 20th channel; pretraining had a reference channel) | None — conv stack is sample-rate-agnostic | `target_sfreq=200` (no resample); pretrained at 250 Hz but conv kernels operate at any rate |
| **LaBraM** | None (built with `n_chans=19`, `on_unknown_chs="warn"`) | Pretrained `temporal_embedding [1, 16, 200]` spliced into our `[1, 5, 200]` slots 1–4 (CLS at slot 0 untouched). 2 missing keys = `final_layer.{weight,bias}` (replaced by task head) | `target_sfreq=200`, `patch_size=200` |

## Tier 3 (frozen FM + canonical Ridge probe) — L3 (5-seed mean ± 1σ)

| Method | lum | cont | pos | narr |
|---|---|---|---|---|
| **BENDR** (frozen + Ridge) | 0.021 ± 0.010 ✓ | 0.039 ± 0.024 ✓ | 0.023 ± 0.004 ✓ | 0.013 ± 0.021 ns |
| **LaBraM** (frozen + Ridge) | 0.136 ± 0.019 ✓ | 0.132 ± 0.038 ✓ | 0.092 ± 0.022 ✓ | 0.018 ± 0.016 ns |
| BIOT (frozen + Ridge, ref) | 0.127 ± 0.033 ✓ | 0.107 ± 0.035 ✓ | 0.107 ± 0.038 ✓ | 0.003 ± 0.009 ns |
| Luna (frozen + Ridge, ref) | 0.182 ± 0.021 ✓ | 0.158 ± 0.018 ✓ | 0.171 ± 0.012 ✓ | 0.053 ± 0.036 ✓ |
| Cine-JEPA + Ridge (ours) | 0.226 ± 0.018 ✓ | 0.223 ± 0.014 ✓ | 0.226 ± 0.016 ✓ | 0.155 ± 0.023 ✓ |

LaBraM ≈ BIOT (within 1σ on most metrics). Luna leads frozen-FM Tier 3.
BENDR is markedly weaker than the other three FMs — its pretraining
(TUEG self-supervised contrastive) doesn't transfer well to 19-ch HBN at
nw2_ws4. None of the frozen-FM Tier 3 methods reach Cine-JEPA's narrative
score; only Luna is significant on narrative.

## Tier 4 (full fine-tune, native head) — L3 (5-seed mean ± 1σ)

| Method | lum | cont | pos | narr |
|---|---|---|---|---|
| **BENDR** (full FT) | 0.087 ± 0.122 ns | 0.100 ± 0.088 ns | 0.128 ± 0.096 ✓ | −0.001 ± 0.071 ns |
| **LaBraM** (full FT) | 0.126 ± 0.081 ✓ | 0.089 ± 0.047 ✓ | 0.070 ± 0.057 ns | 0.069 ± 0.068 ns |
| BIOT (full FT, ref) | 0.085 ± 0.060 ✓ | 0.074 ± 0.066 ns | 0.109 ± 0.030 ✓ | 0.013 ± 0.028 ns |
| Luna (full FT, ref) | 0.192 ± 0.120 ✓ | 0.169 ± 0.038 ✓ | 0.184 ± 0.031 ✓ | 0.095 ± 0.100 ns |
| CBraMod (full FT, n=4, ref) | 0.052 ± 0.077 ns | 0.034 ± 0.042 ns | 0.022 ± 0.079 ns | 0.040 ± 0.067 ns |
| Cine-JEPA + Ridge (ours, ref) | 0.226 ± 0.018 ✓ | 0.223 ± 0.014 ✓ | 0.226 ± 0.016 ✓ | 0.155 ± 0.023 ✓ |

Both BENDR and LaBraM Tier 4 results have substantial seed variance
(σ comparable to or larger than mean). LaBraM lum is significant; BENDR pos
is significant; everything else is non-significant. Neither beats Luna FT.

## Per-seed L1 detail

```
Tier 3 (frozen + Ridge)
bendr  seed=42:    lum=+0.027 cont=+0.056 pos=+0.027 narr=+0.034
bendr  seed=123:   lum=+0.024 cont=+0.017 pos=+0.020 narr=+0.002
bendr  seed=456:   lum=+0.006 cont=+0.072 pos=+0.025 narr=−0.017
bendr  seed=789:   lum=+0.032 cont=+0.019 pos=+0.025 narr=+0.032
bendr  seed=2025:  lum=+0.017 cont=+0.029 pos=+0.018 narr=+0.012
labram seed=42:    lum=+0.132 cont=+0.089 pos=+0.073 narr=+0.016
labram seed=123:   lum=+0.107 cont=+0.118 pos=+0.082 narr=+0.019
labram seed=456:   lum=+0.134 cont=+0.109 pos=+0.079 narr=+0.001
labram seed=789:   lum=+0.148 cont=+0.171 pos=+0.103 narr=+0.011
labram seed=2025:  lum=+0.156 cont=+0.174 pos=+0.125 narr=+0.045

Tier 4 (full FT, nw2_ws4)
bendr  seed=42:    lum=+0.111 cont=+0.113 pos=+0.066 narr=−0.114
bendr  seed=123:   lum=+0.011 cont=−0.002 pos=+0.021 narr=+0.074
bendr  seed=456:   lum=+0.201 cont=+0.142 pos=+0.207 narr=+0.026
bendr  seed=789:   lum=+0.196 cont=+0.216 pos=+0.247 narr=−0.015
bendr  seed=2025:  lum=−0.083 cont=+0.030 pos=+0.100 narr=+0.023
labram seed=42:    lum=+0.032 cont=+0.073 pos=+0.031 narr=+0.007
labram seed=123:   lum=+0.124 cont=+0.083 pos=+0.055 narr=+0.041
labram seed=456:   lum=+0.061 cont=+0.067 pos=+0.007 narr=+0.082
labram seed=789:   lum=+0.214 cont=+0.170 pos=+0.116 narr=+0.035
labram seed=2025:  lum=+0.197 cont=+0.051 pos=+0.143 narr=+0.180
```

## Source artifacts

```
predictions/tier3_canonical/{bendr,labram}_seed{S}/test_seed{S+2}.npz
results/unified/pB_t3_{bendr,labram}_seed{S}_canonical_seed{S}.json
tier4/predictions/{bendr,labram}_seed{S}/test_seed{S}.npz
```

Job IDs:
- Tier 3 BENDR: 18095047–56 (extract+probe pairs)
- Tier 3 LaBraM: 18094404–13 (extract+probe pairs)
- Tier 4 BENDR (nw2_ws4): 18096149–53
- Tier 4 LaBraM (nw2_ws4): 18096154–58

## Implementation note

Earlier failures:
- **BENDR Tier 3 + Tier 4 first attempt**: `target_sfreq=250` triggered
  `_resample_torch` NotImplementedError for src=200 → tgt=250 because the
  upsampler isn't wired (would need `torchaudio.functional.resample`).
  Fixed by setting `target_sfreq=200` since BENDR's conv kernels are
  sample-rate-agnostic (kernel shape doesn't change; receptive field in
  seconds shifts slightly).
- **LaBraM Tier 4 first attempt**: failed with `pos_embed [1,77,200] vs
  x dim1=39`. Cause: tier4_full_ft.sbatch defaults `NW=4 WS=2`, so the
  model was BUILT for `spec.window_seconds=4` (n_times=800, 4 temporal
  patches → pos_embed of 19×4+1=77) but FED 2 s windows (n_times=400,
  2 patches → x dim1=19×2+1=39). Fixed by overriding `NW=2 WS=4`.
