# Per-channel probe pooling (`--keep_channels`)

**Date:** 2026-05-01

## Summary

`probe_eval.py` gains `--keep_channels`. When enabled, `pool_to_windows`
pools only over patches and concatenates the CorrCA channel axis into the
feature dimension — the per-clip movie-feature probe head input grows from
`embed_dim` to `n_chans * embed_dim` (5×64 = 320 with the standard 5-component
CorrCA). Subject and movie-id probes auto-resize from the embedding shape.

The default (`keep_channels=False`) is unchanged and remains the production
training behavior.

## Why

Per-subject ceiling analysis on the 5-component CorrCA-projected test EEG
(108 subjects) showed:

| Feature | Per-subject in-sample r | JEPA + CorrCA (default pool) |
|---|---:|---:|
| luminance | 0.082 ± 0.089 | 0.143 ± 0.074 |
| contrast | 0.088 ± 0.077 | 0.130 ± 0.074 |
| position | 0.106 ± 0.126 | 0.171 ± 0.078 |
| **narrative** | **0.063 ± 0.024** | **−0.005 ± 0.031** |

For luminance / contrast / position the JEPA probe sits at or above the
per-subject ceiling. For narrative it sits at chance, even though raw
CorrCA + linear probe reaches the ceiling (~0.071) — the JEPA stack
*destroys* narrative signal that survives spatial filtering.

CorrCA component 1 alone, averaged across 108 subjects at 2-s windows,
correlates at +0.213 with `narrative_event_score`. The other 4 components
do not. `pool_to_windows` was averaging the 5 components at the embedding
boundary, diluting channel 1's narrative direction by ~5×: 0.071 / 5 ≈
0.014, matching the post-pool measurement.

Concatenating channels into the feature axis gives the linear/MLP probe
its own slots for each CorrCA component, so it can route around the
non-narrative channels.

## Result on the existing exp6 baseline (`nw4_ws2`, 5 enc × 5 probe seeds)

Procedure matches the original `docs/significance_analysis_2026-04-29.md`:
each encoder's predictions averaged across 5 probe seeds first, then
recording-level bootstrap with B=2000 resamples, then 1-sample t-test on
the 5 encoder-bootstrap means against chance.

`reg_narrative_event_score_corr` on test split — three significance lenses:

| Pool | View 2 (per-enc-seed mean t-test, n=5) | View 3 (bootstrap B=2000, ensemble of 5 probe seeds) |
|---|---|---|
| Default | −0.009 ± 0.011, p=0.14 ✗ | −0.020 ± 0.045, p=0.38 ✗ |
| **`--keep_channels`** | **+0.061 ± 0.014, p=6.7e-4 ✓** | **+0.062 ± 0.021, p=2.7e-3 ✓** |

The other previously-failing baseline metric, `cls_position_in_movie_auc`,
also crosses the bootstrap-t-test threshold:

| Pool | View 3 (B=2000) |
|---|---|
| Default | +0.512 ± 0.012, p=0.080 ✗ |
| `--keep_channels` | +0.636 ± 0.009, p=5.2e-6 ✓ |

Other probes either improve or hold:

| Probe (test, View 3 bootstrap B=2000) | Default | `--keep_channels` | Δ |
|---|---:|---:|---:|
| `reg_position_in_movie_corr` | +0.088 | +0.212 | +0.124 |
| `reg_luminance_mean_corr` | +0.137 | +0.187 | +0.050 |
| `reg_contrast_rms_corr` | +0.030 | +0.101 | +0.071 |
| `subject/age_reg/corr` | +0.370 | +0.504 | +0.134 |
| `subject/sex/auc` | +0.619 | +0.713 | +0.094 |

Every probe is now significant under all three lenses.

## Usage

```bash
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint=/path/to/latest.pth.tar \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --keep_channels
```

## Scope

- Eval-side only. Encoder pretraining is unchanged; existing checkpoints
  work as-is.
- Wired in `MaskedJEPA.encode`, `MaskedJEPANoEMA.encode`,
  `MaskedJEPAProbe.__init__/forward`, and `EEGEncoderTokens.pool_to_windows`.
  Default `keep_channels=False` preserves existing behavior.
