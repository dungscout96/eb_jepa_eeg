# trivial_baselines

Sanity baselines that bypass the learned encoder: predict movie features
and subject traits directly from trivial EEG summary statistics
(mean, std, bandpower). The canonical JEPA pipeline should beat these.

## Sweeps

| Launcher | Purpose |
|---|---|
| `sweeps/trivial_baseline_raw.py` | Trivial features on raw 129-ch EEG (no CorrCA), per-rec normalized |
| `sweeps/trivial_shuffle_and_aligned.py` | Trivial features on CorrCA-projected EEG, with shuffled-label control |

## Submit

```bash
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \
    python experiments/trivial_baselines/sweeps/trivial_baseline_raw.py --submit
```

These do NOT call the JEPA pretrainer — they invoke
`experiments/position_leakage/baseline.py` directly (trivial-feature
baseline driver). See `experiments/position_leakage/` for the
underlying implementation.
