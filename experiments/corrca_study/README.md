# corrca_study

Probe-eval studies on CorrCA-preprocessed EEG (129 → 5 stimulus-aligned
components). The canonical pipeline already uses CorrCA; this study
isolates the CorrCA contribution against non-CorrCA baselines.

## Sweeps

| Launcher | Purpose |
|---|---|
| `sweeps/probe_eval_corrca.py` | Probe eval on the CorrCA-preprocessed VICReg best checkpoint |
| `sweeps/probe_eval_corrca_savepreds.py` | Same, but with `--save_predictions_dir` for downstream bootstrap |

## Submit

```bash
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \
    python experiments/corrca_study/sweeps/probe_eval_corrca.py --submit
```

Invokes `python -m eb_jepa.evaluation.probe_eval` (legacy Adam probes).
Swap to `eb_jepa.evaluation.probe_eval_canonical` if you want
spec-faithful sklearn numbers.

## Prerequisite

CorrCA filters at `corrca_filters.npz` (fit on train-only recordings
via `scripts/compute_corrca.py`).
