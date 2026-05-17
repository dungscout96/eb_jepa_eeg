# retrain_best

Retrain the documented best VICReg and SIGReg configurations with
early stopping and (optionally) per-recording normalization to isolate
the contribution of the subject-fingerprint signal.

## Sweeps

| Launcher | Purpose |
|---|---|
| `sweeps/retrain_best.py` | Retrain 7 documented best configs with global norm |
| `sweeps/retrain_best_perrec.py` | Retrain same 7 with `--data.norm_mode=per_recording` |
| `sweeps/probe_eval_retrained_global.py` | Probe eval on global-norm retrained checkpoints |
| `sweeps/probe_eval_retrained_perrec.py` | Probe eval on per-rec retrained checkpoints |
| `sweeps/probe_eval_missing_perrec.py` | Resubmit probe eval for the per-rec checkpoint that timed out |

## Submit

```bash
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \
    python experiments/retrain_best/sweeps/retrain_best.py --submit
```

Pretrain via `python -m eb_jepa.training.jepa_pretrain`; probe eval via
`python -m eb_jepa.evaluation.probe_eval` (or the canonical variant for
spec-faithful numbers).
