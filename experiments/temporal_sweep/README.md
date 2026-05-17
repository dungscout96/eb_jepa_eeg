# temporal_sweep

Sweep `n_windows × window_size_seconds` (temporal vs static pretraining)
to identify the best clip layout for JEPA on HBN movie watching. The
canonical pipeline runs nw=2 ws=4; this study justifies that choice.

## Sweeps

| Launcher | Purpose |
|---|---|
| `sweeps/phase1.py` | Full grid: 11 (nw, ws) configurations × 3 seeds, packed 2-per-job on Delta A40 |
| `sweeps/phase1_resume.py` | Resubmit incomplete Phase 1 jobs |
| `sweeps/nw2_ws4.py` | Single-config dive on the canonical nw=2 ws=4 layout (jamming workstation) |
| `sweeps/probe_eval_phase1.py` | Post-hoc canonical probe eval on all Phase 1 checkpoints |

## Submit

```bash
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \
    python experiments/temporal_sweep/sweeps/phase1.py --submit
```

All launchers invoke `python -m eb_jepa.training.jepa_pretrain` with the
phase-1 fixed config (encoder_depth=2, lr=5e-4, VCLoss(0.25,0.25),
smooth_l1, 100 ep) plus per-config (nw, ws, batch_size) overrides.
