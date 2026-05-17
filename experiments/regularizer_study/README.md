# regularizer_study

Compare collapse-prevention regularizers for the JEPA objective: VICReg
(VCLoss = std + cov) vs SIGReg (slice information minimization), with
and without a projector head, and with optional CorrCA preprocessing.

## Sweeps

### VICReg (`VCLoss`)

| Launcher | Purpose |
|---|---|
| `sweeps/vicreg.py` | Sweep std/cov coefficients + projector ablation |
| `sweeps/vicreg_noproj.py` | Resubmit only the noproj VICReg variants |
| `sweeps/probe_eval_vicreg.py` | Post-hoc probe eval on VICReg checkpoints |

### SIGReg

| Launcher | Purpose |
|---|---|
| `sweeps/sigreg.py` | SIGReg on top 3 temporal configs from phase1 |
| `sweeps/sigreg_corrca.py` | SIGReg + per-recording norm + CorrCA |
| `sweeps/probe_eval_sigreg.py` | Post-hoc probe eval on SIGReg checkpoints |
| `sweeps/probe_eval_sigreg_corrca.py` | Post-hoc probe eval on SIGReg+CorrCA checkpoints |
| `sweeps/probe_eval_exp6_sigreg.py` | Post-hoc probe eval on the exp6 SIGReg run |

## Submit

```bash
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \
    python experiments/regularizer_study/sweeps/vicreg.py --submit
```

All pretrain launchers invoke `python -m eb_jepa.training.jepa_pretrain`.
Probe-eval launchers invoke `python -m eb_jepa.evaluation.probe_eval`
(legacy Adam-trained linear probes) — switch to
`eb_jepa.evaluation.probe_eval_canonical` for spec-faithful numbers.
