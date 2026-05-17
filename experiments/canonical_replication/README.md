# canonical_replication

Spec-faithful 5-seed replication of the locked canonical EEG JEPA protocol
(nw=2, ws=4, encoder_depth=2, predictor_embed_dim=24, per_recording norm,
CorrCA 129→5, lr=5e-4 cosine, VCLoss(0.25,0.25), smooth_l1, 100 ep).

This is the "promoted, working" pipeline. Use it as the reference point
for any new architecture / loss / data study — it's the headline number
the rest of the experiments should beat.

## Pipeline

Each seed runs end-to-end in one SLURM job (≈3 h budget):

1. Pretrain via `python -m eb_jepa.training.jepa_pretrain`
2. Canonical probe eval (sklearn Ridge/LogReg, n_passes=20, probe_seed=42)
   via `python -m eb_jepa.evaluation.probe_eval_canonical`
3. Bootstrap (B=2000, recording-level) via
   `python -m eb_jepa.evaluation.bootstrap_canonical`

After all 5 seeds finish, the aggregate sbatch collapses to L3 + t-test
vs chance via `scripts/aggregate_and_print.py`.

## Sweeps

| Launcher | Purpose |
|---|---|
| `sweeps/canonical_5seed.py` | Submit 5 pretrain+probe+bootstrap jobs |
| `sweeps/canonical_5seed_probe_only.py` | Re-run only the probe+bootstrap stage against existing checkpoints |

## Submit

```bash
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \
    python experiments/canonical_replication/sweeps/canonical_5seed.py --submit

# After all 5 finish, submit the aggregate (the script prints the exact command
# with --dependency=afterok:<jobids>):
sbatch --dependency=afterok:<...> eb_jepa/training/sbatch/canonical_aggregate.sbatch
```

## Reference results

See [docs/canonical_replication_report_2026-05-16.md](../../docs/canonical_replication_report_2026-05-16.md)
for the 5-seed L3 table and t-test vs chance (all 12 canonical metrics
significantly above chance).

## Dependencies

- Library: `eb_jepa.training.jepa_pretrain`, `eb_jepa.evaluation.probe_eval_canonical`,
  `eb_jepa.evaluation.bootstrap_canonical`, `scripts/aggregate_and_print.py`
- Delta: HBN preprocessed data at `/projects/bbnv/kkokate/hbn_preprocessed/`,
  CorrCA filters at `corrca_filters.npz` (in repo root on Delta).
