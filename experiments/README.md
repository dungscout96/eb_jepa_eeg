# Experiments

Each subfolder is a self-contained study with its own sweep launchers,
sbatch files, and (for non-trivial studies) configs. Studies depend on
the `eb_jepa` library and on each other ONLY at the data level (shared
preprocessed HBN, shared CorrCA filters) — they never import each
other's code.

## Library entry points

The reusable training and eval pipeline lives in [`eb_jepa/`](../eb_jepa/):

| Library entry | What it does |
|---|---|
| `python -m eb_jepa.training.jepa_pretrain` | Masked-prediction JEPA pretraining (the foundation of every JEPA study below) |
| `python -m eb_jepa.training.clip_pretrain` | EEG↔V-JEPA-2 CLIP / scene_clip pretraining |
| `python -m eb_jepa.evaluation.probe_eval` | Adam-trained linear probe eval for JEPA checkpoints |
| `python -m eb_jepa.evaluation.bootstrap` | Recording-level bootstrap on probe_eval predictions |
| `python eb_jepa/evaluation/clip_probe/probe.py` | CV-by-recording linear probe on CLIP encoder embeddings |
| `python eb_jepa/evaluation/clip_probe/probe_traintest.py` | ImageNet-style fit-on-train / eval-on-{val,test} probe with bootstrap CI |

Canonical pretrain → probe → bootstrap sbatches:
[`eb_jepa/training/sbatch/canonical_*.sbatch`](../eb_jepa/training/sbatch/).

## Studies

| Study | What it tests |
|---|---|
| [`clip_pretraining/`](clip_pretraining/) | EEG↔V-JEPA-2 CLIP / scene_clip studies (from-scratch + REVE warm-start). |
| [`canonical_replication/`](canonical_replication/) | Spec-faithful 5-seed JEPA replication. The headline JEPA result. |
| [`temporal_sweep/`](temporal_sweep/) | n_windows × window_size grid (Phase 1). |
| [`regularizer_study/`](regularizer_study/) | VICReg vs SIGReg ablations, projector on/off, ± CorrCA. |
| [`retrain_best/`](retrain_best/) | Retrain documented best configs with global vs per-recording norm. |
| [`trivial_baselines/`](trivial_baselines/) | Sanity baselines on raw + CorrCA EEG (no learned encoder). |
| [`corrca_study/`](corrca_study/) | Probe-eval studies on CorrCA-preprocessed checkpoints. |
| [`trf_baseline/`](trf_baseline/) | Supervised TRF / Ridge baseline on movie-feature targets. |
| [`benchmark/`](benchmark/) | Supervised EEGNet / REVE / BIOT / classical ML baselines. |
| [`position_leakage/`](position_leakage/) | Diagnostic: does the encoder leak time-in-movie? |
| [`variance_analysis/`](variance_analysis/) | Per-checkpoint variance decomposition + predictability. |

## Conventions

Every new study folder has the same shape:

```
experiments/<study>/
  README.md                  -- description, sweep list, submit examples
  sweeps/                    -- neurolab launchers
  sbatch/                    -- (optional) study-specific raw SBATCH templates
  cfgs/                      -- (optional) yaml configs if the study overrides the library default
```

**Self-containment rule.** A study folder may invoke `python -m eb_jepa.*`
or `bash eb_jepa/training/sbatch/*.sbatch`. It must NOT import or
invoke code from another `experiments/<other_study>/` folder. If two
studies share code, promote that code to `eb_jepa/` first.

**Promotion policy.** Code authored inside a study folder is exploratory.
When a pattern is validated and worth re-using, promote it to `eb_jepa/`
(usually `eb_jepa/training/`, `eb_jepa/evaluation/`, or a new submodule)
and update study launchers to call the library entry.

## Post-training validation

`eb_jepa.training.jepa_pretrain` auto-invokes
`eb_jepa.evaluation.probe_eval` and `eb_jepa.evaluation.bootstrap` at
end of training when `cfg.eval.auto_run=true` (default). Disable for
fast smoke runs: `--eval.auto_run=false`. For spec-faithful evaluation,
disable auto-eval and run the canonical probe + bootstrap explicitly
(see the canonical_replication study).

## Cluster wrappers

Cluster + data utilities live in [`scripts/`](../scripts/):

- `preprocess_hbn.py` — raw HBN → .fif preprocessing (run once)
- `compute_corrca.py` — fit CorrCA filters on train-only recordings
- `aggregate_and_print.py` — L3 aggregation + t-test vs chance across seeds
- `submit_job_{delta,expanse,jamming}.py` — generic neurolab job submitters
- `pull_wandb.py`, `extract_*_results.py` — W&B aggregation helpers
