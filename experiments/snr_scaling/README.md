# snr_scaling — how much signal is there, and what buys more of it?

Measures the **noise ceiling** for stimulus-locked EEG readout on HBN movie-watching, then
asks which data axis moves a model toward it: more **subjects**, or more **movie anchors**.

## Read these

| file | what it holds |
|---|---|
| [`RESULTS.md`](RESULTS.md) | the within-ThePresent record — ceiling, K-averaging, the (anchors × subjects) surface. Start at its TL;DR. |
| [`RESULTS_cross_task.md`](RESULTS_cross_task.md) | the same checkpoints evaluated on DespicableMe: features transfer, alignment does not. |
| [`RESULTS_model_scaling.md`](RESULTS_model_scaling.md) | the depth-22 (`e04_reve_scaling`) subject-scaling sweep — probe + retrieval, within- and cross-task. |
| [`RESULTS_add_val_set.md`](RESULTS_add_val_set.md) | R5 folded into the pretraining pool (1863 → 2156) so S=1863 becomes drawable in triplicate. **Verdict: depth-22's saturation-then-drop was a single-draw artifact** — three draws put S=1863 *above* the S=1400 peak, and `RESULTS_model_scaling.md`'s headline is retracted. Test split only. |
| [`RELATED_WORK.md`](RELATED_WORK.md) | why this sweep's subject axis scales when Banville et al. 2025 report it does not — per-doubling slopes, log-linear fits, and the methodological differences that explain the contrast. |
| [`PLAN.md`](PLAN.md) | forward-looking design. Experiments not yet run. |

## Layout

```
config/        frozen configs. clip_pretrain.yaml is a snapshot the submit
               scripts cp on the cluster -- do NOT re-sync it to the shared
               template. config_probe_DM.yaml is the shared DespicableMe
               eval config.
src/           measurement, analysis and plotting. Imported by tests via
               filesystem path, and invoked by the submit scripts inside
               cluster job commands.
submit/        neurolab job submitters. Each has a dry-run mode; run from the
               repo root.
figures/       rendered .png/.pdf. Written by src/plot_*.py.
raw_results/   every measured JSON. Read by src/analyse_e03.py and the plots.
```

## Two conventions worth knowing before you touch anything

**Paths are repo-relative and resolved on the cluster.** The submit scripts embed strings like
`experiments/snr_scaling/src/measure_isc.py` into job commands that run from the repo root on
Delta. The cluster checkout is *not* a git checkout of this branch — it is an older commit
with files rsynced in — so **anything newly referenced must be synced before submitting**, or
the `&&` chain short-circuits and the job "completes" in seconds with exit 0. This has
happened; see the neurolab skill's pitfall 8.

**`sacct COMPLETED 0:0` proves nothing.** The generated batch script has no `set -e` and ends
with an `echo`, so its exit status is always that echo's. Verify the artifact — a checkpoint
count, an output JSON — never the status.

## Typical flow

```bash
# 1. submit (dry run first; every submitter supports it)
uv run --group eeg python experiments/snr_scaling/submit/_submit_e03.py --nested-draws
uv run --group eeg python experiments/snr_scaling/submit/_submit_e03.py --nested-draws submit

# 2. pick each cell's checkpoint by SMOOTHED selection, then probe it
uv run --group eeg python experiments/snr_scaling/src/select_and_probe_e03.py \
    --cells-glob='e03_s*_a101_nd*' --skip-existing --probe

# 3. retrieval (presets reproduce every published retrieval number)
uv run --group eeg python experiments/snr_scaling/submit/_submit_retrieval.py tp submit

# 4. analyse + plot
uv run --group eeg python experiments/snr_scaling/src/analyse_e03.py --suffix=_es_best
uv run --group eeg python experiments/snr_scaling/src/plot_e03.py
```

Tests: `tests/test_isc_estimator.py`, `tests/test_noise_ceiling.py`,
`tests/test_k_averaging.py`, `tests/unit/test_e03_selection.py`,
`tests/unit/test_scaling_subsample.py`. They load `src/` modules **by filesystem path**, so
moving a module means updating them.
