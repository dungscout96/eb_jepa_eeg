"""eb_jepa.training -- promoted library training entry points.

This package contains the canonical, reproducible JEPA pretraining pipeline.
Experiments under `experiments/<study>/` should invoke these rather than
depending on each other's local code.

Components:
  - `jepa_pretrain` -- single-file pretraining CLI (Fire entry).
                       Invoke as: `python -m eb_jepa.training.jepa_pretrain ...`
  - `builder` -- shared MaskedJEPA builder used by training and probe-eval.
  - Default OmegaConf config lives at repo-root `config/jepa_pretrain.yaml`.
  - `sbatch/canonical_*.sbatch` -- Delta pipeline (pretrain ->
                                   probe_eval -> bootstrap).

Legacy: `experiments/eeg_jepa/train.py` is preserved untouched as a frozen
reproducibility snapshot of the pre-refactor entry point. Old sweeps under
`experiments/eeg_jepa/sweeps/` continue to invoke that legacy entry. New
sweeps under `experiments/<study>/sweeps/` invoke the library entry here.
"""
