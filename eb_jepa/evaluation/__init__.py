"""Post-training evaluation pipeline for EEG JEPA checkpoints.

Two stages:
  1. probe_eval -- train fresh linear probes on a frozen encoder for movie
     features (regression + classification) and subject traits (per-recording
     pooled). Optionally dumps per-clip predictions to .npz.
  2. bootstrap -- recording-level resampling on the .npz dumps to get
     population CIs that decouple probe-init noise from sampling noise.

Library entry points (importable from anywhere):
  >>> from eb_jepa.evaluation import (
  ...     run_probe_eval, bootstrap_predictions, validation_loop,
  ... )

Both ``run_probe_eval`` and ``bootstrap_predictions`` also have fire CLIs
exposed by their submodules:
  uv run python -m eb_jepa.evaluation.probe_eval --checkpoint=...
  uv run python -m eb_jepa.evaluation.bootstrap  --predictions_dir=...
"""

from eb_jepa.evaluation.bootstrap import run as bootstrap_predictions
from eb_jepa.evaluation.probe_eval import run as run_probe_eval
from eb_jepa.evaluation.validation_loop import validation_loop
from eb_jepa.evaluation.variance_decomposition import run as decompose_variance

__all__ = [
    "run_probe_eval",
    "bootstrap_predictions",
    "validation_loop",
    "decompose_variance",
]
