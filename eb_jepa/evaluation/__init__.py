"""Post-training evaluation pipeline for EEG JEPA checkpoints.

Two stages:
  1. probe_eval -- closed-form sklearn linear probes (Ridge for stim
     regression + age, LogisticRegression for binary cls + 20-way
     movie_id) on a frozen encoder. Per-clip flat predictions saved to
     a single NPZ for downstream bootstrap.
  2. bootstrap -- recording-level resampling on the NPZ produces L1
     point metrics and L2 bootstrap CIs in JSON form.

Library entry points:
  >>> from eb_jepa.evaluation import run_probe_eval, bootstrap_predictions

Both also have fire CLIs:
  uv run python -m eb_jepa.evaluation.probe_eval --checkpoint=...
  uv run python -m eb_jepa.evaluation.bootstrap   --predictions_npz=...
"""

from eb_jepa.evaluation.bootstrap import run as bootstrap_predictions
from eb_jepa.evaluation.probe_eval import run as run_probe_eval
from eb_jepa.evaluation.variance_decomposition import run as decompose_variance

__all__ = [
    "run_probe_eval",
    "bootstrap_predictions",
    "decompose_variance",
]
