"""Cluster-aware paths for the EEG JEPA project.

Centralizes the locations where preprocessed HBN data is expected to live on
each cluster (jamming workstation, Expanse, Delta) so callers don't have to
repeat the lookup. Used by the training entry point and by the library-level
evaluation pipeline (probe_eval, variance_decomposition, etc.).
"""

import os
from pathlib import Path

from eb_jepa.logging import get_logger

logger = get_logger(__name__)


# Known preprocessed data locations, checked in order for auto-detection.
PREPROCESSED_DIRS: list[Path] = [
    Path("/mnt/v1/dtyoung/data/eb_jepa_eeg/hbn_preprocessed"),                  # jamming
    Path("/expanse/projects/nemar/dtyoung/.cache/eb_jepa_eeg/hbn_preprocessed"),  # Expanse
    Path("/work/hdd/bbnv/kkokate/hbn_preprocessed"),                              # Delta
]


def resolve_preprocessed_dir(configured: str | None) -> Path | None:
    """Return preprocessed_dir: explicit config > ``HBN_PREPROCESS_DIR`` > auto-detect.

    The env var was previously ignored here, which was a trap: every submit
    script in this repo sets ``HBN_PREPROCESS_DIR``, and it only ever affected
    ``hbn.PREPROCESSED_DIR`` (the default used when ``preprocessed_dir=None``
    reaches ``load_preprocessed``). The training entry point passes this
    function's result explicitly, so the env var was silently inert there and
    auto-detection won.

    That cost the first extended-cohort run: the jobs exported
    ``HBN_PREPROCESS_DIR=/work/hdd/.../hbn_preprocessed`` (which has R7-R10),
    auto-detection returned the old Delta root (which does not), and all five
    cells died on a missing R7 -- while still reporting ``COMPLETED 0:0``.

    On Delta the env var and the auto-detected path were previously identical,
    so honouring it changes nothing for existing scripts.
    """
    if configured:
        return Path(configured)
    env = os.environ.get("HBN_PREPROCESS_DIR")
    if env:
        logger.info("preprocessed_dir from HBN_PREPROCESS_DIR: %s", env)
        return Path(env)
    for p in PREPROCESSED_DIRS:
        if p.exists():
            logger.info("Auto-detected preprocessed_dir: %s", p)
            return p
    return None
