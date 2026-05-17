"""Cluster-aware paths for the EEG JEPA project.

Centralizes the locations where preprocessed HBN data is expected to live on
each cluster (jamming workstation, Expanse, Delta) so callers don't have to
repeat the lookup. Used by the training entry point and by the library-level
evaluation pipeline (probe_eval, variance_decomposition, etc.).
"""

from pathlib import Path

from eb_jepa.logging import get_logger

logger = get_logger(__name__)


# Known preprocessed data locations, checked in order for auto-detection.
PREPROCESSED_DIRS: list[Path] = [
    Path("/mnt/v1/dtyoung/data/eb_jepa_eeg/hbn_preprocessed"),                  # jamming
    Path("/expanse/projects/nemar/dtyoung/.cache/eb_jepa_eeg/hbn_preprocessed"),  # Expanse
    Path("/projects/bbnv/kkokate/hbn_preprocessed"),                              # Delta
]


def resolve_preprocessed_dir(configured: str | None) -> Path | None:
    """Return preprocessed_dir: use explicit config if set, else auto-detect."""
    if configured:
        return Path(configured)
    for p in PREPROCESSED_DIRS:
        if p.exists():
            logger.info("Auto-detected preprocessed_dir: %s", p)
            return p
    return None
