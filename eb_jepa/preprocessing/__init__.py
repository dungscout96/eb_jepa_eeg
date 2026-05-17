"""Preprocessing utilities for EEG JEPA.

These run once before training (not part of every epoch). Currently:
  - corrca: spatial filters that maximize inter-subject correlation,
    used to project [n_chans] EEG into [n_components] stimulus-driven
    components.
"""

from eb_jepa.preprocessing.corrca import compute_corrca, solve_corrca_eigenproblem

__all__ = ["compute_corrca", "solve_corrca_eigenproblem"]
