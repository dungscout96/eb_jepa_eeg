"""Thin CLI wrapper around eb_jepa.preprocessing.compute_corrca.

CorrCA filter computation is a preprocessing step (run once before
training, not per-experiment), so the implementation lives in the
library. This script just dispatches arguments.

Usage (Delta example):
    PYTHONPATH=. uv run --group eeg python scripts/compute_corrca.py \\
        --output_path corrca_filters.npz \\
        --n_components 5 \\
        --task ThePresent
"""

import argparse

from eb_jepa.preprocessing import compute_corrca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_path", default="corrca_filters.npz")
    parser.add_argument("--n_components", type=int, default=5)
    parser.add_argument("--task", default="ThePresent")
    parser.add_argument("--n_time_bins", type=int, default=100)
    parser.add_argument("--preprocessed_dir", default=None)
    args = parser.parse_args()
    compute_corrca(**vars(args))


if __name__ == "__main__":
    main()
