"""Lane-1 lum-failure diagnostic — does the doc's "5 probe heads disagree
across seeds" mechanism actually hold?

Reads issue8E test predictions for both configs, computes:
  1. Pairwise Pearson corr between the 5 probe-seed prediction vectors,
     per (config, encoder seed). Mean → per-encoder agreement.
  2. Seed-mean prediction → correlation with true labels (= View 3 metric).
  3. Per-seed metric → mean (= View 2 metric).

If View 2 is high but View 3 is low AND pairwise agreement is low,
the doc's mechanism is right. If pairwise agreement is high yet seed
averaging still cancels signal, something else is going on.

Run on Delta:
  /projects/bbnv/kkokate/eb_jepa_eeg/.venv/bin/python \\
      scripts/diagnose_lum_seed_agreement.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np

PRED_BASE = Path("/projects/bbnv/kkokate/eb_jepa_eeg/predictions/issue8E")
ENC_SEEDS = [42, 123, 456, 789, 2025]
PROBE_SEEDS = [7, 13, 42, 1234, 2025]
CONFIGS = ["nw4ws2_baseline", "nw2ws4"]
FEATURE_INDEX = {"contrast": 0, "luminance": 1, "position": 2, "narrative": 3}


def load_seed_preds(config: str, enc: int, feature: str):
    """Load 5 (probe-seed) prediction and target arrays for a given encoder.

    Returns
    -------
    preds : np.ndarray [5, n_clips]   — unnormalized regression predictions
    target : np.ndarray [n_clips]     — ground-truth label (averaged over seeds)
    """
    fi = FEATURE_INDEX[feature]
    preds = []
    targets = None
    for p in PROBE_SEEDS:
        tag = f"{config}_enc{enc}_p{p}"
        npz_path = PRED_BASE / tag / f"test_seed{p}.npz"
        if not npz_path.exists():
            return None, None
        d = np.load(npz_path)
        # preds: [N_rec, T, F] standardized; unnormalize
        reg = d["movie_reg_preds"][:, :, fi]
        f_mean = float(d["feature_mean"][fi])
        f_std = float(d["feature_std"][fi])
        reg_un = reg * (f_std + 1e-8) + f_mean
        preds.append(reg_un.ravel())  # [N_rec * T]
        if targets is None:
            targets = d["movie_targets"][:, :, fi].ravel()
    return np.stack(preds, axis=0), targets


def pairwise_corr(preds: np.ndarray) -> float:
    """Mean Pearson r across all C(5,2)=10 pairs."""
    n_seeds = preds.shape[0]
    rs = []
    for i, j in combinations(range(n_seeds), 2):
        a, b = preds[i], preds[j]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def safe_corr(y_pred, y_true):
    if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
        return float("nan")
    return float(np.corrcoef(y_pred, y_true)[0, 1])


def diagnose(feature: str):
    """Build per-encoder agreement table for one feature."""
    print(f"\n=== feature: {feature} ===")
    print(
        f"{'config':<18}{'enc':>5}  {'pair_r':>9}  {'mean(per_seed_r)':>18}  "
        f"{'r(seed_mean,truth)':>20}"
    )
    print("-" * 75)
    for config in CONFIGS:
        all_pair = []
        all_per = []
        all_view3 = []
        for enc in ENC_SEEDS:
            preds, targets = load_seed_preds(config, enc, feature)
            if preds is None:
                continue
            pr = pairwise_corr(preds)
            per_seed = [safe_corr(p, targets) for p in preds]
            seed_mean = preds.mean(axis=0)
            view3 = safe_corr(seed_mean, targets)
            print(
                f"{config:<18}{enc:>5}  {pr:>9.3f}  "
                f"{np.nanmean(per_seed):>18.3f}  {view3:>20.3f}"
            )
            all_pair.append(pr)
            all_per.append(np.nanmean(per_seed))
            all_view3.append(view3)
        if all_pair:
            print(
                f"{config + ' MEAN':<23}  {np.nanmean(all_pair):>9.3f}  "
                f"{np.nanmean(all_per):>18.3f}  {np.nanmean(all_view3):>20.3f}"
            )


def main():
    print("Probe-seed agreement diagnostic")
    print(
        "pair_r:    mean Pearson r across the C(5,2)=10 pairs of probe-seed "
        "prediction vectors per encoder"
    )
    print(
        "per_seed:  mean across the 5 seed metrics (= View 2)"
    )
    print(
        "seed_mean: r(mean prediction across seeds, truth) (= View 3 logit-avg)"
    )
    for feat in ["luminance", "narrative", "contrast", "position"]:
        diagnose(feat)


if __name__ == "__main__":
    main()
