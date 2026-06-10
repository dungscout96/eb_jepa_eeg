"""Extended Tier-1 analyses from saved per-clip embeddings.

Loads .npz files written by ``tier1_baselines.py --save_embeddings`` and
computes:

  (1) Bootstrap 95% CIs on test regression corrs (per probe, per baseline).
  (2) Permutation null on test regression corrs (per probe, per baseline).
  (3) [Movie-ID for Exp 6 — already in the per-cell metric JSONs.]
  (4) CKA between exp6 and random_init per-clip embeddings (val set, per seed).
  (5) Stacked probe: concat(exp6, raw_corrca) features, retrain probe,
      compare to exp6 alone on test. Tells us whether Exp 6 captured all
      linear stimulus information that raw CorrCA carried.
  (6) Data-efficiency curve: train probe with {1%, 5%, 25%, 100%} of train
      clips, eval on test. Per (baseline, seed).

All operations are CPU-only and trivially fast on the saved embeddings.

Usage
-----
    PYTHONPATH=. uv run --group eeg python \
        experiments/eeg_jepa/tier1_extended_analysis.py \
        --emb_dir=/projects/bbnv/kkokate/eb_jepa_eeg/tier1/embeddings \
        --out_json=/projects/bbnv/kkokate/eb_jepa_eeg/tier1/extended_analysis.json
"""

import json
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.optim import Adam

from eb_jepa.architectures import MovieFeatureHead

FEATURE_NAMES = [
    "contrast_rms",
    "luminance_mean",
    "position_in_movie",
    "narrative_event_score",
]
SEEDS_TIER1 = [42, 123, 456]
SEEDS_EXP6 = [42, 123, 2025]
TIER1_BASELINES = ["raw_corrca", "psd_band", "random_init"]
ALL_BASELINES = TIER1_BASELINES + ["exp6"]


def _load(emb_dir: Path, baseline: str, seed: int, split: str):
    path = emb_dir / f"{baseline}_seed{seed}_{split}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return {"embs": z["embs"], "targets": z["targets"], "positions": z["positions"]}


# ---------------------------------------------------------------------------
# Probe training (linear MLP head, identical to tier1_baselines.py)
# ---------------------------------------------------------------------------


def _train_probe_and_predict(
    train_embs: np.ndarray,
    train_targets: np.ndarray,
    test_embs: np.ndarray,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    *,
    hdec: int = 64,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
):
    """Train a small MLP regression head (mean over windows) and return
    test predictions in original feature scale.

    Inputs are per-clip mean-pooled embeddings [N, D] and per-clip-mean
    targets [N, n_features] (averaged over the n_windows axis).
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(train_embs).float()
    y = torch.from_numpy(train_targets).float()
    Xt = torch.from_numpy(test_embs).float().to(device)

    fmean = torch.from_numpy(feat_mean).float().to(device)
    fstd = torch.from_numpy(feat_std).float().to(device)

    D = X.shape[1]
    n_features = y.shape[1]

    # Single-window equivalent: use a 2-layer MLP D -> hdec -> n_features.
    # MovieFeatureHead expects [B, D, T, 1, 1]; we wrap with T=1.
    head = MovieFeatureHead(D, hdec, n_features).to(device)
    opt = Adam(head.parameters(), lr=lr)

    n = X.shape[0]
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            x = X[idx].to(device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            target = y[idx].to(device)
            target_norm = (target - fmean) / (fstd + 1e-8)
            pred = head(x).squeeze(1)  # [B, n_features]
            loss = nn.functional.mse_loss(pred, target_norm)
            opt.zero_grad()
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        pred_norm = head(
            Xt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ).squeeze(1).cpu().numpy()
    pred = pred_norm * (feat_std + 1e-8) + feat_mean
    return pred  # [N_test, n_features]


# ---------------------------------------------------------------------------
# Analysis 1: Bootstrap CIs on per-feature test corrs
# ---------------------------------------------------------------------------


def bootstrap_ci(pred: np.ndarray, target: np.ndarray, n_boot: int = 1000,
                 seed: int = 0):
    """Per-feature bootstrap 95% CIs on Pearson r."""
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    n_feat = pred.shape[1]
    boot = np.empty((n_boot, n_feat))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for j in range(n_feat):
            p = pred[idx, j]
            t = target[idx, j]
            if np.std(p) > 1e-10 and np.std(t) > 1e-10:
                boot[b, j] = float(pearsonr(p, t).statistic)
            else:
                boot[b, j] = 0.0
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        col = boot[:, j]
        out[name] = {
            "mean": float(col.mean()),
            "ci_low": float(np.percentile(col, 2.5)),
            "ci_high": float(np.percentile(col, 97.5)),
        }
    return out


# ---------------------------------------------------------------------------
# Analysis 2: Permutation null on per-feature test corrs
# ---------------------------------------------------------------------------


def perm_null(pred: np.ndarray, target: np.ndarray, n_perm: int = 1000,
              seed: int = 0):
    """Per-feature permutation p-value: P(|r_shuffled| >= |r_observed|)."""
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        p = pred[:, j]
        t = target[:, j]
        if np.std(p) <= 1e-10 or np.std(t) <= 1e-10:
            out[name] = {"r_obs": 0.0, "p_value": 1.0}
            continue
        r_obs = float(pearsonr(p, t).statistic)
        null = np.empty(n_perm)
        for b in range(n_perm):
            t_shuf = rng.permutation(t)
            null[b] = float(pearsonr(p, t_shuf).statistic)
        # Two-sided
        p_val = float((np.abs(null) >= np.abs(r_obs)).mean())
        out[name] = {"r_obs": r_obs, "p_value": p_val}
    return out


# ---------------------------------------------------------------------------
# Analysis 4: Linear CKA between two embedding matrices
# ---------------------------------------------------------------------------


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two [N, D] embedding matrices (Kornblith et al. 2019)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Faster form: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    denom = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / max(denom, 1e-12))


# ---------------------------------------------------------------------------
# Helpers: clip-mean targets (average over windows)
# ---------------------------------------------------------------------------


def _clip_mean_targets(z: dict) -> np.ndarray:
    # z['targets']: [N, T_win, n_features] → [N, n_features]
    return z["targets"].mean(axis=1)


def _train_feature_stats(train_targets: np.ndarray):
    return train_targets.mean(0), train_targets.std(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    emb_dir: str,
    out_json: str,
    n_boot: int = 1000,
    n_perm: int = 1000,
    data_eff_fracs: tuple = (0.01, 0.05, 0.25, 1.0),
):
    emb_dir = Path(emb_dir)
    results = {
        "bootstrap_ci": defaultdict(dict),
        "permutation": defaultdict(dict),
        "cka_random_init_vs_exp6": {},
        "stacked_probe": {},
        "data_efficiency": defaultdict(dict),
    }

    # Cached predictions, so analyses 1 + 2 can share a probe per (baseline, seed).
    # Key: (baseline, seed) → (pred, target) on test set.
    test_preds_cache = {}

    for baseline in ALL_BASELINES:
        seeds = SEEDS_EXP6 if baseline == "exp6" else SEEDS_TIER1
        for seed in seeds:
            train = _load(emb_dir, baseline, seed, "train")
            test = _load(emb_dir, baseline, seed, "test")
            if train is None or test is None:
                print(f"SKIP {baseline} seed={seed}: missing .npz")
                continue
            tr_t = _clip_mean_targets(train)
            te_t = _clip_mean_targets(test)
            fmean, fstd = _train_feature_stats(tr_t)

            pred = _train_probe_and_predict(
                train["embs"], tr_t, test["embs"], fmean, fstd, seed=seed,
            )
            test_preds_cache[(baseline, seed)] = (pred, te_t)

            # --- Analysis 1
            results["bootstrap_ci"][baseline][seed] = bootstrap_ci(
                pred, te_t, n_boot=n_boot, seed=seed,
            )
            # --- Analysis 2
            results["permutation"][baseline][seed] = perm_null(
                pred, te_t, n_perm=n_perm, seed=seed,
            )
            print(f"  done {baseline} seed={seed}")

    # --- Analysis 4: CKA between random_init and exp6 on val set
    # Pair seeds by ordinal (random_init has 42/123/456; exp6 has 42/123/2025).
    for ri_seed, e6_seed in zip(SEEDS_TIER1, SEEDS_EXP6):
        ri = _load(emb_dir, "random_init", ri_seed, "val")
        e6 = _load(emb_dir, "exp6", e6_seed, "val")
        if ri is None or e6 is None:
            continue
        # Align row count if necessary (should be identical; same dataset)
        n = min(ri["embs"].shape[0], e6["embs"].shape[0])
        cka = linear_cka(ri["embs"][:n], e6["embs"][:n])
        results["cka_random_init_vs_exp6"][f"ri_seed{ri_seed}_vs_e6_seed{e6_seed}"] = cka
        print(f"  CKA(random_init seed={ri_seed}, exp6 seed={e6_seed}) = {cka:.4f}")

    # --- Analysis 5: Stacked probe (concat exp6 + raw_corrca on test).
    # Train on concat'd train embeddings; predict on concat test; report
    # per-feature corrs and Δ vs exp6-alone for the same seed pair.
    for ri_seed, e6_seed in zip(SEEDS_TIER1, SEEDS_EXP6):
        rc_train = _load(emb_dir, "raw_corrca", ri_seed, "train")
        rc_test = _load(emb_dir, "raw_corrca", ri_seed, "test")
        e6_train = _load(emb_dir, "exp6", e6_seed, "train")
        e6_test = _load(emb_dir, "exp6", e6_seed, "test")
        if any(z is None for z in [rc_train, rc_test, e6_train, e6_test]):
            continue
        # Use exp6 targets (same dataset, identical row count expected).
        n_tr = min(rc_train["embs"].shape[0], e6_train["embs"].shape[0])
        n_te = min(rc_test["embs"].shape[0], e6_test["embs"].shape[0])
        Xtr = np.concatenate(
            [e6_train["embs"][:n_tr], rc_train["embs"][:n_tr]], axis=1,
        )
        Xte = np.concatenate(
            [e6_test["embs"][:n_te], rc_test["embs"][:n_te]], axis=1,
        )
        ytr = _clip_mean_targets(e6_train)[:n_tr]
        yte = _clip_mean_targets(e6_test)[:n_te]
        fmean, fstd = _train_feature_stats(ytr)
        pred_stack = _train_probe_and_predict(
            Xtr, ytr, Xte, fmean, fstd, seed=e6_seed,
        )
        # Compare to exp6-alone test predictions cached earlier
        e6_pred, _ = test_preds_cache[("exp6", e6_seed)]
        per_feat = {}
        for j, name in enumerate(FEATURE_NAMES):
            r_stack = float(pearsonr(pred_stack[:, j], yte[:, j]).statistic) \
                if np.std(yte[:, j]) > 1e-10 else 0.0
            r_e6 = float(pearsonr(e6_pred[:n_te, j], yte[:n_te, j]).statistic) \
                if np.std(yte[:n_te, j]) > 1e-10 else 0.0
            per_feat[name] = {
                "r_exp6_alone": r_e6,
                "r_stacked": r_stack,
                "delta": r_stack - r_e6,
            }
        key = f"ri_seed{ri_seed}_e6_seed{e6_seed}"
        results["stacked_probe"][key] = per_feat
        print(f"  stacked probe done for {key}")

    # --- Analysis 6: Data efficiency curve, per (baseline, seed)
    rng = np.random.default_rng(0)
    for baseline in ALL_BASELINES:
        seeds = SEEDS_EXP6 if baseline == "exp6" else SEEDS_TIER1
        for seed in seeds:
            train = _load(emb_dir, baseline, seed, "train")
            test = _load(emb_dir, baseline, seed, "test")
            if train is None or test is None:
                continue
            tr_t_full = _clip_mean_targets(train)
            te_t = _clip_mean_targets(test)
            n_train = train["embs"].shape[0]
            curve = {}
            for frac in data_eff_fracs:
                k = max(int(n_train * frac), 64)
                idx = rng.choice(n_train, size=k, replace=False)
                Xtr = train["embs"][idx]
                ytr = tr_t_full[idx]
                fmean, fstd = _train_feature_stats(ytr)
                pred = _train_probe_and_predict(
                    Xtr, ytr, test["embs"], fmean, fstd, seed=seed,
                )
                per_feat = {}
                for j, name in enumerate(FEATURE_NAMES):
                    if np.std(te_t[:, j]) > 1e-10 and np.std(pred[:, j]) > 1e-10:
                        per_feat[name] = float(pearsonr(pred[:, j], te_t[:, j]).statistic)
                    else:
                        per_feat[name] = 0.0
                curve[f"frac{frac:.2f}"] = {
                    "n_train": k, "per_feature_corr": per_feat,
                }
            results["data_efficiency"][baseline][seed] = curve
            print(f"  data-eff done {baseline} seed={seed}")

    # Make defaultdicts JSON-serializable
    def _to_dict(o):
        if isinstance(o, defaultdict):
            return {k: _to_dict(v) for k, v in o.items()}
        if isinstance(o, dict):
            return {k: _to_dict(v) for k, v in o.items()}
        return o

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_dict(results), indent=2))
    print(f"Wrote analysis to {out_path}")


if __name__ == "__main__":
    fire.Fire(run)
