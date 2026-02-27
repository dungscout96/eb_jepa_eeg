"""
Traditional ML benchmark: classify visual features from EEG band power.

Binary classification (median split) using LogisticRegression, SVM, and LDA
on band power features (5 bands x 128 channels = 640 features).

Usage:
    python experiments/run_traditional_ml_benchmark.py
    python experiments/run_traditional_ml_benchmark.py data.visual_processing_delay_s=0.6
"""

import sys
import json
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import signal as sp_signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from eb_jepa.datasets.hbn import HBNMovieProbeDataset


# ===================== Constants =====================

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# TARGET_FEATURES = ["luminance_mean", "contrast_rms", "sham_random"]
TARGET_FEATURES = ["entropy", "n_faces", "face_area_frac", "scene_natural_score", "scene_open_score", "sham_random"]
# BEST PERFORMING FEATURES IN PRELIMINARY ANALYSIS with SVM (2026-02-23): "contrast_rms", "luminance_mean", "entropy", "scene_natural_score""

CLASSIFIERS = {
    "LogisticRegression": lambda: LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    "SVM": lambda: SVC(kernel="rbf", C=1.0, probability=True, max_iter=5000),
    "LDA": lambda: LinearDiscriminantAnalysis(),
}


# ===================== Band Power =====================

def compute_band_power(X, sfreq):
    """Compute band power features from raw EEG windows via Welch's method.

    Args:
        X: (N, n_chans, n_times) array
        sfreq: sampling frequency in Hz

    Returns:
        X_bp: (N, n_bands * n_chans) array — 5 bands x 128 channels = 640 features
    """
    n_times = X.shape[-1]
    freqs, psd = sp_signal.welch(
        X, fs=sfreq, nperseg=min(n_times, int(sfreq * 2)), axis=-1
    )
    band_powers = []
    for fmin, fmax in BANDS.values():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers.append(psd[:, :, mask].mean(axis=-1))
    return np.concatenate(band_powers, axis=1)


# ===================== Data Loading =====================

def load_split_data(split, cfg):
    """Load one data split, compute band power features and extract targets.

    Returns:
        X_bp: (N, 640) band power feature matrix
        targets: dict of feature_name -> (N,) continuous values
        sfreq: sampling frequency
    """
    print(f"Loading {split} split...")
    dataset = HBNMovieProbeDataset(split=split, cfg=cfg.data)
    sfreq = dataset.sfreq

    X_list = []
    real_features = [f for f in TARGET_FEATURES if f != "sham_random"]
    targets = {f: [] for f in real_features}

    for i in tqdm(range(len(dataset)), desc=f"  {split}", leave=False):
        X_tensor, features = dataset[i]
        X_np = X_tensor.numpy()
        X_np = X_np[:-1, :]  # Remove empty reference channel: 129 -> 128
        X_list.append(X_np)
        for f in real_features:
            targets[f].append(float(features[f]))

    X_all = np.stack(X_list)
    targets = {f: np.array(v) for f, v in targets.items()}

    print(f"  {len(X_all)} windows, shape {X_all.shape}")
    print(f"  Computing band power features...")
    X_bp = compute_band_power(X_all, sfreq)
    print(f"  Band power shape: {X_bp.shape}")

    return X_bp, targets, sfreq


# ===================== Evaluation Metrics =====================

def compute_metrics(pipe, X, y_true):
    """Compute accuracy, balanced accuracy, and AUC for a fitted pipeline."""
    y_pred = pipe.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X)[:, 1]
        else:
            y_prob = pipe.decision_function(X)
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except (ValueError, AttributeError):
        metrics["auc"] = float("nan")
    return metrics


# ===================== Report =====================

def generate_report(results_df, results_dir):
    """Generate summary tables and figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ---- 1. Primary metric table ----
    pivot = results_df.pivot(
        index="feature", columns="classifier", values="test_balanced_accuracy"
    )
    print(f"\n{'='*60}")
    print("  TEST BALANCED ACCURACY")
    print(f"{'='*60}")
    print(pivot.round(4).to_string())
    pivot.to_csv(results_dir / "summary_balanced_accuracy.csv")

    # ---- 2. Full metrics table ----
    metric_cols = [c for c in results_df.columns
                   if c.startswith(("train_", "val_", "test_"))]
    summary = results_df[["feature", "classifier"] + metric_cols + ["elapsed_s"]]
    summary.to_csv(results_dir / "full_metrics.csv", index=False)

    print(f"\n{'='*60}")
    print("  FULL METRICS")
    print(f"{'='*60}")
    print(summary.round(4).to_string(index=False))

    # ---- 3. Heatmap ----
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGn",
        center=0.5, ax=ax, linewidths=0.5, vmin=0.45, vmax=0.65,
    )
    ax.set_title("Test Balanced Accuracy: EEG Band Power -> Binary Visual Feature")
    plt.tight_layout()
    plt.savefig(figures_dir / "balanced_accuracy_heatmap.png", dpi=150)
    plt.close()
    print(f"\n  Saved: {figures_dir / 'balanced_accuracy_heatmap.png'}")

    # ---- 4. Bar chart with chance line ----
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, alpha=0.8)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=2, label="Chance (0.5)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Binary Classification: Band Power Features (test set)")
    ax.legend(title="Classifier")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "classifier_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {figures_dir / 'classifier_comparison.png'}")

    # ---- 5. Val vs test scatter ----
    fig, ax = plt.subplots(figsize=(7, 6))
    for clf_name in results_df["classifier"].unique():
        subset = results_df[results_df["classifier"] == clf_name]
        ax.scatter(
            subset["val_balanced_accuracy"],
            subset["test_balanced_accuracy"],
            label=clf_name, s=100, alpha=0.8,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["feature"],
                (row["val_balanced_accuracy"], row["test_balanced_accuracy"]),
                fontsize=8, textcoords="offset points", xytext=(5, 5),
            )
    lims = [0.44, 0.66]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Val Balanced Accuracy")
    ax.set_ylabel("Test Balanced Accuracy")
    ax.set_title("Validation vs Test Performance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "val_vs_test.png", dpi=150)
    plt.close()
    print(f"  Saved: {figures_dir / 'val_vs_test.png'}")


# ===================== Main =====================

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    results_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")
    print(f"Visual processing delay: {cfg.data.visual_processing_delay_s}s")
    print(f"Config:\n{OmegaConf.to_yaml(cfg.data)}")

    # --- Load all splits ---
    X_train, targets_train, sfreq = load_split_data("train", cfg)
    X_val, targets_val, _ = load_split_data("val", cfg)
    X_test, targets_test, _ = load_split_data("test", cfg)

    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Feature dim: {X_train.shape[1]} (5 bands x {X_train.shape[1] // 5} channels)")

    # --- Add sham random targets (seeded for reproducibility) ---
    rng = np.random.RandomState(42)
    targets_train["sham_random"] = rng.uniform(-1, 1, size=len(X_train))
    targets_val["sham_random"] = rng.uniform(-1, 1, size=len(X_val))
    targets_test["sham_random"] = rng.uniform(-1, 1, size=len(X_test))

    # --- Compute median thresholds from TRAINING set only ---
    thresholds = {}
    print("\nBinary thresholds (training set medians):")
    for feat_name in TARGET_FEATURES:
        vals = targets_train[feat_name]
        thresholds[feat_name] = float(np.median(vals[~np.isnan(vals)]))
        print(f"  {feat_name}: {thresholds[feat_name]:.6f}")

    # --- Binarize targets ---
    def binarize(targets_dict):
        return {f: (targets_dict[f] > thresholds[f]).astype(int) for f in TARGET_FEATURES}

    y_train = binarize(targets_train)
    y_val = binarize(targets_val)
    y_test = binarize(targets_test)

    print("\nClass balance (fraction class=1):")
    for split_name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
        parts = [f"{f}={labels[f].mean():.3f}" for f in TARGET_FEATURES]
        print(f"  {split_name}: {', '.join(parts)}")

    # --- Classification loop ---
    all_results = []

    for feat_name in TARGET_FEATURES:
        print(f"\n{'='*60}")
        print(f"  Feature: {feat_name}")
        print(f"{'='*60}")

        for clf_name, clf_factory in CLASSIFIERS.items():
            print(f"\n  {clf_name}...")
            t0 = time.time()

            pipe = make_pipeline(StandardScaler(), clf_factory())
            pipe.fit(X_train, y_train[feat_name])

            result = {"feature": feat_name, "classifier": clf_name}

            for split_name, X_split, y_split in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                metrics = compute_metrics(pipe, X_split, y_split[feat_name])
                for k, v in metrics.items():
                    result[f"{split_name}_{k}"] = v

            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)

            print(
                f"    train: bal_acc={result['train_balanced_accuracy']:.4f}, "
                f"auc={result['train_auc']:.4f}"
            )
            print(
                f"    val:   bal_acc={result['val_balanced_accuracy']:.4f}, "
                f"auc={result['val_auc']:.4f}"
            )
            print(
                f"    test:  bal_acc={result['test_balanced_accuracy']:.4f}, "
                f"auc={result['test_auc']:.4f}"
            )
            print(f"    time:  {elapsed:.1f}s")

            all_results.append(result)

    # --- Save results ---
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / "results.csv", index=False)

    # --- Generate report ---
    generate_report(results_df, results_dir)

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
