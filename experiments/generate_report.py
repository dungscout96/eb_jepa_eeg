"""
Regenerate reports from saved benchmark results.

Usage:
    python experiments/generate_report.py [--results-dir experiments/results]
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir):
    results_dir = Path(results_dir)

    all_results = []
    all_path = results_dir / "all_results.json"
    if all_path.exists():
        with open(all_path) as f:
            all_results = json.load(f)

    test_results = {}
    test_path = results_dir / "test_results.json"
    if test_path.exists():
        with open(test_path) as f:
            test_results = json.load(f)

    return all_results, test_results


def generate_summary_table(test_results, results_dir):
    """Generate summary CSV and print table."""
    rows = []
    for model_name, tres in test_results.items():
        if "error" in tres:
            continue
        rows.append({
            "Model": model_name,
            "Best Config": tres.get("label", "N/A"),
            "Params": tres.get("n_params", "N/A"),
            "Val R²": round(tres.get("val_avg_r2", 0), 4),
            "Val Corr": round(tres.get("val_avg_corr", 0), 4),
            "Test R²": round(tres.get("test_avg_r2", 0), 4),
            "Test Corr": round(tres.get("test_avg_corr", 0), 4),
        })

    if not rows:
        print("No successful test results found.")
        return None

    df = pd.DataFrame(rows).sort_values("Test R²", ascending=False)
    df.to_csv(results_dir / "summary_table.csv", index=False)
    print("\n=== SUMMARY TABLE ===")
    print(df.to_string(index=False))
    return df


def generate_per_feature_table(test_results, results_dir):
    """Generate per-feature results CSV."""
    rows = []
    for model_name, tres in test_results.items():
        if "error" in tres or "test_per_feature" not in tres:
            continue
        for fname, metrics in tres["test_per_feature"].items():
            rows.append({
                "Model": model_name,
                "Feature": fname,
                "R²": round(metrics["r2"], 4),
                "Pearson r": round(metrics["pearson_r"], 4),
                "MSE": round(metrics["mse"], 4),
            })

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "per_feature_results.csv", index=False)

    r2_pivot = df.pivot(index="Feature", columns="Model", values="R²")
    corr_pivot = df.pivot(index="Feature", columns="Model", values="Pearson r")

    print("\n=== PER-FEATURE R² ===")
    print(r2_pivot.to_string())

    return r2_pivot, corr_pivot


def plot_model_comparison(test_results, figures_dir):
    """Bar chart comparing models on average R²."""
    models, val_r2, test_r2 = [], [], []
    for model_name, tres in test_results.items():
        if "error" in tres:
            continue
        models.append(model_name)
        val_r2.append(tres.get("val_avg_r2", 0))
        test_r2.append(tres.get("test_avg_r2", 0))

    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, val_r2, width, label="Val R²", alpha=0.8)
    ax.bar(x + width / 2, test_r2, width, label="Test R²", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Average R²")
    ax.set_title("Model Comparison: Average R² Across All Features")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_r2.png", dpi=150)
    plt.close()


def plot_heatmap(r2_pivot, figures_dir):
    """Heatmap of R² per feature per model."""
    if r2_pivot is None or r2_pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        r2_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
        center=0, ax=ax, linewidths=0.5,
    )
    ax.set_title("Test R² by Feature and Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_heatmap.png", dpi=150)
    plt.close()


def plot_pretrained_vs_scratch(test_results, figures_dir):
    """Comparison of pretrained vs from-scratch for foundation models."""
    pairs = []
    for base in ["REVE", "BIOT"]:
        sk, pk = f"{base}_scratch", f"{base}_pretrained"
        if sk in test_results and pk in test_results:
            s, p = test_results[sk], test_results[pk]
            if "error" not in s and "error" not in p:
                pairs.append((base, s, p))

    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(8 * len(pairs), 6))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (base, scratch, pretrained) in zip(axes, pairs):
        features = list(scratch["test_per_feature"].keys())
        s_r2 = [scratch["test_per_feature"][f]["r2"] for f in features]
        p_r2 = [pretrained["test_per_feature"][f]["r2"] for f in features]

        x = np.arange(len(features))
        width = 0.35
        ax.barh(x - width / 2, s_r2, width, label="From Scratch", alpha=0.8)
        ax.barh(x + width / 2, p_r2, width, label="Pretrained", alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel("R²")
        ax.set_title(f"{base}: Pretrained vs Scratch")
        ax.legend()
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(figures_dir / "pretrained_vs_scratch.png", dpi=150)
    plt.close()


def plot_pearson_correlation(corr_pivot, figures_dir):
    """Bar chart of Pearson correlation per feature."""
    if corr_pivot is None or corr_pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    corr_pivot.plot(kind="bar", ax=ax, alpha=0.8)
    ax.set_ylabel("Pearson r")
    ax.set_title("Test Pearson Correlation by Feature and Model")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "pearson_correlation.png", dpi=150)
    plt.close()


def plot_sweep_overview(all_results, figures_dir):
    """Scatter plot of all sweep configs: params vs R², colored by model."""
    successful = [r for r in all_results if r.get("status") == "success"]
    if not successful:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(set(r["model_name"] for r in successful))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for model, color in zip(models, colors):
        data = [r for r in successful if r["model_name"] == model]
        params = [r["n_params"] for r in data]
        r2s = [r["avg_r2"] for r in data]
        ax.scatter(params, r2s, c=[color], label=model, s=60, alpha=0.7)

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Validation R²")
    ax.set_title("Hyperparameter Sweep: Parameters vs Performance")
    ax.set_xscale("log")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "sweep_params_vs_r2.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="Path to results directory",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}")
    all_results, test_results = load_results(results_dir)

    if not all_results and not test_results:
        print("No results found. Run run_benchmark.py first.")
        return

    print(f"Loaded {len(all_results)} sweep results, {len(test_results)} test results")

    # Generate all outputs
    summary_df = generate_summary_table(test_results, results_dir)
    r2_pivot, corr_pivot = generate_per_feature_table(test_results, results_dir)

    plot_model_comparison(test_results, figures_dir)
    plot_heatmap(r2_pivot, figures_dir)
    plot_pretrained_vs_scratch(test_results, figures_dir)
    plot_pearson_correlation(corr_pivot, figures_dir)
    plot_sweep_overview(all_results, figures_dir)

    print(f"\nReport generated in {results_dir}")
    print(f"Figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
