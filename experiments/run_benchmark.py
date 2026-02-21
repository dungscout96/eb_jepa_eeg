"""
Benchmark evaluation of EEGNet, REVE, and BIOT on HBN movie feature prediction.

Evaluates each model (with hyperparameter sweep) on predicting movie visual features
from EEG data. Foundation models (REVE, BIOT) are evaluated both from scratch and
with pretrained weights.

Usage:
    python experiments/run_benchmark.py
"""

import sys
import os
import json
import time
import itertools
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from eb_jepa.datasets.hbn import HBNMovieProbeDataset

# Model imports
from braindecode.models import EEGNet

REVE_AVAILABLE = False
BIOT_AVAILABLE = False
try:
    from braindecode.models import REVE
    REVE_AVAILABLE = True
except Exception as e:
    warnings.warn(f"REVE not available: {e}")
try:
    from braindecode.models import BIOT
    BIOT_AVAILABLE = True
except Exception as e:
    warnings.warn(f"BIOT not available: {e}")


# ===================== Configuration =====================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "luminance_mean", "contrast_rms",
    "color_r_mean", "color_g_mean", "color_b_mean",
    "saturation_mean", "edge_density", "spatial_freq_energy",
    "entropy", "motion_energy",
    "n_faces", "face_area_frac",
    "depth_mean", "depth_std", "depth_range",
    "n_objects",
    "scene_category_score", "scene_natural_score", "scene_open_score",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 50
PATIENCE = 7
BATCH_SIZE = 32


# ===================== Dataset Wrapper =====================

class MovieFeatureDataset(Dataset):
    """Wraps HBNMovieProbeDataset to return only numeric features as tensors."""

    def __init__(self, hbn_dataset, feature_names, feature_stats=None):
        self.feature_names = feature_names
        self.feature_stats = feature_stats
        self._precompute(hbn_dataset)

    def _precompute(self, hbn_dataset):
        X_list, y_list = [], []
        print(f"  Precomputing {len(hbn_dataset)} windows...")
        for i in tqdm(range(len(hbn_dataset)), desc="  Loading", leave=False):
            X, features = hbn_dataset[i]
            y = torch.tensor([float(features[f]) for f in self.feature_names],
                             dtype=torch.float32)
            X_list.append(X)
            y_list.append(y)

        self.X_data = torch.stack(X_list)
        self.y_data = torch.stack(y_list)

        # Replace NaN with 0
        self.y_data = torch.nan_to_num(self.y_data, nan=0.0)

        if self.feature_stats is not None:
            self.y_data = (
                (self.y_data - self.feature_stats["mean"])
                / (self.feature_stats["std"] + 1e-8)
            )

    def compute_stats(self):
        return {
            "mean": self.y_data.mean(dim=0),
            "std": self.y_data.std(dim=0),
        }

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


# ===================== Channel Positions =====================

def extract_channel_positions(hbn_dataset):
    """Extract 3D channel positions from the raw EEG data montage."""
    try:
        raw = hbn_dataset.data.datasets[0].raw
        montage = raw.get_montage()
        if montage is not None:
            pos_dict = montage.get_positions()["ch_pos"]
            ch_names = raw.info["ch_names"]
            positions = []
            for ch in ch_names:
                if ch in pos_dict:
                    positions.append(pos_dict[ch])
                else:
                    positions.append([0.0, 0.0, 0.0])
            return torch.tensor(np.array(positions), dtype=torch.float32)
    except Exception as e:
        warnings.warn(f"Could not extract channel positions: {e}")
    return None


def get_chs_info(hbn_dataset):
    """Extract channel info for REVE model."""
    try:
        raw = hbn_dataset.data.datasets[0].raw
        return [{"ch_name": ch} for ch in raw.info["ch_names"]]
    except Exception:
        return None


# ===================== Model Creation =====================

def _load_partial_pretrained(model, repo_id):
    """Load pretrained weights with partial matching (skip mismatched shapes)."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id, "model.safetensors")
        from safetensors.torch import load_file
        pretrained_state = load_file(path)
    except Exception:
        path = hf_hub_download(repo_id, "pytorch_model.bin")
        pretrained_state = torch.load(path, map_location="cpu", weights_only=True)

    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in pretrained_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    print(f"    Partial pretrained load: {loaded} matched, {skipped} skipped")
    return model


def create_model(model_name, model_params):
    """Create a model instance from name and parameters."""
    if model_name == "EEGNet":
        return EEGNet(**model_params)
    elif model_name == "REVE_scratch":
        if not REVE_AVAILABLE:
            raise ImportError("REVE not available")
        return REVE(**model_params)
    elif model_name == "REVE_pretrained":
        if not REVE_AVAILABLE:
            raise ImportError("REVE not available")
        try:
            return REVE.from_pretrained(
                "brain-bzh/reve-base",
                n_outputs=model_params["n_outputs"],
                n_chans=model_params.get("n_chans"),
                n_times=model_params.get("n_times"),
                sfreq=model_params.get("sfreq", 200),
                chs_info=model_params.get("chs_info"),
            )
        except Exception as e:
            # Gated repo or other error: create base config and try partial load
            print(f"    from_pretrained failed ({e}), trying partial load...")
            model = REVE(
                n_outputs=model_params["n_outputs"],
                n_chans=model_params.get("n_chans"),
                n_times=model_params.get("n_times"),
                sfreq=model_params.get("sfreq", 200),
                chs_info=model_params.get("chs_info"),
                embed_dim=512, depth=22, heads=8, head_dim=64,
            )
            return _load_partial_pretrained(model, "brain-bzh/reve-base")
    elif model_name == "BIOT_scratch":
        if not BIOT_AVAILABLE:
            raise ImportError("BIOT not available")
        return BIOT(**model_params)
    elif model_name == "BIOT_pretrained":
        if not BIOT_AVAILABLE:
            raise ImportError("BIOT not available")
        # Channel/shape mismatch expected; create with our params and partial load
        model = BIOT(**model_params)
        return _load_partial_pretrained(
            model, "braindecode/biot-pretrained-six-datasets-18chs"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ===================== Hyperparameter Grids =====================

def get_sweep_configs(n_chans, n_times, sfreq, n_outputs, chs_info=None):
    """Define hyperparameter sweep configurations for each model."""
    configs = {}

    # # --- EEGNet ---
    # configs["EEGNet"] = []
    # for F1, D, dp, lr in itertools.product(
    #     [8, 16],        # F1: temporal filters
    #     [2, 4],         # D: depth multiplier
    #     [0.25, 0.5],    # drop_prob
    #     [1e-3, 1e-4],   # learning rate
    # ):
    #     configs["EEGNet"].append({
    #         "model_params": {
    #             "n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times,
    #             "F1": F1, "D": D, "drop_prob": dp,
    #         },
    #         "lr": lr,
    #         "label": f"F1={F1}_D={D}_dp={dp}_lr={lr}",
    #     })

    # --- REVE from scratch ---
    if REVE_AVAILABLE:
        configs["REVE_scratch"] = []
        for embed_dim, depth, lr in itertools.product(
            [128, 256],   # embed_dim
            [4, 8],       # transformer depth
            [5e-4, 1e-4], # learning rate
        ):
            heads = max(4, embed_dim // 32)
            head_dim = embed_dim // heads
            configs["REVE_scratch"].append({
                "model_params": {
                    "n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times,
                    "sfreq": sfreq,
                    "embed_dim": embed_dim, "depth": depth,
                    "heads": heads, "head_dim": head_dim,
                    "chs_info": chs_info,
                },
                "lr": lr,
                "label": f"ed={embed_dim}_d={depth}_lr={lr}",
            })

        # --- REVE pretrained (fine-tune) ---
        configs["REVE_pretrained"] = []
        for lr in [1e-4, 5e-5, 1e-5]:
            configs["REVE_pretrained"].append({
                "model_params": {
                    "n_outputs": n_outputs,
                    "n_chans": n_chans,
                    "n_times": int(n_times * 200 / sfreq),  # resample to 200Hz
                    "sfreq": 200,
                    "chs_info": chs_info,
                },
                "lr": lr,
                "resample_200": True,
                "label": f"pretrained_lr={lr}",
            })

    # # --- BIOT from scratch ---
    # if BIOT_AVAILABLE:
    #     configs["BIOT_scratch"] = []
    #     for embed_dim, num_layers, dp, lr in itertools.product(
    #         [256, 512],    # embed_dim
    #         [2, 4],        # num_layers
    #         [0.3, 0.5],    # drop_prob
    #         [5e-4, 1e-4],  # learning rate
    #     ):
    #         configs["BIOT_scratch"].append({
    #             "model_params": {
    #                 "n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times,
    #                 "sfreq": sfreq,
    #                 "embed_dim": embed_dim, "num_layers": num_layers,
    #                 "drop_prob": dp,
    #             },
    #             "lr": lr,
    #             "label": f"ed={embed_dim}_nl={num_layers}_dp={dp}_lr={lr}",
    #         })

    #     # --- BIOT pretrained (fine-tune) ---
    #     configs["BIOT_pretrained"] = []
    #     for lr in [1e-4, 5e-5, 1e-5]:
    #         configs["BIOT_pretrained"].append({
    #             "model_params": {
    #                 "n_outputs": n_outputs,
    #                 "n_chans": n_chans,
    #                 "n_times": n_times,
    #                 "sfreq": sfreq,
    #             },
    #             "lr": lr,
    #             "label": f"pretrained_lr={lr}",
    #         })

    return configs


# ===================== Training & Evaluation =====================

def train_and_evaluate(
    model_name, config, train_loader, val_loader,
    channel_positions=None, n_epochs=N_EPOCHS, patience=PATIENCE,
):
    """Train a single model configuration and return validation metrics."""
    resample_200 = config.get("resample_200", False)
    resample_size = None
    if resample_200:
        # Target number of samples at 200Hz for the same duration
        resample_size = config["model_params"]["n_times"]

    try:
        # Create model
        model_params = {k: v for k, v in config["model_params"].items()}
        # Don't pass chs_info to from_pretrained if it will be unused
        model = create_model(model_name, model_params)
        model = model.to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            # --- Train ---
            model.train()
            epoch_train_loss = 0
            n_batches = 0
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                if resample_size is not None:
                    X = F.interpolate(X, size=resample_size, mode="linear")

                optimizer.zero_grad()
                if "REVE" in model_name and channel_positions is not None:
                    pos = channel_positions.unsqueeze(0).expand(
                        X.shape[0], -1, -1
                    ).to(DEVICE)
                    out = model(X, pos=pos)
                else:
                    out = model(X)
                loss = criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train = epoch_train_loss / max(n_batches, 1)
            train_losses.append(avg_train)

            # --- Validate ---
            model.eval()
            epoch_val_loss = 0
            n_val_batches = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    if resample_size is not None:
                        X = F.interpolate(X, size=resample_size, mode="linear")
                    if "REVE" in model_name and channel_positions is not None:
                        pos = channel_positions.unsqueeze(0).expand(
                            X.shape[0], -1, -1
                        ).to(DEVICE)
                        out = model(X, pos=pos)
                    else:
                        out = model(X)
                    loss = criterion(out, y)
                    epoch_val_loss += loss.item()
                    n_val_batches += 1

            avg_val = epoch_val_loss / max(n_val_batches, 1)
            val_losses.append(avg_val)

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best model for final evaluation
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Compute per-feature metrics on validation set
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                if resample_size is not None:
                    X = F.interpolate(X, size=resample_size, mode="linear")
                if "REVE" in model_name and channel_positions is not None:
                    pos = channel_positions.unsqueeze(0).expand(
                        X.shape[0], -1, -1
                    ).to(DEVICE)
                    out = model(X, pos=pos)
                else:
                    out = model(X)
                all_preds.append(out.cpu())
                all_targets.append(y)

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        per_feature = {}
        for i, fname in enumerate(NUMERIC_FEATURES):
            p, t = preds[:, i], targets[:, i]
            mse = float(np.mean((p - t) ** 2))
            if np.std(t) > 1e-10 and np.std(p) > 1e-10:
                r2 = float(r2_score(t, p))
                corr = float(pearsonr(p, t).statistic)
            else:
                r2 = 0.0
                corr = 0.0
            per_feature[fname] = {"mse": mse, "r2": r2, "pearson_r": corr}

        avg_r2 = np.mean([m["r2"] for m in per_feature.values()])
        avg_corr = np.mean([m["pearson_r"] for m in per_feature.values()])

        return {
            "status": "success",
            "model_name": model_name,
            "label": config["label"],
            "lr": config["lr"],
            "n_params": n_params,
            "n_trainable": n_trainable,
            "best_val_loss": float(best_val_loss),
            "best_epoch": len(train_losses) - patience_counter,
            "total_epochs": len(train_losses),
            "avg_r2": float(avg_r2),
            "avg_pearson_r": float(avg_corr),
            "per_feature": per_feature,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_state": best_state,
        }

    except Exception as e:
        return {
            "status": "error",
            "model_name": model_name,
            "label": config.get("label", "unknown"),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def evaluate_on_test(
    model_name, config, test_loader, best_state,
    channel_positions=None,
):
    """Evaluate a trained model on the test set."""
    resample_200 = config.get("resample_200", False)
    resample_size = config["model_params"]["n_times"] if resample_200 else None

    model_params = {k: v for k, v in config["model_params"].items()}
    model = create_model(model_name, model_params)
    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            if resample_size is not None:
                X = F.interpolate(X, size=resample_size, mode="linear")
            if "REVE" in model_name and channel_positions is not None:
                pos = channel_positions.unsqueeze(0).expand(
                    X.shape[0], -1, -1
                ).to(DEVICE)
                out = model(X, pos=pos)
            else:
                out = model(X)
            all_preds.append(out.cpu())
            all_targets.append(y)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    per_feature = {}
    for i, fname in enumerate(NUMERIC_FEATURES):
        p, t = preds[:, i], targets[:, i]
        mse = float(np.mean((p - t) ** 2))
        if np.std(t) > 1e-10 and np.std(p) > 1e-10:
            r2 = float(r2_score(t, p))
            corr = float(pearsonr(p, t).statistic)
        else:
            r2 = 0.0
            corr = 0.0
        per_feature[fname] = {"mse": mse, "r2": r2, "pearson_r": corr}

    avg_r2 = np.mean([m["r2"] for m in per_feature.values()])
    avg_corr = np.mean([m["pearson_r"] for m in per_feature.values()])

    return {
        "per_feature": per_feature,
        "avg_r2": float(avg_r2),
        "avg_pearson_r": float(avg_corr),
    }


# ===================== Main =====================

def main():
    print(f"Device: {DEVICE}")
    print(f"Models available: EEGNet=True, REVE={REVE_AVAILABLE}, BIOT={BIOT_AVAILABLE}")

    # --- Load datasets ---
    print("\n=== Loading datasets ===")
    print("Loading train split...")
    train_raw = HBNMovieProbeDataset(split="train")
    print("Loading val split...")
    val_raw = HBNMovieProbeDataset(split="val")
    print("Loading test split...")
    test_raw = HBNMovieProbeDataset(split="test")

    # Get data dimensions
    sample_X, _ = train_raw[0]
    n_chans, n_times = sample_X.shape
    sfreq = train_raw.sfreq
    n_outputs = len(NUMERIC_FEATURES)
    print(f"Data shape: n_chans={n_chans}, n_times={n_times}, sfreq={sfreq}")
    print(f"Predicting {n_outputs} features: {NUMERIC_FEATURES}")

    # Extract channel info for REVE
    channel_positions = extract_channel_positions(train_raw)
    chs_info = get_chs_info(train_raw)
    if channel_positions is not None:
        print(f"Channel positions extracted: {channel_positions.shape}")
    if chs_info is not None:
        print(f"Channel info: {len(chs_info)} channels")

    # --- Wrap datasets ---
    print("\nPrecomputing datasets...")
    train_ds = MovieFeatureDataset(train_raw, NUMERIC_FEATURES)
    train_stats = train_ds.compute_stats()

    # Re-create train with normalization
    train_ds = MovieFeatureDataset(train_raw, NUMERIC_FEATURES, feature_stats=train_stats)
    val_ds = MovieFeatureDataset(val_raw, NUMERIC_FEATURES, feature_stats=train_stats)
    test_ds = MovieFeatureDataset(test_raw, NUMERIC_FEATURES, feature_stats=train_stats)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} windows")

    # Save normalization stats
    torch.save(train_stats, RESULTS_DIR / "train_stats.pt")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Hyperparameter grids ---
    configs = get_sweep_configs(
        n_chans=n_chans, n_times=n_times, sfreq=sfreq,
        n_outputs=n_outputs, chs_info=chs_info,
    )

    total_configs = sum(len(v) for v in configs.values())
    print(f"\nTotal configurations to sweep: {total_configs}")
    for name, cfgs in configs.items():
        print(f"  {name}: {len(cfgs)} configs")

    # --- Run sweep ---
    all_results = []
    best_per_model = {}  # model_name -> best result (including state dict)
    config_idx = 0

    for model_name, model_configs in configs.items():
        print(f"\n{'='*70}")
        print(f"  Model: {model_name} ({len(model_configs)} configurations)")
        print(f"{'='*70}")

        for i, config in enumerate(model_configs):
            config_idx += 1
            print(f"\n  [{config_idx}/{total_configs}] {model_name} | {config['label']}")
            t0 = time.time()

            result = train_and_evaluate(
                model_name, config, train_loader, val_loader,
                channel_positions=channel_positions,
            )
            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)

            if result["status"] == "success":
                print(
                    f"    val_loss={result['best_val_loss']:.4f} | "
                    f"avg_r2={result['avg_r2']:.4f} | "
                    f"avg_corr={result['avg_pearson_r']:.4f} | "
                    f"params={result['n_params']:,} | "
                    f"epochs={result['total_epochs']} | "
                    f"{elapsed:.1f}s"
                )

                # Track best per model
                if (
                    model_name not in best_per_model
                    or result["best_val_loss"]
                    < best_per_model[model_name]["best_val_loss"]
                ):
                    best_per_model[model_name] = result

                # Save result (without state dict for JSON)
                result_save = {
                    k: v for k, v in result.items() if k != "best_state"
                }
                all_results.append(result_save)
            else:
                print(f"    ERROR: {result['error']}")
                all_results.append(result)

            # Save intermediate results
            with open(RESULTS_DIR / "all_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # --- Evaluate best models on test set ---
    print(f"\n\n{'='*70}")
    print("  Evaluating best models on TEST set")
    print(f"{'='*70}")

    test_results = {}
    for model_name, best in best_per_model.items():
        if best["status"] != "success" or best.get("best_state") is None:
            print(f"  {model_name}: skipped (no successful run)")
            continue

        # Find the config that produced the best result
        model_configs = configs[model_name]
        best_config = None
        for cfg in model_configs:
            if cfg["label"] == best["label"]:
                best_config = cfg
                break
        if best_config is None:
            print(f"  {model_name}: could not find matching config")
            continue

        print(f"\n  {model_name} (best config: {best['label']})")
        try:
            test_metrics = evaluate_on_test(
                model_name, best_config, test_loader, best["best_state"],
                channel_positions=channel_positions,
            )
            test_results[model_name] = {
                "label": best["label"],
                "val_avg_r2": best["avg_r2"],
                "val_avg_corr": best["avg_pearson_r"],
                "test_avg_r2": test_metrics["avg_r2"],
                "test_avg_corr": test_metrics["avg_pearson_r"],
                "test_per_feature": test_metrics["per_feature"],
                "n_params": best["n_params"],
            }
            print(
                f"    Test avg_r2={test_metrics['avg_r2']:.4f} | "
                f"avg_corr={test_metrics['avg_pearson_r']:.4f}"
            )
        except Exception as e:
            print(f"    Test evaluation failed: {e}")
            test_results[model_name] = {"error": str(e)}

    # Save test results
    with open(RESULTS_DIR / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    # --- Generate report ---
    generate_report(all_results, test_results, best_per_model)

    print(f"\nResults saved to {RESULTS_DIR}")


# ===================== Report Generation =====================

def generate_report(all_results, test_results, best_per_model):
    """Generate summary tables and plots from experiment results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    successful = [r for r in all_results if r.get("status") == "success"]
    if not successful:
        print("\nNo successful runs to report.")
        return

    # ---- 1. Summary table (best config per model) ----
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY (Best config per model, TEST set)")
    print(f"{'='*70}")

    summary_rows = []
    for model_name, tres in test_results.items():
        if "error" in tres:
            continue
        row = {
            "Model": model_name,
            "Best Config": tres["label"],
            "Params": tres.get("n_params", "N/A"),
            "Val R²": round(tres["val_avg_r2"], 4),
            "Val Corr": round(tres["val_avg_corr"], 4),
            "Test R²": round(tres["test_avg_r2"], 4),
            "Test Corr": round(tres["test_avg_corr"], 4),
        }
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values("Test R²", ascending=False)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)

    # ---- 2. Per-feature test results table ----
    print(f"\n{'='*70}")
    print("  PER-FEATURE TEST R² (Best config per model)")
    print(f"{'='*70}")

    feature_rows = []
    for model_name, tres in test_results.items():
        if "error" in tres or "test_per_feature" not in tres:
            continue
        for fname, metrics in tres["test_per_feature"].items():
            feature_rows.append({
                "Model": model_name,
                "Feature": fname,
                "R²": round(metrics["r2"], 4),
                "Pearson r": round(metrics["pearson_r"], 4),
                "MSE": round(metrics["mse"], 4),
            })

    if feature_rows:
        feature_df = pd.DataFrame(feature_rows)
        feature_df.to_csv(RESULTS_DIR / "per_feature_results.csv", index=False)

        # Pivot for heatmap
        r2_pivot = feature_df.pivot(
            index="Feature", columns="Model", values="R²"
        )
        print(r2_pivot.to_string())

    # ---- 3. Bar chart: average R² per model ----
    if summary_rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = [r["Model"] for r in summary_rows]
        test_r2 = [r["Test R²"] for r in summary_rows]
        val_r2 = [r["Val R²"] for r in summary_rows]

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
        print(f"\n  Saved: {figures_dir / 'model_comparison_r2.png'}")

    # ---- 4. Heatmap: per-feature R² ----
    if feature_rows and len(test_results) > 1:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            r2_pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5,
        )
        ax.set_title("Test R² by Feature and Model")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_heatmap.png", dpi=150)
        plt.close()
        print(f"  Saved: {figures_dir / 'feature_heatmap.png'}")

    # ---- 5. Pretrained vs scratch comparison ----
    scratch_pretrained_pairs = []
    for base in ["REVE", "BIOT"]:
        scratch_key = f"{base}_scratch"
        pretrained_key = f"{base}_pretrained"
        if scratch_key in test_results and pretrained_key in test_results:
            s = test_results[scratch_key]
            p = test_results[pretrained_key]
            if "error" not in s and "error" not in p:
                scratch_pretrained_pairs.append((base, s, p))

    if scratch_pretrained_pairs:
        fig, axes = plt.subplots(1, len(scratch_pretrained_pairs),
                                 figsize=(8 * len(scratch_pretrained_pairs), 6))
        if len(scratch_pretrained_pairs) == 1:
            axes = [axes]

        for ax, (base, scratch, pretrained) in zip(axes, scratch_pretrained_pairs):
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
        print(f"  Saved: {figures_dir / 'pretrained_vs_scratch.png'}")

    # ---- 6. Pearson correlation bar chart ----
    if feature_rows:
        corr_pivot = feature_df.pivot(
            index="Feature", columns="Model", values="Pearson r"
        )
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
        print(f"  Saved: {figures_dir / 'pearson_correlation.png'}")

    # ---- 7. Hyperparameter sweep overview ----
    sweep_rows = []
    for r in successful:
        sweep_rows.append({
            "Model": r["model_name"],
            "Config": r["label"],
            "LR": r["lr"],
            "Params": r["n_params"],
            "Val Loss": round(r["best_val_loss"], 4),
            "Val R²": round(r["avg_r2"], 4),
            "Val Corr": round(r["avg_pearson_r"], 4),
            "Epochs": r["total_epochs"],
            "Time (s)": r.get("elapsed_s", "N/A"),
        })

    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df = sweep_df.sort_values(["Model", "Val R²"], ascending=[True, False])
        sweep_df.to_csv(RESULTS_DIR / "sweep_all_configs.csv", index=False)

        print(f"\n{'='*70}")
        print("  FULL SWEEP RESULTS")
        print(f"{'='*70}")
        print(sweep_df.to_string(index=False))

    # ---- 8. Training curves for best models ----
    fig, axes = plt.subplots(1, len(best_per_model), figsize=(6 * len(best_per_model), 4))
    if len(best_per_model) == 1:
        axes = [axes]
    for ax, (model_name, best) in zip(axes, best_per_model.items()):
        if best["status"] != "success":
            continue
        ax.plot(best["train_losses"], label="Train", alpha=0.8)
        ax.plot(best["val_losses"], label="Val", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{model_name}\n{best['label']}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"  Saved: {figures_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
