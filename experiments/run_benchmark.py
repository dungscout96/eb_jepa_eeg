"""
Benchmark evaluation of EEGNet, REVE, and BIOT on HBN movie feature prediction.

Evaluates each model (with hyperparameter sweep) on predicting movie visual features
from EEG data. Supports regression and classification (binary/tertile) modes.

Usage:
    python experiments/run_benchmark.py                         # default: binary classification
    python experiments/run_benchmark.py benchmark.task_mode=regression
    python experiments/run_benchmark.py benchmark.task_mode=tertile
"""

import sys
import os
import json
import time
import itertools
import traceback
import warnings
from pathlib import Path
import hydra

import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score, roc_auc_score
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

DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"

NUMERIC_FEATURES = [
    # "luminance_mean", "contrast_rms",
    # "color_r_mean", "color_g_mean", "color_b_mean",
    # "saturation_mean", "edge_density", "spatial_freq_energy",
    # "entropy", "motion_energy",
    # "n_faces", "face_area_frac",
    # "depth_mean", "depth_std", "depth_range",
    # "n_objects",
    "scene_category_score", "scene_natural_score", #"scene_open_score",
    "sham_random",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Dataset Wrapper =====================

class MovieFeatureDataset(Dataset):
    """Wraps HBNMovieProbeDataset to return only numeric features as tensors."""

    def __init__(self, hbn_dataset, feature_names, feature_stats=None):
        self.feature_names = feature_names
        self.feature_stats = feature_stats
        self._precompute(hbn_dataset)

    def _precompute(self, hbn_dataset):
        X_list, y_list = [], []
        real_features = [f for f in self.feature_names if f != "sham_random"]
        print(f"  Precomputing {len(hbn_dataset)} windows...")
        for i in tqdm(range(len(hbn_dataset)), desc="  Loading", leave=False):
            X, features = hbn_dataset[i]
            y = torch.tensor([float(features[f]) for f in real_features],
                             dtype=torch.float32)
            X_list.append(X)
            y_list.append(y)

        self.X_data = torch.stack(X_list)
        self.y_data = torch.stack(y_list)

        # Replace NaN with 0
        self.y_data = torch.nan_to_num(self.y_data, nan=0.0)

        # Append sham column: random uniform [-1, 1] (no relation to EEG)
        if "sham_random" in self.feature_names:
            sham = torch.FloatTensor(len(self.X_data), 1).uniform_(-1, 1)
            self.y_data = torch.cat([self.y_data, sham], dim=1)

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

    def compute_bin_edges(self, task_mode):
        """Compute bin edges from training data for classification modes.

        Returns:
            bin_edges: (n_features, n_bins-1) tensor of quantile thresholds
        """
        if task_mode == "binary":
            # Median split
            return self.y_data.median(dim=0).values.unsqueeze(1)  # (n_features, 1)
        elif task_mode == "tertile":
            q33 = torch.quantile(self.y_data, 0.333, dim=0)
            q66 = torch.quantile(self.y_data, 0.667, dim=0)
            return torch.stack([q33, q66], dim=1)  # (n_features, 2)
        else:
            return None

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

    # --- EEGNet ---
    configs["EEGNet"] = []
    for F1, D, dp, lr in itertools.product(
        [8, 16],        # F1: temporal filters
        [2, 4],         # D: depth multiplier
        [0.25, 0.5],    # drop_prob
        [1e-3, 1e-4],   # learning rate
    ):
        configs["EEGNet"].append({
            "model_params": {
                "n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times,
                "F1": F1, "D": D, "drop_prob": dp,
            },
            "lr": lr,
            "label": f"F1={F1}_D={D}_dp={dp}_lr={lr}",
        })

    # # --- REVE from scratch ---
    # if REVE_AVAILABLE:
    #     configs["REVE_scratch"] = []
    #     for embed_dim, depth, lr in itertools.product(
    #         [128, 256],   # embed_dim
    #         [4, 8],       # transformer depth
    #         [5e-4, 1e-4], # learning rate
    #     ):
    #         heads = max(4, embed_dim // 32)
    #         head_dim = embed_dim // heads
    #         configs["REVE_scratch"].append({
    #             "model_params": {
    #                 "n_chans": n_chans, "n_outputs": n_outputs, "n_times": n_times,
    #                 "sfreq": sfreq,
    #                 "embed_dim": embed_dim, "depth": depth,
    #                 "heads": heads, "head_dim": head_dim,
    #                 "chs_info": chs_info,
    #             },
    #             "lr": lr,
    #             "label": f"ed={embed_dim}_d={depth}_lr={lr}",
    #         })

        # --- REVE pretrained (fine-tune) ---
        # configs["REVE_pretrained"] = []
        # for lr in [1e-4, 5e-5, 1e-5]:
        #     configs["REVE_pretrained"].append({
        #         "model_params": {
        #             "n_outputs": n_outputs,
        #             "n_chans": n_chans,
        #             "n_times": int(n_times * 200 / sfreq),  # resample to 200Hz
        #             "sfreq": 200,
        #             "chs_info": chs_info,
        #         },
        #         "lr": lr,
        #         "resample_200": True,
        #         "label": f"pretrained_lr={lr}",
        #     })

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


# ===================== Target Discretization =====================

def discretize_targets(y, bin_edges):
    """Convert continuous targets to class labels using precomputed bin edges.

    Args:
        y: (batch,) continuous values
        bin_edges: (n_edges,) thresholds. For binary: 1 edge (median).
                   For tertile: 2 edges.
    Returns:
        labels: (batch,) long tensor with class indices
    """
    labels = torch.zeros_like(y, dtype=torch.long)
    for edge in bin_edges:
        labels += (y > edge).long()
    return labels


# ===================== Training & Evaluation =====================

def _forward(model, model_name, X, channel_positions):
    """Run forward pass, handling REVE's positional argument."""
    if "REVE" in model_name and channel_positions is not None:
        pos = channel_positions.unsqueeze(0).expand(X.shape[0], -1, -1).to(X.device)
        return model(X, pos=pos)
    return model(X)


def train_and_evaluate(
    model_name, config, train_loader, val_loader,
    feature_idx, feature_name,
    channel_positions=None, *, trainer_cfg,
    task_mode="regression", bin_edges=None,
):
    """Train a single model on one feature and return validation metrics.

    Args:
        task_mode: "regression", "binary", or "tertile"
        bin_edges: (n_edges,) tensor of thresholds for classification modes
    """
    is_classification = task_mode in ("binary", "tertile")
    resample_200 = config.get("resample_200", False)
    resample_size = config["model_params"]["n_times"] if resample_200 else None

    try:
        model_params = {k: v for k, v in config["model_params"].items()}
        model = create_model(model_name, model_params)
        model = model.to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        n_epochs = trainer_cfg.n_epochs
        patience = trainer_cfg.patience

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=trainer_cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        train_losses, val_losses = [], []

        for epoch in range(n_epochs):
            # --- Train ---
            model.train()
            epoch_train_loss = 0
            n_batches = 0
            for X, y_all in train_loader:
                X = X.to(DEVICE)
                y_feat = y_all[:, feature_idx]
                if is_classification:
                    y = discretize_targets(y_feat, bin_edges).to(DEVICE)
                else:
                    y = y_feat.unsqueeze(1).to(DEVICE)
                if resample_size is not None:
                    X = F.interpolate(X, size=resample_size, mode="linear")

                optimizer.zero_grad()
                out = _forward(model, model_name, X, channel_positions)
                if is_classification:
                    loss = criterion(out, y)  # out: (B, n_classes), y: (B,)
                else:
                    loss = criterion(out, y)  # out: (B, 1), y: (B, 1)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), trainer_cfg.grad_clip_norm)
                optimizer.step()
                epoch_train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_losses.append(epoch_train_loss / max(n_batches, 1))

            # --- Validate ---
            model.eval()
            epoch_val_loss = 0
            n_val_batches = 0
            with torch.no_grad():
                for X, y_all in val_loader:
                    X = X.to(DEVICE)
                    y_feat = y_all[:, feature_idx]
                    if is_classification:
                        y = discretize_targets(y_feat, bin_edges).to(DEVICE)
                    else:
                        y = y_feat.unsqueeze(1).to(DEVICE)
                    if resample_size is not None:
                        X = F.interpolate(X, size=resample_size, mode="linear")
                    out = _forward(model, model_name, X, channel_positions)
                    loss = criterion(out, y)
                    epoch_val_loss += loss.item()
                    n_val_batches += 1

            avg_val = epoch_val_loss / max(n_val_batches, 1)
            val_losses.append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # --- Final evaluation on validation set ---
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y_all in val_loader:
                X = X.to(DEVICE)
                if resample_size is not None:
                    X = F.interpolate(X, size=resample_size, mode="linear")
                out = _forward(model, model_name, X, channel_positions)
                all_preds.append(out.cpu())
                y_feat = y_all[:, feature_idx]
                if is_classification:
                    all_targets.append(discretize_targets(y_feat, bin_edges))
                else:
                    all_targets.append(y_feat)

        preds_t = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)

        metrics = _compute_eval_metrics(preds_t, targets_t, is_classification)

        return {
            "status": "success",
            "model_name": model_name,
            "feature_name": feature_name,
            "label": config["label"],
            "lr": config["lr"],
            "n_params": n_params,
            "n_trainable": n_trainable,
            "best_val_loss": float(best_val_loss),
            "best_epoch": len(train_losses) - patience_counter,
            "total_epochs": len(train_losses),
            **metrics,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_state": best_state,
        }

    except Exception as e:
        return {
            "status": "error",
            "model_name": model_name,
            "feature_name": feature_name,
            "label": config.get("label", "unknown"),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _compute_eval_metrics(preds_t, targets_t, is_classification):
    """Compute metrics from prediction and target tensors."""
    if is_classification:
        probs = torch.softmax(preds_t, dim=1).numpy()
        pred_labels = preds_t.argmax(dim=1).numpy()
        true_labels = targets_t.numpy()
        n_classes = preds_t.shape[1]

        acc = float(accuracy_score(true_labels, pred_labels))
        bal_acc = float(balanced_accuracy_score(true_labels, pred_labels))
        try:
            if n_classes == 2:
                auc = float(roc_auc_score(true_labels, probs[:, 1]))
            else:
                auc = float(roc_auc_score(true_labels, probs, multi_class="ovr"))
        except ValueError:
            auc = 0.0

        return {"accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc}
    else:
        preds = preds_t.numpy()[:, 0]
        targets = targets_t.numpy()
        mse = float(np.mean((preds - targets) ** 2))
        if np.std(targets) > 1e-10 and np.std(preds) > 1e-10:
            r2 = float(r2_score(targets, preds))
            corr = float(pearsonr(preds, targets).statistic)
        else:
            r2, corr = 0.0, 0.0
        return {"r2": r2, "pearson_r": corr, "mse": mse}


def evaluate_on_test(
    model_name, config, test_loader, best_state,
    feature_idx, channel_positions=None,
    task_mode="regression", bin_edges=None,
):
    """Evaluate a trained model on the test set for a single feature."""
    is_classification = task_mode in ("binary", "tertile")
    resample_200 = config.get("resample_200", False)
    resample_size = config["model_params"]["n_times"] if resample_200 else None

    model_params = {k: v for k, v in config["model_params"].items()}
    model = create_model(model_name, model_params)
    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y_all in test_loader:
            X = X.to(DEVICE)
            if resample_size is not None:
                X = F.interpolate(X, size=resample_size, mode="linear")
            out = _forward(model, model_name, X, channel_positions)
            all_preds.append(out.cpu())
            y_feat = y_all[:, feature_idx]
            if is_classification:
                all_targets.append(discretize_targets(y_feat, bin_edges))
            else:
                all_targets.append(y_feat)

    preds_t = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)
    return _compute_eval_metrics(preds_t, targets_t, is_classification)


# ===================== Main =====================

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    results_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")
    print(f"Device: {DEVICE}")
    print(f"Models available: EEGNet=True, REVE={REVE_AVAILABLE}, BIOT={BIOT_AVAILABLE}")

    task_mode = cfg.benchmark.task_mode
    is_classification = task_mode in ("binary", "tertile")
    n_classes = {"binary": 2, "tertile": 3}.get(task_mode, 0)
    print(f"Task mode: {task_mode}" + (f" ({n_classes} classes)" if is_classification else ""))

    # --- Load datasets ---
    print("\n=== Loading datasets ===")
    print("Loading train split...")
    train_raw = HBNMovieProbeDataset(split="train", cfg=cfg.data)
    print("Loading val split...")
    val_raw = HBNMovieProbeDataset(split="val", cfg=cfg.data)
    print("Loading test split...")
    test_raw = HBNMovieProbeDataset(split="test", cfg=cfg.data)

    # Get data dimensions
    sample_X, _ = train_raw[0]
    n_chans, n_times = sample_X.shape
    sfreq = train_raw.sfreq
    n_features = len(NUMERIC_FEATURES)
    print(f"Data shape: n_chans={n_chans}, n_times={n_times}, sfreq={sfreq}")
    print(f"Predicting {n_features} features (per-feature probes): {NUMERIC_FEATURES}")

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

    # Compute bin edges from un-normalized training data (for classification)
    all_bin_edges = train_ds.compute_bin_edges(task_mode)  # (n_features, n_edges) or None

    if is_classification:
        # For classification: keep raw values (binning happens in training loop)
        val_ds = MovieFeatureDataset(val_raw, NUMERIC_FEATURES)
        test_ds = MovieFeatureDataset(test_raw, NUMERIC_FEATURES)
    else:
        # For regression: z-normalize targets
        train_ds = MovieFeatureDataset(train_raw, NUMERIC_FEATURES, feature_stats=train_stats)
        val_ds = MovieFeatureDataset(val_raw, NUMERIC_FEATURES, feature_stats=train_stats)
        test_ds = MovieFeatureDataset(test_raw, NUMERIC_FEATURES, feature_stats=train_stats)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} windows")

    # Save normalization stats
    torch.save(train_stats, results_dir / "train_stats.pt")

    batch_size = cfg.trainer.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Hyperparameter grids ---
    n_outputs = n_classes if is_classification else 1
    configs = get_sweep_configs(
        n_chans=n_chans, n_times=n_times, sfreq=sfreq,
        n_outputs=n_outputs, chs_info=chs_info,
    )

    n_model_configs = sum(len(v) for v in configs.values())
    total_runs = n_features * n_model_configs
    print(f"\nPer-feature sweep: {n_features} features x {n_model_configs} configs = {total_runs} total runs")
    for name, cfgs in configs.items():
        print(f"  {name}: {len(cfgs)} configs")

    # --- Run per-feature sweep ---
    all_results = []
    best_per_model_feature = {}
    run_idx = 0

    for feat_idx, feat_name in enumerate(NUMERIC_FEATURES):
        print(f"\n{'='*70}")
        print(f"  Feature [{feat_idx+1}/{n_features}]: {feat_name}")
        print(f"{'='*70}")

        for model_name, model_configs in configs.items():
            for config in model_configs:
                run_idx += 1
                print(f"\n  [{run_idx}/{total_runs}] {model_name} | {feat_name} | {config['label']}")
                t0 = time.time()

                feat_bin_edges = all_bin_edges[feat_idx] if all_bin_edges is not None else None
                result = train_and_evaluate(
                    model_name, config, train_loader, val_loader,
                    feature_idx=feat_idx, feature_name=feat_name,
                    channel_positions=channel_positions,
                    trainer_cfg=cfg.trainer,
                    task_mode=task_mode, bin_edges=feat_bin_edges,
                )
                elapsed = time.time() - t0
                result["elapsed_s"] = round(elapsed, 1)

                if result["status"] == "success":
                    if is_classification:
                        print(
                            f"    val_loss={result['best_val_loss']:.4f} | "
                            f"acc={result['accuracy']:.4f} | "
                            f"bal_acc={result['balanced_accuracy']:.4f} | "
                            f"auc={result['auc']:.4f} | "
                            f"params={result['n_params']:,} | "
                            f"epochs={result['total_epochs']} | "
                            f"{elapsed:.1f}s"
                        )
                    else:
                        print(
                            f"    val_loss={result['best_val_loss']:.4f} | "
                            f"r2={result['r2']:.4f} | "
                            f"corr={result['pearson_r']:.4f} | "
                            f"params={result['n_params']:,} | "
                            f"epochs={result['total_epochs']} | "
                            f"{elapsed:.1f}s"
                        )

                    key = (model_name, feat_name)
                    if (
                        key not in best_per_model_feature
                        or result["best_val_loss"]
                        < best_per_model_feature[key]["best_val_loss"]
                    ):
                        best_per_model_feature[key] = result

                    result_save = {
                        k: v for k, v in result.items() if k != "best_state"
                    }
                    all_results.append(result_save)
                else:
                    print(f"    ERROR: {result['error']}")
                    all_results.append(result)

                with open(results_dir / "all_results.json", "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

    # --- Evaluate best models on test set ---
    print(f"\n\n{'='*70}")
    print("  Evaluating best per-feature models on TEST set")
    print(f"{'='*70}")

    test_results = []
    for (model_name, feat_name), best in best_per_model_feature.items():
        if best["status"] != "success" or best.get("best_state") is None:
            continue

        feat_idx = NUMERIC_FEATURES.index(feat_name)

        best_config = None
        for c in configs[model_name]:
            if c["label"] == best["label"]:
                best_config = c
                break
        if best_config is None:
            continue

        try:
            feat_bin_edges = all_bin_edges[feat_idx] if all_bin_edges is not None else None
            test_metrics = evaluate_on_test(
                model_name, best_config, test_loader, best["best_state"],
                feature_idx=feat_idx,
                channel_positions=channel_positions,
                task_mode=task_mode, bin_edges=feat_bin_edges,
            )
            row = {
                "model_name": model_name,
                "feature_name": feat_name,
                "label": best["label"],
                "n_params": best["n_params"],
            }
            if is_classification:
                val_keys = ["accuracy", "balanced_accuracy", "auc"]
                for k in val_keys:
                    row[f"val_{k}"] = best[k]
                    row[f"test_{k}"] = test_metrics[k]
                print(
                    f"  {model_name} | {feat_name}: "
                    f"test_acc={test_metrics['accuracy']:.4f} | "
                    f"test_auc={test_metrics['auc']:.4f}"
                )
            else:
                row.update({
                    "val_r2": best["r2"], "val_corr": best["pearson_r"],
                    "test_r2": test_metrics["r2"],
                    "test_corr": test_metrics["pearson_r"],
                    "test_mse": test_metrics["mse"],
                })
                print(
                    f"  {model_name} | {feat_name}: "
                    f"test_r2={test_metrics['r2']:.4f} | "
                    f"test_corr={test_metrics['pearson_r']:.4f}"
                )
            test_results.append(row)
        except Exception as e:
            print(f"  {model_name} | {feat_name}: test eval failed: {e}")

    # Save test results
    with open(results_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)

    # --- Generate report ---
    generate_report(all_results, test_results, results_dir, task_mode=task_mode)

    print(f"\nResults saved to {results_dir}")


# ===================== Report Generation =====================

def generate_report(all_results, test_results, results_dir, *, task_mode="regression"):
    """Generate summary tables and plots from per-feature probe results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    is_classification = task_mode in ("binary", "tertile")

    if not test_results:
        print("\nNo successful test results to report.")
        return

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(results_dir / "per_feature_results.csv", index=False)

    # Determine primary metric
    if is_classification:
        primary_metric = "test_accuracy"
        val_metric = "val_accuracy"
        chance = {
            "binary": 0.5,
            "tertile": 0.333,
        }[task_mode]
    else:
        primary_metric = "test_r2"
        val_metric = "val_r2"
        chance = 0.0

    # ---- 1. Per-feature results table ----
    print(f"\n{'='*70}")
    metric_label = "Accuracy" if is_classification else "R²"
    print(f"  PER-FEATURE TEST {metric_label} (best config per model per feature)")
    print(f"{'='*70}")

    pivot = test_df.pivot(
        index="feature_name", columns="model_name", values=primary_metric,
    )
    print(pivot.to_string())

    # ---- 2. Average per model ----
    print(f"\n{'='*70}")
    print("  AVERAGE TEST METRICS PER MODEL")
    print(f"{'='*70}")

    if is_classification:
        model_avg = test_df.groupby("model_name").agg(
            avg_test_acc=("test_accuracy", "mean"),
            avg_test_bal_acc=("test_balanced_accuracy", "mean"),
            avg_test_auc=("test_auc", "mean"),
            avg_val_acc=("val_accuracy", "mean"),
        ).round(4).sort_values("avg_test_acc", ascending=False)
    else:
        model_avg = test_df.groupby("model_name").agg(
            avg_test_r2=("test_r2", "mean"),
            avg_test_corr=("test_corr", "mean"),
            avg_val_r2=("val_r2", "mean"),
        ).round(4).sort_values("avg_test_r2", ascending=False)
    print(model_avg.to_string())
    model_avg.to_csv(results_dir / "model_averages.csv")

    # ---- 3. Heatmap ----
    model_names = pivot.columns.tolist()
    if len(model_names) >= 1:
        fig, ax = plt.subplots(figsize=(max(8, 3 * len(model_names)), 8))
        cmap = "YlGn" if is_classification else "RdYlGn"
        center = chance if is_classification else 0
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap=cmap,
            center=center, ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Test {metric_label} by Feature ({task_mode} per-feature probes)")
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_heatmap.png", dpi=150)
        plt.close()
        print(f"\n  Saved: {figures_dir / 'feature_heatmap.png'}")

    # ---- 4. Bar chart: per-feature primary metric ----
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot.plot(kind="bar", ax=ax, alpha=0.8)
    ax.set_ylabel(metric_label)
    ax.set_title(f"Test {metric_label} by Feature ({task_mode})")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.axhline(y=chance, color="red", linestyle="--", linewidth=2,
               label=f"Chance ({chance})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "per_feature_metric.png", dpi=150)
    plt.close()
    print(f"  Saved: {figures_dir / 'per_feature_metric.png'}")

    # ---- 5. Sweep overview ----
    successful = [r for r in all_results if r.get("status") == "success"]
    if successful:
        sweep_rows = []
        for r in successful:
            row = {
                "Model": r["model_name"],
                "Feature": r["feature_name"],
                "Config": r["label"],
                "LR": r["lr"],
                "Params": r["n_params"],
                "Val Loss": round(r["best_val_loss"], 4),
                "Epochs": r["total_epochs"],
                "Time (s)": r.get("elapsed_s", "N/A"),
            }
            if is_classification:
                row["Val Acc"] = round(r["accuracy"], 4)
                row["Val BalAcc"] = round(r["balanced_accuracy"], 4)
                row["Val AUC"] = round(r["auc"], 4)
                sort_col = "Val Acc"
            else:
                row["Val R²"] = round(r["r2"], 4)
                row["Val Corr"] = round(r["pearson_r"], 4)
                sort_col = "Val R²"
            sweep_rows.append(row)

        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df = sweep_df.sort_values(
            ["Model", "Feature", sort_col], ascending=[True, True, False]
        )
        sweep_df.to_csv(results_dir / "sweep_all_configs.csv", index=False)

        print(f"\n{'='*70}")
        print("  FULL SWEEP RESULTS")
        print(f"{'='*70}")
        print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
