"""Validation loop for EEG JEPA with movie feature probes."""

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from tqdm import tqdm

@torch.inference_mode()
def validation_loop(
    val_loader,
    jepa,
    regression_probe,
    classification_probe,
    device,
    feature_stats,
    feature_median,
    feature_names,
):
    """Run validation and compute regression + classification metrics.

    Args:
        val_loader: DataLoader yielding (eeg, features, probe_label) tuples.
        jepa: MaskedJEPA model.
        regression_probe: MaskedJEPAProbe with continuous MSE loss.
        classification_probe: MaskedJEPAProbe with binary BCE loss.
        device: Torch device.
        feature_stats: Dict with "mean" and "std" tensors from training set.
        feature_median: Tensor of per-feature medians from training set.
        feature_names: List of feature name strings.

    Returns:
        Dict of validation metrics.
    """
    jepa.eval()
    regression_probe.eval()
    classification_probe.eval()

    all_reg_preds = []
    all_cls_preds = []
    all_targets = []
    reg_loss_sum = 0.0
    cls_loss_sum = 0.0
    n_batches = 0

    for eeg, features, *_ in tqdm(val_loader, desc="Validating", leave=False):
        eeg = eeg.to(device)
        features = features.to(device)  # [B, T, n_features]

        # Losses
        reg_loss = regression_probe(eeg, features)
        cls_loss = classification_probe(eeg, features)

        # Predictions (from heads applied to frozen encoder output)
        state = jepa.encode(eeg)  # [B, D, T, 1, 1]
        reg_pred = regression_probe.head(state)  # [B, T, n_features]
        cls_pred = classification_probe.head(state)  # [B, T, n_features]

        all_reg_preds.append(reg_pred.cpu())
        all_cls_preds.append(cls_pred.cpu())
        all_targets.append(features.cpu())
        reg_loss_sum += reg_loss.item()
        cls_loss_sum += cls_loss.item()
        n_batches += 1

    metrics = {
        "val/reg_loss": reg_loss_sum / max(n_batches, 1),
        "val/cls_loss": cls_loss_sum / max(n_batches, 1),
    }

    # Flatten across batches and time: [N, n_features]
    reg_preds = torch.cat(all_reg_preds).flatten(0, 1).numpy()
    cls_preds = torch.cat(all_cls_preds).flatten(0, 1)
    targets = torch.cat(all_targets).flatten(0, 1).numpy()

    # ------------------------------------------------------------------
    # Regression metrics (unnormalize predictions to original scale)
    # ------------------------------------------------------------------
    mean = feature_stats["mean"].numpy()
    std = feature_stats["std"].numpy()
    reg_preds_unnorm = reg_preds * (std + 1e-8) + mean

    for i, name in enumerate(feature_names):
        pred = reg_preds_unnorm[:, i]
        targ = targets[:, i]
        if np.std(targ) > 1e-10 and np.std(pred) > 1e-10:
            metrics[f"val/reg_{name}_r2"] = float(r2_score(targ, pred))
            metrics[f"val/reg_{name}_corr"] = float(pearsonr(pred, targ).statistic)
        else:
            metrics[f"val/reg_{name}_r2"] = 0.0
            metrics[f"val/reg_{name}_corr"] = 0.0

    # ------------------------------------------------------------------
    # Classification metrics (median-split binary labels)
    # ------------------------------------------------------------------
    cls_probs = torch.sigmoid(cls_preds).numpy()
    cls_labels = (cls_probs > 0.5).astype(int)
    median_np = feature_median.numpy()
    binary_targets = (targets > median_np).astype(int)

    for i, name in enumerate(feature_names):
        pred_label = cls_labels[:, i]
        true_label = binary_targets[:, i]
        prob = cls_probs[:, i]
        metrics[f"val/cls_{name}_acc"] = float(
            accuracy_score(true_label, pred_label)
        )
        metrics[f"val/cls_{name}_bal_acc"] = float(
            balanced_accuracy_score(true_label, pred_label)
        )
        try:
            metrics[f"val/cls_{name}_auc"] = float(
                roc_auc_score(true_label, prob)
            )
        except ValueError:
            metrics[f"val/cls_{name}_auc"] = 0.0

    # Print summary
    print("Validation metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Restore training mode
    jepa.train()
    regression_probe.train()
    classification_probe.train()

    return metrics
