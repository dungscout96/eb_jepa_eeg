"""Sanity check hook for masked JEPA training.

Tracks collapse detection, training stability, and downstream signal metrics,
logging everything to W&B under the ``sanity/`` prefix.

Typical usage in the training loop (after backward(), before optimizer.step()):

    hook = SanityCheckHook(
        embed_dim=64,
        feature_median=train_set.compute_feature_median(),  # luminance fallback
        n_pred_masks_short=2,
        log_every_steps=50,
        probe_every_steps=200,
    )

    # inside the loop (probe_labels = [B] binary float from __getitem__):
    sanity_metrics = hook.step(step, eeg, features, jepa, loss_dict,
                               probe_labels=probe_labels)
    if wandb_run and sanity_metrics:
        wandb.log(sanity_metrics, step=global_step)

Metrics logged
--------------
Collapse detection
    sanity/embedding_variance_mean   per-dim variance mean  (collapse → ~0)
    sanity/embedding_variance_min    per-dim variance min   (collapse → 0)
    sanity/embedding_variance_max    per-dim variance max
    sanity/embedding_variance_std    spread of per-dim variances
    sanity/embedding_l2_mean         mean L2 norm of embeddings
    sanity/cosim_random_pairs_mean   mean cosine sim between random pairs (collapse → 1)
    sanity/cosim_random_pairs_max    max |cosine sim| between random pairs

Loss health
    sanity/loss_trend                second-half minus first-half of rolling window
                                     (negative = decreasing — good)
    sanity/loss_rolling_mean         rolling mean of prediction loss
    sanity/pred_loss_short           pred loss on short-range masks  (periodic)
    sanity/pred_loss_long            pred loss on long-range masks   (should be ≥ short)

Training stability
    sanity/grad_norm                 combined grad norm for encoder + predictor

Downstream signal
    sanity/linear_probe_acc          binary-classification accuracy of a lightweight
                                     linear probe trained on frozen embeddings.
                                     Label source (in priority order):
                                       1. Subject metadata: age > median  OR  sex M/F
                                          (passed in via ``probe_labels`` each step)
                                       2. Luminance fallback: luminance_mean > median
                                          (used when probe_labels are all NaN)
                                     Rising accuracy during training indicates the
                                     representation is becoming semantically meaningful.
"""

from __future__ import annotations

import logging
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SanityCheckHook:
    """W&B logging hook for masked JEPA collapse and health monitoring.

    Designed for MaskedJEPA (V-JEPA style). Pass ``enabled=False`` to make all
    calls no-ops, which allows toggling via config without code changes.

    Args:
        embed_dim: Encoder embedding dimension (used to initialise the linear probe).
        feature_median: [n_features] CPU tensor — per-feature median on the training
            set. Used only as a luminance fallback when no real subject labels are
            available (i.e. ``probe_labels`` are all NaN). Pass ``None`` to disable
            the fallback entirely.
        n_pred_masks_short: Number of short-range prediction masks in the collator.
            The first ``n_pred_masks_short`` entries in ``mask_result.pred_masks``
            are short; the rest are long-range.
        log_every_steps: How often (in global steps) to compute embedding/collapse
            metrics and gradient norms.
        probe_every_steps: How often to train and evaluate the downstream linear probe.
        probe_train_steps: SGD steps applied to the linear probe each evaluation round.
        probe_buffer_size: Max number of (embedding, label) samples retained in the
            rolling buffer used to train the linear probe.
        cosim_n_pairs: Number of random embedding pairs for cosine similarity.
        horizon_every_steps: How often to compute separate short / long pred losses.
        loss_history_len: Number of recent pred-loss values to retain for trend analysis.
        enabled: Set to False to disable all metric computation (hook becomes a no-op).
    """

    # Index of luminance_mean in JEPAMovieDataset.DEFAULT_FEATURES —
    # ["contrast_rms", "luminance_mean", "position_in_movie", "narrative_event_score"]
    LUMINANCE_FEATURE_IDX: int = 1

    def __init__(
        self,
        embed_dim: int,
        feature_median: torch.Tensor | None = None,
        n_pred_masks_short: int = 2,
        log_every_steps: int = 50,
        probe_every_steps: int = 200,
        probe_train_steps: int = 30,
        probe_buffer_size: int = 512,
        cosim_n_pairs: int = 128,
        horizon_every_steps: int = 200,
        loss_history_len: int = 100,
        enabled: bool = True,
    ) -> None:
        self.embed_dim = embed_dim
        self.feature_median = feature_median.cpu() if feature_median is not None else None
        self.n_pred_masks_short = n_pred_masks_short
        self.log_every_steps = log_every_steps
        self.probe_every_steps = probe_every_steps
        self.probe_train_steps = probe_train_steps
        self.probe_buffer_size = probe_buffer_size
        self.cosim_n_pairs = cosim_n_pairs
        self.horizon_every_steps = horizon_every_steps
        self.loss_history_len = loss_history_len
        self.enabled = enabled

        # Lightweight linear probe (single linear layer, trained on frozen embeddings)
        self.linear_probe = nn.Linear(embed_dim, 1)
        self.probe_optimizer = torch.optim.SGD(
            self.linear_probe.parameters(), lr=0.01, momentum=0.9
        )

        # Rolling embedding/label buffers for the linear probe
        self._emb_buffer: list[torch.Tensor] = []    # each [D]
        self._label_buffer: list[torch.Tensor] = []  # each scalar

        # Loss history for trend detection
        self._loss_history: deque[float] = deque(maxlen=loss_history_len)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        step: int,
        eeg: torch.Tensor,          # [B, T, C, W]
        features: torch.Tensor,     # [B, T, n_features]
        jepa,                       # MaskedJEPA
        loss_dict: dict,            # from jepa.forward()
        probe_labels: torch.Tensor | None = None,  # [B] binary float (NaN = unknown)
    ) -> dict:
        """Compute and return sanity metrics for the current training step.

        Call this *after* ``optimizer.step()`` so that ``.grad`` tensors are still
        populated. The returned dict can be directly merged into the W&B log call.

        Args:
            probe_labels: Per-sample binary labels from the dataset's subject metadata
                (age > median, sex M/F, …). Values of NaN indicate no metadata was
                available for that recording. When all values are NaN the hook falls
                back to binarised luminance from ``features``. Pass ``None`` to
                always use the luminance fallback.

        Returns an empty dict when ``enabled=False`` or when nothing is scheduled
        this step.
        """
        if not self.enabled:
            return {}

        # Always track pred-loss for trend analysis
        self._loss_history.append(float(loss_dict.get("pred_loss", 0.0)))

        metrics: dict[str, float] = {}

        if step % self.log_every_steps == 0:
            metrics.update(self._compute_embedding_metrics(eeg, jepa))
            metrics.update(self._compute_grad_norm(jepa))
            metrics.update(self._compute_loss_trend())
            self._update_probe_buffer(eeg, features, jepa, probe_labels)

        if step % self.probe_every_steps == 0 and len(self._emb_buffer) >= 16:
            metrics.update(self._train_and_eval_probe(eeg.device))

        if step % self.horizon_every_steps == 0 and step > 0:
            metrics.update(self._compute_horizon_losses(eeg, jepa))

        return metrics

    # ------------------------------------------------------------------
    # Collapse detection
    # ------------------------------------------------------------------

    def _compute_embedding_metrics(
        self, eeg: torch.Tensor, jepa
    ) -> dict[str, float]:
        """Per-dim variance and random-pair cosine similarity."""
        device = eeg.device
        B = eeg.shape[0]

        with torch.no_grad():
            # Full encoding without masking → [B, N_tokens, D]
            tokens = jepa.context_encoder.encode_tokens(eeg, mask=None)
            # Pool over tokens → [B, D]
            emb = tokens.mean(dim=1)

        # Per-dimension variance across the batch: [D]
        per_dim_var = emb.var(dim=0)

        result: dict[str, float] = {
            "sanity/embedding_variance_mean": per_dim_var.mean().item(),
            "sanity/embedding_variance_min": per_dim_var.min().item(),
            "sanity/embedding_variance_max": per_dim_var.max().item(),
            "sanity/embedding_variance_std": per_dim_var.std().item(),
            "sanity/embedding_l2_mean": emb.norm(dim=-1).mean().item(),
        }

        # Cosine similarity between random pairs
        if B >= 2:
            n_pairs = min(self.cosim_n_pairs, B * (B - 1) // 2)
            # Oversample, then filter self-pairs
            idx_a = torch.randint(0, B, (n_pairs * 2,), device=device)
            idx_b = torch.randint(0, B, (n_pairs * 2,), device=device)
            valid = (idx_a != idx_b).nonzero(as_tuple=True)[0][:n_pairs]
            if len(valid) >= 2:
                cosim = F.cosine_similarity(
                    emb[idx_a[valid]], emb[idx_b[valid]], dim=-1
                )
                result["sanity/cosim_random_pairs_mean"] = cosim.mean().item()
                result["sanity/cosim_random_pairs_max"] = cosim.abs().max().item()

        return result

    # ------------------------------------------------------------------
    # Training stability: gradient norms
    # ------------------------------------------------------------------

    def _compute_grad_norm(self, jepa) -> dict[str, float]:
        """Combined L2 gradient norm for context encoder + predictor parameters."""
        total_sq = 0.0
        n_with_grad = 0
        for module in (jepa.context_encoder, jepa.predictor):
            for p in module.parameters():
                if p.grad is not None:
                    total_sq += p.grad.data.norm(2).item() ** 2
                    n_with_grad += 1
        if n_with_grad == 0:
            return {}
        return {"sanity/grad_norm": total_sq ** 0.5}

    # ------------------------------------------------------------------
    # Loss health: trend detection
    # ------------------------------------------------------------------

    def _compute_loss_trend(self) -> dict[str, float]:
        """Rolling mean and first-vs-second-half trend of prediction loss."""
        if len(self._loss_history) < 20:
            return {}
        history = list(self._loss_history)
        half = len(history) // 2
        first_half_mean = sum(history[:half]) / half
        second_half_mean = sum(history[half:]) / (len(history) - half)
        return {
            # Negative = loss is decreasing (healthy). Positive = rising (bad).
            "sanity/loss_trend": second_half_mean - first_half_mean,
            "sanity/loss_rolling_mean": sum(history) / len(history),
        }

    # ------------------------------------------------------------------
    # Horizon analysis: short vs long mask prediction loss
    # ------------------------------------------------------------------

    def _compute_horizon_losses(
        self, eeg: torch.Tensor, jepa
    ) -> dict[str, float]:
        """Prediction loss separately for short-range and long-range masks.

        Sanity property: ``pred_loss_long >= pred_loss_short`` — predicting
        representations at farther/larger masked regions should be harder.
        """
        device = eeg.device
        n_short = self.n_pred_masks_short

        try:
            with torch.no_grad():
                mask_result = jepa.mask_collator()
                context_mask = mask_result.context_mask.to(device)
                all_pred_masks = [pm.to(device) for pm in mask_result.pred_masks]

                if not all_pred_masks:
                    return {}

                short_masks = all_pred_masks[:n_short]
                long_masks = all_pred_masks[n_short:]

                _, pos_embed = jepa.context_encoder.tokenize(eeg)
                ctx_tokens = jepa.context_encoder.encode_tokens(
                    eeg, mask=context_mask
                )
                tgt_tokens = jepa.target_encoder.encode_tokens(eeg, mask=None)
                ctx_pos = pos_embed[:, context_mask]

                result: dict[str, float] = {}
                for name, masks in [("short", short_masks), ("long", long_masks)]:
                    if not masks:
                        continue
                    pred_indices = torch.cat(masks).unique()
                    tgt_pos = pos_embed[:, pred_indices]
                    tgt_repr = tgt_tokens[:, pred_indices]
                    preds = jepa.predictor(ctx_tokens, ctx_pos, tgt_pos)
                    result[f"sanity/pred_loss_{name}"] = F.mse_loss(
                        preds, tgt_repr
                    ).item()

                return result
        except Exception as exc:
            logger.debug("Horizon loss computation skipped: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Downstream signal: lightweight linear probe
    # ------------------------------------------------------------------

    def _update_probe_buffer(
        self,
        eeg: torch.Tensor,
        features: torch.Tensor,
        jepa,
        probe_labels: torch.Tensor | None,
    ) -> None:
        """Accumulate embeddings + binary labels into a rolling buffer.

        Label priority:
          1. ``probe_labels`` (subject metadata: age > median, sex) — used for
             samples where the value is not NaN.
          2. Luminance fallback — used when probe_labels are None or all NaN,
             provided ``feature_median`` was supplied at construction time.
        """
        with torch.no_grad():
            tokens = jepa.context_encoder.encode_tokens(eeg, mask=None)
            emb = tokens.mean(dim=1).cpu()  # [B, D]

        B = emb.shape[0]

        # Build per-sample labels and validity mask
        labels = torch.full((B,), float("nan"))
        if probe_labels is not None:
            # probe_labels: [B] float tensor, NaN where metadata is absent
            labels = probe_labels.cpu().float()

        valid = ~torch.isnan(labels)

        # For samples without subject metadata, fall back to luminance label
        if self.feature_median is not None and not valid.all():
            lum_idx = self.LUMINANCE_FEATURE_IDX
            lum_vals = features[:, :, lum_idx].mean(dim=1).cpu()
            lum_labels = (lum_vals > self.feature_median[lum_idx]).float()
            # Fill in fallback only for samples without real labels
            labels[~valid] = lum_labels[~valid]
            valid = torch.ones(B, dtype=torch.bool)  # all samples now have a label

        if not valid.any():
            return  # no labels available at all — skip this batch

        for i in range(B):
            if valid[i]:
                self._emb_buffer.append(emb[i])
                self._label_buffer.append(labels[i])

        # Trim to max buffer size (drop oldest)
        excess = len(self._emb_buffer) - self.probe_buffer_size
        if excess > 0:
            self._emb_buffer = self._emb_buffer[excess:]
            self._label_buffer = self._label_buffer[excess:]

    def _train_and_eval_probe(self, device: torch.device) -> dict[str, float]:
        """Train the linear probe on buffered embeddings, evaluate on held-out split.

        Uses an 80/20 train/val split of the current buffer. Reports validation
        accuracy as ``sanity/linear_probe_acc``. Values near 0.5 suggest the
        embeddings carry no signal for this label; values >0.6 and rising during
        training indicate the representation is becoming semantically meaningful.
        """
        embs = torch.stack(self._emb_buffer)    # [N, D]
        labels = torch.stack(self._label_buffer)  # [N]
        N = len(embs)

        if N < 16:
            return {}

        split = int(0.8 * N)
        perm = torch.randperm(N)
        train_emb = embs[perm[:split]].to(device)
        train_lbl = labels[perm[:split]].to(device)
        val_emb = embs[perm[split:]].to(device)
        val_lbl = labels[perm[split:]].to(device)

        self.linear_probe = self.linear_probe.to(device)
        self.linear_probe.train()
        for _ in range(self.probe_train_steps):
            self.probe_optimizer.zero_grad()
            logits = self.linear_probe(train_emb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, train_lbl)
            loss.backward()
            self.probe_optimizer.step()

        self.linear_probe.eval()
        with torch.no_grad():
            val_logits = self.linear_probe(val_emb).squeeze(-1)
            acc = ((val_logits > 0).float() == val_lbl).float().mean().item()

        return {"sanity/linear_probe_acc": acc}
