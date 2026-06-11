import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.logging import get_logger

logging = get_logger(__name__)


class JEPAbase(nn.Module):
    """Base JEPA class for planning and inference only. Use JEPA subclass for training."""

    def __init__(self, encoder, aencoder, predictor):
        """Initialize JEPAbase with encoder, action encoder, and predictor."""
        super().__init__()
        # Observation Encoder
        self.encoder = encoder
        # Action Encoder
        self.action_encoder = aencoder
        # Predictor
        self.predictor = predictor
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file), weights_only=False)

    @torch.no_grad()
    def encode(self, observations):
        """Encode a sequence of observations and return the encoder output."""
        return self.encoder(observations)


class JEPA(JEPAbase):
    """Trainable JEPA with prediction loss and anti-collapse regularizer."""

    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        """Initialize JEPA with regularizer and prediction cost in addition to base components."""
        super().__init__(encoder, aencoder, predictor)
        self.regularizer = regularizer
        self.predcost = predcost
        self.ploss = 0
        self.rloss = 0

    @torch.no_grad()
    def infer(self, observations, actions):
        """Produce single-step predictions over all sequence elements in parallel."""
        preds, _ = self.unroll(
            observations,
            actions,
            nsteps=1,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )
        return preds[0]

    def unroll(
        self,
        observations,
        actions,
        nsteps=1,
        unroll_mode="parallel",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
    ):
        """Unified multi-step prediction with optional loss computation.

        This function supports both training (with loss computation) and planning/inference
        (without loss, just state prediction).

        Usage examples:
        - Training video_jepa: unroll(x, None, nsteps, unroll_mode="parallel", compute_loss=True)
        - Training ac_video_jepa with RNN: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=1, compute_loss=True)
        - Planning with ac_video_jepa: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=k, compute_loss=False)
        - Inference like infern(): unroll(x, a, nsteps, unroll_mode="parallel",
          compute_loss=False, return_all_steps=True)

        Predictor behavior:
        - unroll_mode="parallel" (Conv predictor, is_rnn=False):
          Processes all timesteps in parallel. Uses predictor.context_length to
          determine how many ground truth frames to re-feed at each iteration.
          Output: [B, D, T, H', W'] (same length as input, predictions replace non-context).
          Best for training with full ground truth trajectory available.

        - unroll_mode="autoregressive":
          Step-by-step prediction with sliding window of ctxt_window_time states.
          Each step: takes last ctxt_window_time states, predicts next, appends to sequence.
          Output: [B, D, T_context + nsteps, H', W'] (context + predictions appended).
          Best for planning/inference where future ground truth is not available.
          Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1.

        Args:
            observations: [B, C, T, H, W] - observation sequence
                For training (compute_loss=True): full trajectory with ground truth
                For planning (compute_loss=False): context frames only
            actions: [B, A, T_actions] - action sequence, or None for state-only prediction
                T_actions >= nsteps required for autoregressive mode
            nsteps: number of prediction steps
            unroll_mode: "parallel" or "autoregressive"
                - "parallel": Process all timesteps, refeed GT context on left
                - "autoregressive": Step-by-step, append predictions on right
            ctxt_window_time: Context window size for autoregressive mode.
                For RNN predictors (is_rnn=True), this is effectively 1.
            compute_loss: Whether to compute losses (requires ground truth observations)
            return_all_steps: If True, return list of predictions at each step (like infern).
                If False, return only the final predicted states.

        Returns:
            Tuple of (predicted_states, losses) where:
            - If return_all_steps=False:
              predicted_states: [B, D, T_out, H', W'] - final predicted state sequence
            - If return_all_steps=True:
              predicted_states: List[Tensor] of length nsteps, each [B, D, T_out, H', W']
            - losses: None if compute_loss=False, otherwise tuple of 5 elements:
              (total_loss, reg_loss, reg_loss_unweighted, reg_loss_dict, pred_loss)
        """
        state = self.encoder(observations)
        context_length = getattr(self.predictor, "context_length", 0)

        # Compute regularization loss if needed
        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        # Encode actions
        if actions is not None:
            actions_encoded = self.action_encoder(actions)
        else:
            actions_encoded = None

        # Collect all steps if requested
        all_steps = [] if return_all_steps else None

        # Parallel mode: process all timesteps at once, refeed GT context
        if unroll_mode == "parallel":
            predicted_states = state
            for _ in range(nsteps):
                # Predict all timesteps, discard last (no target for it)
                predicted_states = self.predictor(predicted_states, actions_encoded)[
                    :, :, :-1
                ]
                # Collect step if requested
                if return_all_steps:
                    all_steps.append(predicted_states)
                # Refeed ground truth context on the left
                predicted_states = torch.cat(
                    (state[:, :, :context_length], predicted_states), dim=2
                )
                if compute_loss:
                    ploss += self.predcost(state, predicted_states) / nsteps

        # Autoregressive mode: step-by-step with sliding window
        # Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1
        elif unroll_mode == "autoregressive":
            if actions is not None and nsteps > actions.size(2):
                raise ValueError(
                    f"nsteps ({nsteps}) larger than action sequence length ({actions.size(2)})"
                )
            # For RNN predictors, force ctxt_window_time=1
            effective_ctxt_window = 1 if self.single_unroll else ctxt_window_time

            predicted_states = state[:, :, :effective_ctxt_window]
            for i in range(nsteps):
                # Take last ctxt_window_time states
                context_states = predicted_states[:, :, -effective_ctxt_window:]
                # Take corresponding actions
                if actions_encoded is not None:
                    context_actions = actions_encoded[
                        :, :, max(0, i + 1 - effective_ctxt_window) : i + 1
                    ]
                else:
                    context_actions = None
                # Predict and take only last timestep
                pred_step = self.predictor(context_states, context_actions)[:, :, -1:]
                # Append prediction to sequence
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)
                # Collect step if requested
                if return_all_steps:
                    all_steps.append(predicted_states.clone())
                if compute_loss:
                    ploss += (
                        self.predcost(pred_step, state[:, :, i + 1 : i + 2]) / nsteps
                    )
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        # Compute total loss and return
        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        # Return all steps or just final state
        if return_all_steps:
            return all_steps, losses
        else:
            return predicted_states, losses


class JEPAProbe(nn.Module):
    """JEPA with a trainable prediction head. The JEPA encoder is kept fixed."""

    def __init__(self, jepa, head, hcost):
        """Initialize with a frozen JEPA, prediction head, and head loss function."""
        super().__init__()
        self.jepa = jepa
        self.head = head
        self.hcost = hcost

    @torch.no_grad()
    def infer(self, observations):
        """Encode observations through JEPA and apply the prediction head."""
        state = self.jepa.encode(observations)
        return self.head(state)

    @torch.no_grad()
    def apply_head(self, embeddings):
        """
        Decode embeddings using the head.
        This is useful for generating predictions from an unrolling of the predictor, for example.
        """
        return self.head(embeddings)

    def forward(self, observations, targets):
        """Forward pass for training the head (JEPA encoder gradients are detached)."""
        with torch.no_grad():
            state = self.jepa.encode(observations)
        output = self.head(state.detach())
        return self.hcost(output, targets)


class MaskedJEPA(nn.Module):
    """Masked-prediction JEPA composed with a pluggable anti-collapse strategy.

    A single online encoder produces both context and target tokens. The
    ``anti_collapse`` argument decides:

    - whether the targets come from an EMA copy of the encoder (DINO) or
      from the online encoder (VICReg, SIGReg),
    - whether gradients flow into the targets (SIGReg) or not (DINO, VICReg),
    - whether an auxiliary loss is added and how it combines with the
      prediction loss.

    Args:
        encoder: EEGEncoderTokens instance (the single trainable encoder).
        predictor: MaskedPredictor instance.
        mask_collator: MultiBlockMaskCollator instance.
        anti_collapse: AntiCollapse strategy (DINOAntiCollapse,
            VICRegAntiCollapse, SIGRegAntiCollapse, or the base no-op).
        pred_loss_type: "mse" or "smooth_l1".
    """

    def __init__(self, encoder, predictor, mask_collator, anti_collapse,
                 pred_loss_type="mse",
                 clip_head=None,
                 clip_loss_weight: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.mask_collator = mask_collator
        self.anti_collapse = anti_collapse
        self.pred_loss_type = pred_loss_type
        self.clip_head = clip_head
        self.clip_loss_weight = clip_loss_weight

    def update_target_encoder(self, momentum: float):
        """Delegate to the anti-collapse strategy (no-op for VICReg/SIGReg)."""
        self.anti_collapse.step(self.encoder, momentum)

    def forward(
        self,
        eeg: torch.Tensor,
        global_step: int = 0,
        *,
        frame_embedding_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass: mask, encode, predict, combine prediction + auxiliary loss.

        Args:
            eeg: [B, T, C, W] raw EEG.
            global_step: training step; threaded to anti-collapse strategies
                that need it (SIGReg seeds its random projection with it).

        Returns:
            (total_loss, loss_dict) where loss_dict contains individual loss components.
        """
        device = eeg.device

        mask_result = self.mask_collator()
        context_mask = mask_result.context_mask.to(device)
        pred_masks = [pm.to(device) for pm in mask_result.pred_masks]

        _, pos_embed = self.encoder.tokenize(eeg)  # [B, C*T*P, D]
        ctx_tokens = self.encoder.encode_tokens(eeg, mask=context_mask)  # [B, n_ctx, D]
        tgt_tokens = self.anti_collapse.target_representations(self.encoder, eeg)

        ctx_pos = pos_embed[:, context_mask]

        if len(pred_masks) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"pred_loss": 0.0, "ac_loss": 0.0}

        all_pred_indices = torch.cat(pred_masks).unique()
        tgt_pos = pos_embed[:, all_pred_indices]
        tgt_representations = tgt_tokens[:, all_pred_indices]

        predictions = self.predictor(ctx_tokens, ctx_pos, tgt_pos)

        # Prediction loss. We do NOT call .detach() here: the anti-collapse
        # strategy already controls grad flow into the target (DINO/VICReg
        # have no graph on target_representations; SIGReg lets gradients
        # flow on purpose).
        if self.pred_loss_type == "smooth_l1":
            pred_loss = F.smooth_l1_loss(predictions, tgt_representations)
        else:
            pred_loss = F.mse_loss(predictions, tgt_representations)

        # Diagnostics — distinguish "predictor learning" from "targets
        # expanding under the anti-collapse loss".
        with torch.no_grad():
            tgt_d = tgt_representations.detach()
            pred_d = predictions.detach()
            pred_target_cosim = F.cosine_similarity(pred_d, tgt_d, dim=-1).mean()
            target_var = tgt_d.reshape(-1, tgt_d.shape[-1]).var(dim=0).mean()
            pred_var = pred_d.reshape(-1, pred_d.shape[-1]).var(dim=0).mean()
            pred_loss_norm = pred_loss.detach() / target_var.clamp_min(1e-8)

        loss_dict = {
            "pred_loss": pred_loss.item(),
            "pred_target_cosim": pred_target_cosim.item(),
            "target_var": target_var.item(),
            "pred_var": pred_var.item(),
            "pred_loss_norm": pred_loss_norm.item(),
        }

        # Pool target tokens to per-window embeddings — the same shape probes
        # consume. SIGRegAntiCollapse acts on this; other strategies ignore it.
        pooled_map = self.encoder.pool_to_windows(tgt_tokens)  # [B, D, T, 1, 1]
        D = pooled_map.shape[1]
        pooled = pooled_map.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(-1, D)

        ac_loss, ac_dict = self.anti_collapse.auxiliary_loss(
            context_tokens=ctx_tokens,
            target_tokens=tgt_tokens,
            pooled=pooled,
            global_step=global_step,
        )

        if self.anti_collapse.combine_mode == "convex":
            lam = self.anti_collapse.coeff
            total_loss = (1.0 - lam) * pred_loss + lam * ac_loss
        else:
            total_loss = pred_loss + ac_loss

        loss_dict["ac_loss"] = ac_loss.item() if torch.is_tensor(ac_loss) else float(ac_loss)
        loss_dict.update(ac_dict)

        # Auxiliary supervised loss: symmetric CLIP InfoNCE between per-window
        # EEG embeddings and V-JEPA-2 mean-pooled frame embeddings. Only computed
        # when the head and target are both present, so disabling via config
        # skips the extra encoder pass entirely.
        if (self.clip_head is not None and frame_embedding_target is not None
                and self.clip_loss_weight > 0):
            # Symmetric InfoNCE: each per-window EEG embedding is paired with
            # its V-JEPA-2 mean-pooled vision vector; all other B*T-1 vision
            # vectors in the batch are negatives.
            online_tokens = self.encoder.encode_tokens(eeg, mask=None)
            online_pooled = self.encoder.pool_to_windows(online_tokens)  # [B, D, T, 1, 1]
            z_eeg = self.clip_head.project_eeg(online_pooled)        # [B*T, P]
            tgt_emb = frame_embedding_target.to(z_eeg.dtype)
            z_vis = self.clip_head.project_vision(
                tgt_emb.reshape(-1, tgt_emb.shape[-1])
            )                                                        # [B*T, P]
            scale = self.clip_head.logit_scale.exp().clamp(max=100.0)
            logits = scale * (z_eeg @ z_vis.T)                       # [B*T, B*T]
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss_e2v = F.cross_entropy(logits, labels)
            loss_v2e = F.cross_entropy(logits.T, labels)
            clip_loss = 0.5 * (loss_e2v + loss_v2e)
            total_loss = total_loss + self.clip_loss_weight * clip_loss
            loss_dict["clip_loss"] = clip_loss.item()
            loss_dict["clip_logit_scale"] = scale.item()
            with torch.no_grad():
                loss_dict["clip_top1_e2v"] = (logits.argmax(-1) == labels).float().mean().item()
                loss_dict["clip_top1_v2e"] = (logits.argmax(0) == labels).float().mean().item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    @torch.no_grad()
    def encode(self, eeg: torch.Tensor,
               keep_channels: bool = False) -> torch.Tensor:
        """Encode EEG without masking (for probes).

        Args:
            eeg: [B, T, C, W]
            keep_channels: if True, ``pool_to_windows`` keeps the CorrCA
                channel axis (concatenated into the feature dim →
                [B, C*D, T, 1, 1]) instead of averaging it.

        Returns:
            [B, D, T, 1, 1] (default) or [B, C*D, T, 1, 1] (keep_channels=True).
        """
        tokens = self.encoder.encode_tokens(eeg, mask=None)
        return self.encoder.pool_to_windows(tokens, keep_channels=keep_channels)


class MaskedJEPAProbe(nn.Module):
    """Probe for MaskedJEPA: trains a head on frozen encoder representations.

    Pass ``keep_channels=True`` to expose per-CorrCA-channel state to the
    probe head (probe input dim grows from D to C*D).
    """

    def __init__(self, masked_jepa, head, hcost, keep_channels: bool = False):
        super().__init__()
        self.masked_jepa = masked_jepa
        self.head = head
        self.hcost = hcost
        self.keep_channels = keep_channels

    def forward(self, eeg, targets):
        with torch.no_grad():
            state = self.masked_jepa.encode(eeg, keep_channels=self.keep_channels)
        output = self.head(state.detach())
        return self.hcost(output, targets)
