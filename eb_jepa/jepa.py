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
    """V-JEPA style masked prediction for EEG.

    - context_encoder: processes only unmasked (context) tokens
    - target_encoder: EMA copy, processes ALL tokens (no masking), frozen
    - predictor: predicts masked token representations from context

    Args:
        context_encoder: EEGEncoderTokens instance (trainable)
        target_encoder: EEGEncoderTokens instance (EMA copy, no gradients)
        predictor: MaskedPredictor instance (trainable)
        mask_collator: MultiBlockMaskCollator instance
        regularizer: VCLoss or similar (optional, for anti-collapse)
    """

    def __init__(self, context_encoder, target_encoder, predictor, mask_collator, regularizer=None,
                 pred_loss_type="mse"):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.mask_collator = mask_collator
        self.regularizer = regularizer
        self.pred_loss_type = pred_loss_type

        # Freeze target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        """EMA update of target encoder from context encoder."""
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.lerp_(p_ctx.data, 1.0 - momentum)

    def forward(self, eeg: torch.Tensor, return_all_tokens: bool = False):
        """Forward pass: mask, encode, predict, compute loss.

        Args:
            eeg: [B, T, C, W] raw EEG
            return_all_tokens: if True, also return full unmasked encoder tokens
                for auxiliary losses (contrastive, adversarial) without a second
                forward pass.

        Returns:
            (total_loss, loss_dict) or (total_loss, loss_dict, all_tokens)
        """
        B = eeg.shape[0]
        device = eeg.device

        # 1. Generate masks
        mask_result = self.mask_collator()
        context_mask = mask_result.context_mask.to(device)
        pred_masks = [pm.to(device) for pm in mask_result.pred_masks]

        # 2. Full unmasked forward pass (used for auxiliary losses and context)
        all_tokens, pos_embed = self.context_encoder.tokenize(eeg)  # [B, C*T*P, D]
        all_tokens = self.context_encoder.transformer(all_tokens)  # [B, C*T*P, D]

        # 3. Context tokens: select unmasked positions from full pass
        ctx_tokens = all_tokens[:, context_mask]  # [B, n_ctx, D]

        # 4. Target encoding: all tokens (no masking), no gradients
        with torch.no_grad():
            tgt_tokens = self.target_encoder.encode_tokens(eeg, mask=None)  # [B, C*T*P, D]

        # 5. Gather positional embeddings for context and prediction targets
        ctx_pos = pos_embed[:, context_mask]  # [B, n_ctx, D]

        # Concatenate all prediction mask indices and their targets
        if len(pred_masks) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            result = (zero, {"pred_loss": 0.0, "reg_loss": 0.0})
            return (*result, all_tokens) if return_all_tokens else result

        all_pred_indices = torch.cat(pred_masks).unique()
        tgt_pos = pos_embed[:, all_pred_indices]  # [B, n_pred, D]
        tgt_representations = tgt_tokens[:, all_pred_indices]  # [B, n_pred, D]

        # 6. Predictor: predict representations at masked positions
        predictions = self.predictor(ctx_tokens, ctx_pos, tgt_pos)  # [B, n_pred, D]

        # 7. Prediction loss between predictions and target representations
        if self.pred_loss_type == "smooth_l1":
            pred_loss = F.smooth_l1_loss(predictions, tgt_representations.detach())
        else:
            pred_loss = F.mse_loss(predictions, tgt_representations.detach())

        # 8. Regularizer loss (optional, on context representations)
        loss_dict = {"pred_loss": pred_loss.item()}
        reg_loss = torch.tensor(0.0, device=device)
        if self.regularizer is not None:
            from eb_jepa.losses import SIGRegLoss
            if isinstance(self.regularizer, SIGRegLoss):
                ctx_pooled = ctx_tokens.mean(dim=1)
                reg_loss, reg_loss_unweighted, reg_dict = self.regularizer(ctx_pooled)
            else:
                ctx_for_reg = ctx_tokens.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
                reg_loss, reg_loss_unweighted, reg_dict = self.regularizer(ctx_for_reg)
            loss_dict["reg_loss"] = reg_loss.item()
            loss_dict.update(reg_dict)

        total_loss = pred_loss + reg_loss
        loss_dict["total_loss"] = total_loss.item()

        if return_all_tokens:
            return total_loss, loss_dict, all_tokens
        return total_loss, loss_dict

    @torch.no_grad()
    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        """Encode EEG without masking (for probes).

        Args:
            eeg: [B, T, C, W]

        Returns:
            [B, D, T, 1, 1] pooled per-window representations
        """
        tokens = self.context_encoder.encode_tokens(eeg, mask=None)
        return self.context_encoder.pool_to_windows(tokens)


class MaskedJEPANoEMA(nn.Module):
    """LeWorldModel-style masked prediction for EEG (no EMA target encoder).

    Unlike MaskedJEPA, uses a single encoder for both context and target.
    Gradients flow through both branches. SIGReg prevents collapse instead of EMA.

    Reference: LeWorldModel (arXiv:2603.19312)
    """

    def __init__(self, encoder, predictor, mask_collator, regularizer,
                 pred_loss_type="smooth_l1"):
        super().__init__()
        self.context_encoder = encoder  # single encoder, used for both context & target
        self.predictor = predictor
        self.mask_collator = mask_collator
        self.regularizer = regularizer
        self.pred_loss_type = pred_loss_type

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B = eeg.shape[0]
        device = eeg.device

        # 1. Generate masks
        mask_result = self.mask_collator()
        context_mask = mask_result.context_mask.to(device)
        pred_masks = [pm.to(device) for pm in mask_result.pred_masks]

        # 2. Get positional embeddings
        _, pos_embed = self.context_encoder.tokenize(eeg)

        # 3. Full encoding (ALL tokens, WITH gradients — no EMA, no detach)
        all_tokens = self.context_encoder.encode_tokens(eeg, mask=None)  # [B, C*T*P, D]

        # 4. Context encoding (unmasked tokens only)
        ctx_tokens = self.context_encoder.encode_tokens(eeg, mask=context_mask)  # [B, n_ctx, D]

        # 5. Gather positions and targets
        ctx_pos = pos_embed[:, context_mask]

        if len(pred_masks) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"pred_loss": 0.0, "reg_loss": 0.0}

        all_pred_indices = torch.cat(pred_masks).unique()
        tgt_pos = pos_embed[:, all_pred_indices]
        tgt_representations = all_tokens[:, all_pred_indices]  # gradients flow!

        # 6. Predictor
        predictions = self.predictor(ctx_tokens, ctx_pos, tgt_pos)

        # 7. Prediction loss (NO .detach() — gradients flow through target)
        if self.pred_loss_type == "smooth_l1":
            pred_loss = F.smooth_l1_loss(predictions, tgt_representations)
        else:
            pred_loss = F.mse_loss(predictions, tgt_representations)

        # 8. SIGReg on mean-pooled encoder embeddings [B, D] to avoid OOM
        loss_dict = {"pred_loss": pred_loss.item()}
        reg_loss = torch.tensor(0.0, device=device)
        if self.regularizer is not None:
            from eb_jepa.losses import SIGRegLoss
            if isinstance(self.regularizer, SIGRegLoss):
                all_pooled = all_tokens.mean(dim=1)  # [B, D]
                reg_loss, _, reg_dict = self.regularizer(all_pooled)
            else:
                ctx_for_reg = ctx_tokens.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
                reg_loss, _, reg_dict = self.regularizer(ctx_for_reg)
            loss_dict["reg_loss"] = reg_loss.item()
            loss_dict.update(reg_dict)

        total_loss = pred_loss + reg_loss
        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

    @torch.no_grad()
    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        tokens = self.context_encoder.encode_tokens(eeg, mask=None)
        return self.context_encoder.pool_to_windows(tokens)

    def update_target_encoder(self, momentum: float):
        """No-op — no EMA in this architecture."""
        pass


class MaskedJEPAProbe(nn.Module):
    """Probe for MaskedJEPA: trains a head on frozen encoder representations.

    Similar to JEPAProbe but works with MaskedJEPA's token-based encoder.
    """

    def __init__(self, masked_jepa, head, hcost):
        super().__init__()
        self.masked_jepa = masked_jepa
        self.head = head
        self.hcost = hcost

    def forward(self, eeg, targets):
        """Forward pass: encode with frozen MaskedJEPA, apply head, compute loss."""
        with torch.no_grad():
            # Encode without masking, pool to [B, D, T, 1, 1]
            state = self.masked_jepa.encode(eeg)
        output = self.head(state.detach())
        return self.hcost(output, targets)
