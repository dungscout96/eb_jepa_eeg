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
                 pred_loss_type="mse"):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.mask_collator = mask_collator
        self.anti_collapse = anti_collapse
        self.pred_loss_type = pred_loss_type

    def update_target_encoder(self, momentum: float):
        """Delegate to the anti-collapse strategy (no-op for VICReg/SIGReg)."""
        self.anti_collapse.step(self.encoder, momentum)

    def _pred_criterion(self, pred, target):
        """Prediction loss named by ``pred_loss_type``.

        ``l1`` exists for MJEPA fidelity (arXiv:2606.25225 uses an L1 latent
        prediction loss); ``mse`` and ``smooth_l1`` predate it, and ``mse``
        stays the fallback so existing configs are unaffected.
        """
        if self.pred_loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        if self.pred_loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _combine(self, pred_loss, ac_loss):
        """Combine prediction and auxiliary loss per the strategy's combine_mode.

        ``convex`` is LeJEPA's ``(1-λ)·pred + λ·aux``; ``additive_weighted`` is
        Laya's ``pred + λ·aux``; anything else is a plain sum. Shared with
        ``CrossSubjectJEPA`` so the three modes are defined in one place.
        """
        if self.anti_collapse.combine_mode == "convex":
            lam = self.anti_collapse.coeff
            return (1.0 - lam) * pred_loss + lam * ac_loss
        if self.anti_collapse.combine_mode == "additive_weighted":
            # Laya-style: pred + λ · sigreg (λ from anti_collapse.coeff)
            return pred_loss + self.anti_collapse.coeff * ac_loss
        return pred_loss + ac_loss

    def forward(self, eeg: torch.Tensor, global_step: int = 0) -> tuple[torch.Tensor, dict]:
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
        pred_loss = self._pred_criterion(predictions, tgt_representations)

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

        total_loss = self._combine(pred_loss, ac_loss)

        loss_dict["ac_loss"] = ac_loss.item() if torch.is_tensor(ac_loss) else float(ac_loss)
        loss_dict.update(ac_dict)
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


class CrossSubjectJEPA(MaskedJEPA):
    """Predict subject B's target tokens from subject A's context at the same movie time.

    Within-subject masked prediction is rationally solved by modelling the
    subject fingerprint, which dominates HBN movie EEG (~30 uV against a ~3 uV
    stimulus response). This objective removes that solution: because the
    partner B is drawn independently of A,

        E[z_B | context_A] = E[shared_stimulus_response(t) | context_A]

    so the fingerprint is marginalized out and the Bayes-optimal prediction is
    the shared stimulus response.

    Implementation note — the pair axis is flattened rather than looped. Rows
    ``0..B-1`` hold subject A and rows ``B..2B-1`` hold subject B, so the whole
    forward is ``MaskedJEPA``'s at batch ``2B``, and
    ``torch.roll(targets, shifts=B, dims=0)`` swaps the halves: row ``r < B``
    predicts A(r)->B(r) and row ``r >= B`` predicts B(r-B)->A(r-B). One
    predictor call, exactly symmetric, shared weights. Nothing in the encoder
    mixes across the batch axis, so this is numerically identical to two
    separate passes.

    Subclasses ``MaskedJEPA`` deliberately: submodule names stay
    ``encoder.* / predictor.* / anti_collapse.*``, so ``state_dict()`` is
    layout-compatible and the probe evaluators (which load only ``encoder.*``)
    need no changes.

    Args:
        symmetric: train both directions. False trains A->B only (half the
            supervision for the same compute; diagnostic use).
        context_mask_mode: ``"masked"`` (default) applies the mask collator to
            A's context, matching the masked-JEPA baseline exactly.
            ``"full"`` gives the predictor A's complete token set, which turns
            the objective into a pure learned shared-response regression and
            saves one encoder pass — but makes a near-identity map plus an
            "average subject" correction a viable shortcut, since volume
            conduction leaves neighbouring channels highly correlated.
        pred_target_mode: ``"masked"`` (default) supervises only the masked
            positions; ``"all"`` supervises every token position, decoupling
            "context is partial" from "supervision is sparse".
        within_subject_weight: blend in own-subject prediction. Kept at 0.0 —
            this exists only as the documented rescue lever if the
            cross-subject term collapses (see ``pred_loss_gap`` below).
        diagnostic_every: reserved for gating any future expensive diagnostic.
            The diagnostics currently logged are elementwise ops on resident
            tensors, so they run unconditionally.
    """

    def __init__(self, encoder, predictor, mask_collator, anti_collapse,
                 pred_loss_type="mse", *,
                 symmetric: bool = True,
                 context_mask_mode: str = "masked",
                 pred_target_mode: str = "masked",
                 within_subject_weight: float = 0.0,
                 diagnostic_every: int = 50):
        super().__init__(encoder, predictor, mask_collator, anti_collapse,
                         pred_loss_type=pred_loss_type)
        if context_mask_mode not in ("masked", "full"):
            raise ValueError(
                f"context_mask_mode must be 'masked' or 'full', got {context_mask_mode!r}"
            )
        if pred_target_mode not in ("masked", "all"):
            raise ValueError(
                f"pred_target_mode must be 'masked' or 'all', got {pred_target_mode!r}"
            )
        if not 0.0 <= within_subject_weight <= 1.0:
            raise ValueError(
                f"within_subject_weight must be in [0, 1], got {within_subject_weight}"
            )
        self.symmetric = symmetric
        self.context_mask_mode = context_mask_mode
        self.pred_target_mode = pred_target_mode
        self.within_subject_weight = within_subject_weight
        self.diagnostic_every = diagnostic_every

    #: Alias kept so the diagnostics below read as "criterion"; the dispatch
    #: (including the ``l1`` branch) lives once on MaskedJEPA.
    _criterion = MaskedJEPA._pred_criterion

    def forward(self, eeg_pair: torch.Tensor,
                global_step: int = 0) -> tuple[torch.Tensor, dict]:
        """Cross-subject forward pass.

        Args:
            eeg_pair: [B, 2, T, C, W] — ``[:, 0]`` is subject A, ``[:, 1]`` is
                subject B, both at the same movie time.
            global_step: threaded to anti-collapse strategies that need it.

        Returns:
            (total_loss, loss_dict).
        """
        if eeg_pair.ndim != 5 or eeg_pair.shape[1] != 2:
            raise ValueError(
                f"CrossSubjectJEPA expects eeg of shape [B, 2, T, C, W], got "
                f"{tuple(eeg_pair.shape)}. Did the dataset return a "
                "single-subject batch?"
            )
        device = eeg_pair.device
        B = eeg_pair.shape[0]
        # Rows 0..B-1 = subject A, rows B..2B-1 = subject B.
        eeg = torch.cat([eeg_pair[:, 0], eeg_pair[:, 1]], dim=0)  # [2B, T, C, W]

        mask_result = self.mask_collator()
        context_mask = mask_result.context_mask.to(device)
        pred_masks = [pm.to(device) for pm in mask_result.pred_masks]
        if len(pred_masks) == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"pred_loss": 0.0, "ac_loss": 0.0}

        _, pos_embed = self.encoder.tokenize(eeg)  # [2B, C*T*P, D]

        # Full target token set for both subjects. The anti-collapse strategy
        # owns the grad policy (EMA / stop-grad / gradients-flow).
        tgt_tokens = self.anti_collapse.target_representations(self.encoder, eeg)

        if self.context_mask_mode == "full":
            # Reuse the single full encoder pass instead of a second masked one.
            ctx_tokens = tgt_tokens
            ctx_pos = pos_embed
        else:
            ctx_tokens = self.encoder.encode_tokens(eeg, mask=context_mask)
            ctx_pos = pos_embed[:, context_mask]

        if self.pred_target_mode == "all":
            pred_indices = torch.arange(pos_embed.shape[1], device=device)
        else:
            pred_indices = torch.cat(pred_masks).unique()
        tgt_pos = pos_embed[:, pred_indices]

        predictions = self.predictor(ctx_tokens, ctx_pos, tgt_pos)  # [2B, n_pred, D]

        tgt_same = tgt_tokens[:, pred_indices]
        # The swap: rows <B get subject B's targets, rows >=B get subject A's.
        tgt_cross = torch.roll(tgt_same, shifts=B, dims=0)

        if self.symmetric:
            pred_loss = self._criterion(predictions, tgt_cross)
        else:
            pred_loss = self._criterion(predictions[:B], tgt_cross[:B])

        if self.within_subject_weight > 0.0:
            w = self.within_subject_weight
            pred_loss = (1.0 - w) * pred_loss + w * self._criterion(predictions, tgt_same)

        # Diagnostics. The two that matter are pred_loss_gap and pred_var_ratio:
        # SIGReg constrains only the marginal distribution of the embeddings, so
        # "targets spread isotropically while predictions go constant" is a valid
        # optimum that ac_loss / target_var cannot see. These can.
        with torch.no_grad():
            pred_d = predictions.detach()
            cross_d = tgt_cross.detach()
            same_d = tgt_same.detach()
            # Different subject AND different movie time.
            shuffled_d = torch.roll(cross_d, shifts=1, dims=0)

            cosim = F.cosine_similarity(pred_d, cross_d, dim=-1).mean()
            cosim_within = F.cosine_similarity(pred_d, same_d, dim=-1).mean()
            cosim_shuffled = F.cosine_similarity(pred_d, shuffled_d, dim=-1).mean()

            pred_loss_shuffled = self._criterion(pred_d, shuffled_d)
            target_var = cross_d.reshape(-1, cross_d.shape[-1]).var(dim=0).mean()
            pred_var = pred_d.reshape(-1, pred_d.shape[-1]).var(dim=0).mean()

            loss_dict = {
                "pred_loss": pred_loss.item(),
                "pred_target_cosim": cosim.item(),
                "cosim_within": cosim_within.item(),
                "cosim_shuffled": cosim_shuffled.item(),
                # Stimulus-specific information: -> 0 means the predictor has
                # collapsed to the stimulus-independent mean.
                "align_gap": (cosim - cosim_shuffled).item(),
                "pred_loss_shuffled": pred_loss_shuffled.item(),
                "pred_loss_gap": (pred_loss_shuffled - pred_loss.detach()).item(),
                "target_var": target_var.item(),
                "pred_var": pred_var.item(),
                # -> 0 means the predictor emits a constant.
                "pred_var_ratio": (pred_var / target_var.clamp_min(1e-8)).item(),
                # Across-sample variance per position. The flattened target_var
                # above conflates this with across-position variance, and under a
                # cross-subject objective that distinction is the whole game.
                "target_var_across_batch": cross_d.var(dim=0).mean().item(),
                "target_var_across_pos": cross_d.var(dim=1).mean().item(),
                "pred_loss_norm": (pred_loss.detach() / target_var.clamp_min(1e-8)).item(),
                "n_ctx": int(ctx_tokens.shape[1]),
                "n_pred": int(pred_indices.numel()),
            }

        # Anti-collapse sees both subjects' pooled targets, doubling the sample
        # count SIGReg's Cramer-Wold test gets for free.
        pooled_map = self.encoder.pool_to_windows(tgt_tokens)  # [2B, D, T, 1, 1]
        D = pooled_map.shape[1]
        pooled = pooled_map.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(-1, D)

        ac_loss, ac_dict = self.anti_collapse.auxiliary_loss(
            context_tokens=ctx_tokens,
            target_tokens=tgt_tokens,
            pooled=pooled,
            global_step=global_step,
        )

        total_loss = self._combine(pred_loss, ac_loss)

        loss_dict["ac_loss"] = ac_loss.item() if torch.is_tensor(ac_loss) else float(ac_loss)
        loss_dict.update(ac_dict)
        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict


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
