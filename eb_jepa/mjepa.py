"""MJEPA-adapted: joint intra-modal + cross-modal L1 prediction for EEG.

Adapts MJEPA (Teotia et al., arXiv:2606.25225) to the EEG <-> movie setting.
MJEPA uses one shared encoder and one predictive objective applied both within
and across modalities, and its headline ablation is that a shared encoder
*without* cross-modal prediction degrades below unimodal baselines.

WHY THIS OBJECTIVE, HERE. Every cross-modal loss in this repo is contrastive
(``eb_jepa/clip.py`` is entirely ``F.cross_entropy`` / ``logsumexp``), and
``experiments/snr_scaling/PLAN.md`` diagnoses the consequence: ThePresent is
203.3 s = only 101 distinct 2 s anchors, so "the InfoNCE problem has ~101
classes… almost certainly why every objective saturates at the same place"
(14 contrastive configs all land in delta-r^2 [+0.041, +0.043]). MJEPA's
cross-modal term is an **L1 regression to a continuous 1408-d target**, which
has no class-count ceiling. That is the hypothesis this module exists to test.

SCOPE (deliberate deviation from the paper). V-JEPA-2 stays frozen precomputed
data rather than a second stream through the shared encoder, so there is no
movie tokenizer, no modality embedding, and no change to ``tokenize`` /
``pool_to_windows`` / masking. Three loss terms instead of the paper's nine:

    L_e2e = || predictor(ctx, ctx_pos, tgt_pos) - sg(tgt_tokens) ||   (MaskedJEPA)
    L_e2v = || mlp_ev(pool(z_eeg)) - sg(standardize(vision)) ||_1
    L_v2e = || mlp_ve(vision)      - sg(standardize(pool(z_eeg))) ||_1

    total = lambda_intra * L_e2e (+ anti_collapse) + L_e2v + L_v2e

CONSEQUENCE OF THAT SCOPE, STATED PLAINLY: **L_v2e delivers no gradient to the
encoder.** In MJEPA the video stream flows through the shared encoder, so v->e
trains it; here the movie side is frozen data and the EEG target is stop-grad,
so this term trains only ``mlp_ve``. It is kept because it is a free and
unambiguous collapse detector -- if ``pool(z_eeg)`` collapses, ``ve_loss`` falls
while ``pooled_var_raw`` goes to zero, a joint signature nothing else produces.
Set ``ve_stopgrad=False`` to let gradients through, but note that turns v->e
into a BYOL-style objective whose global optimum is *both sides constant*, so it
then requires a real anti-collapse term.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.architectures import CrossModalMLP
from eb_jepa.jepa import MaskedJEPA


class TargetStandardizer(nn.Module):
    """Per-dim running standardization of an already-detached target.

    The two cross-modal targets live on wildly different scales, measured on
    this data:

    ==========================  ===================  ==============
    quantity                    V-JEPA-2 (centered)  pooled EEG
    ==========================  ===================  ==============
    per-element mean|x|         0.547                0.996
    L1 to the per-dim median    0.530                0.104
    ==========================  ===================  ==============

    90% of the pooled-EEG L1 mass is a *constant offset*, so untreated,
    ``mlp_ve`` fits that offset within a few dozen steps and parks at ~0.10
    while ``L_e2v`` sits at ~0.53 -- a 5x imbalance inside a sum that MJEPA
    intends to be unweighted. Worse, ``EEGEncoderTokens.encode_tokens`` returns
    the raw pre-norm residual stream with no final LayerNorm, so that scale is a
    free parameter that drifts during training.

    Standardizing both targets fixes the imbalance and pins each loss to a known
    "predict the mean" floor, which is what makes a converged loss value
    interpretable at all (see ``PREDICT_MEAN_FLOOR`` below).

    Buffers, not parameters: they checkpoint with the model so eval is
    reproducible, and ``eval()`` freezes the running statistics. This is
    BatchNorm's statistics without its learnable affine -- and critically
    without a gradient path, since the input is already detached.
    """

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[N, dim] -> [N, dim], standardized.

        The statistics update runs under ``no_grad``, and the returned value is
        an affine function of ``x`` using buffers -- so gradients flow through
        ``x`` iff the caller passed a tensor that carries them. That lets the
        same module serve both the stop-grad targets and the ``ve_stopgrad=False``
        ablation without a straight-through hack.
        """
        if self.training and x.shape[0] > 1:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                if self.n_updates == 0:
                    # Initialize from the first batch instead of crawling from
                    # (0, 1); at momentum=0.01 that would take ~500 steps and
                    # make early loss values uninterpretable.
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                else:
                    m = self.momentum
                    self.running_mean.lerp_(batch_mean, m)
                    self.running_var.lerp_(batch_var, m)
                self.n_updates += 1
        return (x - self.running_mean) / (self.running_var + self.eps).sqrt()


#: L1 loss achieved by predicting the per-dim mean of a standardized target.
#: For a standard normal, E|x - 0| = sqrt(2/pi) ~= 0.798. Measured on
#: ThePresent's mean-centered V-JEPA-2 shot means the value is 0.711 (the target
#: is heavier-tailed than Gaussian). **A converged ``ev_loss`` at its floor means
#: the model learned nothing** -- this is the pass/fail line for every run.
PREDICT_MEAN_FLOOR_GAUSSIAN = 0.7979


class MJEPA(MaskedJEPA):
    """Joint intra-modal masked prediction + cross-modal L1 regression.

    Subclasses ``MaskedJEPA`` deliberately, exactly as ``CrossSubjectJEPA``
    does: submodule names stay ``encoder.* / predictor.* / anti_collapse.*``, so
    ``state_dict()`` remains loadable by the probe evaluators, which filter
    ``encoder.*`` into a bare ``EEGEncoderTokens`` and never construct a model.
    The new ``mlp_ev`` / ``mlp_ve`` / ``tgt_norm_*`` keys are simply dropped by
    that filter.

    Args:
        encoder, predictor, mask_collator, anti_collapse, pred_loss_type:
            as ``MaskedJEPA``.
        vision_dim: dim of the frozen movie embedding (V-JEPA-2 = 1408).
        lambda_intra: weight on the intra-modal masked term. 1.0 is MJEPA's
            unweighted sum; **0.0 is the clean test** -- pure cross-modal
            regression, with masking/predictor/target-encoder skipped entirely.
        cross_loss_type: ``l1`` (MJEPA) or ``smooth_l1``.
        predictor_hidden / predictor_depth: the cross-modal MLPs.
        cross_source: where the gradient-carrying pooled EEG comes from.
            ``context`` pools the online masked pass (MJEPA's ``z^m_vis``, free
            when lambda>0); ``full`` runs one unmasked pass (the tensor the probe
            reads); ``auto`` picks ``full`` at lambda=0 and ``context`` above it.
            **Never sourced from ``anti_collapse.target_representations``** --
            three of the four strategies wrap that in ``no_grad``, which would
            silently zero the cross-modal gradient.
        ve_enabled: include ``L_v2e``.
        ve_stopgrad: stop-grad the EEG target of ``L_v2e`` (see module docstring).
        standardize_targets: apply ``TargetStandardizer`` to both targets.
    """

    def __init__(self, encoder, predictor, mask_collator, anti_collapse,
                 pred_loss_type="mse", *,
                 vision_dim: int,
                 lambda_intra: float = 1.0,
                 cross_loss_type: str = "l1",
                 predictor_hidden: int = 2048,
                 predictor_depth: int = 3,
                 cross_source: str = "auto",
                 ve_enabled: bool = True,
                 ve_stopgrad: bool = True,
                 standardize_targets: bool = True,
                 standardizer_momentum: float = 0.01,
                 diagnostic_every: int = 1):
        super().__init__(encoder, predictor, mask_collator, anti_collapse,
                         pred_loss_type=pred_loss_type)
        if vision_dim <= 0:
            raise ValueError(
                f"vision_dim must be > 0, got {vision_dim}. The trainer must pass "
                "train_set.frame_embedding_dim (1408 for V-JEPA-2) -- a 0 here "
                "usually means the dataset was built without recipe_mode."
            )
        if lambda_intra < 0.0:
            raise ValueError(f"lambda_intra must be >= 0, got {lambda_intra}")
        if cross_loss_type not in ("l1", "smooth_l1"):
            raise ValueError(
                f"cross_loss_type must be 'l1' or 'smooth_l1', got {cross_loss_type!r}"
            )
        if cross_source not in ("auto", "context", "full"):
            raise ValueError(
                f"cross_source must be 'auto', 'context' or 'full', got {cross_source!r}"
            )

        self.vision_dim = vision_dim
        self.lambda_intra = float(lambda_intra)
        self.cross_loss_type = cross_loss_type
        self.ve_enabled = ve_enabled
        self.ve_stopgrad = ve_stopgrad
        self.standardize_targets = standardize_targets
        self.diagnostic_every = diagnostic_every

        if cross_source == "auto":
            cross_source = "full" if self.lambda_intra == 0.0 else "context"
        self.cross_source = cross_source

        embed_dim = encoder.embed_dim
        self.embed_dim = embed_dim

        self.mlp_ev = CrossModalMLP(embed_dim, predictor_hidden, vision_dim,
                                    depth=predictor_depth)
        self.mlp_ve = (
            CrossModalMLP(vision_dim, predictor_hidden, embed_dim,
                          depth=predictor_depth)
            if ve_enabled else None
        )
        self.tgt_norm_v = TargetStandardizer(vision_dim, standardizer_momentum)
        self.tgt_norm_e = TargetStandardizer(embed_dim, standardizer_momentum)

    # -- helpers ------------------------------------------------------------

    def _cross_criterion(self, pred, target):
        if self.cross_loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        return F.l1_loss(pred, target)

    def _combine_scaled(self, pred_loss, ac_loss):
        """``MaskedJEPA._combine`` with lambda applied to the PREDICTION term only.

        Scaling the combined value instead (``lam * _combine(...)``) would make
        the SIGReg strength itself a function of lambda, confounding the sweep
        that this whole experiment turns on.
        """
        lam = self.lambda_intra
        mode = self.anti_collapse.combine_mode
        if mode == "convex":
            c = self.anti_collapse.coeff
            return (1.0 - c) * lam * pred_loss + c * ac_loss
        if mode == "additive_weighted":
            return lam * pred_loss + self.anti_collapse.coeff * ac_loss
        return lam * pred_loss + ac_loss

    @staticmethod
    def _flatten_windows(pooled: torch.Tensor) -> torch.Tensor:
        """[B, D, T, 1, 1] -> [B*T, D] in (b, t) row-major order.

        Must match ``vision.reshape(-1, V)`` exactly. This is the same idiom as
        ``MovieCLIPHead.project_eeg``; a silent transpose here produces a
        plausible loss curve and a dead model, so it is pinned by a test.
        """
        B, D, T = pooled.shape[0], pooled.shape[1], pooled.shape[2]
        return pooled.view(B, D, T).permute(0, 2, 1).reshape(B * T, D)

    def _pool_context(self, ctx_tokens, context_mask, n_windows):
        """Mean-pool masked context tokens per window -> [B, D, T, 1, 1].

        Token flattening is (c, t, p) with ``idx = c*(T*P) + t*P + p`` (see
        ``masking.py``), so the window of a token is ``(idx % (T*P)) // P``.
        Only the visible (context) tokens are present, so this is a scatter-mean
        over an irregular subset rather than a reshape.
        """
        B, _, D = ctx_tokens.shape
        P = self.encoder.n_patches_per_window
        T = n_windows
        if T == 1:
            return ctx_tokens.mean(dim=1).view(B, D, 1, 1, 1)
        vis_idx = torch.nonzero(context_mask, as_tuple=False).squeeze(-1)
        win = (vis_idx % (T * P)) // P                            # [n_ctx]
        out = ctx_tokens.new_zeros(B, T, D)
        cnt = ctx_tokens.new_zeros(B, T, 1)
        out.index_add_(1, win, ctx_tokens)
        cnt.index_add_(1, win, torch.ones_like(ctx_tokens[..., :1]))
        pooled = out / cnt.clamp_min(1.0)                         # [B, T, D]
        return pooled.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

    # -- forward ------------------------------------------------------------

    def forward(self, eeg: torch.Tensor, vision: torch.Tensor,
                global_step: int = 0) -> tuple[torch.Tensor, dict]:
        """Joint intra + cross-modal forward.

        Args:
            eeg: [B, T, C, W] raw EEG.
            vision: [B, T, vision_dim] frozen V-JEPA-2 embeddings.
            global_step: threaded to anti-collapse strategies that need it.

        Returns:
            (total_loss, loss_dict).
        """
        if eeg.ndim != 4:
            raise ValueError(f"eeg must be [B, T, C, W], got {tuple(eeg.shape)}")
        if vision.ndim != 3 or vision.shape[-1] != self.vision_dim:
            raise ValueError(
                f"vision must be [B, T, {self.vision_dim}], got {tuple(vision.shape)}"
            )
        if vision.shape[:2] != eeg.shape[:2]:
            raise ValueError(
                f"eeg and vision must agree on (B, T): {tuple(eeg.shape[:2])} vs "
                f"{tuple(vision.shape[:2])}"
            )
        device = eeg.device
        n_windows = eeg.shape[1]
        loss_dict: dict = {}
        lam = self.lambda_intra

        pred_loss = torch.zeros((), device=device)
        ac_loss = torch.zeros((), device=device)
        z_pool_map = None

        if lam > 0.0:
            # ---- intra-modal: verbatim MaskedJEPA computation --------------
            mask_result = self.mask_collator()
            context_mask = mask_result.context_mask.to(device)
            pred_masks = [pm.to(device) for pm in mask_result.pred_masks]
            if len(pred_masks) == 0:
                zero = torch.tensor(0.0, device=device, requires_grad=True)
                return zero, {"pred_loss": 0.0, "ac_loss": 0.0}

            _, pos_embed = self.encoder.tokenize(eeg)
            ctx_tokens = self.encoder.encode_tokens(eeg, mask=context_mask)
            tgt_tokens = self.anti_collapse.target_representations(self.encoder, eeg)

            all_pred_indices = torch.cat(pred_masks).unique()
            predictions = self.predictor(
                ctx_tokens, pos_embed[:, context_mask], pos_embed[:, all_pred_indices]
            )
            tgt_representations = tgt_tokens[:, all_pred_indices]
            pred_loss = self._pred_criterion(predictions, tgt_representations)

            with torch.no_grad():
                tgt_d = tgt_representations.detach()
                pred_d = predictions.detach()
                tv = tgt_d.reshape(-1, tgt_d.shape[-1]).var(dim=0).mean()
                loss_dict.update({
                    "pred_loss": pred_loss.item(),
                    "pred_target_cosim": F.cosine_similarity(pred_d, tgt_d, dim=-1).mean().item(),
                    "target_var": tv.item(),
                    "pred_var": pred_d.reshape(-1, pred_d.shape[-1]).var(dim=0).mean().item(),
                    "pred_loss_norm": (pred_loss.detach() / tv.clamp_min(1e-8)).item(),
                    "n_ctx": int(ctx_tokens.shape[1]),
                    "n_pred": int(all_pred_indices.numel()),
                })

            # Anti-collapse acts on the pooled TARGET tokens, as in MaskedJEPA.
            pooled_map = self.encoder.pool_to_windows(tgt_tokens)
            D = pooled_map.shape[1]
            pooled = pooled_map.squeeze(-1).squeeze(-1).permute(0, 2, 1).reshape(-1, D)
            ac_loss, ac_dict = self.anti_collapse.auxiliary_loss(
                context_tokens=ctx_tokens, target_tokens=tgt_tokens,
                pooled=pooled, global_step=global_step,
            )
            loss_dict.update(ac_dict)
            loss_dict["ac_loss"] = (
                ac_loss.item() if torch.is_tensor(ac_loss) else float(ac_loss)
            )

            if self.cross_source == "context":
                z_pool_map = self._pool_context(ctx_tokens, context_mask, n_windows)

        if z_pool_map is None:
            # cross_source == "full", or lambda == 0 (no masked pass happened).
            tokens = self.encoder.encode_tokens(eeg, mask=None)
            z_pool_map = self.encoder.pool_to_windows(tokens)

        z_pool = self._flatten_windows(z_pool_map)              # [B*T, D]
        v_flat = vision.reshape(-1, self.vision_dim).to(z_pool.dtype)

        # ---- L_e2v : the hypothesis ---------------------------------------
        v_target = v_flat.detach()
        if self.standardize_targets:
            v_target = self.tgt_norm_v(v_target)
        ev_pred = self.mlp_ev(z_pool)
        ev_loss = self._cross_criterion(ev_pred, v_target)

        # ---- L_v2e : trains mlp_ve only (see module docstring) -------------
        ve_loss = torch.zeros((), device=device)
        if self.ve_enabled:
            e_target = z_pool.detach() if self.ve_stopgrad else z_pool
            if self.standardize_targets:
                e_target = self.tgt_norm_e(e_target)
            ve_loss = self._cross_criterion(self.mlp_ve(v_flat), e_target)

        total_loss = self._combine_scaled(pred_loss, ac_loss) + ev_loss + ve_loss

        # ---- collapse diagnostics -----------------------------------------
        # ev_gap is the direct analogue of CrossSubjectJEPA's pred_loss_gap and
        # is the only trustworthy signal: it must be > 0 and growing. A model
        # that has learned nothing scores the same on matched and mismatched
        # targets, so the gap stays at 0 while ev_loss sits at its floor.
        with torch.no_grad():
            ev_shuffled = self._cross_criterion(
                ev_pred.detach(), torch.roll(v_target.detach(), 1, dims=0)
            )
            # Participation ratio of the pooled covariance: ~= the number of
            # effective dimensions. -> 1 is rank collapse (all variance in one
            # direction). Degenerate when the representation is EXACTLY constant
            # (cov is all zeros, so this is 0/0); report 0.0 there and let
            # pooled_var_raw carry that case, which it does unambiguously.
            z_d = z_pool.detach()
            if z_d.shape[0] > 1:
                cov = torch.cov(z_d.T)
                tr = torch.diagonal(cov).sum()
                fro_sq = cov.pow(2).sum()
                pr = (tr * tr) / fro_sq if fro_sq > 1e-12 else torch.zeros((), device=device)
            else:
                pr = torch.zeros((), device=device)
            loss_dict.update({
                "ev_loss": ev_loss.item(),
                "ev_loss_shuffled": ev_shuffled.item(),
                "ev_gap": (ev_shuffled - ev_loss.detach()).item(),
                "ve_loss": ve_loss.item(),
                # RAW, pre-standardization: standardizing the target divides out
                # exactly the shrinkage this is meant to detect.
                "pooled_var_raw": z_pool.detach().var(dim=0).mean().item(),
                "pooled_pr": pr.item(),
                "lambda_intra": lam,
                "total_loss": total_loss.item(),
            })
        return total_loss, loss_dict
