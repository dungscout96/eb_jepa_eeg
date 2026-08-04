"""Unit tests for CrossSubjectJEPA — the A->B token-prediction objective.

Mirrors tests/test_jepa_refactor.py's tiny-config pattern. The two assertions
that matter most:

  * ``test_state_dict_keys_match_masked_jepa`` — the entire probe/eval chain
    loads only ``encoder.*`` keys, so key-layout compatibility with MaskedJEPA
    is what keeps every evaluator working unchanged.
  * ``test_collapse_diagnostics_detect_constant_predictor`` — SIGReg cannot see
    "targets spread, predictions constant", so ``pred_var_ratio`` /
    ``pred_loss_gap`` are the only collapse detectors. They must actually fire.
"""

import copy
import random

import pytest
import torch
import torch.nn as nn

from eb_jepa.anti_collapse import (
    AntiCollapse,
    DINOAntiCollapse,
    SIGRegAntiCollapse,
    VICRegAntiCollapse,
)
from eb_jepa.architectures import EEGEncoderTokens, MaskedPredictor, Projector
from eb_jepa.jepa import CrossSubjectJEPA, MaskedJEPA
from eb_jepa.losses import SIGRegLoss, VCLoss
from eb_jepa.masking import MultiBlockMaskCollator

# Tiny config for fast tests — same values as tests/test_jepa_refactor.py.
N_CHANS = 4
N_WINDOWS = 2
WINDOW_SIZE = 200
EMBED_DIM = 16
BATCH = 3
PATCH_SIZE = 50
PATCH_OVERLAP = 10

MOCK_CHS_INFO = [{"ch_name": name} for name in ("Cz", "Fz", "Pz", "Oz")]


def _make_encoder(seed: int = 0) -> EEGEncoderTokens:
    torch.manual_seed(seed)
    return EEGEncoderTokens(
        n_chans=N_CHANS,
        n_times=WINDOW_SIZE,
        embed_dim=EMBED_DIM,
        depth=1,
        heads=2,
        head_dim=8,
        n_windows=N_WINDOWS,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        freqs=2,
        chs_info=MOCK_CHS_INFO,
        mlp_dim_ratio=2.0,
    )


def _make_predictor() -> MaskedPredictor:
    return MaskedPredictor(
        embed_dim=EMBED_DIM, depth=1, heads=2, head_dim=8,
        mlp_dim_ratio=2.0, predictor_dim=None,
    )


def _make_mask_collator(encoder: EEGEncoderTokens) -> MultiBlockMaskCollator:
    return MultiBlockMaskCollator(
        n_channels=N_CHANS,
        n_windows=N_WINDOWS,
        n_patches_per_window=encoder.n_patches_per_window,
        n_pred_masks_short=1,
        n_pred_masks_long=1,
        short_channel_scale=(0.2, 0.4),
        short_patch_scale=(0.2, 0.4),
        long_channel_scale=(0.3, 0.5),
        long_patch_scale=(0.5, 0.8),
        min_context_fraction=0.15,
    )


def _make_anti_collapse(strategy: str, encoder):
    if strategy == "dino":
        return DINOAntiCollapse(copy.deepcopy(encoder))
    if strategy == "vicreg":
        proj = Projector(f"{EMBED_DIM}-{EMBED_DIM * 2}-{EMBED_DIM * 2}")
        return VICRegAntiCollapse(VCLoss(std_coeff=1.0, cov_coeff=1.0, proj=proj))
    if strategy == "sigreg":
        return SIGRegAntiCollapse(SIGRegLoss(num_slices=64, coeff=0.05))
    if strategy == "none":
        return AntiCollapse()
    raise ValueError(strategy)


def _make_xsubj(strategy: str = "sigreg", **kwargs) -> CrossSubjectJEPA:
    encoder = _make_encoder(seed=0)
    return CrossSubjectJEPA(
        encoder,
        _make_predictor(),
        _make_mask_collator(encoder),
        _make_anti_collapse(strategy, encoder),
        pred_loss_type=kwargs.pop("pred_loss_type", "mse"),
        **kwargs,
    )


@pytest.fixture
def eeg_pair():
    """[B, 2, T, C, W] — subject A and subject B at the same movie time."""
    torch.manual_seed(123)
    return torch.randn(BATCH, 2, N_WINDOWS, N_CHANS, WINDOW_SIZE)


# -- forward / gradients ----------------------------------------------------


@pytest.mark.parametrize("strategy", ["dino", "vicreg", "sigreg", "none"])
@pytest.mark.parametrize("context_mask_mode", ["masked", "full"])
@pytest.mark.parametrize("pred_target_mode", ["masked", "all"])
def test_forward_runs_and_returns_diagnostics(
    strategy, context_mask_mode, pred_target_mode, eeg_pair
):
    jepa = _make_xsubj(
        strategy,
        context_mask_mode=context_mask_mode,
        pred_target_mode=pred_target_mode,
    )
    loss, loss_dict = jepa(eeg_pair, global_step=7)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    for required in (
        "pred_loss", "ac_loss", "total_loss", "pred_target_cosim",
        "cosim_within", "cosim_shuffled", "align_gap",
        "pred_loss_shuffled", "pred_loss_gap",
        "target_var", "pred_var", "pred_var_ratio",
        "target_var_across_batch", "target_var_across_pos",
        "pred_loss_norm", "n_ctx", "n_pred",
    ):
        assert required in loss_dict, f"{strategy}: missing key {required}"

    n_tokens = N_CHANS * N_WINDOWS * jepa.encoder.n_patches_per_window
    if pred_target_mode == "all":
        assert loss_dict["n_pred"] == n_tokens
    else:
        assert 0 < loss_dict["n_pred"] <= n_tokens
    if context_mask_mode == "full":
        assert loss_dict["n_ctx"] == n_tokens
    else:
        assert loss_dict["n_ctx"] < n_tokens


@pytest.mark.parametrize("strategy", ["dino", "vicreg", "sigreg"])
def test_loss_propagates_gradients_to_encoder_and_predictor(strategy, eeg_pair):
    jepa = _make_xsubj(strategy)
    loss, _ = jepa(eeg_pair, global_step=0)
    loss.backward()

    for name, module in (("encoder", jepa.encoder), ("predictor", jepa.predictor)):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
        assert has_grad, f"{strategy}: {name} received no gradient signal"


def test_rejects_single_subject_batch():
    """A [B, T, C, W] batch means the plain dataset was wired up by mistake."""
    jepa = _make_xsubj()
    bad = torch.randn(BATCH, N_WINDOWS, N_CHANS, WINDOW_SIZE)
    with pytest.raises(ValueError, match=r"\[B, 2, T, C, W\]"):
        jepa(bad)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"context_mask_mode": "bogus"}, "context_mask_mode"),
        ({"pred_target_mode": "bogus"}, "pred_target_mode"),
        ({"within_subject_weight": 1.5}, "within_subject_weight"),
    ],
)
def test_rejects_invalid_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make_xsubj(**kwargs)


# -- checkpoint compatibility (protects the whole eval chain) ---------------


def test_state_dict_keys_match_masked_jepa():
    """Every evaluator loads only ``encoder.*``; key layout must not drift.

    probe.py / probe_traintest.py / retrieval.py never construct a MaskedJEPA —
    they filter ``encoder.*`` out of the checkpoint and load it into a bare
    EEGEncoderTokens. So as long as this holds, a cross-subject checkpoint is
    evaluated by exactly the same code path as a masked one.
    """
    encoder = _make_encoder(seed=0)
    predictor = _make_predictor()
    collator = _make_mask_collator(encoder)
    ac = _make_anti_collapse("sigreg", encoder)

    masked = MaskedJEPA(encoder, predictor, collator, ac, pred_loss_type="mse")
    xsubj = CrossSubjectJEPA(encoder, predictor, collator, ac, pred_loss_type="mse")

    assert set(masked.state_dict()) == set(xsubj.state_dict())


def test_no_forbidden_top_level_submodule_names():
    """check_old_checkpoint_format hard-fails on these prefixes."""
    from eb_jepa.training.builder import check_old_checkpoint_format

    sd = _make_xsubj("dino").state_dict()
    check_old_checkpoint_format(sd)  # must not raise
    top_level = {k.split(".")[0] for k in sd}
    assert top_level <= {"encoder", "predictor", "anti_collapse"}, top_level


# -- objective semantics ----------------------------------------------------


def test_symmetric_loss_is_invariant_to_swapping_the_pair(eeg_pair):
    """roll(targets, B) trains A->B and B->A equally, so the loss is symmetric.

    Swapping the pair axis permutes the (prediction, target) rows without
    changing the set of pairs, so the mean loss is unchanged.
    """
    jepa = _make_xsubj("none")
    jepa.eval()

    # MultiBlockMaskCollator draws from the `random` module (masking.py:13), not
    # torch, and is called inside forward — so seed `random` to fix the mask.
    random.seed(0)
    _, d_ab = jepa(eeg_pair)
    random.seed(0)
    _, d_ba = jepa(eeg_pair.flip(dims=[1]))

    assert d_ab["pred_loss"] == pytest.approx(d_ba["pred_loss"], abs=1e-5)


def test_asymmetric_uses_only_the_a_half(eeg_pair):
    """With symmetric=False the B->A rows are excluded from the loss."""
    jepa = _make_xsubj("none", symmetric=False)
    jepa.eval()

    random.seed(0)
    _, d_ab = jepa(eeg_pair)
    random.seed(0)
    _, d_ba = jepa(eeg_pair.flip(dims=[1]))

    # A->B only is direction-dependent, unlike the symmetric case above.
    assert d_ab["pred_loss"] != pytest.approx(d_ba["pred_loss"], abs=1e-6)


def test_identical_subjects_collapse_cross_onto_within(eeg_pair):
    """If B's EEG equals A's, the cross target IS the within target."""
    jepa = _make_xsubj("none")
    jepa.eval()
    identical = torch.stack([eeg_pair[:, 0], eeg_pair[:, 0]], dim=1)

    _, d = jepa(identical)

    assert d["pred_target_cosim"] == pytest.approx(d["cosim_within"], abs=1e-5)


def test_within_subject_weight_one_targets_own_tokens(eeg_pair):
    """within_subject_weight=1.0 degenerates to within-subject prediction."""
    jepa = _make_xsubj("none", within_subject_weight=1.0)
    jepa.eval()
    torch.manual_seed(0)
    _, d_within = jepa(eeg_pair)

    jepa_cross = _make_xsubj("none", within_subject_weight=0.0)
    jepa_cross.eval()
    torch.manual_seed(0)
    _, d_cross = jepa_cross(eeg_pair)

    # Different targets => different loss. Random EEG has no shared structure,
    # so predicting your own tokens is the easier of the two.
    assert d_within["pred_loss"] != pytest.approx(d_cross["pred_loss"], abs=1e-6)


# -- the collapse detector must actually detect -----------------------------


def test_collapse_diagnostics_detect_constant_predictor(eeg_pair):
    """A constant predictor must show pred_var_ratio ~ 0 and pred_loss_gap ~ 0.

    This is the failure mode SIGReg cannot see: targets can stay isotropically
    spread (so ac_loss / target_var look healthy) while the predictor emits the
    stimulus-independent mean. If this test ever stops failing for a constant
    predictor, the kill-switch protocol in the experiment config is worthless.
    """
    jepa = _make_xsubj("sigreg")

    class ConstantPredictor(nn.Module):
        """Emits the same vector regardless of context — the collapsed solution."""

        def __init__(self):
            super().__init__()
            self.n_pred_seen = 0

        def forward(self, ctx, ctx_pos, tgt_pos):
            self.n_pred_seen = tgt_pos.shape[1]
            return torch.zeros(
                tgt_pos.shape[0], tgt_pos.shape[1], EMBED_DIM,
                device=tgt_pos.device, dtype=tgt_pos.dtype,
            )

    jepa.predictor = ConstantPredictor()
    _, d = jepa(eeg_pair)

    assert jepa.predictor.n_pred_seen > 0
    assert d["pred_var_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert d["pred_loss_gap"] == pytest.approx(0.0, abs=1e-6)
    assert d["align_gap"] == pytest.approx(0.0, abs=1e-6)
    # And the target side still looks perfectly healthy — which is the point.
    assert d["target_var_across_batch"] > 0


def test_shuffled_control_differs_from_matched_target(eeg_pair):
    """pred_loss_gap is only meaningful if the shuffled target really differs."""
    jepa = _make_xsubj("none")
    _, d = jepa(eeg_pair)
    assert d["pred_loss_shuffled"] != pytest.approx(d["pred_loss"], abs=1e-9)


# -- builder dispatch -------------------------------------------------------


def test_builder_dispatches_on_loss_objective():
    from omegaconf import OmegaConf

    from eb_jepa.training.builder import build_jepa

    base = {
        "model": {
            "encoder_embed_dim": EMBED_DIM, "encoder_depth": 1,
            "encoder_heads": 2, "encoder_head_dim": 8,
            "patch_size": PATCH_SIZE, "patch_overlap": PATCH_OVERLAP,
            "freqs": 2, "mlp_dim_ratio": 2.0,
            "predictor_depth": 1, "predictor_embed_dim": None,
        },
        "masking": {"n_pred_masks_short": 1, "n_pred_masks_long": 1},
        "loss": {"anti_collapse": "sigreg",
                 "sigreg": {"num_slices": 64, "coeff": 0.05},
                 "pred_loss_type": "mse"},
    }
    kwargs = dict(n_chans=N_CHANS, n_times=WINDOW_SIZE,
                  chs_info=MOCK_CHS_INFO, n_windows=N_WINDOWS)

    # default -> masked
    assert type(build_jepa(OmegaConf.create(base), **kwargs)) is MaskedJEPA

    cs = copy.deepcopy(base)
    cs["loss"]["objective"] = "cross_subject"
    cs["loss"]["cross_subject"] = {
        "symmetric": False, "context_mask_mode": "full",
        "pred_target_mode": "all", "within_subject_weight": 0.25,
        "diagnostic_every": 10,
    }
    model = build_jepa(OmegaConf.create(cs), **kwargs)
    assert isinstance(model, CrossSubjectJEPA)
    assert model.symmetric is False
    assert model.context_mask_mode == "full"
    assert model.pred_target_mode == "all"
    assert model.within_subject_weight == 0.25

    bad = copy.deepcopy(base)
    bad["loss"]["objective"] = "nonsense"
    with pytest.raises(ValueError, match="Unknown loss.objective"):
        build_jepa(OmegaConf.create(bad), **kwargs)
