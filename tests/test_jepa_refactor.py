"""Unit tests for the post-refactor MaskedJEPA + AntiCollapse composition.

Exercises the three canonical anti-collapse strategies (DINO, VICReg,
SIGReg) end-to-end on a tiny synthetic EEG batch.
"""

import copy

import pytest
import torch

from eb_jepa.anti_collapse import (
    AntiCollapse,
    DINOAntiCollapse,
    SIGRegAntiCollapse,
    VICRegAntiCollapse,
)
from eb_jepa.architectures import EEGEncoderTokens, MaskedPredictor, Projector
from eb_jepa.jepa import MaskedJEPA
from eb_jepa.losses import SIGRegLoss, VCLoss
from eb_jepa.masking import MultiBlockMaskCollator


# Tiny config for fast tests
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


def _make_jepa(strategy: str) -> MaskedJEPA:
    encoder = _make_encoder(seed=0)
    predictor = _make_predictor()
    mask_collator = _make_mask_collator(encoder)

    if strategy == "dino":
        ac = DINOAntiCollapse(copy.deepcopy(encoder))
    elif strategy == "vicreg":
        proj = Projector(f"{EMBED_DIM}-{EMBED_DIM * 2}-{EMBED_DIM * 2}")
        ac = VICRegAntiCollapse(VCLoss(std_coeff=1.0, cov_coeff=1.0, proj=proj))
    elif strategy == "sigreg":
        ac = SIGRegAntiCollapse(SIGRegLoss(num_slices=64, coeff=0.05))
    elif strategy == "none":
        ac = AntiCollapse()
    else:
        raise ValueError(strategy)

    return MaskedJEPA(encoder, predictor, mask_collator, ac, pred_loss_type="mse")


@pytest.fixture
def eeg_batch():
    torch.manual_seed(123)
    return torch.randn(BATCH, N_WINDOWS, N_CHANS, WINDOW_SIZE)


@pytest.mark.parametrize("strategy", ["dino", "vicreg", "sigreg", "none"])
def test_forward_runs_and_returns_loss_dict(strategy, eeg_batch):
    jepa = _make_jepa(strategy)
    loss, loss_dict = jepa(eeg_batch, global_step=7)

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    for required in ("pred_loss", "ac_loss", "total_loss", "pred_target_cosim",
                     "target_var", "pred_var", "pred_loss_norm"):
        assert required in loss_dict, f"{strategy}: missing key {required}"


@pytest.mark.parametrize("strategy", ["dino", "vicreg", "sigreg"])
def test_loss_propagates_gradients_to_encoder(strategy, eeg_batch):
    jepa = _make_jepa(strategy)
    loss, _ = jepa(eeg_batch, global_step=0)
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in jepa.encoder.parameters()
    )
    assert has_grad, f"{strategy}: encoder received no gradient signal"


def test_dino_step_updates_target_encoder(eeg_batch):
    jepa = _make_jepa("dino")
    assert isinstance(jepa.anti_collapse, DINOAntiCollapse)

    # Perturb online encoder so EMA has something to move toward.
    with torch.no_grad():
        for p in jepa.encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    target_before = [p.detach().clone() for p in jepa.anti_collapse.target_encoder.parameters()]
    jepa.update_target_encoder(momentum=0.5)
    target_after = list(jepa.anti_collapse.target_encoder.parameters())

    moved = any(
        not torch.equal(b, a) for b, a in zip(target_before, target_after)
    )
    assert moved, "DINO EMA update did not change any target encoder parameter"


@pytest.mark.parametrize("strategy", ["vicreg", "sigreg", "none"])
def test_non_dino_step_is_noop(strategy, eeg_batch):
    jepa = _make_jepa(strategy)
    # Anti-collapse strategy itself should not have a target encoder to mutate.
    snapshot_encoder = [p.detach().clone() for p in jepa.encoder.parameters()]
    snapshot_ac = [p.detach().clone() for p in jepa.anti_collapse.parameters()]

    jepa.update_target_encoder(momentum=0.5)

    for before, after in zip(snapshot_encoder, jepa.encoder.parameters()):
        assert torch.equal(before, after), f"{strategy}: step() mutated encoder"
    for before, after in zip(snapshot_ac, jepa.anti_collapse.parameters()):
        assert torch.equal(before, after), f"{strategy}: step() mutated anti_collapse params"


def test_dino_target_has_no_grad(eeg_batch):
    jepa = _make_jepa("dino")
    for p in jepa.anti_collapse.target_encoder.parameters():
        assert p.requires_grad is False


def test_sigreg_combines_convexly(eeg_batch):
    jepa = _make_jepa("sigreg")
    assert jepa.anti_collapse.combine_mode == "convex"
    assert 0.0 < jepa.anti_collapse.coeff < 1.0


def test_vicreg_and_dino_combine_additively():
    for strategy in ("vicreg", "dino", "none"):
        jepa = _make_jepa(strategy)
        assert jepa.anti_collapse.combine_mode == "additive", strategy
