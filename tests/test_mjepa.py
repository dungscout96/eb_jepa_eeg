"""Unit tests for MJEPA — joint intra-modal + cross-modal L1 prediction.

Follows the tiny-config pattern of tests/test_cross_subject_jepa.py. The tests
that matter most, in order:

  * ``test_encoder_state_dict_keys_match_masked_jepa`` and
    ``test_load_encoder_weights_roundtrip`` — the probe/eval chain loads ONLY
    ``encoder.*`` into a bare EEGEncoderTokens, so key-layout compatibility is
    what keeps every evaluator working.
  * ``test_row_alignment_pooled_vs_vision`` — a silent transpose between pooled
    EEG rows and vision rows produces a plausible loss curve and a dead model.
  * ``test_ev_gradient_reaches_encoder`` — three of the four anti-collapse
    strategies wrap ``target_representations`` in ``no_grad``; sourcing the
    cross-modal input from there would silently zero the encoder gradient.
  * ``test_collapse_diagnostics_fire`` — ``ev_gap`` / ``pooled_pr`` are the only
    signals that distinguish "learned nothing" from "learning".
"""

import copy

import pytest
import torch
import torch.nn as nn

from eb_jepa.anti_collapse import (
    AntiCollapse,
    DINOAntiCollapse,
    SIGRegAntiCollapse,
    VICRegAntiCollapse,
)
from eb_jepa.architectures import CrossModalMLP, EEGEncoderTokens, MaskedPredictor, Projector
from eb_jepa.jepa import MaskedJEPA
from eb_jepa.losses import SIGRegLoss, VCLoss
from eb_jepa.masking import MultiBlockMaskCollator
from eb_jepa.mjepa import MJEPA, TargetStandardizer

N_CHANS = 4
N_WINDOWS = 2
WINDOW_SIZE = 200
EMBED_DIM = 16
VISION_DIM = 32          # small for speed; one test uses the real 1408
BATCH = 5
PATCH_SIZE = 50
PATCH_OVERLAP = 10

MOCK_CHS_INFO = [{"ch_name": name} for name in ("Cz", "Fz", "Pz", "Oz")]


def _make_encoder(seed: int = 0) -> EEGEncoderTokens:
    torch.manual_seed(seed)
    return EEGEncoderTokens(
        n_chans=N_CHANS, n_times=WINDOW_SIZE, embed_dim=EMBED_DIM, depth=1,
        heads=2, head_dim=8, n_windows=N_WINDOWS, patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP, freqs=2, chs_info=MOCK_CHS_INFO,
        mlp_dim_ratio=2.0,
    )


def _make_predictor() -> MaskedPredictor:
    return MaskedPredictor(embed_dim=EMBED_DIM, depth=1, heads=2, head_dim=8,
                           mlp_dim_ratio=2.0, predictor_dim=None)


def _make_mask_collator(encoder) -> MultiBlockMaskCollator:
    return MultiBlockMaskCollator(
        n_channels=N_CHANS, n_windows=N_WINDOWS,
        n_patches_per_window=encoder.n_patches_per_window,
        n_pred_masks_short=1, n_pred_masks_long=1,
        short_channel_scale=(0.2, 0.4), short_patch_scale=(0.2, 0.4),
        long_channel_scale=(0.3, 0.5), long_patch_scale=(0.5, 0.8),
        min_context_fraction=0.15,
    )


def _make_anti_collapse(strategy: str, encoder):
    if strategy == "dino":
        return DINOAntiCollapse(copy.deepcopy(encoder))
    if strategy == "vicreg":
        proj = Projector(f"{EMBED_DIM}-{EMBED_DIM * 2}-{EMBED_DIM * 2}")
        return VICRegAntiCollapse(VCLoss(std_coeff=1.0, cov_coeff=1.0, proj=proj))
    if strategy == "sigreg":
        return SIGRegAntiCollapse(SIGRegLoss(num_slices=32, coeff=0.05))
    if strategy == "none":
        return AntiCollapse()
    raise ValueError(strategy)


def _make_mjepa(strategy: str = "none", vision_dim: int = VISION_DIM, **kwargs) -> MJEPA:
    encoder = _make_encoder(seed=0)
    return MJEPA(
        encoder, _make_predictor(), _make_mask_collator(encoder),
        _make_anti_collapse(strategy, encoder),
        pred_loss_type=kwargs.pop("pred_loss_type", "mse"),
        vision_dim=vision_dim, **kwargs,
    )


@pytest.fixture
def batch():
    torch.manual_seed(123)
    return (torch.randn(BATCH, N_WINDOWS, N_CHANS, WINDOW_SIZE),
            torch.randn(BATCH, N_WINDOWS, VISION_DIM))


# -- forward / gradients ----------------------------------------------------


@pytest.mark.parametrize("lam,strategy", [(0.0, "none"), (0.5, "sigreg"), (1.0, "sigreg")])
def test_forward_runs_and_returns_diagnostics(lam, strategy, batch):
    eeg, vision = batch
    m = _make_mjepa(strategy, lambda_intra=lam)
    loss, d = m(eeg, vision, global_step=3)

    assert torch.is_tensor(loss) and loss.ndim == 0 and torch.isfinite(loss)
    for key in ("ev_loss", "ev_loss_shuffled", "ev_gap", "ve_loss",
                "pooled_var_raw", "pooled_pr", "total_loss", "lambda_intra"):
        assert key in d, f"lam={lam}: missing {key}"
    if lam == 0.0:
        # The masked branch must be skipped entirely, not merely zero-weighted.
        assert "pred_loss" not in d and "ac_loss" not in d
    else:
        assert "pred_loss" in d and "ac_loss" in d


@pytest.mark.parametrize("strategy", ["none", "dino", "vicreg", "sigreg"])
def test_ev_gradient_reaches_encoder(strategy, batch):
    """The cross-modal term must train the encoder under EVERY strategy.

    `none`, `dino` and `vicreg` all wrap `target_representations` in `no_grad`;
    if `z_eeg` were sourced from there the encoder would receive no cross-modal
    gradient and the run would look healthy while training nothing.
    """
    eeg, vision = batch
    lam = 0.0 if strategy == "none" else 1.0
    m = _make_mjepa(strategy, lambda_intra=lam)
    loss, _ = m(eeg, vision)
    loss.backward()

    for name, module in (("encoder", m.encoder), ("mlp_ev", m.mlp_ev),
                         ("mlp_ve", m.mlp_ve)):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in module.parameters()), f"{strategy}: {name} got no gradient"


def test_ev_alone_trains_encoder_at_lambda_zero(batch):
    """At lambda=0, L_e2v is the ONLY term that trains the encoder."""
    eeg, vision = batch
    m = _make_mjepa("none", lambda_intra=0.0, ve_enabled=False)
    loss, d = m(eeg, vision)
    loss.backward()
    assert "pred_loss" not in d
    assert d["ve_loss"] == 0.0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.encoder.parameters())


def test_ve_stopgrad_gives_no_encoder_gradient(batch):
    """Documents the adapted-scope consequence as a contract.

    With the movie side frozen and the EEG target stop-grad, L_v2e trains only
    mlp_ve. If this ever starts failing, the stop-grad was dropped and L_v2e
    became a BYOL-style objective that needs an anti-collapse term.
    """
    eeg, vision = batch
    m = _make_mjepa("none", lambda_intra=0.0, ve_stopgrad=True)
    z = m._flatten_windows(m.encoder.pool_to_windows(m.encoder.encode_tokens(eeg)))
    ve = m._cross_criterion(m.mlp_ve(vision.reshape(-1, VISION_DIM)),
                            m.tgt_norm_e(z.detach()))
    ve.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in m.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.mlp_ve.parameters())


def test_ve_stopgrad_false_does_reach_encoder(batch):
    eeg, vision = batch
    m = _make_mjepa("none", lambda_intra=0.0, ve_stopgrad=False, ve_enabled=True)
    z = m._flatten_windows(m.encoder.pool_to_windows(m.encoder.encode_tokens(eeg)))
    ve = m._cross_criterion(m.mlp_ve(vision.reshape(-1, VISION_DIM)),
                            m.tgt_norm_e(z))
    ve.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.encoder.parameters())


# -- row alignment ----------------------------------------------------------


def test_row_alignment_pooled_vs_vision():
    """Row b*T+t of the flattened pooled EEG must be window t of sample b.

    This is the pairing the whole objective rests on; a transpose here silently
    pairs each EEG window with the wrong movie moment.
    """
    B, D, T = 4, 6, 3
    pooled = torch.arange(B * D * T, dtype=torch.float32).view(B, D, T)
    flat = MJEPA._flatten_windows(pooled.unsqueeze(-1).unsqueeze(-1))
    assert flat.shape == (B * T, D)
    for b in range(B):
        for t in range(T):
            assert torch.equal(flat[b * T + t], pooled[b, :, t])


def test_row_alignment_matches_movie_clip_head():
    """Must agree with MovieCLIPHead.project_eeg's flatten, the repo convention."""
    from eb_jepa.architectures import MovieCLIPHead
    B, D, T = 3, EMBED_DIM, 2
    pooled = torch.randn(B, D, T, 1, 1)
    head = MovieCLIPHead(eeg_in_dim=D, vision_in_dim=VISION_DIM,
                         vision_passthrough=False, proj_dim=D)
    head.eeg_proj = nn.Identity()
    expected = torch.nn.functional.normalize(
        pooled.view(B, D, T).permute(0, 2, 1).reshape(B * T, D), dim=-1)
    got = torch.nn.functional.normalize(MJEPA._flatten_windows(pooled), dim=-1)
    assert torch.allclose(got, expected, atol=1e-6)


# -- checkpoint compatibility ----------------------------------------------


def test_encoder_state_dict_keys_match_masked_jepa():
    encoder = _make_encoder(seed=0)
    predictor, coll = _make_predictor(), _make_mask_collator(encoder)
    ac = _make_anti_collapse("sigreg", encoder)
    masked = MaskedJEPA(encoder, predictor, coll, ac)
    mj = MJEPA(encoder, predictor, coll, ac, vision_dim=VISION_DIM)
    ek = lambda m: {k for k in m.state_dict() if k.startswith("encoder.")}
    assert ek(masked) == ek(mj)


def test_no_forbidden_top_level_submodule_names():
    from eb_jepa.training.builder import check_old_checkpoint_format
    sd = _make_mjepa("dino").state_dict()
    check_old_checkpoint_format(sd)
    assert {k.split(".")[0] for k in sd} <= {
        "encoder", "predictor", "anti_collapse", "mlp_ev", "mlp_ve",
        "tgt_norm_v", "tgt_norm_e",
    }


def test_load_encoder_weights_roundtrip(tmp_path):
    """An MJEPA checkpoint must load cleanly into a bare encoder — the probe path."""
    from eb_jepa.training_utils import load_encoder_weights
    m = _make_mjepa("sigreg", lambda_intra=1.0)
    ckpt = tmp_path / "latest.pth.tar"
    torch.save({"model_state_dict": m.state_dict()}, ckpt)

    fresh = _make_encoder(seed=1)
    info = load_encoder_weights(fresh, str(ckpt))
    assert info["n_loaded"] > 0
    assert info["missing"] == [] and info["unexpected"] == []
    # the cross-modal heads are dropped, not mistaken for encoder tensors
    assert any(k.startswith("mlp_ev.") for k in info["dropped"])


# -- collapse diagnostics ---------------------------------------------------


def test_collapse_diagnostics_fire(batch):
    """A constant encoder must show ev_gap ~ 0 and pooled_pr ~ 1."""
    eeg, vision = batch
    m = _make_mjepa("none", lambda_intra=0.0)

    class ConstantEncoder(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.embed_dim = inner.embed_dim
            self.n_patches_per_window = inner.n_patches_per_window

        def encode_tokens(self, eeg, mask=None):
            n = self.inner.encode_tokens(eeg, mask=mask)
            return torch.ones_like(n)

        def pool_to_windows(self, tokens, keep_channels=False):
            return self.inner.pool_to_windows(tokens, keep_channels=keep_channels)

    m.encoder = ConstantEncoder(m.encoder)
    _, d = m(eeg, vision)
    # A constant encoder produces a constant prediction, which scores identically
    # on matched and shuffled targets (the shuffle is a permutation, so the
    # multiset of |pred - target| terms is unchanged) -> the gap is exactly 0.
    assert d["ev_gap"] == pytest.approx(0.0, abs=1e-6)
    assert d["pooled_var_raw"] == pytest.approx(0.0, abs=1e-8)
    # cov is all-zero here, so the participation ratio is degenerate (0/0);
    # pooled_var_raw is the signal for this case.
    assert d["pooled_pr"] == pytest.approx(0.0, abs=1e-6)


def test_pooled_pr_is_one_for_rank_one_representation(batch):
    """Rank collapse WITH nonzero variance is what pooled_pr is for."""
    eeg, vision = batch
    m = _make_mjepa("none", lambda_intra=0.0)

    class RankOneEncoder(nn.Module):
        """All samples differ only by a scalar along one fixed direction."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.embed_dim = inner.embed_dim
            self.n_patches_per_window = inner.n_patches_per_window

        def encode_tokens(self, eeg, mask=None):
            t = self.inner.encode_tokens(eeg, mask=mask)
            direction = torch.zeros_like(t)
            direction[..., 0] = 1.0
            scale = t.mean(dim=-1, keepdim=True)
            return direction * scale

        def pool_to_windows(self, tokens, keep_channels=False):
            return self.inner.pool_to_windows(tokens, keep_channels=keep_channels)

    m.encoder = RankOneEncoder(m.encoder)
    _, d = m(eeg, vision)
    assert d["pooled_var_raw"] > 0, "this case must have nonzero variance"
    assert d["pooled_pr"] == pytest.approx(1.0, abs=1e-3)


def test_ev_gap_is_positive_when_predictions_track_the_target(batch):
    """Unit-test the gap formula itself, not a training run.

    A perfect predictor scores 0 on the matched target and > 0 on a shuffled
    one, so the gap must be strictly positive. This pins the direction and sign
    convention that the kill-switch protocol reads.
    """
    eeg, vision = batch
    m = _make_mjepa("none", lambda_intra=0.0, ve_enabled=False,
                    standardize_targets=False)

    class PerfectPredictor(nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = target

        def forward(self, x):
            return self.target

    m.mlp_ev = PerfectPredictor(vision.reshape(-1, VISION_DIM))
    _, d = m(eeg, vision)
    assert d["ev_loss"] == pytest.approx(0.0, abs=1e-6)
    assert d["ev_loss_shuffled"] > 0.1
    assert d["ev_gap"] > 0.1


# -- TargetStandardizer -----------------------------------------------------


def test_standardizer_normalizes_and_freezes():
    torch.manual_seed(0)
    s = TargetStandardizer(8, momentum=0.5)
    x = torch.randn(256, 8) * 5.0 + 3.0
    s.train()
    for _ in range(50):
        out = s(x)
    assert out.mean().abs() < 0.2
    assert abs(out.std().item() - 1.0) < 0.2

    s.eval()
    before = s.running_mean.clone()
    s(torch.randn(256, 8) * 100.0)
    assert torch.equal(s.running_mean, before), "eval() must freeze the statistics"


def test_standardizer_initializes_from_first_batch():
    """Avoids ~500 steps of crawl from (0,1) at momentum=0.01."""
    s = TargetStandardizer(4, momentum=0.01)
    x = torch.full((32, 4), 7.0) + torch.randn(32, 4) * 0.01
    s.train()
    s(x)
    assert torch.allclose(s.running_mean, torch.full((4,), 7.0), atol=0.05)


def test_standardizer_preserves_gradient_when_input_requires_it():
    s = TargetStandardizer(4)
    x = torch.randn(16, 4, requires_grad=True)
    s(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


# -- misc -------------------------------------------------------------------


def test_real_vision_dim_shapes():
    """Exercise the true V-JEPA-2 width once."""
    m = _make_mjepa("none", vision_dim=1408, lambda_intra=0.0)
    eeg = torch.randn(3, N_WINDOWS, N_CHANS, WINDOW_SIZE)
    vision = torch.randn(3, N_WINDOWS, 1408)
    loss, d = m(eeg, vision)
    assert torch.isfinite(loss)
    assert m.mlp_ev.net[-1].out_features == 1408
    assert m.mlp_ve.net[-1].out_features == EMBED_DIM


@pytest.mark.parametrize("bad,match", [
    ({"vision_dim": 0}, "vision_dim"),
    ({"cross_loss_type": "bogus"}, "cross_loss_type"),
    ({"cross_source": "bogus"}, "cross_source"),
    ({"lambda_intra": -1.0}, "lambda_intra"),
])
def test_rejects_invalid_options(bad, match):
    with pytest.raises(ValueError, match=match):
        _make_mjepa("none", **bad)


def test_rejects_mismatched_vision_shape(batch):
    eeg, _ = batch
    m = _make_mjepa("none", lambda_intra=0.0)
    with pytest.raises(ValueError, match="vision must be"):
        m(eeg, torch.randn(BATCH, N_WINDOWS, VISION_DIM + 1))
    with pytest.raises(ValueError, match=r"\(B, T\)"):
        m(eeg, torch.randn(BATCH + 1, N_WINDOWS, VISION_DIM))


def test_combine_scaled_applies_lambda_to_pred_only():
    """lambda must not scale the anti-collapse term — that would confound the sweep."""
    m = _make_mjepa("sigreg", lambda_intra=0.5)
    pred = torch.tensor(2.0)
    ac = torch.tensor(10.0)
    c = m.anti_collapse.coeff
    # sigreg default combine_mode is "convex": (1-c)*lam*pred + c*ac
    assert m._combine_scaled(pred, ac).item() == pytest.approx(
        (1 - c) * 0.5 * 2.0 + c * 10.0, rel=1e-6)


def test_cross_modal_mlp_has_no_batchnorm():
    """BatchNorm would silently standardize and partly rescue a collapsed encoder."""
    mlp = CrossModalMLP(8, 16, 4, depth=3)
    assert not any(isinstance(mod, nn.BatchNorm1d) for mod in mlp.modules())
    assert sum(1 for mod in mlp.net if isinstance(mod, nn.Linear)) == 3
