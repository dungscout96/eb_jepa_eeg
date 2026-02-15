"""
Test script to verify output formats of EEG JEPA's unroll() function.

This tests:
1. Model assembly matching the eeg_jepa example (main.py)
2. Output format of unroll() in parallel mode with and without loss
3. Output format of unroll() in autoregressive mode
4. return_all_steps functionality
5. infer() method

Setup under test:
    encoder = EEGEncoder(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor_model = MLPEEGPredictor(cfg.model.dstc*2, cfg.model.hpre, cfg.model.dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{cfg.model.dstc}-{cfg.model.dstc*4}-{cfg.model.dstc*4}")
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)
"""

import torch

from eb_jepa.architectures import (
    EEGEncoder,
    MLPEEGPredictor,
    StateOnlyPredictor,
    Projector,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VCLoss


# ============================================================================
# Config values from examples/eeg_jepa/cfgs/default.yaml
# ============================================================================
DOBS = 1       # Input channels
HENC = 32      # Hidden dimension in encoder
DSTC = 16      # Output representation dimension
HPRE = 32      # Hidden dimension in predictor
STEPS = 4      # Number of prediction steps during training
STD_COEFF = 10.0
COV_COEFF = 100.0

# Mock channel info for REVE (1 channel to match dobs=1)
MOCK_CHS_INFO = [{"ch_name": "CZ"}]


# ============================================================================
# Helper functions
# ============================================================================
def set_seed(seed=42):
    """Set random seed for reproducibility in tests."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_eeg_jepa_model(device="cpu"):
    """Create an EEG JEPA model matching the eeg_jepa example main.py."""
    encoder = EEGEncoder(DOBS, HENC, DSTC, chs_info=MOCK_CHS_INFO)
    predictor = MLPEEGPredictor(DSTC, HPRE, DSTC)
    predictor_model = MLPEEGPredictor(DSTC*2, HPRE, DSTC)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{DSTC}-{DSTC*4}-{DSTC*4}")
    regularizer = VCLoss(STD_COEFF, COV_COEFF, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)
    return jepa


# ============================================================================
# Tests for unroll() function in parallel mode
# ============================================================================


def test_unroll_parallel_mode_output_format():
    """
    Test unroll() output format in parallel mode (no loss, return_all_steps=True).

    Usage pattern:
        preds, losses = jepa.unroll(x, actions=None, nsteps=nsteps,
                                    unroll_mode="parallel", compute_loss=False,
                                    return_all_steps=True)
    """
    print("=" * 60)
    print("Testing EEG JEPA unroll() parallel mode output format")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(42)
    jepa = create_eeg_jepa_model(device)
    jepa.eval()

    # Verify predictor properties
    assert not jepa.single_unroll, "MLPEEGPredictor should have single_unroll=False"
    assert jepa.predictor.context_length == 2, "MLPEEGPredictor context_length should be 2"
    print(f"  Predictor: single_unroll={jepa.single_unroll}, context_length={jepa.predictor.context_length}")

    # Create fake EEG input: [B, C, T, n_channels, n_times]
    # C=1 (dobs), n_channels=1 (matching dobs for REVE), n_times=1000
    B, C, T, N_CH, N_TIMES = 4, 1, 10, DOBS, 1000
    x = torch.randn(B, C, T, N_CH, N_TIMES, device=device)

    print(f"\nInput shape: {x.shape}")
    print(f"  B={B}, C={C}, T={T}, N_CH={N_CH}, N_TIMES={N_TIMES}")

    nsteps = STEPS
    print(f"\nCalling: jepa.unroll(x, actions=None, nsteps={nsteps}, ...")
    print(f"         unroll_mode='parallel', compute_loss=False, return_all_steps=True)")

    with torch.no_grad():
        preds, losses = jepa.unroll(
            x,
            actions=None,
            nsteps=nsteps,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )

    print(f"\n--- unroll() Output Analysis ---")
    print(f"Return type of preds: {type(preds)}")
    print(f"Length of preds list: {len(preds)}")
    print(f"Expected length (nsteps): {nsteps}")
    print(f"losses: {losses} (expected: None when compute_loss=False)")

    # Analyze each prediction step
    print(f"\nPer-step shapes:")
    for i, pred in enumerate(preds):
        print(f"  preds[{i}] shape: {pred.shape}")

    # Assertions
    assert isinstance(preds, list), f"Expected list, got {type(preds)}"
    assert len(preds) == nsteps, f"Expected {nsteps} steps, got {len(preds)}"
    assert losses is None, f"Expected losses=None, got {losses}"
    print("\n  All assertions passed!")

    print("=" * 60)
    return preds


def test_unroll_parallel_mode_with_loss():
    """
    Test unroll() output format in parallel mode with loss computation.

    Usage pattern (from eeg_jepa main.py):
        _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
            x, actions=None, nsteps=cfg.model.steps,
            unroll_mode="parallel", compute_loss=True, return_all_steps=False)
    """
    print("\n" + "=" * 60)
    print("Testing EEG JEPA unroll() parallel mode with loss computation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(42)
    jepa = create_eeg_jepa_model(device)
    jepa.train()

    B, C, T, N_CH, N_TIMES = 4, 1, 10, DOBS, 1000
    x = torch.randn(B, C, T, N_CH, N_TIMES, device=device)

    print(f"\nInput shape: {x.shape}")

    nsteps = STEPS
    print(f"\nCalling: jepa.unroll(x, actions=None, nsteps={nsteps}, ...")
    print(f"         unroll_mode='parallel', compute_loss=True)")

    predicted_states, losses = jepa.unroll(
        x, actions=None, nsteps=nsteps, unroll_mode="parallel", compute_loss=True
    )
    loss, rloss, rloss_unweight, rloss_dict, ploss = losses

    print(f"\n--- unroll() Output Analysis ---")
    print(f"  predicted_states shape: {predicted_states.shape}")
    print(f"\nlosses tuple contains 5 elements:")
    print(f"  1. total_loss:      shape={loss.shape}, value={loss.item():.6f}")
    print(f"  2. reg_loss:        shape={rloss.shape}, value={rloss.item():.6f}")
    print(f"  3. reg_unweighted:  shape={rloss_unweight.shape}, value={rloss_unweight.item():.6f}")
    print(f"  4. reg_loss_dict:   {rloss_dict}")
    print(f"  5. pred_loss:       shape={ploss.shape}, value={ploss.item():.6f}")

    # Assertions
    assert loss.shape == torch.Size([]), f"total_loss should be scalar, got {loss.shape}"
    assert rloss.shape == torch.Size([]), f"reg_loss should be scalar, got {rloss.shape}"
    assert ploss.shape == torch.Size([]), f"pred_loss should be scalar, got {ploss.shape}"

    # reg_loss_dict should contain expected keys for VCLoss
    expected_keys = {"std_loss", "cov_loss"}
    assert set(rloss_dict.keys()) == expected_keys, \
        f"Expected keys {expected_keys}, got {set(rloss_dict.keys())}"
    print(f"\n  reg_loss_dict contains expected keys: {expected_keys}")
    print("  All assertions passed!")

    print("=" * 60)
    return loss, rloss, rloss_unweight, rloss_dict, ploss


def test_infer_method():
    """Test the infer() method which uses unroll() internally."""
    print("\n" + "=" * 60)
    print("Testing EEG JEPA infer() method")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    jepa = create_eeg_jepa_model(device)
    jepa.eval()

    B, C, T, N_CH, N_TIMES = 2, 1, 8, DOBS, 1000
    x = torch.randn(B, C, T, N_CH, N_TIMES, device=device)

    with torch.no_grad():
        infer_result = jepa.infer(x, actions=None)

    print(f"infer() output shape: {infer_result.shape}")
    print(f"  infer() returns single tensor (first step from unroll)")

    print("=" * 60)


# ============================================================================
# Tests for unroll() function in autoregressive mode
# ============================================================================


def test_unroll_autoregressive_mode_shapes():
    """
    Test unroll() in autoregressive mode (no loss, for planning/inference).

    Usage pattern:
        predicted_states, _ = jepa.unroll(obs_init, actions=None, nsteps=nsteps,
                                          unroll_mode="autoregressive",
                                          ctxt_window_time=2, compute_loss=False)
    """
    print("\n" + "=" * 60)
    print("Testing EEG JEPA unroll() autoregressive mode shapes")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(42)
    jepa = create_eeg_jepa_model(device)
    jepa.eval()

    # Verify this is NOT an RNN predictor (Conv-like sliding window)
    assert not jepa.single_unroll, "MLPEEGPredictor should have single_unroll=False"
    print("  Confirmed non-RNN predictor (single_unroll=False)")

    # Create test input: context frames only
    B, C, T_CONTEXT, N_CH, N_TIMES = 2, 1, 3, DOBS, 1000
    nsteps = 5
    ctxt_window_time = 2

    set_seed(42)
    obs = torch.randn(B, C, T_CONTEXT, N_CH, N_TIMES, device=device)

    print(f"\nInput shape: {obs.shape}")
    print(f"nsteps: {nsteps}")
    print(f"ctxt_window_time: {ctxt_window_time}")

    print(f"\nCalling: jepa.unroll(obs, actions=None, nsteps={nsteps}, ...")
    print(f"         unroll_mode='autoregressive', ctxt_window_time={ctxt_window_time})")

    with torch.no_grad():
        predicted_states, losses = jepa.unroll(
            obs,
            actions=None,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=ctxt_window_time,
            compute_loss=False,
            return_all_steps=False,
        )

    expected_T_out = ctxt_window_time + nsteps
    print(f"\n  Output shape: {predicted_states.shape}")
    print(f"  Expected time dimension: {expected_T_out} (ctxt_window_time + nsteps)")

    assert predicted_states.shape[2] == expected_T_out, \
        f"Time dim mismatch: got {predicted_states.shape[2]}, expected {expected_T_out}"
    print(f"  Time dimension correct: {predicted_states.shape[2]}")

    assert losses is None, f"Expected losses=None, got {losses}"
    print(f"  losses is None when compute_loss=False")

    print("\n  All assertions passed!")
    print("=" * 60)

    return predicted_states


def test_unroll_autoregressive_with_loss():
    """
    Test unroll() autoregressive mode with loss computation.
    """
    print("\n" + "=" * 60)
    print("Testing EEG JEPA unroll() autoregressive mode with loss")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_seed(42)
    jepa = create_eeg_jepa_model(device)
    jepa.train()

    B, C, T, N_CH, N_TIMES = 2, 1, 12, DOBS, 1000
    nsteps = 6
    ctxt_window_time = 2

    set_seed(42)
    x = torch.randn(B, C, T, N_CH, N_TIMES, device=device)

    print(f"\nInput shape: {x.shape}")
    print(f"nsteps: {nsteps}, ctxt_window_time: {ctxt_window_time}")

    predicted_states, losses = jepa.unroll(
        x,
        actions=None,
        nsteps=nsteps,
        unroll_mode="autoregressive",
        ctxt_window_time=ctxt_window_time,
        compute_loss=True,
    )
    loss, rloss, rloss_unweight, rloss_dict, ploss = losses

    print(f"\n--- unroll() Output Analysis ---")
    print(f"  predicted_states shape: {predicted_states.shape}")
    print(f"\nlosses tuple contains 5 elements:")
    print(f"  1. total_loss:      shape={loss.shape}")
    print(f"  2. reg_loss:        shape={rloss.shape}")
    print(f"  3. reg_unweighted:  shape={rloss_unweight.shape}")
    print(f"  4. reg_loss_dict:   keys={list(rloss_dict.keys())}")
    print(f"  5. pred_loss:       shape={ploss.shape}")

    # Assertions
    assert loss.shape == torch.Size([]), f"total_loss should be scalar, got {loss.shape}"
    assert rloss.shape == torch.Size([]), f"reg_loss should be scalar, got {rloss.shape}"
    assert ploss.shape == torch.Size([]), f"pred_loss should be scalar, got {ploss.shape}"

    expected_keys = {"std_loss", "cov_loss"}
    assert set(rloss_dict.keys()) == expected_keys, \
        f"Expected keys {expected_keys}, got {set(rloss_dict.keys())}"
    print(f"\n  All assertions passed!")

    print("=" * 60)
    return loss, rloss, rloss_unweight, rloss_dict, ploss


def test_unroll_return_all_steps_format():
    """
    Test that return_all_steps=True returns the correct format.
    """
    print("\n" + "=" * 60)
    print("Testing EEG JEPA unroll() return_all_steps format")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with parallel mode
    print("\n--- Parallel mode ---")
    set_seed(42)
    jepa = create_eeg_jepa_model(device)
    jepa.eval()

    B, C, T, N_CH, N_TIMES = 2, 1, 8, DOBS, 1000
    x = torch.randn(B, C, T, N_CH, N_TIMES, device=device)
    nsteps = 3

    with torch.no_grad():
        all_steps, _ = jepa.unroll(
            x,
            actions=None,
            nsteps=nsteps,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )

    assert isinstance(all_steps, list), f"Expected list, got {type(all_steps)}"
    assert len(all_steps) == nsteps, f"Expected {nsteps} steps, got {len(all_steps)}"
    print(f"  Parallel mode returns list of {len(all_steps)} tensors")
    for i, step in enumerate(all_steps):
        print(f"    Step {i}: shape={step.shape}")

    # Test with autoregressive mode
    print("\n--- Autoregressive mode ---")
    set_seed(42)
    jepa = create_eeg_jepa_model(device)
    jepa.eval()

    obs = torch.randn(B, C, 3, N_CH, N_TIMES, device=device)

    with torch.no_grad():
        all_steps_ar, _ = jepa.unroll(
            obs,
            actions=None,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=2,
            compute_loss=False,
            return_all_steps=True,
        )

    assert isinstance(all_steps_ar, list), f"Expected list, got {type(all_steps_ar)}"
    assert len(all_steps_ar) == nsteps, f"Expected {nsteps} steps, got {len(all_steps_ar)}"
    print(f"  Autoregressive mode returns list of {len(all_steps_ar)} tensors")
    for i, step in enumerate(all_steps_ar):
        print(f"    Step {i}: shape={step.shape}")

    # Verify autoregressive steps grow in time dimension
    for i in range(1, len(all_steps_ar)):
        assert all_steps_ar[i].shape[2] == all_steps_ar[i - 1].shape[2] + 1, \
            f"Autoregressive steps should grow by 1: step {i-1}={all_steps_ar[i-1].shape[2]}, step {i}={all_steps_ar[i].shape[2]}"
    print("  Autoregressive steps correctly grow in time dimension")

    print("\n  All assertions passed!")
    print("=" * 60)

    return True


def run_all_tests():
    """Run all tests for EEG JEPA unroll() function."""
    print("\n" + "#" * 60)
    print("# EEG JEPA UNROLL() FUNCTION TEST SUITE")
    print("#" * 60)

    results = {}

    # Parallel mode tests
    try:
        test_unroll_parallel_mode_output_format()
        results["unroll parallel mode output"] = "PASSED"
    except Exception as e:
        results["unroll parallel mode output"] = f"FAILED: {e}"

    try:
        test_unroll_parallel_mode_with_loss()
        results["unroll parallel mode with loss"] = "PASSED"
    except Exception as e:
        results["unroll parallel mode with loss"] = f"FAILED: {e}"

    try:
        test_infer_method()
        results["infer method"] = "PASSED"
    except Exception as e:
        results["infer method"] = f"FAILED: {e}"

    # Autoregressive mode tests
    try:
        test_unroll_autoregressive_mode_shapes()
        results["unroll autoregressive mode shapes"] = "PASSED"
    except Exception as e:
        results["unroll autoregressive mode shapes"] = f"FAILED: {e}"

    try:
        test_unroll_autoregressive_with_loss()
        results["unroll autoregressive with loss"] = "PASSED"
    except Exception as e:
        results["unroll autoregressive with loss"] = f"FAILED: {e}"

    try:
        test_unroll_return_all_steps_format()
        results["return_all_steps format"] = "PASSED"
    except Exception as e:
        results["return_all_steps format"] = f"FAILED: {e}"

    # Summary
    print("\n" + "#" * 60)
    print("# TEST SUMMARY")
    print("#" * 60)
    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result == "PASSED" else "FAIL"
        print(f"  [{status}] {test_name}: {result}")
        if result != "PASSED":
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# EEG JEPA Output Format Test Suite")
    print("#" * 60 + "\n")

    all_passed = run_all_tests()

    print("\n" + "#" * 60)
    if all_passed:
        print("# All tests completed successfully!")
    else:
        print("# Some tests FAILED - see details above")
    print("#" * 60)
