# Sweep Observations — sweep/mar30

## Exp 1: Baseline (defaults, epochs=30)
- **Run**: jrepxrw9 | **Commit**: f230a22
- **Config**: all defaults (lr=3e-4, std_coeff=1.0, cov_coeff=1.0, embed_dim=64, depth=4)
- **Results**: pred_loss=0.182, cosim=0.940, embed_std=0.449, probe_acc=0.500, val_reg=0.828
- **Observation**: Cosine similarity at 0.94 indicates significant representation collapse.
  Probe accuracy at chance (0.50). Pred loss is decreasing but representations aren't diverse.

## Exp 2: std_coeff=10.0 (DISCARDED)
- **Run**: t6iialwu | **Commit**: 7b44178 (reverted)
- **Config**: std_coeff=10.0 (10x increase), everything else default
- **Results**: pred_loss=0.161, cosim=0.996, embed_std=0.278, probe_acc=0.422, val_reg=0.833
- **Observation**: Collapse got WORSE despite 10x std regularization.
  - cosim 0.940 → 0.996 (nearly total collapse)
  - embed_std 0.449 → 0.278 (less variance)
  - VC loss dominated total loss (3.76/3.92) but didn't prevent collapse
  - Epoch 19 had a pred_loss spike to 2.66 (instability)
- **Hypothesis**: The std penalty is being minimized by the optimizer without actually
  diversifying representations — the model finds a way to satisfy the penalty locally
  while still collapsing globally. The collapse may be structural (architecture/masking)
  rather than a regularization strength issue.

## Exp 3: lr=1e-4 (DISCARDED)
- **Run**: fjz1riz8 | **Commit**: c7674a5 (to revert)
- **Config**: lr=1e-4 (3x lower), everything else default
- **Results**: pred_loss=0.189, cosim=0.952, embed_std=0.317, probe_acc=0.516, val_reg=0.832
- **Observation**: Lower lr didn't prevent collapse — cosim slightly worse (0.940→0.952),
  embed_std dropped (0.449→0.317). Pred loss slightly worse too (slower convergence).
  Grad norm lower (0.247 vs 0.414) indicating more stable but not more useful training.
  Probe acc marginally above chance (0.516) but not meaningful.
- **Conclusion**: Collapse is not caused by lr being too high. The default lr=3e-4 is fine.
  The problem is deeper — likely in the masking/architecture.

## Exp 4: min_context_fraction=0.30 (DISCARDED)
- **Run**: 20qsvrfm | **Commit**: fac59e5 (to revert)
- **Config**: min_context_fraction=0.30 (2x baseline 0.15), everything else default
- **Results**: pred_loss=0.182, cosim=0.940, embed_std=0.449, probe_acc=0.500, val_reg=0.828
- **Observation**: Results are IDENTICAL to baseline. The masking override had zero effect
  on any metric. W&B config confirms 0.30 was set. Two possibilities:
  1. min_context_fraction doesn't meaningfully change the masking — the mask collator
     may already produce masks with >30% context at default settings
  2. The model converges to the same collapsed attractor regardless of context amount
- **Action needed**: Inspect the actual mask statistics (what fraction of tokens are
  context vs masked) to understand if the masking is actually changing.

## Key Takeaways So Far
1. **VC regularization alone can't fix collapse** — increasing std_coeff made it worse.
   The model may be "gaming" the variance penalty.
2. **Collapse is the primary blocker** — until cosim comes down significantly, probes
   will stay at chance.
3. **lr doesn't explain collapse** — both 1e-4 and 3e-4 collapse similarly.
4. **min_context_fraction change had zero effect** — identical results to baseline.
   Need to verify the masking collator is actually respecting this parameter.
5. **Next directions to try** (prioritized):
   - Investigate masking code — verify masks are actually changing with config
   - Model capacity — depth=2 or embed_dim=32 to prevent shortcut collapse
   - EMA momentum — higher starting momentum (0.999) slows target encoder updates
