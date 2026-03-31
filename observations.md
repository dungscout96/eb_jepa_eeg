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

## Exp 5: 2x mask scales (~63% masked vs 38% baseline)
- **Run**: js7tp4cq | **Commit**: 38fa14e
- **Config**: short_ch=[0.15,0.30], short_p=[0.5,0.8], long_ch=[0.30,0.60], long_p=[0.7,1.0]
- **Results**: pred_loss=0.101, cosim=0.946, embed_std=0.527, probe_acc=0.484, val_reg=0.829
- **Note**: First attempt OOM'd — two processes ran simultaneously on GPU. Resubmitted OK.
- **Observation**: Pred loss much lower (0.101 vs 0.182) — the model handles the harder
  prediction task well. embed_std slightly up (0.527 vs 0.449). But cosim still ~0.94.
  The masking ratio increased substantially but collapse persists.
- **Insight**: Verified min_context_fraction was non-binding (exp4 explanation confirmed).
  With P=32, the mask blocks only cover ~38% at default scales. 2x scales reached ~63%.
  Even at 63% masking, collapse is unchanged. The model finds a near-constant representation
  that still achieves low prediction loss — suggesting the predictor is powerful enough to
  compensate even with limited encoder information.
- **Status**: Marginal improvement in embed_std, keep for now but collapse unsolved.

## Exp 6: predictor_depth=1 (DISCARDED)
- **Run**: m7uvrncw | **Commit**: 8f6411b (to revert)
- **Config**: predictor_depth=1 (halved from 2), everything else default
- **Results**: pred_loss=0.214, cosim=0.960, embed_std=0.062, probe_acc=0.438, val_reg=0.597
- **Observation**: Collapse got MUCH worse — embed_std cratered to 0.062 (from 0.449!),
  cosim up to 0.96. The weaker predictor couldn't compensate, but instead of forcing the
  encoder to be more informative, the whole system collapsed harder. val_reg=0.597 looks
  better but is misleading — with near-zero variance embeddings, the regression head
  simply learned the target mean.
- **Conclusion**: Reducing predictor capacity alone destabilizes training and accelerates
  collapse. The predictor needs enough capacity to learn the prediction task; without it,
  the gradient signal to the encoder degrades rather than becoming more informative.

## Exp 7: encoder_depth=2 (BEST SO FAR — KEEP)
- **Run**: yfzgaoy8 | **Commit**: e2b5a53
- **Config**: encoder_depth=2 (halved from 4), everything else default
- **Results**: pred_loss=0.193, cosim=0.947, embed_std=0.795, probe_acc=0.469, val_reg=0.803
- **Observation**: BEST result yet! While cosim is still ~0.94, embed_std nearly doubled
  (0.795 vs 0.449 baseline) and downstream probes improved across the board:
  - val_reg 0.803 vs 0.828 (lower = better)
  - cls_entropy_auc 0.580 vs 0.503 (notable improvement)
  - cls_luminance_auc 0.575 vs 0.549
  - reg_entropy_corr 0.146 vs 0.119
  - reg_luminance_corr 0.194 vs 0.164
- **Insight**: A shallower encoder can't take shortcuts as easily — with only 2 transformer
  layers, the encoder must preserve more input structure in its representations, leading to
  higher embed_std and better downstream performance despite similar global cosim.
  The cosim metric may be misleading — local structure can improve even when global
  similarity is high.
- **Status**: KEEP. Try combining with other improvements (2x masks, EMA momentum).

## Exp 8: encoder_depth=2 + 2x mask scales
- **Run**: 1od352tn | **Commit**: 2740405
- **Config**: encoder_depth=2, short_ch=[0.15,0.30], short_p=[0.5,0.8], long_ch=[0.30,0.60], long_p=[0.7,1.0]
- **Results**: pred_loss=0.056, cosim=0.951, embed_std=0.802, probe_acc=0.484, val_reg=0.803
- **Observation**: Combination is roughly on par with depth=2 alone. Pred loss much lower
  (0.056 — model easily handles harder masking task), but embed_std (0.80) and probe
  metrics are within noise of exp7. The 2x masks don't add meaningful value on top of
  the shallower encoder.
- **Conclusion**: The encoder depth is the dominant factor. 2x mask scales are orthogonal
  but not additive for representation quality.
- **Status**: Keep (on par with exp7), but depth=2 alone is simpler and equally effective.

## Exp 9: encoder_depth=2 + ema_momentum=0.999
- **Run**: uwxc8euh | **Commit**: dcb7251
- **Config**: encoder_depth=2, ema_momentum=0.999 (default 0.996)
- **Results**: pred_loss=0.156, cosim=0.947, embed_std=0.776, probe_acc=0.469, val_reg=0.804
- **Observation**: Essentially identical to depth=2 alone. EMA momentum=0.999 didn't
  add any value. The depth=2 finding is robust — it produces the same embed_std (~0.78-0.80)
  and probe metrics regardless of masking or EMA changes.
- **Status**: Discard (no improvement over simpler depth=2 alone).

## Key Takeaways So Far
1. **VC regularization alone can't fix collapse** — increasing std_coeff made it worse.
   The model may be "gaming" the variance penalty.
2. **Collapse is the primary blocker** — until cosim comes down significantly, probes
   will stay at chance.
3. **lr doesn't explain collapse** — both 1e-4 and 3e-4 collapse similarly.
4. **min_context_fraction change had zero effect** — identical results to baseline.
   Need to verify the masking collator is actually respecting this parameter.
5. **min_context_fraction was non-binding** — default masks only cover ~38% of 4128
   cells, well below the 85% cap. Confirmed by code analysis (P=32 with 129 channels).
6. **2x mask scales (63% masked) didn't break collapse** — pred_loss improved but
   cosim unchanged. The predictor compensates for limited encoder info.
7. **predictor_depth=1 made collapse much worse** — embed_std=0.062, the system needs
   a capable predictor to maintain any signal. Don't reduce predictor further.
8. **encoder_depth=2 is best so far** — embed_std=0.795, downstream probes improved.
   Shallower encoder forces more informative representations.
9. **depth=2 + 2x masks ≈ depth=2 alone** — masking doesn't add on top of shallower encoder.
10. **depth=2 + ema=0.999 ≈ depth=2 alone** — EMA momentum doesn't matter here.
11. **Next directions to try** (prioritized):
   - encoder_depth=2 + lr=1e-3 — smaller model may benefit from higher lr
   - encoder_depth=2 for 100 epochs — see if longer training helps probes
   - Try entirely different approach if lr=1e-3 doesn't help
