## 03/01/2026
### Summary
The experiment pipeline is now sound and ready for experimentation. Here's what was done:

**Issues Found & Fixed**
1. **REVE encoder missing channel positions** - The REVE model requires EEG channel positions for its attention mechanism. Fixed by extracting `chs_info` from the GSN-HydroCel-129 standard montage and passing it to the encoder.

2. **`n_times` mismatch** - REVE was hardcoded with `n_times=1000` but actual data has `n_times=200` (2s windows at 100Hz). Made `n_times` a parameter auto-detected from data.

3. **Massively overparameterized model** - Default REVE (70M params) for 20 training recordings. Reduced to a small config (64 embed_dim, 4 depth, 4 heads → 755K params) by making REVE architecture configurable via the YAML config.

4. **EEG data not normalized** - Values were in volts (~1e-5 scale), causing the encoder to produce constant representations regardless of input. Added per-channel z-normalization computed from training set and applied to both train and val sets.

5. **Regression probe LR too low** - Was `lr/10`, removed the reduction since there are very few gradient updates.

6. **VC loss coefficients too high** - `cov_coeff=100, std_coeff=10` caused regularization to dominate. Reduced both to 1.0.

**wandb Integration**
- Enabled in config (`log_wandb: true`)
- Per-step training metrics: `train_step/{jepa_loss, vc_loss, pred_loss, reg_loss, cls_loss, std_loss, cov_loss}`
- Per-epoch validation metrics: per-feature R2, correlation, accuracy, balanced accuracy, AUC
- W&B project: `eb_jepa` at https://wandb.ai/sccn/eb_jepa

**Results After Fixes**
- JEPA loss converges (1.5 → 0.60)
- Encoder representations are now diverse across time (std=2.35) and recordings (diversity=1.53)
- Probes show above-chance performance (entropy bal_acc: 0.54, contrast_rms corr: 0.09)
- All losses are balanced (VC0.44, pred0.16, reg0.44, cls0.69)

**Training dynamic observation**
> The train/cov_loss is going up while train/std_loss is going down. Does that mean the features are getting more and more correlated?

Yes, exactly. The covariance loss penalizes off-diagonal elements of the covariance matrix, so it going up means the encoder features are becoming more correlated over training.

The dynamics make sense when you consider the interplay:

std_loss is decreasing (0.60 → 0.35): The encoder is successfully spreading features to have higher variance (good — prevents collapse)
pred_loss is decreasing (0.20 → 0.15): The predictor is getting better at predicting next states
cov_loss is increasing (0.01 → 0.09): As the encoder learns to encode more information from the EEG and make features more variable, it's doing so by creating correlated features
This is a common tension in VC-regularized models: as std_loss pushes features to have high variance (anti-collapse), the easiest way for the encoder to do that while also minimizing pred_loss is to make features that are correlated copies of each other. The cov_loss is supposed to counteract this, but with cov_coeff=1.0 it's being overpowered.

The fix is to increase cov_coeff to penalize the growing correlations more strongly. The original config had cov_coeff=100 which was too aggressive with the old 70M-param model, but with the smaller model and normalized data, a moderate increase should help.

### Ready for Experimentation

You can now experiment with:

- Architecture: model.encoder_embed_dim, encoder_depth, encoder_heads, encoder_head_dim, dstc, hpre
- Training: optim.lr, optim.epochs, loss.std_coeff, loss.cov_coeff
- Data: data.n_windows, data.window_size_seconds, data.batch_size
- Overrides via CLI: `PYTHONPATH=. .venv/bin/python experiments/eeg_jepa/main.py --model.dstc=128 --optim.lr=1e-3`