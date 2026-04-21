# Experiment 6: How It Works, Component by Component

Exp 6 is a masked JEPA trained on movie-watching EEG where every component solves a specific failure mode that earlier experiments hit. This document walks through the full pipeline and explains the role of each piece.

---

## The Core Problem We're Solving

Raw HBN EEG has the decomposition:
```
EEG = stimulus_response (~3μV, 0.3% var)
    + subject_fingerprint (~30μV, 26% var)
    + neural noise (~50μV, 73% var)
```

Stimulus SNR per single trial is **-24 dB**. A naive self-supervised model latches onto subject identity because that's 90× stronger than the stimulus signal. Every component of Exp 6 fights this.

---

## Data Pipeline (runs every `__getitem__`)

### 1. Load 4 contiguous 2-second windows per sample
```
eeg_raw: [4 windows, 129 channels, 400 samples @ 200Hz]
```
4 windows × 2s = 8s context per training sample. Enough to capture the slow delta/theta dynamics (125ms–1s periods) where ISC is highest.

### 2. Per-recording z-normalization (from Exp 2, `34fee98`)
```python
rec_mean = eeg.mean(dim=(0, 2), keepdim=True)   # [1, 129, 1]
rec_std  = eeg.std(dim=(0, 2), keepdim=True)
eeg = (eeg - rec_mean) / rec_std
```

**What it does:** Each recording is standardized independently.

**Why it helps:** Subject-specific amplitude scale (from skull thickness, impedance, setup) is ~26% of total variance. Global normalization preserves this — the encoder can discriminate subjects by "how loud" their EEG is. Per-recording norm destroys this channel of information at the input level. This was the single biggest win in Exp 2 (age bal_acc 0.483 → 0.587, +21%).

### 3. CorrCA spatial projection (Exp 6 core innovation)
```python
eeg = torch.einsum("wct,ck->wkt", eeg, W_corrca)
# [4, 129, 400] → [4, 5, 400]
```

**What it does:** Projects 129 raw channels onto 5 stimulus-driven "virtual channels" via a precomputed spatial filter matrix `W [129, 5]`.

**How `W` is computed (offline, once):**
1. Collect time-aligned EEG from all 701 training subjects watching the same 203s movie
2. Build two covariance matrices:
   - `R_b` (between-subject): what's shared across subjects at the same movie timepoint = **stimulus response**
   - `R_w` (within-subject): total variance per subject
3. Solve generalized eigenvalue problem: `R_b w = λ R_w w`
4. Top-k eigenvectors maximize Inter-Subject Correlation (ISC) — by construction, subject-specific patterns cancel out

Our top-5 ISC values: 0.019, 0.012, 0.006, 0.004, 0.002.

**Why it helps:** This is a **mathematical extraction of the stimulus signal**. 96% of the 129-channel EEG is noise+fingerprint; the 5 CorrCA components concentrate the 0.3% stimulus signal. Reduces task SNR from -24 dB to an estimated -8 to -12 dB, giving the encoder something to actually learn.

---

## Model Pipeline

### 4. Patch extraction & linear embedding
```python
patches = eeg.unfold(dim=3, size=50, step=30)  # [4, 5, 12, 50]
tokens = Linear(50, 64)(patches)                # [4, 5, 12, 64]
tokens = tokens.reshape(B, 5*4*12, 64)          # [B, 240, 64]
```

250ms patches (50 samples @ 200Hz) with 40% overlap. 240 tokens per sample. 250ms captures one theta cycle — well-matched to where CorrCA ISC is concentrated.

### 5. 4D Fourier positional encoding
```python
# For each token: (x, y, z, t)
# x,y,z = 3D position of the CorrCA component's peak-weight channel
# t = window_idx * 12 + patch_idx
pos_embed = fourier4d(positions) + mlp4d(positions)
tokens = tokens + pos_embed
```

**Why 4D:** EEG has both spatial (where on scalp) and temporal (when in window) structure. Giving the transformer position information for both lets it route attention to semantically related tokens.

**Why use peak-weight channel positions for CorrCA components:** A CorrCA component has no physical location, but we approximate each with the scalp position of its most heavily weighted input channel. This gives the encoder a sensible spatial prior.

### 6. Block masking (V-JEPA multi-block)
Generate 4 prediction blocks on the `[C=5, P=12]` grid, replicated across all T=4 windows:
- 2 short blocks (small channel+patch extent)
- 2 long blocks (large channel+patch extent)
- Keep at least 15% visible as context
- ~38% of tokens masked on average

**Key invariant:** If (channel c, patch p) is masked, it's masked in **all 4 windows**. This prevents temporal information leakage from EEG's high autocorrelation (adjacent windows are nearly identical).

**Why it helps:** The predictor must use context from visible tokens (other channels / earlier patches) to infer masked ones. This forces the encoder to produce representations with enough structure that nearby tokens are predictable from farther ones — i.e., it has to encode temporal dynamics, not just reconstruct the input.

### 7. Context encoder (trainable, depth=2 REVE Transformer)
```python
all_tokens = context_encoder.transformer(tokens)  # [B, 240, 64]
ctx_tokens = all_tokens[:, context_mask]           # [B, ~148, 64]  (visible only)
```

2 layers, 4 heads × 16 head_dim, GEGLU FFN. Small because 240 tokens from only 5 channels don't need a huge model. Sweep run with depth=4 actually hurt performance.

### 8. EMA target encoder (frozen)
```python
with torch.no_grad():
    tgt_tokens = target_encoder.encode_tokens(eeg, mask=None)  # [B, 240, 64]

# Per-step update:
momentum = cosine_schedule(0.996 → 1.0)
target.lerp_(context_encoder, 1 - momentum)
```

**What it does:** A slowly-updated copy of the context encoder produces the prediction targets.

**Why it helps:** This is the JEPA trick that avoids collapse. If targets were from the same encoder (shared weights), the predictor could output a constant and the encoder could map everything to that constant → zero loss, zero information. EMA decouples target from prediction: the target is yesterday's encoder, which knows slightly different things, so there's always a non-trivial gap to predict.

### 9. Narrow predictor bottleneck (`fb444af`, from Exp 1)
```python
ctx_proj = input_proj(ctx_tokens)          # 64 → 24
ctx_pos_proj = pos_proj(ctx_pos)           # 64 → 24
mask_tokens = self.mask_token.expand(...)  # [B, ~92, 24]  learnable
x = cat([ctx_proj + ctx_pos_proj, mask_tokens + tgt_pos_proj])
x = predictor_transformer(x)               # 2 layers in 24-dim
predictions = output_proj(x[:, n_ctx:])    # 24 → 64
```

**What it does:** Squeezes 64-dim context into a 24-dim space (37.5% ratio), processes, then unprojects back to 64.

**Why it helps:** This is the single most important architectural choice in Exp 6. Without the bottleneck, the predictor is powerful enough to do all the heavy lifting — the encoder only needs to produce raw patch info and the predictor figures out the structure. Representations collapse to a low-rank subspace (Exp 1's cosim 0.97 bug).

With the bottleneck, the predictor has limited capacity, so the encoder is forced to pre-organize information into structured features that fit through the 24-dim pipe. Participation ratio jumped from 3.4/64 to 19.1/64. Run B in the sweep confirmed: widening the bottleneck back to 48 destroyed position/luminance/contrast probes.

### 10. Prediction loss (Smooth L1 / Huber)
```python
pred_loss = F.smooth_l1_loss(predictions, tgt_tokens[masked].detach())
```

**What it does:** Predictor output must match the EMA target's embedding at each masked position.

**Why Smooth L1 over MSE:** MSE amplifies large errors quadratically, making training sensitive to outliers in EEG (which is plentiful — blinks, muscle artifacts). Smooth L1 behaves like MSE near zero but L1 for large errors — robust to outliers while still providing useful gradients near convergence. V-JEPA paper recommends this.

### 11. VCLoss anti-collapse regularizer
```python
projector = MLP(64 → 256 → 256)
fx = projector(ctx_tokens_flat)              # [B*n_ctx, 256]
std_loss = hinge_std_loss(fx, margin=1.0)    # push per-dim std → 1
cov_loss = covariance_loss(fx)                # off-diagonal → 0
reg_loss = 0.25 * std_loss + 0.25 * cov_loss
```

**What it does:**
- **Std loss:** Each embedding dimension should have std ≥ 1 (measured across the batch). Penalizes dead dimensions.
- **Cov loss:** Off-diagonal covariance entries should be 0. Penalizes redundant dimensions.

**Why it helps:** JEPA's prediction loss alone doesn't prevent collapse (map everything to same vector → pred_loss=0). The EMA target encoder makes this harder but not impossible. VICReg-style variance and covariance penalties explicitly guarantee embeddings span the full representation space.

**Why a projector MLP:** Applying VCLoss directly to the encoder's 64-dim output would constrain what the encoder can express. Projecting to 256-dim first lets the regularizer act in an expanded space, gently pushing the encoder toward diversity without micromanaging which directions it uses. VICReg paper found this projector is critical.

**Why 0.25 coefficients:** Iter 4 in the optimization sweep showed that cranking these to 1.0+ (VICReg default ratio) **destroys stimulus probes** — the regularizer starts dominating the prediction loss and forces embeddings into a uniform-distribution that throws away fine-grained information. 0.25 is the sweet spot for the -24 dB SNR regime.

### 12. Total loss & optimization
```python
total_loss = pred_loss + reg_loss   # That's it — no auxiliary losses
total_loss.backward()
torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
optimizer.step()                     # Adam, lr=5e-4
scheduler.step()                     # cosine decay 5e-4 → 1e-6, 5-epoch warmup
target_encoder.lerp_(context_encoder, 1 - ema_momentum)
```

No contrastive loss, no adversarial loss. Just JEPA.

### 13. Early stopping (`18a6dc9`)
Track best `val/reg_loss` (online probe), save `best.pth.tar` when it improves, stop after 20 epochs without improvement.

**Why it helps:** Exp 1 showed a clear U-shape in val loss — best around epoch 50, degraded by epoch 99 as the model started overfitting to prediction while losing downstream utility. Early stopping catches the model at its usefulness peak.

---

## Evaluation (the part that measures success)

Encoder is **frozen**. Fresh linear probes are trained on train-set embeddings, then evaluated on val/test.

- **Movie feature probes** (per clip, ~8s embeddings): regression and binary classification on contrast, luminance, position, narrative
- **Subject probes** (per recording, pooled over clips): age, sex

This is the standard SSL evaluation protocol — it measures how much linearly-accessible information is in the frozen representations, not how well we can fine-tune.

---

## How the Components Interact

Here's the causal chain from raw EEG to downstream probe accuracy:

```
129ch raw EEG
    │  (dominated by subject fingerprint)
    ▼
Per-rec norm              ← REMOVES amplitude fingerprint (~26% var gone)
    │
    ▼
CorrCA projection         ← EXTRACTS stimulus-driven subspace (5ch, 0.019 ISC)
    │  (now +15-20 dB stimulus SNR)
    ▼
Patch + position encoding ← Gives transformer spatial+temporal structure
    │
    ▼
Masked JEPA prediction    ← Forces encoder to learn predictable structure
    │
    ├─ EMA target          ← Provides non-trivial targets without collapse
    ├─ Narrow predictor    ← Bottleneck forces encoder to do the work
    └─ VCLoss              ← Guarantees embedding space is fully used
    │
    ▼
Frozen 64-dim embedding   ← Contains stimulus features linearly decodable
    │
    ▼
Linear probe              ← Measures: "did we encode stimulus content?"
```

Each step either removes subject information from the input or forces the model to organize stimulus information into something useful. The final result is a frozen encoder whose representations produce:

- Position corr 0.176±0.048 (vs 0.0 chance, p<0.001)
- Luminance corr 0.168±0.059 (vs 0.0 chance, p<0.001)
- Contrast corr 0.115±0.054 (vs 0.0 chance, p<0.001)
- Age bal_acc 0.637±0.024, Sex AUC 0.618±0.007

across 5 seeds. These are the first statistically significant movie-stimulus probes on this dataset.

---

## What's Doing the Heavy Lifting

Ranked by contribution:

1. **CorrCA projection** (Exp 6 core): +15-20 dB stimulus SNR. Without this, stimulus probes are at chance even with everything else.
2. **Per-recording normalization** (Exp 2): +21% age bal_acc alone. Removes the subject fingerprint that would otherwise dominate.
3. **Narrow predictor bottleneck** (Exp 1): Prevents collapse. Run B confirmed: widen it and stimulus encoding dies.
4. **EMA target encoder** (V-JEPA): Standard but essential. Without it, the predictor would match its own encoder trivially.
5. **VCLoss regularizer** (VICReg): Anti-collapse backstop. Coefficients tuned down to 0.25 to not overwhelm the fragile stimulus signal.
6. **Block masking**: Makes the prediction task non-trivial but not impossible.
7. **Smooth L1, early stopping, 4D pos encoding**: Polish — each buys a few percent, none are the breakthrough.

The bottom line: **input preprocessing (1+2) does most of the SNR work**, while the **architecture (3+4+5) prevents the model from sabotaging itself**. The masked prediction objective itself is standard — it's the combination of stimulus-aligned input and collapse-resistant architecture that makes it work.

---

## Exact Training Configuration

```bash
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --optim.epochs=100 \
    --optim.early_stopping_patience=20 \
    --model.encoder_depth=2 \
    --model.predictor_embed_dim=24 \
    --optim.lr=5e-4 \
    --optim.lr_min=1e-6 \
    --optim.warmup_epochs=5 \
    --data.norm_mode=per_recording \
    --data.corrca_filters=corrca_filters.npz \
    --loss.std_coeff=0.25 \
    --loss.cov_coeff=0.25 \
    --loss.pred_loss_type=smooth_l1
```

Offline preprocessing (once):
```bash
PYTHONPATH=. uv run --group eeg python scripts/compute_corrca.py \
    --output_path corrca_filters.npz \
    --n_components 5 \
    --task ThePresent
```

Probe evaluation (after training):
```bash
PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint=/path/to/best.pth.tar \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --splits=val,test
```
