# Phase 1 — Pre-training Loss Analysis (2026-05-02)

**Source data:** `logs/issue8D_17927885..17927889.out` — Phase-D nw4_ws2 baseline,
5 enc seeds (42, 123, 456, 789, 2025), 100 epochs each. Per-epoch logged
metrics: train `loss`, `vc_loss`, `pred_loss`; val `reg_loss` (online probe MSE),
val per-feature `reg_<feature>_corr` (online linear probe correlation).

## Headline

The masked-prediction loss **does not decrease across 100 epochs of training**,
yet val stim correlations rise to a peak somewhere mid-training and then
**decline by epoch 99**. The encoder is **acquiring useful stimulus features
in early epochs and unlearning them in later ones**. We have been probing
the *latest* checkpoint, which is systematically worse than checkpoints from
the peak window.

## Per-seed summary

| seed | pred[0] → pred[99] | val_pos peak (ep) | val_pos @ ep99 | val_lum peak (ep) | val_lum @ ep99 |
|---:|---:|---:|---:|---:|---:|
| s42   | 1.18 → 1.39 | **+0.298 @ 72** | +0.025 | +0.245 @ 40 | +0.032 |
| s123  | 1.24 → 1.69 | +0.267 @ 51 | +0.218 | +0.314 @ 100 | +0.314 |
| s456  | 1.37 → 1.25 | +0.251 @ 32 | +0.110 | +0.207 @ 32 | +0.087 |
| s789  | 1.02 → 1.44 | +0.253 @ 91 | +0.120 | +0.202 @ 84 | +0.154 |
| s2025 | 1.03 → 1.46 | +0.232 @ 94 | +0.197 | +0.215 @ 77 | +0.196 |

Three patterns, every seed:
1. **Train pred_loss does not converge.** 4 of 5 seeds show pred_loss
   *increasing* over training; the 5th is roughly flat.
2. **Val stim correlations DO rise then peak.** Position correlation hits
   +0.23–0.30 at the peak — meaningfully above the +0.124 we measured
   probing the final (`latest.pth.tar`) checkpoint with kc + MLP.
3. **The peak is followed by a decline.** s42 lost 92 % of its peak
   pos-corr (+0.298 @ ep72 → +0.025 @ ep99). s456 lost 56 %. s2025 held
   most of it (lost only 15 %). Highly seed-dependent.

## What this means

### Train pred_loss is not a useful learning signal

The smooth_l1 loss against a moving EMA target is roughly stationary at ≈ 1.2
across training. This is consistent with:
- The target encoder is itself moving (EMA momentum < 1), so the target
  representation drifts as training proceeds. The student chases an
  oscillating target and the loss stays bounded.
- EEG is high-noise. The masked-token prediction problem has an irreducible
  noise floor; the loss is sitting at that floor from the start.
- VCLoss prevents collapse but doesn't push pred_loss down.

The implication: **we cannot use train pred_loss to decide when to stop or
whether more data would help.** We need a *task-relevant* val signal, and
the obvious one — val stim correlations from the online probe — shows the
peak-and-decline pattern above.

### The encoder learns stim, then drifts away from it

The fact that val correlations rise from chance to +0.25 over the first
~30–80 epochs proves the masked-prediction objective is *not orthogonal* to
stim. The encoder builds stim-relevant features incidentally — they emerge
because they help with masked-token prediction in the early-encoder /
short-distance regime.

The decline phase is the pathology. Possible mechanisms:
- **Shrinking effective predictor capacity.** As the EMA target gets smoother
  (representations stabilize and become low-rank), the predictor's job gets
  easier; the encoder is rewarded for producing low-rank features that the
  predictor can match. Stim is high-rank (channels and time), so the
  encoder discards it.
- **VCLoss saturation.** vc_loss decays from 0.15 to 0.006 by epoch 20.
  After that the regularizer has effectively stopped pushing the encoder
  away from a degenerate solution; the only signal is the prediction loss,
  which doesn't reward stim.
- **EMA target collapse onto subject-identity.** Identity features are
  trivially predictable from neighbors (a recording's spectral fingerprint
  is the same at all positions). The EMA target becomes a smoothed
  identity representation; the encoder is then trained to match it. Stim
  features, which vary across positions, get downweighted.

Without a separate experiment we can't pick between these. But all three
predict the SAME observation: **stim peaks early, then declines as the
target stabilizes.**

### Will more data help?

**No, not directly.** The loss is sitting at its noise floor; more data
won't reduce it. More data without a different loss would just give the
encoder more capacity to learn the same identity-dominated solution faster.

More data **could** help indirectly:
- Larger dataset = noisier per-step gradients = effective regularization,
  may slow the drift toward identity. Modest, no quantitative claim.
- More subjects = harder for the encoder to memorize subject-specific
  features. Marginal.

The bottleneck is **what the loss rewards**, not how many examples we have.

### Will more encoder layers help?

**Probably not, possibly worse.** Phase-1 finding 1 already showed block 1
of the existing 2-block encoder adds nothing for stim probes. Adding
block 2, 3, 4 would give the encoder more capacity to:
- Learn deeper identity-aware features (the path of least loss)
- Smear stim signal across more layers, giving more places for the target
  to "iron out" temporally-varying components

The loss-curve evidence specifically argues against scaling depth alone:
the encoder doesn't have a depth bottleneck (block 1 isn't being used);
it has a *loss-design* bottleneck.

Larger encoders **could** help if combined with a stim-aware auxiliary
loss — depth then has a use. Without that pairing, deeper = no better, and
plausibly worse.

## Concrete recommendations

### Immediate, no code change
1. **Evaluate at `epoch_40.pth.tar` instead of `latest.pth.tar` for stim
   probes.** Per the table, peak val_pos for s42 was at epoch 72 (+0.298),
   and the production probe at the *latest* checkpoint gave +0.124. A
   30–50 % stim-corr lift is plausible by checkpoint selection alone.
   Verification jobs queued (12 jobs, 4 epochs × 3 probe seeds, kc + MLP).
2. **Per-seed early-stopping by val stim corr.** Pick checkpoint per enc
   seed at its individual peak — the heterogeneity (epoch 32 to 100) means
   a single global epoch won't work.

### Phase 2 — Loss / objective design (high priority, drives findings)
1. **Auxiliary stim-regression head during SSL.** Add a small head on
   pooled encoder tokens that predicts the 4 stim features from EEG, with
   coefficient `env_coeff` (envelope-aux is already partially scaffolded
   in `MaskedJEPA.forward`). Tune so the encoder must keep stim. This is
   the most direct fix; literature support coming from the SSL research
   memo (in flight).
2. **Per-clip normalization** (or no per-recording norm). Removes the slow
   subject-identity signal the encoder is currently latching onto.
3. **Weighted loss with declining VCLoss schedule.** Currently vc decays
   from 0.15 → 0.006 by epoch 20. Try keeping it higher longer — may
   prevent the encoder from collapsing to identity.
4. **Predict at block 0**, not final (re-target the prediction loss).
   Phase-1 finding 1 showed stim is at block 0; reward keeping it there.

### What NOT to do
1. Don't add more pretraining data without a loss change.
2. Don't add more encoder layers as the only change.
3. Don't widen the predictor bottleneck — Phase-1 finding 3 (under mp;
   kc test pending) already showed it preserves stim.

## Open questions for Phase 2

- What does the val stim trajectory look like with auxiliary stim loss
  added? If the peak-then-decline disappears, that confirms the loss
  diagnosis.
- Is the EMA target encoder's representation collapsing to identity? Test
  by probing teacher checkpoints across epochs the same way we did
  Phase-1 final. If teacher-stim drops earlier than student-stim, target
  is the leading indicator and the suggested fix is masking schedule /
  stop-grad changes.
- Do we get the same peak-then-decline on a smaller subset (200 subjects
  vs 700)? If yes, it's a loss-pathology signature, not data-related. If
  no (smaller dataset peaks earlier and stays there), it's overfit.
