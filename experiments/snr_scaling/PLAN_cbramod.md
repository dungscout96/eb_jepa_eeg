# E1.2 — An HBN-free warm start: CBraMod on the subject-scaling axis

Status: **infrastructure done, sweep pending** (drafted 2026-09-09, branch
`cbramod-experiments`).

## Why

Every warm-started arm in this study (E0.4, E0.6/addval, E1.0) starts from
`reve-base`. Its pretraining corpus is HBN R1–R9 — the val release R5 and the
test release R6 included, movie-watching recordings and all (memory note
`reve-pretraining-includes-hbn-r6`, verified 2026-09-05 against the REVE paper
and the authors' dataset card). So the paper's central pairing — *the warm
start sets the level, capacity sustains the slope* (E0.4 vs E0.5) — rests on
an encoder that had seen the evaluation subjects before alignment began. The
random-init arms are clean; the pretrained arm is "held out from cross-modal
alignment", not "unseen subjects".

The fix is not a caveat but a replication with a pretrained encoder that has
**never seen HBN**. Two candidates exist with public weights, both pretrained
on the Temple University corpus and neither on any HBN release.

## LUNA vs CBraMod — the decision

| | **CBraMod** (Wang et al., ICLR 2025) | **LUNA** (Döner et al., NeurIPS 2025) |
|---|---|---|
| pretraining corpus | TUEG only, ~9k h, 19-ch 10-20 unipolar | TUEG + Siena, ~21.9k h, 20/22/29-ch, TUEG in *bipolar* double-banana |
| HBN exposure | none | none |
| native rate / patch | **200 Hz, 1 s patch, 30 s sample** — our 2 s / 200 Hz windows map to (129, 2, 200) with no resampling | 256 Hz, 40-sample patch (0.156 s), 5 s samples — our data would need 200→256 resampling and 2 s ≠ integer patches (512/40) |
| channel handling | asymmetric conditional PE: depthwise (19,7) conv over the (channel, patch) grid; **no coordinates** | 3-D electrode coordinates → NeRF encoding; learned queries compress C channels into Q=4–8 latents |
| high-density precedent | fine-tuned on 62-ch SEED-V, 64-ch PhysioNet-MI, 64-ch BCIC2020-3 | 62-ch SEED-V ("unseen montage") |
| hidden dim / params | **200-d**, 4.9M | 64-d / 7M (Base), 96-d / 43M (Large), 128-d / 311M (Huge) |
| readout consequence | a 200-d pooled embedding for the 12-feature ridge probe | 64–128-d pooled embedding; the probe has less to read at every S |
| weights | `weighting666/CBraMod` (MIT), single `pretrained_weights.pth` | `PulpBio/LUNA` (CC BY-ND 4.0), three safetensors |
| port size | 2 files, ~340 lines, no new deps; state dict maps by a prefix rename | model + 5 module files, timm/einops RoPE blocks, a Lightning task; coordinates must be plumbed through the dataset |

**CBraMod, for four reasons in order of weight.** (1) Its native geometry is
ours — 200 Hz and 1 s patches — so the port introduces no resampling and no
patch-count compromise, and the only free knob is an input scale. (2) It is
the cheaper and better-conditioned readout: 200-d against LUNA's 64–128-d,
and at 4.9M parameters the full 26-cell sweep costs less than four REVE
cells. (3) The upstream paper fine-tunes the same weights on 62–64-channel
unipolar montages, which is the situation we are in. (4) MIT weights and a
two-file port that loads under the strict guard with zero missing or
unexpected keys, verified bit-exact against the upstream forward pass.

**What CBraMod costs, and must be disclosed.** It carries no electrode
coordinates: channel identity comes only from index order through a 19-wide
conv, so the spatial prior it brings to a 129-channel EGI net is weaker than
REVE's 4-D Fourier positions, and the pretraining saw 19 channels where we
present 129. The expected warm-start gain is therefore a *conservative* test
of the E0.4 claim: if a coordinate-free, 19-channel, clinical-EEG prior still
lifts the level, subject exposure was not what did it. If it does not, the
result is "this prior does not transfer", not "the REVE gain was leakage" —
LUNA becomes the next experiment, not a footnote.

## What runs

Two arms, paired cell-for-cell, on E0.4's split (pool R1–R4 + R7–R10 = 1863,
val R5, test R6) so the val CV probe stays the headline and the E0.4 null
protocol applies unchanged:

| arm | prefix | init | cells |
|---|---|---|---|
| warm | `e12cb` | `cbramod_eet_init.pth.tar` (TUEG weights) | S ∈ {50, 200, 701, 1400} × draws {11, 22, 33} + S=1863 |
| random | `e12cbr` | none | same 13 cells, same cohorts |

Everything else is `train_e04_cell.sbatch`'s protocol: soft_target_clip
α=0.5 / τ=0.05, lr 1e-4 adam warmup 5, batch 64, n_windows 1, 400 epochs at
`epoch_size=703` (4400 steps in every cell), `meta.seed=2026`, `save_every=25`.
Frozen config: [`clip_pretrain_cbramod.yaml`](clip_pretrain_cbramod.yaml) —
CBraMod's released geometry (d=200, 12 layers, 8 heads, patch 200, ff 800,
dropout 0.1) plus `input_scale=0.2` to put z-scored input in the µV/100
regime the weights were trained on, and `proj_dim=200` per the recipe's
no-expansion rule.

Readouts at fixed epoch 325 (the paper's convention), three per cell:
`probe_val` (Δr² vs the CBraMod-shape null), `tt_test` (r/ceiling on test),
`retr_test`. Four nulls at the CBraMod shape (`e12cb_*_random.json`, copied
to `e12cbr_*`). Aggregation via `aggregate_nested.py` per arm.

Scripts: `submit_e12_cbramod_sweep.sh`, `submit_e12_readouts.sh`,
`experiments/clip_pretraining/scene_clip_from_checkpoint/prepare_cbramod_checkpoint.py`.

## What counts as reproducing the main results

1. **Level.** At matched S, warm Δr² / random Δr² > 1 with non-overlapping
   draw spread at ≥ 4 of 5 S values. E0.4/E0.5 measured ~1.7×; any ratio
   clearly above 1 that is roughly constant in S reproduces the *shape* of the
   finding. A ratio that shrinks toward 1 with S would be new information.
2. **Slope.** Both arms rise monotonically in S; the local exponent over
   701→1863 is positive in the warm arm. E0.4's warm exponent was +0.30, its
   random-init depth-22 twin +0.27, depth-12 +0.05.
3. **Absolute level is not a target.** A 4.9M-parameter, 200-d encoder is
   expected to sit below the 69M REVE arm at every S; what is compared across
   studies is the warm/random ratio and the exponents, never raw Δr².

## Caveats known before the first job

- `input_scale=0.17` is measured, not assumed: 24 random R1 ThePresent
  recordings give a per-recording median channel std of 16.6 µV (IQR
  14.7–20.9), i.e. a z-score→µV/100 factor of 0.166. Both arms use the same
  value, so it cannot bias the level ratio, only the warm arm's headroom.
- Fixed epoch 325 for both arms: RESULTS_epoch_curve.md shows from-scratch
  cells peak later at high S. The 25-epoch grid is saved; a complement-selected
  re-read is a follow-up, not part of the preliminary claim.
- Δr² here is against the CBraMod-shape null. It must not be plotted on the
  same axis as E0.4's numbers without saying so; `aggregate_nested.py` will
  refuse a null with the wrong prefix, which is the intended friction.
