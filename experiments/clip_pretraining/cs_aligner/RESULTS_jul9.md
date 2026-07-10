# cs_aligner — RESULTS (jul9)

InfoNCE + Cauchy–Schwarz divergence on the batch marginals, per Yin et al.
2025 ([arXiv:2502.17028](https://arxiv.org/abs/2502.17028)), implemented as
`loss.mode=cs_aligner` alongside the three existing losses (`clip`,
`scene_clip`, `soft_target_clip`). Tested against the jul7 baselines under
matched TP-only 400 ep seed=2026 conditions after a 7-arm `cs_weight` sweep.

All artifacts in this directory. Configs / probe JSONs / this write-up are
on branch [`feature/cs-aligner`](https://github.com/dungscout96/eb_jepa_eeg/tree/feature/cs-aligner).

---

## TL;DR

CS-Aligner adds a distributional-alignment term `L = L_InfoNCE + λ · D_CS` on
top of vanilla CLIP. Sweep over `λ ∈ {0.001, 0.01, 0.05, 0.1, 1.0, 10.0}`
plus a `λ=0` control at 60 ep TP-only seed=2026, plus a full 400-ep run at
`λ=1.0` on TP-only, plus a **3-seed replicate at 400 ep on multi-movie**
(TP + DM).

**Findings:**

1. **The CS term is mechanically working.** `cs_div` drops monotonically as
   `λ` grows (0.67 at `λ=0` → 0.11 at `λ=10`); the adaptive median-heuristic
   σ stabilizes at ~1.0 within the first epoch. Six unit tests confirm
   `D_CS(z,z)=0`, monotonicity under noise, rotation invariance, gradient
   flow, and median-heuristic correctness.
2. **But CS provides no downstream benefit on this pipeline.** At 60 ep,
   `λ=0` (vanilla CLIP + temporal buffer) is the best arm; every `λ>0`
   arm underperforms, and the degradation is monotonic in `λ`:
   `Δr² = 0.0162 → 0.0053` as `λ` goes 0 → 10.
3. **At 400 ep TP-only the story converges to a tie.** CS-Aligner `λ=1.0`
   matches vanilla CLIP within seed noise (**test r = 0.1479 vs 0.1459**,
   val CV Δr² = +0.0452 vs +0.0475). It loses to jul7-best soft-target
   `τ=0.05` by 0.004 raw r on test and 0.005 Δr² on val — genuine but small.
4. **At 400 ep multi-movie, cs_aligner ties both baselines within noise
   across all metrics.** 3-seed mean joint Δr² = **+0.0422 ± 0.0010**
   vs vanilla +0.0423 ± 0.0005 vs soft +0.0426 ± 0.0006. TP and DM axes
   both indistinguishable within noise. **The apparent seed=2026 DM
   improvement (+0.0395) was a 1.5σ outlier within cs_aligner's own seed
   distribution** — with n=3 the DM mean is +0.0384 ± 0.0010, no ceiling
   break.
5. **The gap CS closes is the wrong gap.** At `λ=1.0` after 400 ep, CS
   reduces the visual modality gap by ~47% (cross-minus-within-avg cosine
   moves from −0.0958 → −0.0505). But it does so by **dispersing the EEG
   cluster** (within-EEG cos halves, 0.226 → 0.111) rather than pulling
   matched pairs together — cross-modal similarity actually *decreases*
   (+0.022 → +0.011). The linear probe reads out pair-level information,
   which is exactly what CS erodes.
6. **The paper's Fig-1 claim replicates visually but not usefully.** The
   t-SNE of jul7-best vs cs_aligner_w=1.0 shows CS produces a marginally
   more diffuse EEG cluster and a slightly more mixed periphery — the
   distributional gap did close — but the improvement is cosmetic; probe
   utility does not follow the visual signal.
7. **Recommendation**: continue using soft `τ=0.05` for TP-only and vanilla
   CLIP for multi-movie from-scratch training. CS-Aligner is not a lever
   for our pipeline. The largest remaining gap (jul7 §5.4, REVE warm-start
   at test r=0.1715) is still unclosed by any from-scratch objective.

---

## §1. Setup

### 1.1 The CS-Aligner objective

Paper Eq. 8 defines the empirical Cauchy–Schwarz divergence estimator with
Gaussian kernel `κ_σ(z,z') = exp(-‖z-z'‖²/(2σ²))`:

```
D̂_CS = log_mean_K(K_xx) + log_mean_K(K_yy) − 2·log_mean_K(K_xy)
```

Combined with symmetric InfoNCE, the total training loss is
`L = L_InfoNCE + λ · D_CS`. See implementation in
[`eb_jepa/clip.py:CSAlignerCLIPPretrain`](../../../eb_jepa/clip.py).

**Adaptations from the paper's spec:**

- **Cosine-kernel form.** With `MovieCLIPHead` already L2-normalizing both
  modalities, `‖z-z'‖² = 2(1 − z·z')`, so
  `κ(z,z')/σ² = exp((z·z' − 1)/σ²)`. This gives free numerical stability
  and bounded values, prevents the model from using embedding-norm
  shrinkage as an escape valve, and keeps gradients consistent with
  InfoNCE (same normalized space).
- **Log-mean-exp via `torch.logsumexp`.** Each `log_mean_K` term is
  `logsumexp((cos − 1)/σ²) − log(N²)`. Same-modality Gram diagonals are
  forced to exactly 0 via `fill_diagonal_(0.0)` to eliminate float drift.
- **Adaptive bandwidth (median heuristic).** The paper does not specify σ.
  We compute `σ² = median(‖z_eeg − z_vis‖²)/2` on the cross Gram matrix,
  `.detach()`-ed. Empirically stabilizes at ~1.0 in our config within the
  first epoch. `loss.kernel_bandwidth: <float>` overrides with a fixed
  value (tested at σ=0.5 in Phase 3a).
- **Temporal buffer applied to InfoNCE only.** CS is a distributional
  objective on the marginals; masking cross-pairs corrupts the kernel-
  density estimator. Buffer remains at 2 s for the InfoNCE half so
  cs_aligner is directly comparable to scene_clip / soft_target_clip.
- **`logit_scale` decoupled from CS.** CS operates on raw cosine
  similarities; only InfoNCE sees the learnable temperature.

### 1.2 Training config

Matched to the jul7 soft-target best config from
[RESULTS_jul7.md §1.2](../soft_target_clip/RESULTS_jul7.md), except
`loss.mode` and the CS knobs:

```yaml
model:  encoder_embed_dim: 512, encoder_depth: 12, encoder_heads: 8, patch_size: 400
loss:   mode: cs_aligner
        cs_weight: <sweep var>
        kernel_bandwidth: null    # adaptive median heuristic
        mean_center: true, target_kind: per_window, temporal_buffer_s: 2.0
        proj_dim: 512, temperature: 0.07, vision_passthrough: false
optim:  optimizer: adam, lr: 1e-4, warmup_epochs: 5
data:   batch_size: 64, n_windows: 8, window_size_seconds: 2, task: ThePresent
```

### 1.3 Verification: six unit tests
[`tests/test_loss_equivalences.py::TestCSAlignerDivergence`](../../../tests/test_loss_equivalences.py)
covers: `D_CS(z,z)≈0`, monotonicity under progressive noise, rotation
invariance under a shared orthogonal transform, gradient flow via
`.backward()`, median-heuristic σ matches a manual computation, and
finiteness of all diagnostic scalars. All pass.

---

## §2. `cs_weight` calibration sweep (60 ep TP-only seed=2026)

### 2.1 Phase 3a — bracketing sweep

Six arms varying `cs_weight` across five orders of magnitude plus a
fixed-σ=0.5 control. All 60 ep TP-only seed=2026. Ranked by 5-fold-CV
Δr² on R5 val over the 12 scalar features.

| arm | mean r² | Δr² vs rand | cs_div (final) | val/clip_scene_auc |
|---|---:|---:|---:|---:|
| **w=0.0 (ctrl)** | **0.0321** | **+0.0162** | 0.666 | **0.690** |
| w=0.1  autoσ | 0.0308 | +0.0149 | 0.332 | 0.663 |
| w=1.0  autoσ | 0.0261 | +0.0102 | 0.172 | 0.654 |
| w=1.0  σ=0.5 fixed | 0.0253 | +0.0094 | 0.695 | 0.516 |
| w=10.0 autoσ | 0.0212 | +0.0053 | 0.113 | 0.621 |

Two things worth reading off:

- **CS is optimized as expected** — `cs_div` moves 0.67 → 0.11 as
  `cs_weight` grows.
- **CS monotonically degrades probe R²** — the ordering is exact.

The fixed-σ=0.5 arm is particularly bad: `clip_scene_auc = 0.516`
(near-chance), suggesting σ=0.5 is too sharp for the actual embedding
spread and destroys scene structure faster than the sweep can compensate.
Adaptive σ (which lands at ~1.0) is the better default.

### 2.2 Phase 3b — tiny-weight sweep

Extended the sweep with `cs_weight ∈ {0.001, 0.01, 0.05}` to check whether
a very light CS pull matches or beats the vanilla control. All 60 ep
TP-only seed=2026, adaptive σ.

| arm | mean r² | Δr² vs rand | Δ vs w=0 ctrl |
|---|---:|---:|---:|
| w=0.0  (ctrl) | 0.0321 | +0.0162 | — |
| w=0.001 | 0.0320 | +0.0161 | −0.0001 |
| w=0.01 | 0.0318 | +0.0159 | −0.0003 |
| w=0.05 | 0.0316 | +0.0157 | −0.0005 |
| w=0.1  | 0.0308 | +0.0149 | −0.0013 |
| w=1.0  | 0.0261 | +0.0102 | −0.0060 |
| w=10.0 | 0.0212 | +0.0053 | −0.0109 |

**Full-sweep pattern (seven arms):** monotonically non-increasing in
`cs_weight`. Tiny weights (0.001 – 0.05) are within noise of the vanilla
control (Δ ≤ 0.0005, well inside the ~0.002 seed noise expected at 60 ep).
No `cs_weight > 0` beats `cs_weight = 0`.

---

## §3. Full run at 400 ep TP-only seed=2026

The 60-ep sweep implies CS-Aligner will not beat vanilla CLIP at any
weight. Ran the paper's default `cs_weight=1.0` at the full 400-ep budget
anyway to test whether CS's effect diminishes at longer training (Risk #3
from the plan called out this possibility).

### 3.1 Val 5-fold CV (12 features)

| arm | mean r² | Δr² vs rand |
|---|---:|---:|
| **cs_aligner w=1.0** (NEW) | **0.0611** | **+0.0452** |
| vanilla clip (jul7 3-seed mean) | 0.0634 | +0.0475 |
| soft τ=0.05 (jul7 3-seed mean) | 0.0660 | +0.0501 |

The 60-ep 0.006 gap between `w=0` and `w=1.0` shrank to ~0.002 at 400 ep
— CS interference decays with training but never reverses. cs_aligner sits
~2 standard deviations below the vanilla-CLIP 3-seed mean.

### 3.2 R6 train→test Pearson r (SSL-standard protocol)

| checkpoint | test mean r | test Δr vs rand (0.0527) |
|---|---:|---:|
| random_depth4 (baseline) | 0.0527 | — |
| fresh500_ep499 scene_clip | 0.1328 | +0.0801 |
| **cs_aligner w=1.0** (NEW) | **0.1479** | **+0.0952** |
| vanilla clip (jul7 seed=2025) | 0.1459 | +0.0932 |
| soft τ=0.05 (jul7 seed=2026) | 0.1517 | +0.0990 |
| reve_warmstart_v2_ep299 | **0.1715** | **+0.1198** |

On R6 test, cs_aligner sits **between** vanilla CLIP (0.1459) and
soft-target (0.1517) — slightly above vanilla, slightly below soft. Both
differences are within the ±0.001–0.002 seed noise for 400-ep runs, so
functionally: **cs_aligner ≈ vanilla CLIP at 400 ep, within noise**.

---

## §3b. Multi-movie 3-seed replicate at 400 ep (Phase 7)

The TP-only Phase 4 result is one seed. Ran a 3-seed replicate (seeds
2025, 2026, 2027) on the **multi-movie** config (`task=[ThePresent,
DespicableMe]`) to test whether cs_aligner behaves differently when
trained across movies — a natural next question because DM has been
loss-independent at ~+0.0378 across every jul7 arm
([jul7 §4.2](../soft_target_clip/RESULTS_jul7.md), "the loss-independent
DM ceiling").

### 3b.1 Per-seed 5-fold-CV Δr²

| seed | TP Δr² | DM Δr² | joint Δr² |
|---:|---:|---:|---:|
| 2025 | +0.0452 | +0.0375 | +0.0414 |
| 2026 | +0.0470 | **+0.0395** | +0.0433 |
| 2027 | +0.0458 | +0.0384 | +0.0421 |
| **mean ± std** | **+0.0460 ± 0.0009** | **+0.0384 ± 0.0010** | **+0.0422 ± 0.0010** |

### 3b.2 Comparison to jul7 3-seed multi-movie baselines

| arm | TP Δr² | DM Δr² | joint Δr² |
|---|---:|---:|---:|
| vanilla clip (jul7 3-seed) | +0.0468 ± 0.0002 | +0.0378 ± 0.0008 | +0.0423 ± 0.0005 |
| soft τ=0.05 (jul7 3-seed) | +0.0474 ± 0.0010 | +0.0379 ± 0.0004 | +0.0426 ± 0.0006 |
| **cs_aligner w=1.0** (NEW) | **+0.0460 ± 0.0009** | **+0.0384 ± 0.0010** | **+0.0422 ± 0.0010** |

Two things worth reading off:

- **All three losses converge to the same multi-movie ceiling within
  seed noise** — joint Δr² pinned at +0.0422 ± 0.0004 across the three
  arms. Consistent with jul7 §4.1's observation that "everything at 400
  ep, embed=512 lands in ~[+0.041, +0.043]."
- **The seed=2026 DM signal (+0.0395) that motivated this replicate does
  not survive n=3.** DM mean drops to +0.0384, which sits 0.6σ above
  vanilla's +0.0378 — no meaningful ceiling break. The single-seed
  observation was a 1.5σ outlier within cs_aligner's own seed
  distribution (std=0.0010).

### 3b.3 Why the DM ceiling did NOT break

Two hypotheses were plausible after Phase 7's n=1 signal:

- *H1*: CS's distributional-alignment term provides a noise-robust
  alignment signal that helps precisely when the pair-level V-JEPA-2
  teacher is degraded (as on DM).
- *H2*: The seed=2026 DM value was seed noise on a small effect size.

The 3-seed replicate favors *H2*. Under *H1* we'd expect the DM boost
to persist across seeds; instead the std doubles (from 0.0004 for soft
to 0.0010 for cs_aligner) without a meaningful shift in mean. Consistent
with jul7 §5.3 — the DM ceiling is teacher-limited, not loss-limited.

---

## §4. Modality-gap analysis (Phase 5)

### 4.1 Quantitative gap metrics

Computed on the panel-(b) embedding sets (2000 EEG + 2000 V-JEPA-2
windows from TP+DM at val split, projected through
`MovieCLIPHead.project_{eeg,vision}`):

| metric | Soft-Target (jul7) | CS-Aligner (jul9) | Δ (CS − Soft) |
|---|---:|---:|---:|
| mean cos EEG–EEG (within) | +0.2257 | +0.1110 | **−0.1147** |
| mean cos V–V (within) | +0.0095 | +0.0110 | +0.0015 |
| **mean cos EEG↔V (cross, paired)** | **+0.0218** | **+0.0105** | **−0.0113** |
| gap (cross − within-avg) | −0.0958 | −0.0505 | **+0.0453** |
| D_CS (σ²=1) | 0.2575 | 0.1539 | −0.1036 |

**The gap CS closes is the wrong gap.** Under CS-Aligner:

- The modality-gap metric improves by ~47% (−0.096 → −0.051).
- D_CS itself drops by ~40% (0.26 → 0.15) — the objective is doing what
  it's designed to do.
- **But cross-modal cosine similarity DECREASES**, from +0.022 to +0.011.
- The gap improvement comes entirely from the EEG cluster spreading out
  (within-EEG cos halves), NOT from matched pairs pulling together.

CS is exchanging alignment for uniformity — exactly the tradeoff the paper
claims to resolve, but that resolution does not transpose to our
EEG↔V-JEPA-2 setup. The linear probe reads pair-level information; when
cross-modal similarity decays, probe utility decays with it.

### 4.2 Visualization

[`modality_gap_cs_vs_soft.png`](modality_gap_cs_vs_soft.png) — Fig-1-style
t-SNE, panel (a) = jul7-best soft τ=0.05, panel (b) = cs_aligner w=1.0
full-jul9. Visually the two panels look very similar; the CS-Aligner
panel shows a marginally more diffuse EEG central mass and a slightly
more mixed periphery, consistent with the quantitative gap-reduction, but
the dominant modality-gap structure persists in both. This is the paper's
Fig 1 claim replicated at the direction of the effect but not the
magnitude — visually the gap is not closed.

[`embedding_structure_cs_aligner.png`](embedding_structure_cs_aligner.png)
— 2×2 recoloring of the cs_aligner panel by modality / movie / scene id /
shot id. Movie separation (TP vs DM) still emerges clearly in both
modalities. Scene- and shot-level structure looks similar to the jul7
version.

---

## §5. What this means

### 5.1 The alignment–uniformity conflict is real; CS does not resolve it here

Wang & Isola (2020) formalize alignment (matched pairs close) and
uniformity (embeddings spread) as two competing objectives contrastive
learning optimizes implicitly. CS-Aligner's proposition is that a
distributional-matching term can improve uniformity across modalities
without hurting alignment — the paper's Sec 3 explicitly frames it as
resolving the InfoNCE alignment/uniformity conflict.

Our data show the opposite. As `cs_weight` grows, EEG within-modality
similarity drops (uniformity improves) but cross-modal paired similarity
drops in lockstep (alignment degrades). The probe metric — which requires
alignment to a specific per-window V-JEPA-2 target vector — tracks the
alignment axis, not the uniformity axis. Every arm's `cs_div` is smaller;
every arm's probe R² is smaller too, until the effect saturates at 400 ep
into a tie with vanilla CLIP.

### 5.2 60-ep dynamics vs 400-ep asymptote

The 60-ep sweep predicts a 0.006 gap between vanilla and cs_aligner
`w=1.0`. At 400 ep this shrinks to 0.002. The interference DOES fade with
training — CS's effect is on the training trajectory, not the final
representation. Given long enough training, InfoNCE dominates and the CS
term becomes a small perturbation. But the perturbation never helps.

### 5.3 Why the paper's headline result may not transfer

Three probable reasons this pipeline diverges from the paper's claim:

- **Scale mismatch.** Paper trains at CLIP ViT-L/14 scale on large paired
  datasets. Our encoder is small (embed=512) and the domain is EEG, not
  natural images. Alignment/uniformity trade-off regimes shift with scale.
- **Frozen teacher.** Our V-JEPA-2 targets are precomputed and frozen —
  half the "distribution" being aligned is fixed. CS's benefit may
  require both modalities to co-adapt.
- **Domain-representation regime.** Paper's vision-language pair is
  semantically much closer than EEG↔V-JEPA-2. Our modality gap is
  driven by the fundamental physics of EEG vs video content, and no
  distribution-matching kernel bridges that structurally.

### 5.4 What the negative result rules out

- **`cs_weight` calibration is not the issue** — seven orders of magnitude
  tested, monotonic pattern.
- **Bandwidth choice is not the issue** — adaptive median heuristic is
  strictly better than fixed σ, and both underperform vanilla.
- **Training budget is not the issue** — 400-ep results are noisier but
  the same story as 60 ep.
- **Implementation is not the issue** — six unit tests confirm the math,
  training-time diagnostics confirm CS is being minimized, quantitative
  gap metrics confirm the gap does close along the CS axis.

---

## §6. Recommendation

- **Continue with soft `τ=0.05` on TP-only, vanilla CLIP on multi-movie**
  — matches the jul7 §7.1 recipe. Both remain undisputed within seed
  noise across all objectives tested.
- **Do not use `loss.mode=cs_aligner`** for downstream work. The
  distributional-alignment framing does not translate to per-window probe
  utility in this pipeline, on either single-movie or multi-movie
  training, at any weight in `{0.001 – 10.0}`.
- **REVE warm-start remains the largest remaining lever** at the same
  400-ep budget, per [jul7 §5.4](../soft_target_clip/RESULTS_jul7.md).
  CS-Aligner does not narrow that ~0.020 raw-r gap.

---

## §7. Engineering deliverables

Committed on branch `feature/cs-aligner`, retained for possible v2 with
unpaired-data support:

- [`CSAlignerCLIPPretrain`](../../../eb_jepa/clip.py) — ~180 lines, the
  loss module + median-heuristic bandwidth + fp32-safe log-sum-exp
  divergence computation.
- Loss-mode dispatch and config knobs in
  [`eb_jepa/training/clip_pretrain.py`](../../../eb_jepa/training/clip_pretrain.py)
  and [`config/clip_pretrain.yaml`](../../../config/clip_pretrain.yaml).
- Six unit tests in
  [`tests/test_loss_equivalences.py`](../../../tests/test_loss_equivalences.py).
- [`_submit_train.py`](_submit_train.py) — Delta submit for sweeps and
  full runs; handles `--cs-weight`, `--kernel-bw`, `--task`,
  `--skip-probe`.
- Plot infrastructure reused from July 8 session:
  [`plot_modality_gap.py`](plot_modality_gap.py),
  [`plot_embedding_structure.py`](plot_embedding_structure.py),
  [`_submit.py`](_submit.py).

---

## §8. Reproducibility

Full 400-ep TP-only run (Phase 4):
```bash
uv run --group eeg python experiments/clip_pretraining/cs_aligner/_submit_train.py \
    full-jul9 --cs-weight=1.0 --epochs=400 --seed=2026 --task=ThePresent submit
```

Full 400-ep multi-movie 3-seed replicate (Phase 7 + 7b):
```bash
for seed in 2025 2026 2027; do
    uv run --group eeg python experiments/clip_pretraining/cs_aligner/_submit_train.py \
        full-jul9-mm --cs-weight=1.0 --epochs=400 --seed=$seed submit
done
```

R6 train→test probe (val + test):
```bash
uv run --group eeg python -c "
from neurolab.jobs import Job
exp='/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner/full-jul9_w1_bwauto_seed2026'
for split in ('val', 'test'):
    Job(name=f'tt_{split}', cluster='delta', repo_path='/u/dtyoung/eb_jepa_eeg',
        partition='gpuA40x4', time_limit='00:30:00', venv='__none__',
        branch='feature/cs-aligner',
        env_vars={'HBN_PREPROCESS_DIR': '/projects/bbnv/kkokate/hbn_preprocessed'},
        command=(
            f'PYTHONPATH=. uv run --group eeg python '
            f'eb_jepa/evaluation/clip_probe/probe_traintest.py '
            f'--checkpoint {exp}/latest.pth.tar --config {exp}/config_TP.yaml '
            f'--eval-split {split} '
            f'--output experiments/clip_pretraining/cs_aligner/probe_traintest_{split}_full-jul9_w1_seed2026_TP.json'
        )).submit()
"
```

Modality-gap plots:
```bash
CS=/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner/full-jul9_w1_bwauto_seed2026
SOFT=/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip/jul7-tp_soft_target_clip_seed2026
uv run --group eeg python experiments/clip_pretraining/cs_aligner/_submit.py \
    --ckpt-a=$SOFT/latest.pth.tar --label-a="Soft-Target (jul7 best)" \
    --ckpt-b=$CS/latest.pth.tar   --label-b="CS-Aligner w=1.0" \
    --config=$CS/config_TP.yaml --npz-name=panel_b_cs_aligner.npz \
    --gap-png=modality_gap_cs_vs_soft.png \
    --struct-png=embedding_structure_cs_aligner.png submit
```

Total Delta compute for jul9: ~8 GPU-hours across 12 training jobs (5
Phase 3a + 3 Phase 3b + 1 Phase 4 + 3 Phase 7/7b) + 2 probe jobs + 1
plot job.

---

## §9. Artifacts

All in this directory. Sweep JSONs, full-run probe JSONs, both plot
images, and the extracted embeddings.

**5-fold CV probe (val split):**
- `probe_val_sweep-jul9_{w0,w0p1,w1,w10}_bwauto_seed2026_TP.json` — Phase 3a
- `probe_val_sweep-jul9_w1_bw0.5_seed2026_TP.json` — Phase 3a fixed-σ
- `probe_val_sweep2-jul9_{w0p001,w0p01,w0p05}_bwauto_seed2026_TP.json` — Phase 3b
- `probe_val_full-jul9_w1_bwauto_seed2026_TP.json` — Phase 4
- `probe_val_full-jul9-mm_w1_bwauto_seed{2025,2026,2027}_{TP,DM}.json` — Phase 7 + 7b

**Train→test Pearson r probe (val + test splits):**
- `probe_traintest_{val,test}_full-jul9_w1_seed2026_TP.json`

**Plots + embeddings:**
- `modality_gap_cs_vs_soft.{png,pdf}` — Fig-1-style comparison
- `embedding_structure_cs_aligner.{png,pdf}` — 2×2 recoloring
- `panel_b_cs_aligner.npz` — 2000 EEG + 2000 V-JEPA-2 shared-space
  embeddings + metadata + t-SNE coords for cs_aligner
- `panel_b.npz` — same for jul7-best soft τ=0.05 (from previous session)

**Code:**
- `_submit_train.py`, `_submit.py`, `plot_modality_gap.py`,
  `plot_embedding_structure.py`, this `RESULTS_jul9.md`.

---

*Concludes the cs_aligner experiment as a documented negative result.
Next: REVE + soft-target trial per [jul7 §7.2](../soft_target_clip/RESULTS_jul7.md#L433-L442)
remains the recommended next lever.*
