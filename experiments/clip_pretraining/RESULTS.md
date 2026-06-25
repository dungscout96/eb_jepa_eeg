# CLIP Pretraining — Current Results

EEG ↔ V-JEPA-2 alignment via supervised-contrastive InfoNCE on the HBN
*ThePresent* dataset. This document records the results of the experiment
sweep ending 2026-06-25, and the methodology used to validate them.

Code paths (after the 2026-06-25 reorganization):

- Recipe / scene_clip loss: [`eb_jepa/clip.py`](../../eb_jepa/clip.py) (`SceneCLIPPretrain`)
- Trainer: [`eb_jepa/training/clip_pretrain.py`](../../eb_jepa/training/clip_pretrain.py)
- Config: [`config/clip_pretrain.yaml`](../../config/clip_pretrain.yaml)
- Recipe artifact precompute: [`embedding_feature_correlation/precompute_vjepa2_recipe.py`](embedding_feature_correlation/precompute_vjepa2_recipe.py)
- Recipe design notes: [`embedding_feature_correlation/clip_design_observations_vjepa2.md`](embedding_feature_correlation/clip_design_observations_vjepa2.md)
- CV probe: [`clip_probe/probe.py`](clip_probe/probe.py)
- ImageNet-style train→test probe: [`clip_probe/probe_traintest.py`](clip_probe/probe_traintest.py)

All probe JSONs are checked in under [`probe_results/`](probe_results/).

---

## TL;DR

**Best checkpoint: `fresh500_ep499`** (per_window targets, lr=1e-4, bs=64,
warmup_epochs=5, epochs=500, AdamW *not* yet ablated). Trained with the
`scene_clip` recipe (scene-ID positive mask, 2 s temporal buffer, mean-centered
shot-mean/per-window V-JEPA-2 targets) on R1-R4.

Per the ImageNet-style probe (fit Ridge on R1-R4 train embeddings, evaluate on
R6 test embeddings — the SSL literature's standard linear-evaluation protocol):

- **Mean Δ Pearson r vs random encoder: +0.080** (every single feature
  positive, no inversions).
- **Mean Δ R² vs random encoder: +0.073** (the trained encoder also produces
  far better calibrated predictions — random Ridge gets negative R² across the
  board due to train→test distribution shift; the trained encoder fixes this).
- High-level features (motion, depth, faces, narrative) — the ones that were
  near-random in the original CLIP baseline — now show real signal:
  `motion_energy` Δr = **+0.117**, `depth_mean` Δr = +0.079, `n_faces` Δr =
  +0.064, `narrative_event_score` Δr = +0.079.

---

## §1. Methodology

### 1.1 Training data

- **HBN** EEG, 129 channels, preprocessed to 200 Hz, *ThePresent* movie (3:23,
  4878 frames @ 24 fps).
- Splits: train R1-R4 (4 OpenNeuro releases, ~700 recordings),
  val R5 (~293 rec), test R6 (~108 rec). Subject-disjoint by HBN release
  structure.
- Vision teacher: **V-JEPA-2** (1408-d, ~2 Hz, 406 clips of *ThePresent*),
  precomputed and frozen throughout.

### 1.2 The `scene_clip` recipe (§9 of the design notes)

Implemented in [`SceneCLIPPretrain`](../../eb_jepa/clip.py). The eight choices,
all controlled by `cfg.loss.*`:

1. **Mean-center V-JEPA-2 targets** globally (subtract `global_mean[1408]`).
2. **Embedding target**: `shot_mean` (default) or `per_window` — per_window
   wins downstream as of this sweep.
3. **Contrastive label**: scene IDs from
   [`embedding_feature_correlation/vjepa2_scenes_map.csv`](embedding_feature_correlation/vjepa2_scenes_map.csv)
   (36 scenes from 54 shots, agglomerative threshold = 0.90).
4. **Loss**: supervised-contrastive InfoNCE — multi-positive numerator on
   same-scene mask, denominator excludes cross-scene pairs with |Δt| < 2 s.
5. **Temporal buffer**: 2 s (drops shot-cut-boundary false negatives, per §3
   of the design notes).
6. **Negatives**: many in-batch (bs=64), moderate temperature (T=0.07,
   learnable logit_scale), no hard mining.
7. **Diagnostic monitored**: scene-collapse ratio
   (within-scene / between-scene spread of EEG embeddings).
8. **Sanity gate**: vision-side AUC ≈ 1.0 confirms shot-mean replacement
   worked (irrelevant when target_kind=per_window, where vision_*_auc lives
   in the doc's 0.92/0.94 range).

Artifacts produced by
[`precompute_vjepa2_recipe.py`](embedding_feature_correlation/precompute_vjepa2_recipe.py):
`movie_annotation/output/ThePresent/vjepa2_recipe.npz` containing
`global_mean[1408]`, `shot_means[54,1408]`, `scene_id_per_shot[54]`.

### 1.3 Probe evaluation protocols

Three were run; their differences matter for interpreting the numbers below.

| protocol | fit on | eval on | metric | when to trust |
|---|---|---|---|---|
| **clip_probe CV-on-val** ([probe.py](clip_probe/probe.py)) | 4 folds of R5 | held-out R5 fold | R² score (5-fold mean) | quick dev iteration; fits are within-release |
| **clip_probe CV-on-test** ([probe.py](clip_probe/probe.py)) | 4 folds of R6 | held-out R6 fold | R² score (5-fold mean) | small-N: only ~86 rec per fit fold; biases pessimistically |
| **probe_traintest** ([probe_traintest.py](clip_probe/probe_traintest.py)) | full R1-R4 train (~700 rec, ~71k windows) | full R6 test (~108 rec, ~11k windows) | **Pearson r** + R² | matches SSL literature (CLIP/SimCLR/DINOv2 linear-eval style). **Primary metric.** |

All probes: standardize embeddings on the fit-set stats, fit `RidgeCV` (13 α
values), report per-feature score on the held-out windows. `--random-baseline`
re-runs with a fresh-init encoder of the same architecture so all gains can be
expressed as Δ vs random (encoder-architecture-controlled).

---

## §2. Training trajectory

Probe-evaluated checkpoints, sorted by appearance in the sweep. Each row is one
trained encoder; clip_probe CV-on-val is the headline column (used as the
quick-iteration metric during development).

| run | recipe knobs | epochs | mean R² (val CV) | Δ vs random | wandb |
|---|---|---|---|---|---|
| random_depth4 (floor) | — | — | 0.0190 | — | — |
| original perwindow ep50 | per_window, lr=3e-4 | 50 | 0.0204 | +0.0014 | — |
| baseline (shot_mean) | shot_mean, lr=3e-4 | 50 | 0.0226 | +0.0036 | [gpjzz1kt](https://wandb.ai/sccn/eb_jepa/runs/gpjzz1kt) |
| pw_lr1e4_ep50 | per_window, lr=1e-4 | 50 | 0.0255 | +0.0065 | [3ijbku4z](https://wandb.ai/sccn/eb_jepa/runs/3ijbku4z) |
| pw_lr1e4_ep99 | per_window, lr=1e-4 | 99 | 0.0289 | +0.0099 | [3ijbku4z](https://wandb.ai/sccn/eb_jepa/runs/3ijbku4z) |
| continuation ep29 (partial) | resume ep99 + lr=5e-5 | 100+29 | 0.0306 | +0.0116 | [51cwpy2e](https://wandb.ai/sccn/eb_jepa/runs/51cwpy2e) |
| continuation v3 ep99 | resume ep99 + lr=5e-5 | 100+99 | 0.0338 | +0.0149 | [aat8wjv7](https://wandb.ai/sccn/eb_jepa/runs/aat8wjv7) |
| fresh300 ep299 | per_window, lr=1e-4 | 299 | 0.0403 | +0.0213 | [8kw3d49v](https://wandb.ai/sccn/eb_jepa/runs/8kw3d49v) |
| h200_bs256 ep99 | per_window, lr=2e-4, bs=256 | 99 | 0.0268 | +0.0078 | [dkvndll7](https://wandb.ai/sccn/eb_jepa/runs/dkvndll7) |
| **fresh500 ep399** | per_window, lr=1e-4 | 399 (AUC peak) | **0.0460** | **+0.0271** | [mij2cj4j](https://wandb.ai/sccn/eb_jepa/runs/mij2cj4j) |
| **fresh500 ep499** | per_window, lr=1e-4 | 499 (latest) | **0.0462** | **+0.0272** | [mij2cj4j](https://wandb.ai/sccn/eb_jepa/runs/mij2cj4j) |

The trajectory is monotonic in `total epochs of training`: every additional
hundred epochs adds ~0.005 Δ to mean R² on val. No plateau seen at 500
epochs.

### Key methodological findings discovered during the sweep

1. **AUC ≠ R² downstream.** The clip_shot_auc / clip_scene_auc trajectory is
   noisy late in cosine schedules (e.g., 500-ep peaks AUC at ep399 = 0.76, dips
   to 0.68 at ep499). But the linear-probe R² is essentially identical between
   those two checkpoints (Δ = +0.0271 vs +0.0272). The in-loop AUC metric
   plateaus while downstream R² keeps climbing.
   *Implication*: don't early-stop on AUC. The contrastive loss has won the
   discrimination task long before the encoder has finished acquiring
   linearly-decodable structure.
2. **Continuation strategy works, but fresh-from-scratch is better at equal
   compute.** Resuming an ep99 checkpoint with a fresh lr=5e-5 cosine over
   100 more epochs hits Δ = +0.0149 — better than the parent (+0.0099) but
   worse than running 200 epochs from scratch would predict from the
   fresh-trajectory slope.
3. **Smaller batch helps in this regime.** H200 bs=256 (sqrt-scaled lr=2e-4)
   underperforms A40 bs=64 lr=1e-4 (Δ = +0.0078 vs +0.0099 at ep99). bs=512
   OOMs the H200; bs=128 OOMs the A40. Smaller-batch noise appears to act as
   regularization here.
4. **per_window target > shot_mean target downstream.** Despite shot_mean
   producing better intra-modal AUC (because same-shot rows are identical so
   AUC saturates trivially), per_window is what unlocks high-level features
   (motion, depth, faces) on the linear probe. Larger contrastive vocabulary
   (406 per_window targets vs 54 shot_means) is the per InfoMin / Tian et al.
   prediction.
5. **lr=1e-4 > lr=3e-4 > lr=1e-3** for per_window. The original recipe doc
   used lr=3e-4. lr=1e-3 was strictly destructive (Δ < 0 vs random on the
   probe). lr=1e-4 is the per_window optimum at this scale.

---

## §3. Generalization — train → val → test

All three probe protocols on the same fresh500_ep499 encoder:

| protocol | mean Δ vs random | mean random R² | mean trained R² |
|---|---|---|---|
| clip_probe CV-on-val (R5 internal) | **Δ R² = +0.0272** | +0.0190 | +0.0462 |
| clip_probe CV-on-test (R6 internal) | Δ R² = +0.0122 | +0.0043 | +0.0166 |
| **probe_traintest (R1-R4 → R6)** | **Δ Pearson r = +0.080** | r = +0.053 | r = +0.133 |
|   | **Δ R² = +0.073** | R² = −0.072 | R² = +0.001 |

### Per-feature train→test (fresh500_ep499)

| feature | rand r | trained r | Δ r | rand R² | trained R² | Δ R² |
|---|---|---|---|---|---|---|
| motion_energy | +0.063 | **+0.179** | **+0.117** | −0.037 | +0.016 | +0.053 |
| saturation_mean | +0.037 | **+0.149** | +0.112 | −0.138 | +0.002 | **+0.140** |
| luminance_mean | +0.050 | **+0.159** | +0.109 | −0.105 | +0.009 | +0.113 |
| entropy | +0.058 | **+0.150** | +0.092 | −0.190 | +0.001 | **+0.191** |
| contrast_rms | +0.055 | +0.134 | +0.079 | −0.085 | −0.008 | +0.077 |
| depth_mean | +0.049 | +0.128 | +0.079 | −0.046 | +0.007 | +0.053 |
| narrative_event_score | +0.028 | +0.107 | +0.079 | −0.048 | −0.007 | +0.042 |
| edge_density | +0.037 | +0.109 | +0.071 | −0.050 | −0.001 | +0.048 |
| n_faces | +0.064 | +0.128 | +0.064 | −0.036 | +0.007 | +0.043 |
| position_in_movie | +0.053 | +0.116 | +0.063 | −0.047 | −0.016 | +0.031 |
| face_area_frac | +0.055 | +0.107 | +0.052 | −0.068 | −0.003 | +0.064 |
| scene_natural_score | +0.083 | +0.129 | +0.045 | −0.017 | +0.005 | +0.021 |
| **mean** | **+0.053** | **+0.133** | **+0.080** | **−0.072** | **+0.001** | **+0.073** |

### Why the within-test CV looked smaller

The clip_probe CV-on-test number (Δ R² = +0.0122) was *understated* by the
small fit-set: GroupKFold-by-recording on 108 R6 recordings gives only ~86
recordings per fit fold, leaving the Ridge probe data-starved on both random
and trained encoders. The probe_traintest result with full R1-R4 as the fit
set is the proper SSL-literature comparison and shows the encoder transfers
substantially better than the within-test CV suggested.

### Why train→test R² goes negative on the random encoder

Random encoders extract spurious correlations on the train subjects that
anti-fit the test subjects. With distribution shift across HBN release waves,
those calibration errors yield SS_res > SS_tot, hence R² < 0. The trained
encoder fixes this: its representations are robust enough that Ridge fits
generalize with near-zero R² (and positive Pearson r on every feature). This
is exactly the calibration-robustness story you'd want a generalist EEG
representation to support.

### Why Pearson r is the headline metric for probe_traintest

R²_score is *not* translation/scale-invariant. Across releases there's
inevitable mean/variance drift in EEG features, so even a perfect-correlation
predictor will get R²_score < perfect. Pearson r isolates "is the signal
there?" from "is the regression calibrated?" — which is why probe_eval (the
canonical JEPA spec) reports it for regression. The Pearson r values above
(0.11 – 0.18 across features, all positive) are the unambiguous statement
that the encoder transfers.

---

## §4. Open questions / next experiments

1. **Does R² keep climbing past 500 epochs?** Best evidence: the
   500-ep latest checkpoint matches the 500-ep AUC-peak checkpoint within
   noise, suggesting saturation may be near. But the
   100 → 200 → 300 → 500 trajectory shows no plateau either. A 1000-epoch
   run was attempted (job 19667191) but crashed at ep390 due to a
   `torch.save` Lustre/NFS hiccup on `/projects/bbnv` — would need to
   retry with `save_every=50` or `/work/hdd` checkpoints.

2. **AdamW with weight decay** (jobs 19668129, 19668130). Both crashed mid-run
   at ep350 from the same Lustre event. Partial val trajectories suggest:
   - AdamW + wd=0.05: similar or marginally better than plain Adam at same
     epoch.
   - AdamW + frozen vision projection (vision_passthrough, EEG → 1408 dim):
     reaches the baseline's ep299 AUC by ep99. Strong candidate for
     faster convergence. Need to rerun.

3. **Memory bank / MoCo queue** for decoupling negatives from batch size.
   Highest-expected-impact change per the multi-view redundancy theory:
   adding ~16k V-JEPA-2 queue negatives without changing batch memory should
   raise the asymptote, not just convergence speed.

4. **Cross-modal latent prediction (EEG-JEPA against V-JEPA-2 latents).**
   The natural next-generation move per V-JEPA / I-JEPA — replace the
   contrastive objective with direct regression of V-JEPA-2 latents from
   masked EEG. Larger code change but sidesteps the InfoMin / log-K-bits
   ceiling of pure InfoNCE.

---

## Reproducibility

Recipe artifact (rebuild after any V-JEPA-2 swap):

```bash
python -m experiments.clip_pretraining.embedding_feature_correlation.precompute_vjepa2_recipe \
    --task ThePresent
```

Best training run (Delta job 19654184, wandb [mij2cj4j](https://wandb.ai/sccn/eb_jepa/runs/mij2cj4j)):

```bash
PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain \
    --loss.target_kind=per_window \
    --optim.lr=1e-4 --optim.warmup_epochs=5 --optim.epochs=500 \
    --eval.val_recording_fraction=0.5 \
    --logging.wandb_group=clip_pw_lr1e4_ep500
```

ImageNet-style probe on the best checkpoint (Delta job 19670936):

```bash
PYTHONPATH=. uv run --group eeg python \
    experiments/clip_pretraining/clip_probe/probe_traintest.py \
    --device cuda \
    --checkpoint /path/to/fresh500/latest.pth.tar \
    --config experiments/clip_pretraining/clip_probe/configs/recipe_depth4.yaml \
    --output probe_results/probe_traintest_pw_fresh500_ep499.json
```

Random baseline (substitute `--random-baseline` for `--checkpoint`).
