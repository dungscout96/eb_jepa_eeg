# SSL Research Memo: Pushing a Masked-Prediction Encoder Toward Stimulus Signal

Date: 2026-05-02
Author: literature scan, written for Phase 2 grid planning
Scope: 11 references (2020-2026), focused on (i) loss-function levers that bias encoders toward task-relevant subspaces, (ii) depth/scale behavior of masked-prediction SSL, and (iii) EEG-specific SSL state of the art.

Phase 1 finding being addressed: our EB-JEPA encoder (REVE depth=2, dim=64, predictor bottleneck=24, VC=0.25/0.25, smooth-l1 against EMA target, multi-block masking, 5 CorrCA channels, per-recording z-norm) saturates stimulus probes at the *patch_embed projection* — the second transformer block adds nothing for stimulus probes, while subject identity (age r=+0.50, sex AUC=+0.71) is encoded strongly. Narrative correlation +0.026 vs cross-subject ceiling +0.213. The objective is paying for trait-stable features and ignoring stimulus-locked signal. The three sections below identify which loss/architecture/data-regime levers in the recent literature address exactly this failure mode.

---

## Q1 — Loss-function modifications that push encoders toward task-relevant subspaces

The default JEPA loss `L = smooth_l1(predictor(ctx, mask), sg(target_enc(x)[mask]))` rewards *anything* that lets the predictor reconstruct masked tokens. In a 5-channel CorrCA-projected EEG dataset where every channel is a low-frequency mixture of subject-stable cortical sources, the easiest predictable signal is the per-recording DC mixture (subject identity) — exactly what we observe. The literature offers four mechanistically distinct fixes.

### 1.1 Subject-adversarial / subject-aware contrastive auxiliary head — Cheng et al. 2020, "Subject-Aware Contrastive Learning for Biosignals" (arXiv:2007.04871)

The most direct fix. Cheng et al. add two auxiliaries on top of a standard SimCLR-style contrastive backbone:

- **Subject-specific contrastive loss `L_subj`**: when forming positive/negative pairs in a batch, mask out cross-subject pairs and only contrast within subject. This forces invariances learned by the contrastive loss to be *within-subject* invariances (i.e., ignores nuisance that already varies within a subject like trial noise, but cannot use cross-subject differences as a free shortcut).
- **Adversarial subject classifier head `L_adv`**: a 2-layer MLP attached after the encoder predicts subject ID from the embedding. The encoder is trained with a gradient-reversal layer to *maximize* `L_adv` while the head minimizes it. Net effect: the encoder is penalized for any direction in feature space that linearly separates subjects.

Combined objective: `L = L_contrastive + λ_subj·L_subj − λ_adv·L_adv` (with GRL implementing the negative coefficient). Reported gains on Stanford-EEG/PhysioNet tasks: subject-invariance term recovered ~3-5 percentage points over baseline contrastive on cross-subject test splits, and was complementary to the subject-specific positive sampling.

**Applicability to our setup**: trivially compatible with JEPA. Add a 2-layer MLP head on the mean-pooled token output, train it as a 700-way subject classifier, attach a gradient reversal between encoder and head. Cost: one extra MLP and a `λ_adv` hyperparameter. Critically, this directly attacks the subject-identity-as-shortcut failure mode our Phase 1 surfaced.

### 1.2 Stimulus-aligned contrastive head — InfoNCE with cross-subject same-time positives

Naturalistic-stimulus EEG has a free positive-pair structure that almost no SSL paper exploits: **same movie timepoint, different subjects** — the inter-subject-correlation literature (Dmochowski et al., 2012; Kaneshiro et al., 2024, *Eur. J. Neurosci.*) shows this signal is dominated by stimulus-driven cortical components and is exactly the CorrCA target. Concretely, build an InfoNCE auxiliary:

- Positive: window `(subj_i, t)` paired with `(subj_j, t)` where `j != i` watching the same movie at the same timestamp.
- Negative: `(subj_k, t')` with `|t'−t| > Δ` (time-shuffled within or across subjects).

Loss: `L_stim = −log[exp(sim(z_i,t / τ)) / Σ_neg exp(sim(z_neg / τ))]` with cosine sim and τ=0.1. This is mechanically a cross-subject CPC (van den Oord et al., 2018, arXiv:1807.03748) restricted to stimulus-aligned positives. It is the converse of the Cheng adversarial — instead of *removing* subject info, it *forces* the embedding to carry signal that is shared across subjects, which is by construction the stimulus-driven subspace.

**Why this is a stronger lever than 1.1 for our setup**: HBN has dense same-stimulus alignment (all subjects watch the same DM/TP movies). The CorrCA preprocessing already projects into the spatial subspace where this signal lives, so the auxiliary head only needs to recover its temporal coding. Cross-subject CPC has been used in BCI (e.g., motor-imagery alignment), but to our knowledge has not been combined with a JEPA backbone on naturalistic stimuli.

### 1.3 VICReg / LeJEPA — variance-covariance regularization with theoretical grounding (Bardes et al., 2022, arXiv:2105.04906; LeJEPA, arXiv:2511.08544)

We already use VC(0.25/0.25). Two observations from the recent literature suggest tuning rather than abandonment:

- **C-JEPA (arXiv:2410.19560)** ablates VICReg-on-JEPA and reports: with V/Cov regularization on context tokens, ImageNet-1% linear probe goes from 63.7% → 68.3% at 100 epochs and 72.9% → 73.7% at 600 epochs (ViT-B/16). The effect is largest when the encoder is small/shallow — exactly our regime.
- **LeJEPA (Bauchau et al., 2025, arXiv:2511.08544)** replaces VICReg with **SIGReg** (Sketched Isotropic Gaussian Regularization), which has a *single hyperparameter* and theoretically minimizes downstream prediction risk under the assumption that the embedding distribution should be isotropic Gaussian. ViT-H/14 reaches 79% linear-probe ImageNet — competitive with DINOv2 — with no stop-gradient, no teacher-student, no schedulers. The repo says ~50 lines of distributed code. Our codebase already contains a `SIGRegLoss` (`eb_jepa/losses.py`), so the migration cost is small.

**Applicability**: replace `VCLoss` with `SIGRegLoss` and sweep its single coefficient, or stack both with std/cov turned down. A LeJEPA-style isotropy constraint *across the batch* is more aggressive than per-dim variance and would penalize any low-rank subject-identity subspace that currently dominates our embeddings.

### 1.4 Deep self-supervision / contextualized targets — V-JEPA 2.1, data2vec (arXiv:2603.14482; arXiv:2202.03555)

V-JEPA 2.1 introduces two changes over V-JEPA, both relevant to our finding that "block 2 adds nothing":

- **Dense Predictive Loss**: visible context tokens *also* contribute to the loss with distance-weighted coefficients `λ_i = λ / √(d_min(i, M))`, weighting context tokens close to masked regions higher. Loss: `L_dense = L_predict + L_ctx` with `L_ctx = (1/|C|) Σ λ_i · ||P_φ(E_θ(x), Δ_y)_i − sg(E_θ̄(y)_i)||_1`. Ablation on ADE20K mIoU: 22.2 → 33.8 with context loss alone.
- **Deep Self-Supervision (DSS)**: concatenate outputs from 3 intermediate encoder blocks plus the final layer; a lightweight MLP fuses them and feeds the predictor. The same self-supervised loss is applied at *every* level, not just the top. Ablation: ADE20K mIoU 33.8 → 40.8, NYUv2 RMSE 0.473 → 0.418. The mechanism: the loss gradient now reaches every block directly, preventing the "all useful features collapse into block 1" failure mode we are observing.

data2vec (Baevski et al., 2022, arXiv:2202.03555) uses a related trick: targets are an average of the top-K teacher layers rather than the final layer alone, which empirically gives richer, less position-specific targets and improved low-shot ImageNet by ~1-2 points over MAE.

**Applicability**: directly addresses our Phase 1 finding. With depth=2, DSS reduces to "supervise both layer 1 and layer 2 outputs," which forces layer 2 to do non-trivial work or be penalized. Trivially implementable: add a second predictor head on the mid-layer activations, share the EMA target. data2vec-style top-K averaging is a one-line change to `MaskedJEPA.forward`.

### Recommendations for our setup (Q1, ranked)

1. **Cross-subject stimulus-aligned InfoNCE auxiliary** (§1.2). Highest expected gain because (a) it directly injects the stimulus signal we are trying to learn, (b) HBN movie-watching has the alignment structure for free, (c) the CorrCA preprocessing has already isolated the subspace. Concrete formula: `L_total = L_jepa + 0.5 · L_stim_nce`, positives = same-stimulus-time-different-subject, negatives = time-shuffled within batch, τ=0.1, projector = 2-layer MLP 64→128→64.
2. **Subject-adversarial head with gradient reversal** (§1.1). Direct, well-documented, complementary to (1). `L_total += −0.1 · L_subj_classifier` with GRL. Sweep λ_adv ∈ {0.05, 0.1, 0.3}.
3. **Deep self-supervision** (§1.4). Add a second predictor on layer-1 tokens, same EMA target. Tests directly whether block 2 *can* learn stimulus signal under explicit supervision. If both layers are supervised and block 2 still adds nothing, the encoder is depth-saturated, not optimization-starved.
4. **SIGReg replacement of VC loss** (§1.3). One-hyperparameter swap; complementary to (1)-(3). Worth one cell in the grid.

---

## Q2 — Encoder depth in masked-prediction SSL (do more blocks help?)

### 2.1 I-JEPA scale curve (Assran et al., 2023, arXiv:2301.08243)

I-JEPA reports linear probe ImageNet-1K at 600 epochs:
- ViT-B/16: 72.9%
- ViT-L/16: 77.5%
- ViT-H/14: 79.3% (300 epochs)
- ViT-H/16₄₄₈: 81.1% (300 epochs at 448²)

Encoder depth scales monotonically with linear probe — **but each step doubles parameters and uses ImageNet-1K's 1.28M images**. The paper's most-cited architectural ablation is on the *predictor* (Tables 12, 14): 12-layer predictor beats 6-layer (66.9% vs 64.0% on ImageNet-1%); 384-dim narrow predictor beats 1024-dim wide (70.7% vs 68.4%). **The narrow-deep predictor is what we already have.** What I-JEPA does *not* claim is that adding encoder layers without proportionally scaling data helps — and it does not perform any small-data ablation.

### 2.2 V-JEPA 2.1 explicitly diagnoses "shallow encoders learn block-1-only features"

V-JEPA 2.1 (arXiv:2603.14482) names the problem we observed and offers DSS (§1.4) as the architectural fix. The relevant message: when masked-prediction loss is applied only at the top, *deeper encoders concentrate semantic features in early layers and use the rest for prediction-specific bookkeeping*. With DSS, every layer is forced to be a usable representation. ViT-L → ViT-G under DSS gave +5.0 mIoU on ADE20K and −0.111 RMSE on NYUv2 — a steeper depth-payoff curve than vanilla V-JEPA.

### 2.3 Depth in MAE under small data (Xue et al., 2022 ICML, "A Study on Transformer Configuration and Training Objective", arXiv:2205.10505)

This paper systematically scans depth × width × objective on ImageNet-1K, ImageNet-100, and small subsets. Headline result: **under masked-prediction objectives, deeper-and-narrower beats wider-and-shallower at fixed parameter count** — the opposite of supervised training. They propose "Bamboo": ViT depth 24-48 layers with width 192-256, beating standard ViT-B/L at the same parameter count under MAE pretraining.

But the paper *also* reports diminishing returns past depth ~24 on ImageNet-100 (130k images), and explicit overfitting on a 10k-image subset past depth 12. The lesson for us: depth-vs-data is the key axis, and our ~35 hours of EEG (≈350k 2s windows) is closer to ImageNet-100 than ImageNet-1K. A depth-of-6 to depth-of-12 encoder is a defensible target; depth=2 is almost certainly underprovisioned, depth=24 is over.

### 2.4 DINOv2 / DINOv3 scaling laws (Oquab et al., 2023, arXiv:2304.07193; DINOv3, arXiv:2508.10104)

DINOv2 trains ViT-g (1.1B params) on a curated 142M-image dataset and distills to smaller models. Linear probe scales monotonically with both depth and data (ViT-S: 79.2%, ViT-B: 82.1%, ViT-L: 84.1%, ViT-g: 86.5% on ImageNet-1K). The crucial caveat: DINOv2's gains over DINOv1 are largely attributable to data curation and training stability, not naive depth. DINOv3 (2025) extends this to 7B parameters but explicitly reports that *gains saturate without proportional dataset growth*, with patch-level features benefiting more than image-level features. Neither paper claims that extra depth helps when data is fixed.

### 2.5 Specific evidence on fine-grained vs invariant features

V-JEPA 2.1's depth ablation is the cleanest data point: dense features (depth, segmentation — fine-grained, position-sensitive) benefit *more* from depth-with-DSS than from depth alone, while clip-level action recognition (invariant features) saturates earlier. This maps onto our problem directly: stimulus-locked narrative correlation is a fine-grained, time-coded signal (more like depth estimation than action recognition); subject identity is a static, invariant feature (more like image classification). The literature predicts that the first will benefit much more than the second from deeper encoders trained with intermediate-layer supervision.

### Recommendations for our setup (Q2, ranked)

1. **Depth=4 with deep self-supervision** as the primary Phase 2 architecture. Concretely: 4 REVE blocks, predictor on each block's output (shared EMA target), loss = mean of per-layer L1s. This tests both "is depth-2 underprovisioned?" and "is the saturation an optimization artifact?" in one cell.
2. **Depth=6 vanilla** as a control to isolate the DSS effect from the depth effect.
3. **Depth=2 with DSS** (i.e., supervise both blocks) as a cheap control showing whether DSS alone helps even at fixed depth.
4. **Avoid depth ≥ 8 without first solving the loss-function question.** With 35 hours of EEG, the Bamboo / DINOv3 evidence says we will overfit before we benefit. Hold depth-scaling back until the loss is fixed.

---

## Q3 — EEG-specific SSL literature

### 3.1 BENDR (Kostas, Aroca-Ouellette, Rudzicz, 2021, arXiv:2101.12037)

Convolutional feature extractor (Wav2Vec-style strided convs producing ~2.67 Hz tokens) followed by an 8-layer transformer encoder. Loss: contrastive — for each masked BENDR vector, the transformer must output a representation closer to the original BENDR vector at that position than to randomly sampled distractors from the same sequence (InfoNCE with K=20 negatives). Pretraining: TUEG (~1.5 TB EEG, mixed clinical). Downstream: motor-imagery, P300, MMI, sleep-EDF — all clinical/BCI, no naturalistic stimulus tasks. **Subject confounding is not addressed.**

### 3.2 LaBraM (Jiang, Zhao et al., ICLR 2024 spotlight, arXiv:2405.18765)

Two-stage: (1) train a vector-quantized "neural tokenizer" via spectral prediction — encode 200ms EEG patches into a discrete codebook of 8192 codes; (2) pretrain a transformer to predict masked codebook indices (BERT-style). Sizes: LaBraM-Base (5.8M), Large (46M), Huge (369M), all 12-layer transformers. Pretraining data: ~2500 hours, ~20 datasets, all clinical. Downstream: TUAB (abnormal vs normal), TUEV (event type), SEED-V (emotion), Mumtaz2016 (depression). LaBraM-Huge reports balanced accuracy 0.8258 on TUAB, 0.6616 on TUEV — SOTA at submission. **No naturalistic stimulus eval. Subject identity not probed; SEED-V is intra-subject so subject confounding inflates emotion accuracy.**

### 3.3 Brant (Zhang et al., NeurIPS 2023; "Brant: A Foundation Model for Intracranial Neural Signals")

500M parameters, intracranial EEG (281k channel-hours). Architecture: temporal encoder + spatial encoder + linear decoder. SSL objective: 40% random masking of time × channel patches, MSE reconstruction on raw signal *plus* power-spectral targets. Downstream: seizure prediction, neural response forecasting. iEEG-only — does not transfer cleanly to scalp 129-channel EEG.

### 3.4 EEGFormer (Chen, Ren et al., arXiv:2401.10278, Jan 2024)

VQ-VAE with discrete codebook (512 / 1024 / 2048 codes for S/B/L). 6/8/12-layer transformer encoder, 3-layer shallow decoder. Loss: reconstruction + codebook commitment. Pretraining: TUH EEG (~10k hours). Downstream: TUSZ (seizure) AUPRC 0.556 vs prior SOTA 0.491; neonate seizure AUPRC 0.544 vs 0.499. **Clinical only.**

### 3.5 BIOT (Yang et al., NeurIPS 2023)

Tokenizes each channel separately into fixed-length spectrogram segments, then runs a transformer over the concatenated multi-channel sequence with learned channel and relative-position embeddings. Contrastive supervised pretraining. Notable for handling variable channel counts across datasets — relevant if we ever combine HBN with other EEG corpora. 3.3M parameters, lighter than the foundation models above. Clinical downstream tasks only.

### 3.6 EEG2Rep (Foumani et al., KDD 2024, arXiv:2402.17772)

JEPA-style: predict masked patches in **latent space** rather than raw signal. Key innovation: **Semantic Subsequence Preserving (SSP) masking** — instead of random masks, it chooses masks that *preserve* semantically important segments and mask the rest. Reports 50% preserve ratio is optimal across 6 datasets. SOTA on emotion/mental-state/epilepsy across 6 public datasets, beating TS2Vec, Mixing-up, BENDR, MAE. **The closest published method to ours, and the SSP masking is a direct candidate for our masking layer.** Clinical only.

### 3.7 CBraMod (Wang et al., ICLR 2025, arXiv:2412.07236)

12-layer "criss-cross" transformer with separate spatial and temporal attention heads (4+4 of 8 total). Hidden dim 200, FFN 800. SSL: 50% mask ratio, MSE reconstruction of raw patches. Pretraining: TUEG. SOTA across 10 downstream BCI tasks (FACED emotion κ=0.504, PhysioNet-MI κ=0.522, SHU-MI AUROC 0.699). Architectural takeaway: **separating spatial and temporal attention substantially improves cross-task transfer.** With our 5 CorrCA channels the spatial dimension is small but non-trivial; the criss-cross design is worth considering.

### 3.8 NeuroLM (Jiang et al., ICLR 2025, arXiv:2409.00101)

1.7B-parameter EEG foundation model; treats EEG as a "foreign language" by tokenizing into a vocabulary aligned with a frozen LLM, then doing autoregressive next-token prediction. Largest published EEG model. Multi-task instruction-tuned. **Not directly applicable to us — the autoregressive objective is opposite to our masked-prediction, and the scale is 4 orders of magnitude beyond our budget.** Useful only as a "what does pretraining at scale buy?" reference point.

### 3.9 ContraWR (Yang et al., JMIR AI 2023, arXiv:2110.15278)

Contrastive with a "world representation" — global average over the dataset — used as the negative anchor. Designed for sleep staging. Reports +4% accuracy with <2% labels vs supervised baselines. Architecturally simple (CNN backbone). Mostly relevant for the negative-construction trick: when dataset-global averages are good negatives, batch-level negatives are not.

### 3.10 BrainBERT (Wang, Mamashli et al., ICLR 2023, arXiv:2302.14367) — *the only naturalistic-stimulus EEG-class paper*

Intracranial recordings during Hollywood-movie watching. SSL objective: masked reconstruction in spectrogram space (STFT magnitude). 6-layer transformer. **Demonstrates that stimulus-driven probes (linguistic features during movie listening) work well with masked SSL on naturalistic data — this is the only published existence proof for our setup.** Caveats: iEEG is much higher SNR than scalp EEG, ~10 subjects, and the probes are semantic content of speech rather than visual narrative.

The Brain Treebank dataset paper (arXiv:2411.08343) and Neuroprobe benchmark (arXiv:2509.21671) extend this line, but on iEEG only.

### 3.11 COMET (Wang et al., NeurIPS 2023, arXiv:2310.14017) — *multi-level contrastive for medical TS*

Defines four levels of contrast — observation, sample, trial, patient — with separate InfoNCE losses at each: `L_total = α_o L_obs + α_s L_sample + α_t L_trial + α_p L_patient`. Patient-level: positives = same patient different visits, negatives = different patients. Trial-level: positives = same trial different windows. The hierarchical structure is the prototype for what a stimulus-aware EEG SSL would look like — substitute "movie" for "patient" and "movie-timepoint" for "trial." Reports F1 improvements of 5-10 points over TS2Vec and SimCLR on AD/PD EEG with 1-10% labels.

### Cross-cutting observations

- **No EEG SSL paper from 2021-2025 explicitly evaluates stimulus-vs-subject trade-offs.** Every clinical task has subject identity as a strong confound (sleep stage and seizure both have subject-stable baselines). The Cheng 2020 paper is the only one that directly attacks subject confounding via loss design.
- **All foundation EEG models are 6-12 layers deep.** None demonstrate gains from depth past 12 with anything less than ~1000 hours of data. Our 35-hour budget puts depth=4-8 in the right zone.
- **BrainBERT is the only published naturalistic-stimulus EEG SSL paper, and it uses spectrogram MAE — not JEPA.** Our setup is genuinely under-explored: JEPA-style masked-prediction on naturalistic-stimulus scalp EEG with stimulus-decoding probes.
- **EEG2Rep's SSP masking and CBraMod's criss-cross attention are the two architectural ideas most directly relevant to our masking and 5-channel-spatial setup respectively.**

### Recommendations for our setup (Q3, ranked)

1. **Adopt EEG2Rep's Semantic Subsequence Preserving masking** in place of random multi-block masking. Preserve high-variance / high-CorrCA-energy patches and mask the rest. Concrete change: in `eb_jepa/masking.py`, score each patch by RMS energy in CorrCA space, mask the bottom-50% rather than random 50%. Cost: one heuristic, no new parameters.
2. **Add a hierarchical-contrastive auxiliary in the spirit of COMET**, with three levels for our setup: observation (within-window patches), trial (same movie-time across subjects → positive, this is the §1.2 InfoNCE), subject (within-subject across-time → adversarial penalty, this is §1.1). This is the cleanest unifying frame for combining the §1.1 and §1.2 recommendations.
3. **Adopt CBraMod's spatial/temporal-separated attention** in REVE if expanding past depth=2. With 5 spatial × 12 temporal patches, splitting the 4 attention heads into 2 spatial + 2 temporal is a free architectural improvement that has been shown to help cross-task generalization.
4. **Use BrainBERT as the existence-proof reference** for stimulus probes on naturalistic neural recordings — its spectrogram-space MAE is a fallback ablation if JEPA-in-time-domain continues to fail.
5. **Do not adopt NeuroLM-style autoregressive tokenization** — wrong objective and wrong scale for our budget.

---

## Final Phase 2 Grid Recommendations (synthesis)

Highest-leverage cells, ordered by expected gain on stimulus probes:

| # | Loss change                                                                    | Architecture change                          | Expected to fix                                       | Cost |
|---|--------------------------------------------------------------------------------|----------------------------------------------|-------------------------------------------------------|------|
| 1 | + cross-subject stimulus-aligned InfoNCE (§1.2), λ=0.5, τ=0.1                  | (none)                                       | direct stimulus signal injection                      | low  |
| 2 | + subject-adversarial GRL head (§1.1), λ_adv ∈ {0.1}                           | (none)                                       | removes subject-identity shortcut                     | low  |
| 3 | DSS: per-layer predictor, shared EMA target (§1.4 / 2.5)                       | depth 2 → 4                                  | tests "block 2 adds nothing" directly                 | med  |
| 4 | replace VC loss with SIGReg (§1.3)                                             | (none)                                       | better isotropy, kills low-rank subject subspace      | low  |
| 5 | EEG2Rep SSP masking (§3.6)                                                     | (none, masking module only)                  | stops the loss from being satisfied by easy tokens    | low  |
| 6 | (1)+(2) combined + DSS                                                          | depth 4, criss-cross attention (§3.7)        | stress test of all three levers                       | high |

The single highest-information cell in Phase 2 is #1 (stimulus-aligned InfoNCE) because it is the only intervention that explicitly *requires* stimulus encoding — the others remove subject-identity shortcuts but do not on their own create a gradient signal toward stimulus-locked features.

## References (with arXiv IDs)

1. Cheng, Goh, Dogrusoz, Tuzel, Azemi (2020). Subject-Aware Contrastive Learning for Biosignals. arXiv:2007.04871.
2. van den Oord, Li, Vinyals (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.
3. Bardes, Ponce, LeCun (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. arXiv:2105.04906 (ICLR 2022).
4. Bauchau et al. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning. arXiv:2511.08544.
5. Mur-Labadia et al. (2026). V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. arXiv:2603.14482.
6. Baevski, Hsu, Xu, Babu, Gu, Auli (2022). data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language. arXiv:2202.03555 (ICML 2022).
7. Assran et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). arXiv:2301.08243.
8. Xue et al. (2022). A Study on Transformer Configuration and Training Objective. arXiv:2205.10505 (ICML 2023).
9. Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193.
10. Kostas, Aroca-Ouellette, Rudzicz (2021). BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn from Massive Amounts of EEG Data. arXiv:2101.12037 (Frontiers Hum. Neurosci.).
11. Jiang, Zhao et al. (2024). LaBraM: Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI. arXiv:2405.18765 (ICLR 2024 spotlight).
12. Chen, Ren et al. (2024). EEGFormer: Towards Transferable and Interpretable Large-Scale EEG Foundation Model. arXiv:2401.10278.
13. Foumani et al. (2024). EEG2Rep: Enhancing Self-supervised EEG Representation Through Informative Masked Inputs. arXiv:2402.17772 (KDD 2024).
14. Wang et al. (2025). CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. arXiv:2412.07236 (ICLR 2025).
15. Wang, Mamashli et al. (2023). BrainBERT: Self-supervised Representation Learning for Intracranial Recordings. arXiv:2302.14367 (ICLR 2023).
16. Wang et al. (2023). COMET: A Hierarchical Contrastive Framework for Medical Time-Series. arXiv:2310.14017 (NeurIPS 2023).
17. Yue et al. (2022). TS2Vec: Towards Universal Representation of Time Series. arXiv:2106.10466 (AAAI 2022).
18. Yang et al. (2023). Self-supervised EEG Representation Learning for Automatic Sleep Staging (ContraWR). arXiv:2110.15278 (JMIR AI 2023).
19. Hojjati et al. (2025). From Video to EEG: Adapting Joint Embedding Predictive Architecture (EEG-VJEPA). arXiv:2507.03633.
20. Wu et al. (2024). Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning (C-JEPA). arXiv:2410.19560.
