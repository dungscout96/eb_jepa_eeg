# NeurIPS Workshop (Foundation Models for EEG) — 4-page outline

**Working title:** *Subject Identity Is the Objective: Why Self-Supervised EEG
Pretraining Fails on Stimulus-Locked Readout*

**Thesis.** For stimulus-locked EEG, the bottleneck is not the SSL objective.
Every within-subject objective is rationally solved by modelling the subject
fingerprint, which carries ~96% of single-trial variance and is orthogonal to
(or anti-correlated with) stimulus content. We show this with a convergent
falsification across 6 objectives, explain it with an SNR argument, and propose
a cross-subject predictive target that provably marginalizes the fingerprint out.

**Contribution triangle:**
1. **Diagnosis** — a stimulus-locked evaluation protocol (continuous-feature
   probe + bidirectional retrieval against a frozen video FM) on which the
   field's standard recipes are indistinguishable from random init.
2. **Explanation** — SNR decomposition + one-to-many pairing structure; the
   v→e-at-chance asymmetry is the predicted signature, not a bug.
3. **Fix** — cross-subject predictive JEPA. `E[z_B | ctx_A] = E[shared(t) | ctx_A]`,
   so the Bayes-optimal solution *is* the shared stimulus response.

---

## Page 1 — §1 Introduction (~0.75 p)

Four beats:

1. **The field's shape.** EEG foundation models (LaBraM, EEGPT, BIOT, CBraMod,
   NeuroLM, Brant, and now LeJEPA/Laya-style JEPA) are near-universally unimodal
   masked/contrastive SSL, evaluated on **recording-level labels** — pathology,
   sleep stage, motor imagery, subject state. Every such label is a property of
   the subject or recording.
2. **The gap.** No standard benchmark tests **stimulus-locked, continuous,
   single-trial** readout — the regime that matters for naturalistic
   neuroscience and for any EEG↔video/audio multimodal model. We introduce one.
3. **The finding.** On it, six objectives spanning cross-modal contrastive and
   masked-predictive SSL land within noise of each other, and masked SSL lands
   at or below random init.
4. **The reason and the fix.** One sentence each; forward-reference §5 and §6.

**Positioning sentence (verified against the sources — see
[RESULTS.md §1.2b](../experiments/snr_scaling/RESULTS.md)):**

> Where naturalistic EEG decoding has succeeded, it has bought SNR from
> somewhere: NICE/THINGS-EEG averages 80 test repetitions per image and trains
> per subject (13.8 % top-1 subject-dependent vs 6.2 % subject-independent);
> Benchetrit et al. and the headline Défossez et al. results are MEG. Run
> single-trial on EEG with a cross-subject model — Défossez's own EEG datasets —
> the same architecture yields ~5 % top-1 against ~41 % on MEG. Continuous
> naturalistic viewing affords neither repetition nor MEG, so SNR must be
> bought *inside the objective*, via cross-subject aggregation.

**Do NOT write "prior work is trial-averaged image decoding."** That was an
earlier draft's claim and it is false for Défossez et al., which is
single-trial and is speech, not images. The ~8× MEG-over-EEG gap on an
otherwise identical pipeline is the stronger argument and it survives
checking; the averaging argument only applies to NICE.

**Related work, one compressed paragraph.** EEG FMs (above); neural decoding
with contrastive losses; the modality gap (Liang et al. 2022) and
alignment/uniformity (Wang & Isola 2020); ISC/CorrCA and hyperalignment as the
classical cross-subject tools we recast as a *pretraining objective*.

---

## Page 1–2 — §2 Setup and evaluation protocol (~0.5 p)

- **Data.** HBN movie-watching, 129 ch @ 200 Hz. *ThePresent* (TP) and
  *DespicableMe* (DM). Splits R1–R4 train / R5 val / R6 test, ~71 K train and
  ~11 K test windows, 108 test recordings. Subject-disjoint by construction.
- **Encoder.** REVE-style transformer, depth 12 / dim 512 / 8 heads,
  2 s windows.
- **Probe (primary).** `probe_traintest.py` — fit ridge on frozen R1–R4
  embeddings, evaluate on R6. Mean Pearson *r* over **12 continuous movie
  features** (luminance, contrast, edge density, saturation, entropy, motion
  energy, n_faces, face area, depth, scene-naturalness, narrative event,
  position). Random-init baseline *r* = 0.0527.
- **Retrieval (secondary, and the more diagnostic of the two).**
  `retrieval.py` — Top-K at three pool granularities (time / shot / scene),
  **both directions**, chance = K/N_pool. This is the metric that carries §4.
- **Why this protocol is the contribution.** Continuous targets, single-trial,
  subject-disjoint test, and a bidirectional metric. Contrast explicitly with
  the clinical-label benchmarks the field currently optimizes.

---

## Page 2 — §3 Result 1: cross-modal objective engineering saturates (~0.6 p)

**Claim.** Four cross-modal objectives, three seeds, two data regimes, 400 ep —
all within ±0.003 Δr². Capacity and compute don't move it either.

### Table 1 — objectives are indistinguishable
Source: [RESULTS_jul7.md §3.2, §4.1](../experiments/clip_pretraining/soft_target_clip/RESULTS_jul7.md),
[RESULTS_jul9.md §3b.2](../experiments/clip_pretraining/cs_aligner/RESULTS_jul9.md).
Multi-movie joint Δr², 3 seeds:

| objective | joint Δr² (mean ± std) | R6 test mean *r* |
|---|---:|---:|
| vanilla CLIP | +0.0423 ± 0.0005 | 0.1459 |
| scene_clip (sup-contrastive multi-positive) | +0.0413 (jul2 iter 1) | — |
| soft-target distillation (τ=0.05) | +0.0426 ± 0.0006 | **0.1517** |
| CS-Aligner (λ=1.0) | +0.0422 ± 0.0010 | 0.1479 |
| — random init | — | 0.0527 |

Add a footnote row for the null capacity/compute sweeps: 2× depth (+0.0419),
2× width (+0.0420), 2× epochs (+0.0382) — all inside the same band
(jul7 §4.1, 14 configurations).

**Sharp sub-result worth its own two sentences (jul9 §4.1).** CS-Aligner closes
the distributional modality gap by 47% (−0.096 → −0.051) but does it by
*dispersing the EEG cluster* (within-EEG cos 0.226 → 0.111) while **cross-modal
paired similarity drops** (0.022 → 0.011). Distributional alignment ≠ pair
alignment. This is a falsification of the CS-Aligner claim (Yin et al. 2025) in
a new regime and costs one table row.

---

## Page 2–3 — §4 Result 2: masked SSL is at or below random init (~0.7 p)

**Claim.** Two published JEPA recipes reproduced faithfully on HBN transfer
*negatively* to stimulus-locked readout.

### Table 2 — masked JEPA vs. random init
Source: [probe_results_jul10/](../experiments/jepa_pretraining/probe_results_jul10/),
R6 test, mean Pearson *r* over 12 features.

| encoder | pretraining | test mean *r* |
|---|---|---:|
| REVE-512 | **random init** | **0.0515** |
| REVE-512 | LeJEPA (SIGReg, convex) — TP only | 0.0344 |
| REVE-512 | LeJEPA — multi-movie | 0.0416 |
| REVE-384 | Laya (additive SIGReg, temporal masking) — TP only | 0.0377 |
| REVE-384 | Laya — multi-movie | 0.0494 |
| REVE-384 | **random init** | 0.0251 |

> ⚠️ **Blocking control — see §Runs, item 1.** Without the subject-trait probe,
> a reviewer reads this table as "your JEPA training is broken." With it, the
> same table becomes the paper's central mechanism evidence.

### Figure 1 — the retrieval asymmetry (the paper's best object)
Source: [RESULTS_jul7.md §3.6](../experiments/clip_pretraining/soft_target_clip/RESULTS_jul7.md).
Grouped bars, ×chance on the y-axis, one group per pool level, two directions.

| level | N | e→v Top-1 (×chance) | v→e Top-1 (×chance) |
|---|---:|---:|---:|
| time | 101 | 0.054 (**5.5×**) | 0.010 (**1.0×**) |
| shot | 49 | 0.084 (**4.1×**) | 0.020 (**1.0×**) |
| scene | 35 | 0.106 (**3.7×**) | 0.029 (**1.0×**) |

**Read:** EEG→video works well above chance at every granularity. Video→EEG is
at *exactly* chance at every granularity. In image–text CLIP both directions
work. §5 explains why this one cannot.

**Second data point, same figure.** REVE warm-start — i.e. *more EEG
pretraining data*, not a better loss — is the only intervention that ever moved
v→e, doubling it to 2.0× chance at all three levels (jul7 §3.6). Consistent with
an SNR-limited rather than objective-limited regime.

---

## Page 3 — §5 Analysis: why (~0.5 p)

**5.1 The SNR decomposition.** Source: [experiments.md § "Core Problem Identified"](../experiments.md#core-problem-identified).

```
EEG(s,c,t) = stimulus_response(c,τ) + subject_fingerprint(c) + noise(c,t)
                  ~3 µV                     ~30 µV               ~50 µV
```

Single-trial stimulus SNR ≈ **−24 dB** (0.4% of variance); the fingerprint is
~96%. Spatial masking is near-trivial (adjacent-channel *r* > 0.9 from volume
conduction), so masked prediction is solved by spatial interpolation plus
fingerprint modelling — neither of which contains stimulus content. Band
evidence: δ/θ ISC = 0.10–0.28 while α ISC < 0.05, and α dominates the variance.

**5.2 Why v→e must be at chance.** EEG↔video is not a bijective pairing. It is
**one-to-many with the "many" axis (subjects) carrying 96% of variance
orthogonal to the alignment target.** Given a video anchor, the correct EEG
window is one draw from a subject-conditioned cloud whose within-class spread
exceeds between-class separation, so nearest-neighbour retrieval is at chance by
construction. The e→v direction survives because each EEG window still has a
unique correct video target. State this as a short proposition, not a theorem.

**5.3 Corollary.** Objective-level interventions reweight *which* function of
the input the encoder fits; they cannot change the fact that the target of
within-subject prediction is dominated by the nuisance. Hence Tables 1 and 2.

### Figure 2 — embedding geometry
Reuse [modality_gap_cs_vs_soft.png](../experiments/clip_pretraining/cs_aligner/modality_gap_cs_vs_soft.png)
(2-panel t-SNE) *or* the cleaner
[embedding_structure_cs_aligner.png](../experiments/clip_pretraining/cs_aligner/embedding_structure_cs_aligner.png)
(2×2 recolouring by modality / movie / scene / shot). Space permitting, keep only
one and push the other to appendix. Caption carries the 4.1 numbers.

---

## Page 3–4 — §6 Cross-subject predictive JEPA (~0.7 p)

**6.1 The objective.** Replace the within-subject masked target with a
**same-movie-time, different-subject** target: predict subject B's tokens from
subject A's context. Implementation:
[`PairedSubjectJEPADataset`](../eb_jepa/datasets/paired.py),
[cross_subject/config.yaml](../experiments/jepa_pretraining/cross_subject/config.yaml).
`model.*` and `masking.*` are bit-identical to the LeJEPA arm, so the probe
delta is attributable to the target swap alone.

**6.2 Why it is the right fix (the one-paragraph argument).** B is drawn
independently of A, so

```
E[z_B | ctx_A] = E[shared_stimulus_response(t) | ctx_A]
```

The fingerprint is marginalized out and the Bayes-optimal predictor *is* the
shared stimulus response. Contrast with the classical cross-subject tools
(ISC/CorrCA, hyperalignment, MCCA), which are post-hoc linear analyses; here the
same principle becomes the pretraining objective of a foundation model.

**6.3 Collapse is the failure mode, and SIGReg does not cover it.** At −24 dB
the MSE gap between "recover the shared response" and "emit the constant mean"
is small. SIGReg constrains only the *marginal* embedding distribution, so
"targets spread isotropically, predictions go constant" is a valid optimum it
cannot see. Report the three diagnostics from step 1: `pred_loss_gap` (matched
minus mismatched targets, ~0 at init by construction — its opening is the *only*
evidence anything time-specific was learned), `pred_var_ratio`, and
`target_var_across_batch`.

### Table 3 — the headline (TO RUN)
Matched conditions, same encoder, same masking, same budget.

| pretraining | test probe *r* | *r* / noise ceiling | scene v→e Top-1 (×chance) | subject-trait AUC |
|---|---:|---:|---:|---:|
| random init | 0.0515 | — | 1.0× | — |
| LeJEPA (within-subject) | 0.0344 | — | — | *(expect ↑)* |
| **cross-subject JEPA** | **?** | **?** | **?** | *(expect ↓)* |

**The double dissociation is the claim**: stimulus readout ↑ *while* subject-trait
readout ↓, from a single change to the prediction target.

### Figure 3 — the √K partner-averaging curve (TO RUN; best figure in the paper)
Average the cross-subject target over K partners, K ∈ {1, 2, 4, 8, 16}. Plot
probe *r* vs K with the predicted √K SNR trend overlaid. A measured curve
matching a predicted trend converts "we got +0.03" into "we characterized the
regime," and it is the figure a reviewer remembers.

---

## Page 4 — §7 Discussion, limitations, and what this implies (~0.4 p)

- **For EEG FM design:** if a benchmark's labels are recording-level properties,
  fingerprint-modelling inflates it. Report at least one stimulus-locked task.
- **For multimodal learning:** the low-SNR, one-to-many, nuisance-dominated
  pairing regime is under-studied. Contrastive alignment's failure mode there is
  directional (§5.2), and standard modality-gap remedies attack the wrong gap
  (§3, CS-Aligner).
- **Limitations, stated up front (do not let a reviewer find these first):**
  single dataset (HBN); two movies; absolute *r* is small — mitigated by
  reporting against a measured noise ceiling; V-JEPA-2 teacher fidelity is a
  confound on DM specifically, where the ceiling is teacher-limited at ~+0.038
  Δr² regardless of loss, capacity, or compute (jul7 §5.3).

---

## Runs required before writing

Ordered. Items 1–2 are blocking; 3–4 are what make it a *contribution* rather
than a negative result; 5–6 are reviewer-proofing.

1. **Subject-trait probe on the LeJEPA / Laya / random checkpoints.**
   `evaluation/probe_eval.py`. Cheap. **This is the linchpin** — without it,
   Table 2 reads as a broken training run; with it, Table 2 *is* the mechanism.
   Predicted: subject-trait AUC ↑ while stimulus *r* ↓.

2. **Rerun the `*_from_jepa_checkpoint` CLIP arms matched to jul7.**
   Current configs use `epochs=100, lr=1e-5`
   ([config_lejepa.yaml:74-77](../experiments/clip_pretraining/soft_target_clip_from_jepa_checkpoint/config_lejepa.yaml))
   vs jul7's `epochs=400, lr=1e-4`. The observed collapse (val Δr² 0.019 vs
   0.063; scene e→v 2.4–3.0× vs 5.1× chance) is **confounded and cannot be
   claimed as-is**. Also verify the `strict=False` load actually populated the
   encoder keys — log the matched/dropped key counts.

3. **Cross-subject JEPA, matched to the LeJEPA arm.** Then the full evaluation
   suite: probe_traintest on R6, bidirectional retrieval, subject-trait probe.
   Watch the three collapse diagnostics from step 1, not epoch 100. Rescue
   order if it flattens: `pred_target_mode=all` → sigreg 0.05→0.1 →
   `within_subject_weight=0.3` → `context_mask_mode=full`.

4. **K-partner sweep** for Figure 3.

5. **Noise ceiling.** Probe subject-*averaged* EEG at matched movie times, or
   reuse [corrca_study/](../experiments/corrca_study/). Report *r*/ceiling
   throughout. "60% of the achievable ceiling" survives review; "*r* = 0.15"
   invites "that's small."

6. **Held-out movie.** Everything above on DM as well as TP; the DM ceiling
   result (jul7 §5.3) already gives the honest caveat.

## Framing to avoid

Do **not** pitch as SOTA on EEG↔video alignment — that loses on numbers to
trial-averaged decoding work whose numbers are not comparable anyway. Pitch as
**diagnosis → theory → principled objective**. Workshops reward that shape.
