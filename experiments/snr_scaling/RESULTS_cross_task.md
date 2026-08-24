# Cross-task generalization — TP-trained checkpoints evaluated on DespicableMe

Companion to [`RESULTS.md`](RESULTS.md), which is the within-movie record. Every E0.3
checkpoint was trained on **ThePresent only**; this file asks whether what they learned is
stimulus-general or movie-specific.

---

## TL;DR

1. **The features transfer; the alignment does not.** With only a ridge head refit, a
   TP-trained encoder reaches **58 % of DM-native** by CV-val Δr² (§2.1) and **69 %** by
   train→eval Δ*r* on both val and test (§2.2). But *zero-shot* retrieval — nothing refit —
   lands **below a random encoder**, while a DM-native model clears it. The representation
   carries cross-movie information; the CLIP-aligned space does not.
2. **Transfer improves steeply with training subjects**, from *below random* at S ≤ 20 to
   69 % of DM-native at S=1863. Subject scaling buys *generalization*, not just within-movie
   accuracy — the strongest argument yet for the subject axis, and it comes from a movie the
   model never saw.
3. **Transfer saturates where within-movie performance does**, at S≈1000 — the same knee as
   §2.11, now seen on three independent readouts (§2.1, §2.2 val, §2.2 test).
4. **Measured against its own within-movie score, transfer is strong: 81–85 %.** At every
   S ≥ 701 a checkpoint reads DespicableMe at ~⅘ of how well it reads ThePresent, which the
   near-equal ceilings (§2.9) make a fair comparison. "58 % of native" understates how much
   of the *encoder* survives the movie change; what fails is the head.

---

## §1. Setup

| | |
|---|---|
| Checkpoints | E0.3 cells, TP-trained, epoch 325 (smoothed selection), one draw per S |
| DM data | train **1832** (R1–R4 + R7–R10), val 299 after filtering, test 108 |
| Eval config | one shared `config/config_probe_DM.yaml` — `data.task=DespicableMe`, scaling knobs cleared |
| Features | all 12 present in DM's parquet, none constant (checked against TP) |

**Two readouts, deliberately answering different questions:**

- **Zero-shot retrieval (DM test).** Encoder *and* CLIP projection are TP-trained; nothing
  is refit. Tests whether the TP-aligned space itself transfers.
- **Linear transfer (CV probe, DM val).** Encoder frozen, only a ridge head refit on DM.
  Tests whether the features carry DM information even if the alignment does not.

> **On 1832 vs the 1841 this file used to quote.** 1841 is the `max_subjects` cap requested
> for the DM-native cells; the realised DM train pool is **1832**, measured identically by all
> 26 train→eval artifacts here and, independently, by 84 depth-22 `e04` cross-task artifacts.
> A cap above the pool is a silent no-op. The 9-recording gap is presumably a few DM
> recordings failing the annotation/duration checks ThePresent's pass. It changes nothing in
> the comparison — every cell fits its head on the same 1832 — but the cell named
> `e03_s1841_a85_dm` is the **full** DM pool, not a 1841-subject draw from a larger one.

**DM-native reference.** Two cells trained *on DM* with the identical recipe (S=701 and
S=1841, `max_anchors=85` for DM's 170.6 s / 85-anchor grid, `epoch_size=703` so they run the
same 4400 steps). Without this anchor, transfer could only be compared to random, which
cannot separate "transfers well" from "DM is easy".

**Verified the evaluation actually used DM**, since a silently-TP run would look entirely
plausible: the probe reports **299** val recordings / 25 415 windows against TP's 293 /
29 593, and retrieval reports pools of **85/32/26** against TP's 101/49/35. The 299 matches
§2.9's independent DM measurement exactly.

---

## §2. Linear transfer — the features do transfer, and it scales

Two protocols, reported separately because they differ in more than arithmetic: §2.1
cross-validates *within* DM val, §2.2 fits the head on DM train and evaluates on a split it
never saw, on **both** val and test.

### §2.1 CV probe on DM val — 58 % of DM-native

Against a DM random baseline of r² = 0.00681. The 100 % anchor is DM-native S=1841
(Δr² = 0.03901).

| S (TP training subjects) | DM Δr² | % of DM-native |
|---:|---:|---:|
| 10 | 0.00073 | 2 % |
| 20 | 0.00098 | 3 % |
| 50 | 0.00468 | 12 % |
| 100 | 0.00604 | 15 % |
| 200 | 0.00872 | 22 % |
| 400 | 0.01504 | 39 % |
| 701 | 0.02065 | 53 % |
| 1000 | 0.02309 | **59 %** |
| 1400 | 0.02316 | **59 %** |
| 1863 | 0.02255 | 58 % |

> **Transfer is real, substantial, and bought with subjects.** 2 % at S=10 rising to 58 % at
> S=1863 — a 30× improvement in cross-movie generalization from subject scaling alone, on a
> film the encoder never saw.

**It saturates at the same place within-movie performance does.** 59 / 59 / 58 % at
S = 1000 / 1400 / 1863 mirrors §2.11's knee at ~700–1000. So the subject axis stops paying
for generalization and for within-movie accuracy at the same point — one knee, not two.

For reference, the DM-native cells themselves: S=701 → 0.03615, S=1841 → 0.03901 (+8 %),
so DM-native also improves with subjects, just from a higher base.

### §2.2 Train→eval Pearson *r*, on val **and** test

The `probe_traintest` protocol of RESULTS.md §2.12, now on DespicableMe: RidgeCV head fit on
the full DM train pool (1832 recordings, identical for every cell — only the encoder's
training cohort differs), evaluated on a split it never saw. Mean Pearson *r* over the 12
scalar features. Δ is over the DM random baseline; **% nat** is Δ as a fraction of DM-native
S=1841's Δ; **DM/TP** compares each checkpoint's DM transfer to its *own* within-movie
ThePresent Δ on test, from the §2.12 table.

| S | val *r* | Δ | % nat | test *r* | Δ | % nat | DM/TP (test) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.1107 | — | — | 0.0989 | — | — | — |
| 10 | 0.1066 | −0.0041 | −3 % | 0.0921 | −0.0069 | −6 % | — |
| 20 | 0.1092 | −0.0015 | −1 % | 0.0945 | −0.0044 | −4 % | — |
| 50 | 0.1245 | +0.0138 | 12 % | 0.1054 | +0.0064 | 6 % | 41 % |
| 100 | 0.1338 | +0.0231 | 20 % | 0.1174 | +0.0185 | 17 % | 67 % |
| 200 | 0.1456 | +0.0349 | 29 % | 0.1343 | +0.0354 | 33 % | 69 % |
| 400 | 0.1653 | +0.0545 | 46 % | 0.1487 | +0.0498 | 47 % | 70 % |
| 701 | 0.1853 | +0.0746 | 63 % | 0.1709 | +0.0720 | 68 % | 85 % |
| 1000 | 0.1900 | +0.0792 | 67 % | 0.1698 | +0.0709 | 67 % | 82 % |
| 1400 | 0.1910 | +0.0803 | 68 % | 0.1713 | +0.0724 | 68 % | 81 % |
| 1863 | **0.1918** | **+0.0811** | **69 %** | **0.1732** | **+0.0743** | **70 %** | **85 %** |
| DM-native S=701 | 0.2195 | +0.1088 | 92 % | 0.1941 | +0.0952 | 90 % | — |
| DM-native S=1841 | 0.2291 | +0.1183 | 100 % | 0.2052 | +0.1063 | 100 % | — |

**Three things this adds that §2.1 could not.**

**1. It holds on test.** §2.1 is CV *within* val — the split checkpoint selection was done on.
Test is untouched by both, and the val and test columns agree to within a percentage point of
Δ at every S ≥ 200. The transfer result is not an artifact of the selection split.

**2. At S ≤ 20 transfer is negative, on both splits.** The encoder is *worse than random* for
DespicableMe (−0.0069 test at S=10). §2.1 reported "2 %" there, which reads as "a little
transfer"; it is more honest to say a model trained on 10 subjects of one movie carries **no**
usable information about another, and slightly damages the features a random projection would
have given. The sign flips between S=20 and S=50.

**3. Against its own within-movie score, transfer is far better than 58 % suggests.** The same
checkpoint reads DespicableMe at **81–85 %** of how well it reads the movie it was trained on,
for every S ≥ 701. That comparison is only fair because §2.9 measured the two movies' ceilings
as near-equal (0.304 vs 0.313) on a 98.9 %-shared cohort — otherwise "85 %" could just mean DM
is easier.

> **The two protocols agree on shape, not on magnitude.** Both show transfer rising with
> subjects and flattening at S ≈ 700–1000. But the fraction-of-native is **58 %** by Δr² on
> CV-val and **69–70 %** by Δ*r* on train→eval. These are different summaries of different
> fits, and 0.69² = 0.48 does not recover 0.58 either, so the gap is protocol, not units.
> **Quote the metric with the number.** RESULTS.md §2.12 measured the same protocol gap
> within ThePresent at ~25 %, in the same direction — CV lower.

**Saturation replicates, a third time.** val 0.1900 → 0.1910 → 0.1918 and test 0.1698 → 0.1713
→ 0.1732 across S = 1000 → 1400 → 1863: monotone but flat. Nearly doubling the cohort buys
**+2.4 % of Δ on val and +4.8 % on test**, against **+114 % and +103 %** for the same 1.75×
step from 200 → 701. Same knee as §2.1 and as RESULTS.md §2.11. One small difference from
§2.12: there, S=1863 fell *below* S=1400 within ThePresent; here it is the top cell on both
splits, so that dip was noise rather than extended-pool dilution.

**DM-native still improves with subjects** — 0.2195 → 0.2291 val, 0.1941 → 0.2052 test from
S=701 to the full pool (+4.4 % / +5.7 % in *r*, +9 % / +12 % in Δ), from a base whose Δ is
**1.4×** the best transfer cell's.

---

## §3. Zero-shot retrieval — the alignment does not transfer

DM test, e→v scene level (pool N=26, chance@1 = 0.0385).

| model | top-1 | top-5 | top-10 |
|---|---:|---:|---:|
| DM random | 0.1061 | 0.3307 | 0.5163 |
| **DM-native S=701** | **0.1464** | **0.4486** | — |
| DM-native S=1841 | 0.1276 | 0.4351 | — |
| TP-trained S=1863 → DM | 0.0526 | 0.2674 | 0.4778 |
| TP-trained S=10 → DM | 0.0682 | 0.2211 | 0.3832 |

**TP-trained checkpoints score below random on every K, and top-1 *declines* with more TP
training** (0.0682 at S=10 → 0.0526 at S=1863). More ThePresent makes DespicableMe retrieval
worse.

> **A caveat that the DM-native reference resolves.** The DM random baseline sits at
> **2.76× chance** (0.1061 vs 0.0385) — a random encoder cannot genuinely beat chance, and
> this is the same modal-answer collapse documented in §2.7/§2.12. Taken alone, "trained <
> random" would be uninterpretable.
>
> **The DM-native cells settle it.** They clear the same inflated baseline (0.1464 vs
> 0.1061), so the metric *can* detect genuine DM alignment. A TP-trained model failing to is
> therefore a real failure, not a baseline artifact. This is exactly why the reference was
> worth training.

Top-5 does rise with TP subjects (0.221 → 0.267), so some coarse structure survives — but
never enough to reach the baseline.

**v→e is at chance (1.00×) for every TP-trained cell on DM**, matching §2.13's finding that
it is 1.00× at every S from 10 to 1863 within TP. Cross-movie changes nothing.

---

## §4. What this means

**The split is clean, and it is the same split as saturation (§2.12) — sharper here.**

| | transfers? |
|---|---|
| Encoder features (probe, head refit) | **yes — 58–69 % of native, 81–85 % of its own within-movie score, scaling with subjects** |
| CLIP alignment (retrieval, nothing refit) | **no — below a random encoder** |

The encoder learns something about EEG-during-movie-watching that is not specific to
ThePresent. The projection head that maps it into V-JEPA-2 space is specific to ThePresent.
That is consistent with jul2's domain-shift finding — the two movies' V-JEPA-2 target
distributions differ enough that one alignment does not serve both — and it localises the
problem to the *head*, which is the cheap part to retrain.

**Practical consequence.** A TP-trained encoder is a reasonable initialisation for a new
movie provided the alignment is refit; deploying it zero-shot is not viable. And the way to
improve cross-movie transfer is more *subjects*, not more of the same movie.

**Caveats.** One draw per S (between-draw sd on the TP probe was 0.0012, small, but not
measured here). One target movie. The DM-native anchor is n=2 cells, and its own
subject-scaling is only sampled at two points. §2.2's DM/TP column compares a transfer Δ to a
within-movie Δ measured on a different split's baseline; the near-equal ceilings (§2.9) make
that fair to first order, not exactly.

---

## §5. Artifacts

- `raw_results/xtask_probe_DMval_*.json` — §2.1 CV probe on DM val, 10 TP cells + random + 2 DM-native
- `raw_results/xtask_tt_DM{val,test}_*.json` — §2.2 train→eval probe, both splits, same 13 (26 files)
- `raw_results/xtask_retr_DMtest_*.json` — §3 zero-shot retrieval on DM test, same set
- `config/config_probe_DM.yaml` — the shared eval config all of the above used
- DM-native checkpoints: `/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling/e03_s{701,1841}_a85_dm/`
- Submitted via `_submit_e03.py --dm-reference` (training), `_submit_retrieval.py cross|dm-ref`
  (§3) and `_submit_traintest.py cross|dm-ref` (§2.2)
- Reproduce the §2.2 table: `src/summarise_traintest.py --family xtask`

> **Check artifacts, not exit codes — this section is why.** The first §2.2 submission omitted
> `HBN_TRAIN_RELEASES`, so the ridge fit on the R1–R4 default (741 recordings) instead of the
> extended 1832. All five jobs reported `COMPLETED 0:0` and wrote 26 well-formed, plausible,
> wrong JSONs; only `n_train_recordings` distinguished them. `_submit_traintest.py <preset>
> verify` now checks that field against the expected pool for every artifact, and the
> superseded files are kept at `raw_results/_wrongpool_R1R4/` on Delta.
