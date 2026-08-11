# Cross-task generalization — TP-trained checkpoints evaluated on DespicableMe

Companion to [`RESULTS.md`](RESULTS.md), which is the within-movie record. Every E0.3
checkpoint was trained on **ThePresent only**; this file asks whether what they learned is
stimulus-general or movie-specific.

---

## TL;DR

1. **The features transfer; the alignment does not.** With only a ridge head refit, a
   TP-trained encoder reaches **58 % of DM-native performance** on DespicableMe. But
   *zero-shot* retrieval — nothing refit — lands **below a random encoder**, while a
   DM-native model clears it. The representation carries cross-movie information; the
   CLIP-aligned space does not.
2. **Transfer improves steeply with training subjects: 2 % → 58 %** of DM-native from
   S=10 to S=1863. Subject scaling buys *generalization*, not just within-movie accuracy —
   the strongest argument yet for the subject axis, and it comes from a movie the model
   never saw.
3. **Transfer saturates where within-movie performance does**, at S≈1000 (59 %, 59 %, 58 %
   at S=1000/1400/1863) — the same knee as §2.11.

---

## §1. Setup

| | |
|---|---|
| Checkpoints | E0.3 cells, TP-trained, epoch 325 (smoothed selection), one draw per S |
| DM data | train 1841 (R1–R4 + R7–R10), val 299 after filtering, test 108 |
| Eval config | one shared `config/config_probe_DM.yaml` — `data.task=DespicableMe`, scaling knobs cleared |
| Features | all 12 present in DM's parquet, none constant (checked against TP) |

**Two readouts, deliberately answering different questions:**

- **Zero-shot retrieval (DM test).** Encoder *and* CLIP projection are TP-trained; nothing
  is refit. Tests whether the TP-aligned space itself transfers.
- **Linear transfer (CV probe, DM val).** Encoder frozen, only a ridge head refit on DM.
  Tests whether the features carry DM information even if the alignment does not.

**DM-native reference.** Two cells trained *on DM* with the identical recipe (S=701 and
S=1841, `max_anchors=85` for DM's 170.6 s / 85-anchor grid, `epoch_size=703` so they run the
same 4400 steps). Without this anchor, transfer could only be compared to random, which
cannot separate "transfers well" from "DM is easy".

**Verified the evaluation actually used DM**, since a silently-TP run would look entirely
plausible: the probe reports **299** val recordings / 25 415 windows against TP's 293 /
29 593, and retrieval reports pools of **85/32/26** against TP's 101/49/35. The 299 matches
§2.9's independent DM measurement exactly.

---

## §2. Linear transfer — 58 % of DM-native, and it scales

CV probe on DM val, against a DM random baseline of r² = 0.00681. The 100 % anchor is
DM-native S=1841 (Δr² = 0.03901).

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
| Encoder features (probe, head refit) | **yes — 58 % of native, scaling with subjects** |
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
subject-scaling is only sampled at two points.

---

## §5. Artifacts

- `raw_results/xtask_probe_DMval_*.json` — CV probe on DM val, 10 TP cells + random + 2 DM-native
- `raw_results/xtask_retr_DMtest_*.json` — zero-shot retrieval on DM test, same set
- `config/config_probe_DM.yaml` — the shared eval config all of the above used
- DM-native checkpoints: `/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling/e03_s{701,1841}_a85_dm/`
- Submitted via `_submit_e03.py --dm-reference` (adds `--task`, `DM_CELLS`)
