# E0.4 — Full-REVE numbers on the tracked metrics

Status: **proposed, not started.** Drafted 2026-08-08 against `main` @ `a68c5e4`.

**Goal:** re-run the E0.3 subject-scaling axis with the full REVE encoder and report the
same three readouts we already track, so the E0.4 rows drop into the existing tables.

Not a new study design. Same cells, same protocol, same metrics — one shape change.

---

## 1. What runs

Exactly one arm. Everything except the encoder shape and the init is held at E0.3's
values so the rows are comparable.

| | E0.3 (existing) | **E0.4 (this plan)** |
|---|---|---|
| `encoder_embed_dim` | 512 | 512 |
| `encoder_depth` | 12 | **22** |
| `patch_size` / `patch_overlap` | 400 / 0 | **200 / 20** |
| init | random | **`reve-base` → EET** |
| params | 37.9 M | **69.2 M** |
| tokens/sample | 129 | **258** |
| loss | `soft_target_clip`, α=0.5, τ=0.05 | same |
| lr / optim | 1e-4 / adam / warmup 5 | same |
| batch / n_windows | 64 / 1 | same |
| epochs / steps | 400 / 4400 | same |
| `meta.seed` | 2026 | same |
| cells | S ∈ {400,701,1000,1400} × draw {11,22,33} + S=1863, A=101 | same |

**Two of those rows are forced, not chosen.** `to_patch_embedding.weight` in the REVE
checkpoint is `(512, 200)`, so loading it into a patch-400 model raises `RuntimeError`.
Taking REVE's weights means taking REVE's geometry — patch 400 is not available in this
arm. Everything else is deliberately pinned to E0.3.

`lr=1e-4` is kept from E0.3 rather than the REVE record's `3e-4`; the record's own
results section states the recipe is robust across `[1e-4, 3e-4]` and that the two
**tie on test** (+0.198 vs +0.197), so holding E0.3's value removes a confound.

---

## 2. The metrics we track

Per cell, computed on the epoch chosen by smoothed `val/clip_scene_auc` — the same
selection E0.3's nested cells used (they all landed on epoch 325).

| # | metric | script | output |
|---|---|---|---|
| 1 | **Δr² over random init**, 12-feature CV probe, val | `eb_jepa/evaluation/clip_probe/probe.py --split val --cv-splits 5` | `e04_probe_val_{slug}.json` |
| 2 | **Pearson r**, train→test ridge probe, val **and** test, `--bootstrap 2000` | `eb_jepa/evaluation/clip_probe/probe_traintest.py --eval-split {val,test}` | `e04_tt_{val,test}_{slug}.json` |
| 3 | **Scene-level retrieval top-1/5/10**, e→v, **test** | `eb_jepa/evaluation/clip_probe/retrieval.py --split test --topks 1 5 10` | `e04_retr_test_{slug}.json` |

Then: mean/sd across the three draws per S, step exponents, and the §2.12-style
three-readout table.

**Reporting rules that apply to these numbers** (house rules, from `RESULTS.md`):
- Report `r`/ceiling from the **same split**; never raw `r` alone.
- Use **test** retrieval, not val — a random encoder scores 3.79× chance at scene level
  on val while sitting below chance on test.
- Say "the 12-feature linear probe saturates near N subjects", never "the subject axis
  saturates" — readouts 1 and 3 disagreed at depth 12 and may again.

**A fresh depth-22 random-init null is mandatory for all three.** Δr² is defined against
"a random encoder of identical shape"; the existing null (r² = 0.014127) is depth-12 and
cannot be reused. This is 3 short jobs and it gates every number in the table.

### What to expect

Depth ≥ 22 has already been measured as a **tie** twice, both at the 703-recording scale:
jul1 iter 14 (depth 12→22 at patch 400, Δ +0.00009, "discard") and jul2 (depth 24, ~tied).
`RESULTS_jul7.md` §7.3 lists capacity scaling as "off-limits — nothing to gain past
embed=512 depth=12 **at this data scale**."

That qualifier is why E0.4 is worth running: crossing depth with the *extended* pool
(1863) is the one framing not yet ruled out. But if the numbers come back flat, that is
consistent with three prior measurements, not a bug — and the pretrained init is the
part most likely to move them, since REVE warm-start holds the repo's record
(test Δr +0.197 vs +0.099 from-scratch).

---

## 3. Blockers to clear first

All confirmed on Delta / in source, not suspected.

1. **`_submit_e03.py` points at another account.** `REPO = "/u/dtyoung/eb_jepa_eeg"`
   (line 40) is permission-denied here; `CKPT_ROOT` is under `dtyoung` too.

2. **Silent depth truncation — the dangerous one.** `load_encoder_weights` uses
   `load_state_dict(..., strict=False)` (`training_utils.py:272`). Loading REVE-base into
   a depth-12 model gives `missing=0, unexpected=61` and trains on a truncated REVE
   without complaint. It produces plausible numbers, so it will not be caught by looking
   at the loss curve.

3. **`analyse_e03.py` / `plot_e03.py` are broken on a fresh checkout** — commit `6cabc01`
   moved the JSONs into `raw_results/`, the scripts still resolve the old paths.

4. **No aggregator for nested draws.** `analyse_e03.py::load` keys by `(S, A)`, so three
   draws at one S collide, and `s_axis` is hardcoded. The §2.11/§2.12 tables were computed
   ad hoc and exist only as prose — E0.4 needs this written to produce its own table.

5. **`config/clip_pretrain_from_reve.yaml` is not the record recipe** (`shot_mean`,
   lr 1e-5, 100 ep vs `per_window`, 3e-4, 300 ep). Do not launch from it unmodified.

---

## 4. Steps

**Phase 0 — code, no GPU**

1. `eb_jepa/training_utils.py`: strict-load guard in `load_encoder_weights` — raise
   unless `missing == []` and `unexpected ⊆ {"_position_bank.embedding"}`, behind
   `allow_partial=False`. Unit test in `tests/unit/`.
2. `experiments/snr_scaling/clip_pretrain_reve.yaml`: new frozen snapshot. Copy the
   existing frozen file; change **only** `encoder_depth: 22`, `patch_size: 200`,
   `patch_overlap: 20`. Header stating what it is frozen against and not to edit it.
3. `experiments/snr_scaling/_submit_e04.py`: clone of `_submit_e03.py` with
   `REPO=/projects/bbnv/kkokate/eb_jepa_eeg`,
   `CKPT_ROOT=/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling`, slug prefix `e04_`,
   `--time-limit 03:00:00`, copies `clip_pretrain_reve.yaml`, threads
   `--meta.encoder_init_from`.
4. Fix the `raw_results/` path drift in the three analysis scripts.
5. `experiments/snr_scaling/aggregate_nested.py`: mean/sd per S across draws, step
   exponents, three-readout table. Key by `(S, draw_seed)`, discover by glob, serve both
   E0.3 and E0.4.
6. Convert the checkpoint on Delta — `prepare_reve_checkpoint.py` →
   `/work/hdd/bbnv/kkokate/eb_jepa/reve_base_eet_init.pth.tar`. HF cache already present.

**Phase 1 — one smoke cell (~20 min)**
20 epochs at S=701. Confirms the checkpoint loads with zero missing keys, measures real
s/epoch, and checks memory headroom. Settles the repo's own 2.7× disagreement on d22/p200
throughput (27 s/ep in `mjepa/_submit.py:121` vs 10 s/ep in `mjepa/NOTES.md`) before
13 jobs are sized against it.

**Phase 2 — nulls (3 jobs, ~1 GPU-h)**
Random-init depth-22 at the same geometry, all three readouts.

**Phase 3 — the sweep (13 cells)**
```
uv run --group eeg python experiments/snr_scaling/_submit_e04.py \
    --nested-draws --suffix=_nd --save-every=25 --skip-probe submit
```

**Phase 4 — select + read out**
`select_and_probe_e04.py --cells-glob 'e04_s*_a101_nd*' --probe`, then metrics 2 and 3.

**Phase 5 — aggregate → table → `RESULTS.md` section.**

---

## 5. Cost — measured, not extrapolated

| shape | tokens | s/epoch | source |
|---|---|---|---|
| d12 / p400 | 129 | ~3.4–5.0 | `results.tsv`, 25 min / 300 ep incl. probe |
| d22 / p400 | 129 | 6.6 | `results.tsv` commit `35784f0` |
| **d22 / p200** | **258** | **11.5** | commit `080339b`, Delta job 20400451 (57:26 / 299 ep) |

Depth alone is cheap — 1.56× wall, ~1 GB extra memory (14.0 vs 13.0 GB). **Halving
`patch_size` costs about as much as adding 10 layers**, because attention is O(n²) in
tokens. Geometry is the budget driver.

Per cell at 400 ep: ~77 min train, ~85 min with probe → **`--time-limit 03:00:00`**
(the S=1863 cells ran 1.2× the S=701 cells in E0.3, and three `jul7-e400` jobs already
timed out at 1 h 39).

- 13 cells ≈ **30–36 GPU-h**; whole plan ≈ **40 GPU-h** against a **27,653 GPU-h** balance
- disk ≈ 14 GB/cell × 13 ≈ **180 GB**; 47 TB free on `/work`

Compute is not the constraint. The analysis-code debt in §3 is.

---

## 6. Invariants preserved

- **Constant gradient steps, not epochs** — `data.epoch_size=703` → 11 steps/ep → 4400.
- **Probe sees full data** — `config_probe.yaml` clears `max_subjects/max_anchors/epoch_size`.
- **val = R5, test = R6**, never extended.
- **S counts subjects**; draws nest (permute once, take prefix).
- **`meta.seed=2026` fixed**; only `subsample_seed` varies.
- **Never edit `clip_pretrain.yaml`** (frozen, line 18) — freeze a new one.
- Slug prefix `e04_` so nothing clobbers E0.3.
- Cells discovered by glob, never a second hardcoded list.
- PEP 8; imports at top; `black` + `isort` over `eb_jepa experiments tests`.

## 7. Infrastructure — verified

| item | status |
|---|---|
| REVE-base weights | cached, `/u/kkokate/.cache/huggingface/hub/models--eeg-telecom-paris--reve-base` |
| R1–R6 | `/projects/bbnv/kkokate/hbn_preprocessed` — 703 train / 296 val / 109 test |
| R7–R10 | readable at `/work/hdd/bbnv/dtyoung/hbn_preprocessed` — no re-preprocessing |
| scratch | `/work/hdd/bbnv/kkokate`, 47 TB free |
| allocation | 27,653 GPU-h of 30,000 on `bbnv-delta-gpu`; queue empty |
