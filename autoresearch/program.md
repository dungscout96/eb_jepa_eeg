# EEG-JEPA Autoresearch — encoder body search

Adapted from Karpathy's `program.md`. The agent is the loop: edit one file,
submit a Delta job, parse the result, decide keep/discard, advance or reset
the branch, repeat.

## Setup

- **Branch**: `autoresearch/apr26` (already created from `main`).
- **Mutable file** (the only one you edit each iteration):
  `eb_jepa/encoder_search.py` — exports `build_encoder_body(embed_dim) -> nn.Module`.
- **Locked files** — do NOT edit:
  - `experiments/eeg_jepa/main.py` (training harness)
  - `experiments/eeg_jepa/eval.py` (metric computation)
  - `experiments/eeg_jepa/cfgs/autoresearch.yaml` (fixed evaluation config)
  - `eb_jepa/architectures.py` (tokenization, pooling, predictor)
  - `eb_jepa/masking.py`, `eb_jepa/jepa.py`, `eb_jepa/datasets/hbn.py`
- **Untracked**: `autoresearch/results.tsv`, `run.log`.

## Per-experiment budget

- **Wall-clock training**: 30 min (gated by `optim.time_budget_minutes` in `autoresearch.yaml`).
- **SLURM time_limit**: 1 h (leaves ~30 min for startup + final eval).
- **GPU**: 1 × A40 on Delta `gpuA40x4`.

## Decision rule

Three layers — a hard gate, an optimisation target, and a consistency check.

### Layer 1: hard collapse gate (auto-discard if either fails)

```
sanity_emb_var_min   > 0.01     # else: encoder dimension(s) collapsed
sanity_cosim_max     < 0.99     # else: pathological pairwise similarity
```

Failures are auto-marked `status=discard` by `parse_log.py` regardless of any
other metric. Cheap, unambiguous; catches degenerate runs that would otherwise
"score" through luck or shortcut features.

### Layer 2: optimisation target (`val_bpb` analog)

```
val_corr_weighted = 0.30 * val/reg_position_in_movie_corr
                  + 0.30 * val/reg_contrast_rms_corr
                  + 0.30 * val/reg_luminance_mean_corr
                  + 0.10 * val/reg_narrative_event_score_corr
```

Two flavors emitted per run:
- `val_corr_weighted` — final state (after the time budget cuts training).
- `val_corr_weighted_max` — running max across all per-epoch validations.

**Use `val_corr_weighted_max` for keep/discard decisions.** Avoids penalizing
architectures that peak then degrade.

### Layer 3: consistency check (`rep_score`, diagnostic only)

For each candidate, after standardising across the last ~10 runs in
`results.tsv`:

```
rep_score = z(sanity_lin_probe_acc)
          + 0.5 * z(sanity_emb_var_mean)
          - 0.5 * z(sanity_cosim_mean)
```

If `val_corr_weighted_max` improves but `rep_score` drops sharply (Δ > 1 std
below the running mean), **flag the run as suspicious** (likely position-leak
or probe-overfit win) but do NOT auto-discard — leave to operator judgment.
Selection rationale in [autoresearch/analysis/metric_correlation_report.md](analysis/metric_correlation_report.md).

## What you CAN change in `eb_jepa/encoder_search.py`

Anything that produces an `nn.Module` mapping `[B, N, embed_dim] -> [B, N, embed_dim]`:

- transformer depth, heads, head_dim, mlp_dim_ratio
- different attention variants (sliding-window, linear attention, etc.)
- conv-mixer / MLP-mixer over tokens
- mamba / state-space bodies
- residual MLP stacks, gated MLPs
- hybrid stems (depthwise conv → transformer → ...)

## What you CANNOT change

- the `[B, N, embed_dim]` input/output contract (breaks JEPA + probes)
- `embed_dim` itself (it's set by `cfg.model.encoder_embed_dim=64`; mutating
  internally is fine, but I/O must stay 64)
- tokenization, masking, positional encoding (LOCKED in `architectures.py`)
- the predictor or pool layers
- dependencies in `pyproject.toml`

## The experiment loop

1. **State check**: `git rev-parse --short HEAD`, `cat autoresearch/results.tsv`.
2. **Edit `eb_jepa/encoder_search.py`** with one architectural idea.
3. **Commit + push**:
   `git add eb_jepa/encoder_search.py && git commit -m "<idea>" && git push origin autoresearch/apr26`
4. **Submit**: `python autoresearch/submit.py --label "<idea>"` — submits a Delta job that does
   `git fetch && git checkout <commit> && uv run python experiments/eeg_jepa/main.py --fname=experiments/eeg_jepa/cfgs/autoresearch.yaml > run.log 2>&1`.
5. **Wait**: poll `sacct -j <job_id>` until COMPLETED / FAILED / TIMEOUT.
6. **Pull log**: rsync `run.log` from the run directory.
7. **Parse**: `python autoresearch/parse_log.py run.log` → JSON of summary block + `val_corr_weighted`.
8. **Log row** (TSV, 9 columns):
   ```
   commit  val_corr_weighted  val_reg_position  val_reg_contrast  val_reg_luminance  val_reg_narrative  peak_vram_gb  status  description
   ```
   Use `0.000000` / `0.0` for crashed runs.
9. **Decide**:
   - **improved** vs. running best → status=`keep`, leave commit.
   - **equal/worse** → status=`discard`, `git reset --hard HEAD~1 && git push --force-with-lease origin autoresearch/apr26`.
   - **crashed** (no summary block) → status=`crash`, reset.
10. **Loop**.

## Crash handling

Run `tail -n 50 run.log` to read the stack trace. If it's a typo / import error / shape mismatch in your encoder body, fix and re-submit. If the idea is fundamentally broken (e.g. body doesn't preserve N), discard and move on.

## Time-out handling

The Delta SLURM `time_limit=01:00:00`. If a job hits this without producing a summary block, treat as crash. The 30-min training budget should always finish well within 1 h, so a TIMEOUT is a real signal something is wrong (probably a body that's far slower than expected).

## NEVER STOP

Once the loop has begun, do not pause to ask "should I keep going?". The user expects you to iterate autonomously. If you run out of obvious ideas, re-read the in-scope files for new angles, combine previous near-misses, try more radical changes (different attention, conv-mixers, state-space bodies). The loop runs until manually stopped.
