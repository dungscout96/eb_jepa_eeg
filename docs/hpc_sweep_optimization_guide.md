# HPC Sweep Optimization Guide

Lessons learned from running large ML hyperparameter sweeps on SLURM clusters
(A40 GPUs, Delta HPC). Applies broadly to any GPU training sweep where you want
to minimize cost (GPU-hours) and wall time.

---

## 1. Understanding GPU Accounting

Most HPC clusters charge **per GPU-hour** regardless of actual GPU utilization.
On Delta's `gpuA40x4` partition, 1 SU = 1 A40-hour (16 CPU cores, 62.5 GB RAM).

**The core problem:** Small or fast experiments leave the GPU mostly idle.
If a training run uses 2 GB of a 48 GB GPU and finishes in 20 minutes, you still
paid for a full hour.

**The solution:** Pack multiple experiments into one allocation and run them
either in parallel or sequentially.

---

## 2. Parallel vs Sequential Packing

### 2a. Parallel execution (`& + wait`)

```bash
train.py --config=A &
train.py --config=B &
wait
```

Both processes run simultaneously on the same GPU.

- **Wall time** = `max(T_A, T_B)`
- **GPU-hours charged** = `max(T_A, T_B)` × 1 GPU
- **Best for:** Small configs where combined VRAM fits comfortably and neither
  process saturates compute.

**When it works:** Both experiments are small enough that they don't compete
for GPU compute or memory bandwidth. In practice this means each experiment
uses ≤ ~30% of theoretical GPU throughput.

**When it breaks:** Two large experiments competing for the same GPU can slow
each other down by 5–15×. A job that would take 2.5h solo instead takes 14h
with a competing process — costing 5.6× more GPU-hours than running sequentially.

### 2b. Sequential execution (`&&`)

```bash
train.py --config=A &&
train.py --config=B
```

The second experiment starts only after the first finishes (and only if it
succeeded).

- **Wall time** = `T_A + T_B`
- **GPU-hours charged** = `T_A + T_B` × 1 GPU
- **Best for:** Large configs where parallel execution would cause contention.

**Key insight:** Sequential pairing uses the *same total GPU-hours* as two
separate SLURM job submissions, but requires only **one queue slot**. On a
busy cluster this matters — fewer submissions means less scheduler overhead
and potentially faster overall throughput.

### 2c. When to use each

The decision threshold is based on **GPU memory + compute saturation**. A useful
proxy is `n_windows × window_size` (tokens per sample), but any measure of
per-step compute works.

| Config size | Proxy value | Execution mode | GPU-hour savings |
|-------------|-------------|----------------|-----------------|
| Small (< 4 GB VRAM each) | nw × ws ≤ 4 | Parallel | `T_A` (save `T_B` entirely) |
| Medium–Large (≥ 4 GB each) | nw × ws > 4 | Sequential | Same GPU-hrs, saves 1 queue slot |

```python
PARALLEL_THRESHOLD = 4  # nw * ws ≤ this → parallel safe

def _can_parallelize(nw, ws):
    return nw * ws <= PARALLEL_THRESHOLD

# In job builder:
if len(chunk) == 2 and all(_can_parallelize(r.nw, r.ws) for r in chunk):
    cmd = " &\n".join(cmds) + " &\nwait"
    time_limit = max(T_A, T_B)        # wall = max
else:
    cmd = " &&\n".join(cmds)
    time_limit = T_A + T_B            # wall = sum
```

**Calibrate your threshold empirically:** Submit one paired job of your medium
config. If per-epoch time is within ~20% of solo time → parallel is fine.
If it's 2× or more → switch to sequential.

---

## 3. Estimating Time Limits

Time limits must be conservative enough to avoid TIMEOUT but not so large that
they block queue slots.

### Per-epoch time estimation

Run a short timing test (10–20 epochs) with your largest expected config.
Key variables:

- **Sequence length / token count**: attention is O(n²) — doubling tokens
  quadruples compute per sample
- **Batch size**: halving batch size doubles the number of steps per epoch;
  combined with the O(n²) attention, large configs with forced small batch sizes
  can be 8× slower per epoch than baseline
- **DataLoader workers**: usually not the bottleneck for GPU-bound training
  (see Section 5)

### Scaling rule of thumb

Once you have a baseline measurement (e.g., 65s/epoch for nw4_ws2):

```
T_epoch(nw, ws, bs) ≈ T_baseline × (nw × ws / nw_base × ws_base) × (bs_base / bs)
```

This gives rough O(n²) attention scaling. Add a **1.5× safety margin** for the
SLURM time limit.

### Example calibration (A40, EEG JEPA project)

| Config | nw×ws | batch | measured T/epoch (solo) | safety × 1.5 for 100ep |
|--------|-------|-------|------------------------|------------------------|
| nw4_ws2 | 8 | 64 | 65s | ~2.5h |
| nw4_ws4 | 16 | 32 | ~2.6 min | ~6.5h |
| nw8_ws1 | 8 | 64 | ~1.5 min | ~3.75h |
| nw8_ws2 | 16 | 32 | ~7 min | ~17.5h |

---

## 4. Checkpoint Resuming

When a SLURM job times out, training state is preserved in the last saved
checkpoint (`latest.pth.tar`). You can resume from it in a follow-up job.

### Pattern

```python
# In the training script: load checkpoint if flag is set
if cfg.meta.get("load_model"):
    ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
    ckpt_info = load_checkpoint(ckpt_path, model, optimizer)
    start_epoch = ckpt_info["epoch"]
```

```bash
# Resume command: pass the absolute path to the saved checkpoint
python train.py \
  --meta.load_model=true \
  --meta.load_checkpoint=/abs/path/to/latest.pth.tar \
  [all other original args]
```

**Python pathlib note:** `Path(new_exp_dir) / "/absolute/path"` resolves to
`/absolute/path`. This lets you pass a full path as a config override without
changing the training script's path join logic.

### What to track

Each resumed run creates a **new** experiment directory (new timestamp) and a
new W&B run. For analysis, identify runs by their config (hyperparams in the
run name) and filter by `wandb_group`. The final metric at epoch 99 is all that
matters for comparing configs.

### Estimating remaining time

```
remaining_hours = (total_epochs - saved_epoch) × T_epoch_solo × 1.5
```

Use the saved epoch count to set tighter time limits on the resume job rather
than re-using the full original limit.

---

## 5. Data Loading: Workers, pin_memory, persistent_workers

### Is data loading the bottleneck?

**Test:** Compare per-epoch time with `num_workers=2` vs `num_workers=16`.
If the difference is < 10%, training is **GPU-bound** — data loading is not
the bottleneck.

In the EEG JEPA project, this test confirmed GPU-bound behavior for the
baseline config. For larger configs (longer sequences, more compute per step),
the GPU bottleneck is even stronger.

**Practical implication:** For GPU-bound training, increasing `num_workers`
beyond 2–4 has negligible effect on throughput. Use enough workers to keep
the GPU fed without wasting CPU resources.

### pin_memory

Allocates CPU tensors in page-locked (pinned) memory, enabling faster
CPU → GPU DMA transfers.

- **Pro:** Faster host-to-device transfer (~2× for large batches)
- **Con:** Increases host RAM usage; pinned memory is a limited resource
- **Enable when:** Batch size is large and CPU → GPU transfer is measurable
- **Skip when:** Training is compute-bound and batches are small

```python
DataLoader(..., pin_memory=True)  # enable if CPU→GPU transfer shows in profiling
```

### persistent_workers

Keeps DataLoader worker processes alive between epochs instead of forking/joining.

- **Pro:** Eliminates worker startup overhead (~5–30s per epoch for complex datasets)
- **Con:** Workers hold dataset state in memory continuously; increases RAM
- **Enable when:** Dataset initialization per epoch is expensive (e.g., opening
  many files, computing statistics)

```python
DataLoader(..., persistent_workers=(num_workers > 0))
```

### Caching expensive initialization

If your dataset computes statistics (e.g., normalization mean/std) over the
whole dataset at startup, save them to disk and load on subsequent runs:

```python
cache_file = data_dir / "normalization_stats.npz"
if cache_file.exists():
    cached = np.load(cache_file)
    mean, std = cached["mean"], cached["std"]
else:
    mean, std = compute_over_dataset(...)  # expensive
    np.savez(cache_file, mean=mean, std=std)
```

In the EEG JEPA project, this reduced initialization from ~8 minutes to
~3 seconds per run — significant when submitting 30+ experiments.

### Recommended defaults

```python
num_workers = 4             # per process; use 8 for solo jobs with 16 CPU cores
pin_memory = True           # always safe to enable for GPU training
persistent_workers = (num_workers > 0)
```

When packing 2 parallel processes on a 16-core node, use `num_workers=4` per
process (8 total ≤ 16 available).

---

## 6. Common Failure Modes and Fixes

### 6a. Git index.lock (concurrent git pulls)

**Symptom:** Job fails immediately with "Another git process seems to be running".

**Cause:** Multiple SLURM jobs run `git pull` simultaneously on the same repo
clone. SLURM can schedule multiple jobs to the same node, causing them to
conflict on the shared git lock file.

**Fix:**
```bash
# Add a random sleep before git pull in each SLURM script
sleep $((RANDOM % 15)) && git fetch origin && git pull --ff-only
```

The 0–15 second jitter is enough to stagger concurrent pulls on the same node.

If it still happens, manually remove the lock:
```bash
rm /path/to/repo/.git/index.lock
```

### 6b. Silent process crash (no error message)

**Symptom:** Only 1 of 2 parallel processes logs "Training complete!"; no
traceback anywhere.

**Cause:** When multiple SLURM jobs land on the same node simultaneously, the
total number of parallel processes can exceed what was planned. E.g., two
2-parallel jobs on the same node = 4 processes competing. The Linux OOM killer
or a CUDA context failure terminates one process silently (SIGKILL doesn't
produce a Python traceback).

**Detection:** Check `squeue` to see if multiple jobs ran on the same node at
the same time. Compare node names in job log headers.

**Fix:** Use sequential packing for medium/large configs (reduces from 4
competing processes to 2), or request `--exclusive` to prevent co-scheduling
(costs more SUs).

### 6c. TIMEOUT on large configs

**Symptom:** Job exits with `CANCELLED AT ... DUE TO TIME LIMIT`, checkpoint
saved at epoch N < 100.

**Cause:** Time limit was estimated assuming parallel execution at full speed,
but contention caused 5–15× slowdown for large configs.

**Fix:**
1. Switch to sequential execution for large configs
2. Use per-epoch timing data to set accurate time limits
3. Implement checkpoint resuming so partial runs can be continued

### 6d. Experiment naming collisions (W&B / checkpoint conflicts)

**Symptom:** Multiple runs share the same checkpoint directory; W&B runs
resume each other's state.

**Cause:** Experiment names don't encode all swept dimensions. E.g., only
encoding `batch_size` and `lr` when also sweeping `n_windows` and
`window_size`.

**Fix:** Include ALL swept hyperparameters in the experiment directory name:

```python
exp_name = (
    f"model_bs{cfg.data.batch_size}"
    f"_lr{cfg.optim.lr}"
    f"_nw{cfg.data.n_windows}"        # include all swept dims
    f"_ws{cfg.data.window_size}s"
    f"_seed{cfg.meta.seed}"
)
```

---

## 7. Sweep Architecture

### Two-phase sweep pattern

For expensive hyperparameter searches, use a two-phase approach:

**Phase 1 — Wide sweep (many configs, fewer seeds):**
Sweep the structural hyperparameters that are most uncertain (e.g., model
architecture, objective function). Use 2–3 seeds. Goal: eliminate unpromising
regions cheaply.

**Phase 2 — Narrow sweep (top configs, more seeds):**
Take the top 2–3 configs from Phase 1 and sweep training hyperparameters
(lr, regularization coefficients). Use 3–5 seeds. Goal: robust comparison of
the finalists.

This typically cuts total GPU-hours by 5–10× compared to a flat grid search.

### Filtering Phase 2 candidates

From Phase 1, select configs that satisfy all of:
1. Primary metric above threshold (e.g., `probe_acc > 0.60`)
2. No collapse (e.g., `cosim < 0.90`)
3. No divergence (loss not growing at end of training)

### Seed strategy

3 seeds is the minimum for a meaningful mean ± std. 5 seeds gives tighter
confidence intervals but costs more. For Phase 1 screening, 2 seeds per config
can be sufficient to detect clear winners.

---

## 8. Monitoring Checklist

```bash
# Job status
ssh cluster "squeue -u $USER"
ssh cluster "squeue -u $USER | wc -l"

# Completed jobs with timing
ssh cluster "sacct -u $USER --starttime=today -o jobid,jobname,state,elapsed,maxrss"

# Quick check: how many training runs completed
grep -c "Training complete" logs/jobname_JOBID.out

# Check for failures
grep -l "Error\|Traceback\|CUDA" logs/*.err
```

W&B grouping: use `--logging.wandb_group=sweep_phaseN` so all runs for a
phase appear together in the W&B dashboard filter.

---

## 9. Quick Reference: Decision Flowchart

```
For each pair of experiments to submit:
│
├─ Are both small (nw×ws ≤ threshold)?
│   ├─ YES → run PARALLEL (&+wait), time_limit = max(T1, T2)
│   └─ NO  → run SEQUENTIAL (&&), time_limit = T1 + T2
│
├─ Does either experiment have a partial checkpoint?
│   ├─ YES → add --meta.load_model=true --meta.load_checkpoint=/abs/path
│   │         time_limit = remaining_epochs × T_epoch × 1.5
│   └─ NO  → fresh start, use full epoch time estimate
│
└─ Is training GPU-bound?
    ├─ YES → num_workers=4 is sufficient; don't waste cores on workers
    └─ NO  → increase num_workers, enable pin_memory + persistent_workers
```
