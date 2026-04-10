# Plan: Scale EEG JEPA Sweeps to Delta Cluster

## Context

After 33 experiments on jamming, best config: `depth=2, embed_dim=64, lr=5e-4, VCLoss(0.25,0.25), Huber, 100ep, bs=64` → probe_acc=0.641, cosim=0.853, pred_loss=0.206.

**Goal:** Determine whether temporal multi-window masking is better than static single-window masking for subject trait prediction. Then sweep hyperparams for the winners.

**Optimization targets (in priority order):**
1. Maximize `probe_acc` (subject trait classification)
2. Minimize `pred_loss` (JEPA prediction quality)
3. Minimize `cosim` (collapse)

## GPU Utilization Strategy

Delta charges 1 SU = 1 A40-hour (16 cores, 62.5GB mem) regardless of actual usage. Small configs use ~2GB VRAM and finish in ~45 min. **Run multiple experiments in parallel on the same GPU** using background processes (`&` + `wait`).

| Config size | VRAM est. | Parallel jobs per A40 | Time alloc |
|-------------|-----------|----------------------|------------|
| Small (≤3K tokens) | ~2 GB | 4 (8 GB total) | 1h |
| Medium (3-7K tokens) | ~6-12 GB | 3-4 (24-48 GB) | 1.5h |
| Large (>7K tokens) | ~12-24 GB | 2 (24-48 GB) | 2h |

Each SLURM job launches N training processes as background jobs, then `wait` for all to complete. With `num_workers=2` per job (down from 4) to stay within 16 CPU cores.

This reduces wall time ~4x AND SU cost ~4x vs one-job-per-allocation.

## Phase 1: Pretraining Objective Sweep

**11 configs × 3 seeds = 33 experiments, packed into ~9 SLURM jobs**

| n_windows | window_size | P (patches) | tokens | bs | size class |
|-----------|-------------|-------------|--------|----|------------|
| 1 | 1s | 6 | 774 | 64 | small |
| 1 | 2s | 12 | 1,548 | 64 | small |
| 1 | 4s | 26 | 3,354 | 64 | medium |
| 2 | 1s | 12 | 1,548 | 64 | small |
| 2 | 2s | 12 | 3,096 | 64 | medium |
| 2 | 4s | 26 | 6,708 | 64 | medium |
| 4 | 1s | 6 | 3,096 | 64 | medium |
| 4 | 2s | 12 | 6,192 | 64 | medium |
| 4 | 4s | 26 | 13,416 | 32 | large |
| 8 | 1s | 6 | 6,192 | 64 | medium |
| 8 | 2s | 12 | 12,384 | 32 | large |

Fixed: `depth=2, embed_dim=64, lr=5e-4, VCLoss(0.25,0.25), smooth_l1, 100ep`. Seeds: [2025, 42, 7].

**Packing plan:** Group by size class, run ~4 experiments **in parallel** per SLURM job (background `&` + `wait`). Each SLURM job requests 1-2h on 1 A40 with `num_workers=2`. Total: ~9 SLURM jobs running ~1h each (vs 33 × 1h without packing).

**Decision criteria for Phase 2:** Rank by mean `probe_acc` across seeds. Require `cosim < 0.90` and `pred_loss` not diverged. Advance top 2-3 configs.

## Phase 2: Hyperparameter Sweep for Winners

For top 2-3 Phase 1 configs, sweep:
- `loss.std_coeff = loss.cov_coeff` ∈ [0.1, 0.25, 0.5, 1.0] — 4 values
- `optim.lr` ∈ [3e-4, 5e-4, 1e-3] — 3 values
- `model.encoder_depth` ∈ [2, 3] — 2 values

= 24 combos × 3 seeds = 72 jobs per winner. Packed similarly (~18 SLURM jobs per winner).

## Implementation Steps

### Step 1: Add Delta preprocessed data path

**File:** `experiments/eeg_jepa/main.py` line 62

Add to `_PREPROCESSED_DIRS`:
```python
Path("/projects/bbnv/kkokate/hbn_preprocessed"),  # Delta
```

### Step 2: Create `scripts/sweep_phase1_delta.py`

Structure:
- Define all 33 `(n_windows, window_size, batch_size, seed)` configs
- Group into SLURM jobs by size class (small: chain 4-5, medium: chain 3-4, large: chain 2)
- Each SLURM job launches experiments as parallel background processes (`&`) then `wait`
- Uses `--data.num_workers=2` (down from 4) to fit 16 CPU cores across parallel jobs
- Uses `neurolab.jobs.Job` per SLURM job with `partition="gpuA40x4"`, `time_limit="04:00:00"`, `mem_gb=64`, `gpus=1`
- Each experiment sets `--logging.wandb_group=sweep_phase1`
- Dry-run mode prints all generated scripts; submit mode sends them

### Step 3: Create `scripts/sweep_phase2_delta.py` (template)

Same packing strategy, parameterized by Phase 1 winners. Filled in after analysis.

### Step 4: Verify Delta environment (single test job)

Before full sweep, submit one job to verify: git checkout, uv dependencies, preprocessed data path, W&B logging, A40 partition name.

## Files to Create/Modify

| File | Action |
|------|--------|
| `experiments/eeg_jepa/main.py:62` | Add Delta preprocessed path |
| `scripts/sweep_phase1_delta.py` | Create — Phase 1 sweep launcher |
| `scripts/sweep_phase2_delta.py` | Create — Phase 2 template |

## Verification

1. `uv run python scripts/sweep_phase1_delta.py` → preview all SLURM scripts (dry run)
2. **Timing test on Delta:** Submit a single baseline job (n_windows=4, ws=2s, the current config) with `num_workers=2` to verify it finishes within ~45 min on A40. If it's significantly slower than `num_workers=4`, adjust packing (fewer parallel jobs, longer allocation).
3. **Parallel test on Delta:** Submit one SLURM job with 4 small parallel experiments to verify GPU sharing works (no CUDA OOM, no contention issues).
4. Submit full Phase 1 → monitor W&B `sweep_phase1` group
5. Analyze results → select Phase 2 winners
