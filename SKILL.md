# EEG JEPA Delta Sweep — Continuation Guide

## Current State (2026-04-04)

### Active Job
- **Delta timing test** (SLURM job `17249801`): Baseline config on A40 with `num_workers=2`
  - W&B run: `0bs5q328`, group `delta_test`
  - Check status: `ssh delta "squeue -u dtyoung"`
  - Check output: `ssh delta "tail -30 /u/dtyoung/logs/eeg_jepa_timing_test_17249801.out"`
  - Check errors: `ssh delta "tail -30 /u/dtyoung/logs/eeg_jepa_timing_test_17249801.err"`

### What to Do Next (in order)

#### Step 1: Check timing test result
```bash
ssh delta "squeue -u dtyoung"  # Is it still running?
ssh delta "head -3 /u/dtyoung/logs/eeg_jepa_timing_test_17249801.out"   # Start time
ssh delta "tail -5 /u/dtyoung/logs/eeg_jepa_timing_test_17249801.out"   # End time / current epoch
```
- If total runtime ≤ 50 min → parallel packing (4 jobs/GPU) works with 1-2h allocations
- If runtime 50-90 min → reduce to 3 parallel jobs, use 2h allocations
- If runtime > 90 min → reduce to 2 parallel, use 3h allocations. Update `PARALLEL` and `TIME_LIMITS` dicts in `scripts/sweep_phase1_delta.py`

#### Step 2: Submit full Phase 1 sweep
```bash
cd ~/Documents/Research/eb_jepa_eeg   # or wherever the repo is
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab python scripts/sweep_phase1_delta.py          # dry run
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab python scripts/sweep_phase1_delta.py --submit  # actual submit
```
This sends 12 SLURM jobs to Delta (33 experiments total, packed with parallel GPU execution).

#### Step 3: Monitor
```bash
ssh delta "squeue -u dtyoung"                    # All jobs
ssh delta "squeue -u dtyoung | wc -l"            # Count
ssh delta "sacct -u dtyoung --starttime=today -o jobid,jobname,state,elapsed,maxrss"  # Completed jobs
```
W&B dashboard: filter by group `sweep_phase1` in project `eb_jepa`

#### Step 4: After Phase 1 completes
- Pull W&B metrics for all `sweep_phase1` runs
- For each of 11 configs, compute mean `probe_acc` across 3 seeds
- Advance top 2-3 configs (probe_acc > 0.60, cosim < 0.90) to Phase 2
- Create `scripts/sweep_phase2_delta.py` (template exists in plan)

---

## Phase 1 Sweep Details

**11 configs × 3 seeds = 33 experiments** sweeping temporal vs static masking:

| n_windows | window_size | Description | Size class |
|-----------|-------------|-------------|------------|
| 1 | 1s, 2s, 4s | Static (no temporal context) | small/medium |
| 2 | 1s, 2s, 4s | Temporal (2 windows) | small/medium |
| 4 | 1s, 2s, 4s | Temporal (4 windows) — current default is 4×2s | medium/large |
| 8 | 1s, 2s | Temporal (8 windows) | medium/large |

Fixed hyperparams: `depth=2, embed_dim=64, lr=5e-4, VCLoss(0.25,0.25), smooth_l1, 100ep, bs=64` (bs=32 for large configs)

Seeds: [2025, 42, 7]

**Optimization targets:** maximize `probe_acc` (subject traits), minimize `pred_loss`, minimize `cosim`

---

## Phase 2 (after Phase 1 analysis)

For top 2-3 winning configs, sweep:
- VCLoss coeff: [0.1, 0.25, 0.5, 1.0]
- lr: [3e-4, 5e-4, 1e-3]
- encoder_depth: [2, 3]

= 24 combos × 3 seeds = 72 jobs per winner

---

## Best Config from Prior Sweep (33 jamming experiments)
```
encoder_depth=2, embed_dim=64, lr=5e-4, VCLoss(0.25,0.25), Huber, 100ep, bs=64
probe_acc=0.641, cosim=0.853, pred_loss=0.206
```

## Key Files
| File | Purpose |
|------|---------|
| `scripts/sweep_phase1_delta.py` | Phase 1 sweep launcher (12 SLURM jobs) |
| `scripts/submit_job_delta.py` | Single Delta job template |
| `scripts/sweep_phase2_delta.py` | Phase 2 (to create after Phase 1) |
| `experiments/eeg_jepa/main.py` | Training entry point |
| `experiments/eeg_jepa/cfgs/default.yaml` | Config defaults |
| `results.tsv` | Results from 33 jamming experiments |

## Delta Environment
- **Repo**: `/u/dtyoung/eb_jepa_eeg` (main branch, up to date)
- **Data**: `/projects/bbnv/kkokate/hbn_preprocessed` (env var: `HBN_PREPROCESS_DIR`)
- **Partition**: `gpuA40x4` (A40 48GB, 16 cores/SU)
- **uv**: `/u/dtyoung/.local/bin/uv`
- **W&B**: credentials in `/u/dtyoung/.netrc`

## neurolab Dependency
```bash
# Required to run sweep scripts locally:
PYTHONPATH=/path/to/scalable-infra-for-EEG-research/neurolab python scripts/sweep_phase1_delta.py
```

## Jamming Experiment Loop (legacy, for reference)
For quick single experiments on jamming (no SLURM):
1. Edit `scripts/submit_job_jamming.py` command string
2. Commit + push
3. Run: `PYTHONPATH=.../neurolab python scripts/submit_job_jamming.py`
4. Check: `ssh dung@100.113.196.11 "ps aux | grep main.py | grep -v grep"`
5. Logs: `ssh dung@100.113.196.11 "tail -20 /home/dung/Documents/eb_jepa_eeg/logs/eeg_jepa_sanity_checks.out"`
