---
name: delta-deploy
description: Deploy, monitor, and debug SLURM jobs on NCSA Delta for this repo. Use when submitting training/eval jobs, running sweeps, diagnosing job failures, or adapting an existing sbatch to new hyperparameters. Triggers on mentions of "Delta", "sbatch", "SLURM job", "deploy to GPU", "run sweep on cluster".
type: runbook
---

# Delta deployment runbook

Concrete patterns for this repo on NCSA Delta. Read the sections in order; each one assumes the previous one is done.

## 1. Fixed facts about this environment

- **SSH alias**: `delta` → `dt-loginNN.delta.ncsa.illinois.edu`, user `kkokate`.
- **Repo path on Delta**: `/projects/bbnv/kkokate/eb_jepa_eeg`.
- **Preprocessed data**: `/projects/bbnv/kkokate/hbn_preprocessed` (`_PREPROCESSED_DIRS` in `main.py` auto-detects this).
- **Checkpoint output root**: `/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/` (per-sweep subdirs go under `/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/<sweep_name>/`).
- **Conda env name**: `eb_jepa` (Python 3.12).
- **Account / partition**: `bbnv-delta-gpu` / `gpuA40x4`. Default node type: A40 40GB.
- **Wandb auth**: `~/.wandb_key` holds the API key; sbatch sources `cat ~/.wandb_key` into `WANDB_API_KEY`. Wandb also reads `~/.netrc` as a fallback — do not add `WANDB_API_KEY` to shell rc files.
- **Python note**: Delta's compute nodes have an older Python than local. PEP 604 union syntax like `int | None` must be **quoted in type hints** (`"int | None"`) in files that run on Delta. See commit `7f73fde`.

## 2. Pre-submit checklist

Run these three in order before any `sbatch` call. They cover 90% of first-time failures.

### 2.1 Push the branch to origin
Delta pulls from origin. If you sbatch without pushing, you get stale code.
```bash
git push origin <your-branch>
```

### 2.2 Pre-checkout the branch on Delta
**Do not `git checkout` inside an sbatch script.** Parallel jobs race on `.git/index.lock`. Always pre-checkout on the login node:
```bash
ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && git fetch origin && git checkout <your-branch> && git pull --ff-only"
```
This is the lesson from commit `2eaf5c0` ("drop per-job git checkout in sweep sbatch").

### 2.3 Confirm preprocessed data + CorrCA filter presence
```bash
ssh delta "ls -la /projects/bbnv/kkokate/hbn_preprocessed | head && ls /projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"
```
If the CorrCA file is missing (or you want a different `n_components`), compute it first via `scripts/compute_corrca.sbatch` and wait for it to finish before submitting downstream jobs.

## 3. sbatch template

Every new sbatch in this repo should follow this shape. Copy `scripts/train_exp6_sweep.sbatch` as the template — it already embodies the lessons below.

```bash
#!/bin/bash
#SBATCH --job-name=<prefix>
#SBATCH --account=bbnv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/<prefix>_%j.out
#SBATCH --error=logs/<prefix>_%j.err

set -euo pipefail

export HBN_PREPROCESS_DIR="${HBN_PREPROCESS_DIR:-/projects/bbnv/kkokate/hbn_preprocessed}"
export UV_LINK_MODE=copy  # avoids hardlink errors across projects/bbnv

: "${SEED:?need SEED}"             # require every varying knob as an env var
# ... other knob checks ...

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate eb_jepa

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT="${WANDB_PROJECT:-eb_jepa}"

# NEVER git checkout here — see §2.2

# Output dir MUST include every varying knob or runs will collide
EXP_ID="<knob1>${KNOB1}_<knob2>${KNOB2}_seed${SEED}"
EXP_DIR="/projects/bbnv/kkokate/eb_jepa_eeg/checkpoints/<sweep_name>/${EXP_ID}"
mkdir -p "$EXP_DIR"

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py \
    --meta.seed="${SEED}" \
    --meta.model_folder="${EXP_DIR}" \
    --<knob1>="${KNOB1}" \
    --<knob2>="${KNOB2}" \
    --logging.wandb_group="<sweep_name>_${EXP_ID}"

CKPT="${EXP_DIR}/best.pth.tar"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: no best.pth.tar at $CKPT"
    exit 1
fi

PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/probe_eval.py \
    --checkpoint="$CKPT" \
    --splits=val,test \
    --norm_mode=per_recording \
    --corrca_filters=corrca_filters.npz \
    --wandb_group="<sweep_name>_eval_${EXP_ID}"
```

### Key invariants
- **One varying knob per env var**. Do not hard-code sweep values inside the sbatch.
- **`EXP_ID` must include every varying knob**, or `best.pth.tar` paths collide and later jobs overwrite earlier runs.
- **W&B group name must be unique per cell** so the aggregator can disambiguate.
- **`--time=04:00:00` is the A40 limit for this account**. Jobs that need longer should checkpoint + resubmit, not raise the cap.

## 4. Submit pattern

A submit script is a shell (or Python) file that loops over the grid and emits env-var'd `sbatch` calls. Keep it in `scripts/submit_<sweep_name>.sh`. Example:

```bash
#!/bin/bash
set -euo pipefail
PD_VALUES=(16 24)
SEEDS=(42 123 2025)
N=0
for PD in "${PD_VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    N=$((N+1))
    echo "[$N] pd=$PD seed=$SEED"
    PRED_DIM="$PD" SEED="$SEED" sbatch scripts/train_<sweep_name>.sbatch
    sleep 2   # stagger to avoid wandb/file-creation races at startup
  done
done
echo "Submitted $N jobs."
```

### Invariants
- `sleep 2` between submissions — empirically avoids wandb-init and file-creation races.
- Submit from the repo root **on Delta** (the sbatch uses `cd "$SLURM_SUBMIT_DIR"`).
- Print `N` at the end so you can cross-check against `squeue` counts.

## 5. Deploying: the exact command sequence

From your local machine, after the pre-submit checklist:

```bash
# (local) ensure Delta has the branch at the right commit
git push origin <your-branch>
ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && git fetch origin && git checkout <your-branch> && git pull --ff-only"

# (remote) submit
ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && mkdir -p logs && bash scripts/submit_<sweep_name>.sh"

# (local) confirm
ssh delta "squeue -u kkokate | head"
```

## 6. Monitoring

```bash
# Queue overview
ssh delta "squeue -u kkokate"

# Tail a specific job
ssh delta "tail -f /projects/bbnv/kkokate/eb_jepa_eeg/logs/<prefix>_<jobid>.out"

# Scan all recent logs for errors
ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && grep -l 'Error\|Traceback\|OOM' logs/<prefix>_*.err 2>/dev/null | head"

# Cell-level status (did it finish the eval stage?)
ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && grep -L 'Cell .* complete' logs/<prefix>_*.out 2>/dev/null"
```

## 7. Known failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `fatal: Unable to create '.git/index.lock'` in job logs | Concurrent jobs running `git checkout` | Pre-checkout on login node; do NOT checkout inside sbatch. See §2.2, commit `2eaf5c0`. |
| Only a few `best.pth.tar` files exist despite many jobs finishing | `EXP_ID` missing a varying knob → output dir collision, later jobs overwrite | Add the missing knob to `EXP_ID`. |
| `SyntaxError: PEP 604 union syntax` | Delta's Python older than local | Quote the union in type hints: `"int | None"`. Commit `7f73fde`. |
| Job dies at epoch 0 with `CUDA out of memory` | A40 has 40GB; large batch × windows × patches | Halve `--data.batch_size` or drop `--data.num_workers`. |
| `wandb: Network error` repeatedly | Login-node wandb cache conflict OR missing `WANDB_API_KEY` | Confirm `~/.wandb_key` exists on Delta. If network flakiness, set `WANDB_MODE=offline` and `wandb sync` later. |
| `corrca_filters.npz` not found | File not in repo root on Delta | Either compute it (`scripts/compute_corrca.sbatch`) or scp it from local: `scp corrca_filters.npz delta:/projects/bbnv/kkokate/eb_jepa_eeg/`. |
| `uv: hardlink failed, falling back to copy` warning | Default `UV_LINK_MODE=hardlink` fails across filesystems on Delta | Export `UV_LINK_MODE=copy` in sbatch (already in template). |
| Job TIMEOUT at `--time` boundary | Training + eval > 4 h | Reduce `optim.epochs`, increase `early_stopping_patience` decay, or split train/eval into separate sbatch. |
| Jobs stuck in PENDING forever | bbnv-delta-gpu queue is busy | `scontrol show job <jobid>` to see reason; nothing to fix — wait or reduce priority competition. |
| W&B "run already exists" in aggregator | Two cells share the same `wandb_group` | Include all varying knobs in the group name. |

## 8. Canceling

```bash
# By job id
ssh delta "scancel <jobid>"

# All your pending jobs (careful)
ssh delta "scancel -u kkokate --state=PENDING"

# All jobs from one sweep (by name)
ssh delta "scancel -u kkokate --name=<prefix>"
```

## 9. Aggregating results

After all jobs finish:

```bash
ssh delta "cd /projects/bbnv/kkokate/eb_jepa_eeg && \
  python scripts/aggregate_<sweep_name>.py --logs_dir logs --prefix <prefix> --out <sweep>_results.md"

# pull results locally
scp delta:/projects/bbnv/kkokate/eb_jepa_eeg/<sweep>_results.md .
scp delta:/projects/bbnv/kkokate/eb_jepa_eeg/<sweep>_results.json .
```

If you're writing a new aggregator, follow `scripts/aggregate_exp6_sweep.py`: it reads stdout files (not W&B API — more robust to partial failures), parses the `cfg:` header line that the sbatch prints, and greps out `probe_eval/test/*` metrics.

## 10. Adding a new knob to an existing sweep

The minimal steps when extending an existing sweep:

1. Add `NEW_KNOB` env-var check to the sbatch (`: "${NEW_KNOB:?need NEW_KNOB}"`).
2. Pass it through to `main.py` as `--<cfg.path>="${NEW_KNOB}"`.
3. Include it in `EXP_ID` so output dirs don't collide.
4. Include it in `wandb_group`.
5. Include it in the `cfg:` header `echo` line so the aggregator can parse it.
6. Extend the aggregator regex and column list to recognize the new knob.
7. Update the submit script's nested loops.

Miss any of steps 3–6 and results will be silently wrong (overwritten files, unparseable logs, or merged cells in the aggregator).

## 11. When debugging a failed cell

1. Read `logs/<prefix>_<jobid>.err` first — tracebacks land here.
2. Read `logs/<prefix>_<jobid>.out` for progress.
3. Check `squeue -j <jobid>` (if still queued) or `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode` for post-hoc status.
4. If only some cells failed and others succeeded, resubmit **only the failed ones** by passing their specific env vars to a single `sbatch` call — do not rerun the whole grid.
5. If the same error fires across many cells, it's almost certainly a code or config bug, not a Delta bug. Fix, push, re-checkout, resubmit.
