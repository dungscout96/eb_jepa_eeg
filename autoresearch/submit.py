"""Submit one autoresearch experiment to Delta.

Submits a single Delta job that pins to the current local HEAD commit, runs
the training script with the locked autoresearch config, and writes
stdout/stderr to a per-job run.log.

Usage
-----
    # commit + push your encoder_search.py edit first, then:
    python autoresearch/submit.py --label "deeper-transformer"
    python autoresearch/submit.py --label "deeper-transformer" --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Resolve neurolab from sibling repo (matches scripts/retrain_best_configs_perrec_delta.py).
NEUROLAB = Path.home() / "Documents/Research/scalable-infra-for-EEG-research/neurolab"
if NEUROLAB.exists() and str(NEUROLAB.parent) not in sys.path:
    sys.path.insert(0, str(NEUROLAB.parent))

from neurolab.jobs import Job  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DELTA_REPO = "/u/dtyoung/eb_jepa_eeg"
RUN_DIR_BASE = "/u/dtyoung/autoresearch_runs"  # remote run.log dump


def _short_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()


def _branch() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:40]


def build_job(label: str, commit: str | None = None, seed: int | None = None) -> Job:
    commit = commit or _short_hash()
    branch = _branch()
    slug = _slug(label)
    run_dir = f"{RUN_DIR_BASE}/{commit}_{slug}"

    seed_arg = f" --meta.seed={seed}" if seed is not None else ""

    cmd = (
        f"mkdir -p {run_dir} && cd {DELTA_REPO} &&"
        f" git fetch origin {branch} && git checkout {commit} &&"
        f" PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py"
        f"  --fname=experiments/eeg_jepa/cfgs/autoresearch.yaml{seed_arg}"
        f"  > {run_dir}/run.log 2>&1"
    )

    return Job(
        name=f"ar_{commit}_{slug}",
        cluster="delta",
        repo_path=DELTA_REPO,
        command=cmd,
        venv="__none__",
        branch="",
        partition="gpuA40x4",
        time_limit="01:00:00",
        mem_gb=64,
        gpus=1,
        env_vars={
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            "WANDB_PROJECT": "eb_jepa",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="short description of this experiment")
    p.add_argument("--commit", default=None, help="commit hash to pin (default: current HEAD)")
    p.add_argument("--seed", type=int, default=None, help="override meta.seed (default: from yaml)")
    p.add_argument("--dry-run", action="store_true", help="print sbatch script without submitting")
    args = p.parse_args()

    commit = args.commit or _short_hash()
    job = build_job(args.label, commit=commit, seed=args.seed)

    print("=" * 72)
    print(f"Job: {job.name}")
    print(f"Commit: {commit}  Branch: {_branch()}  Label: {args.label}")
    print("=" * 72)
    print(job.submit(dry_run=True))
    print()

    if args.dry_run:
        print("dry-run: not submitted")
        return

    job_id = job.submit()
    print(f"Submitted: {job.name}  job_id={job_id}")
    print(f"Remote run.log: /u/dtyoung/autoresearch_runs/{commit}_{_slug(args.label)}/run.log")
    print(f"Poll: ssh delta 'sacct -j {job_id} --format=JobID,State,Elapsed,ExitCode'")


if __name__ == "__main__":
    main()
