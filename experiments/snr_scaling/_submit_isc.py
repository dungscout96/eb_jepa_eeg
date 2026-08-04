"""Submit the E0.1 rho1 / ISC measurement jobs on Delta.

The work is pure numpy over memory-mapped FIFs and needs no GPU, but this
account only holds a ``bbnv-delta-gpu`` allocation -- there is no CPU
allocation, so Delta's ``cpu`` partition is not submittable even though
``sinfo`` lists it. We therefore run on ``gpuA40x4`` and, to avoid holding
several GPU nodes for CPU work, **pack every (split, task) into a single
sequential job** by default. Use ``--separate-jobs`` to fan out instead.

Memory is the binding constraint, not compute: the script holds a dense
[n_recordings, n_anchors, n_chans, n_times] float32 array, ~21 MB per
recording at 2 s windows. R6 test (108 recordings) is ~2.3 GB; R5 val (293)
is ~6.2 GB; train (~703) would be ~15 GB, so train is capped by default.
The cluster default of mem_gb=64 covers all of these.

Usage:
    uv run --group eeg python experiments/snr_scaling/_submit_isc.py            # dry run
    uv run --group eeg python experiments/snr_scaling/_submit_isc.py submit
    uv run --group eeg python experiments/snr_scaling/_submit_isc.py \
        --tasks=ThePresent,DespicableMe submit

NOTE: ``branch=""`` -- the job runs no git operations, so
``experiments/snr_scaling/`` must already exist in the remote checkout.
"""
import argparse
import sys

from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
OUT_DIR = "/work/hdd/bbnv/dtyoung/eb_jepa/snr_scaling"
RESULTS_DIR = "experiments/snr_scaling"

COMMON_ENV = {
    "WANDB_MODE": "disabled",
    "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
    "MNE_DATA": "/u/dtyoung/mne_data",
}

# split -> (time limit, --max-recordings). None means "use every recording".
SPLIT_SETTINGS = {
    "test": ("01:00:00", None),
    "val": ("02:00:00", None),
    "train": ("04:00:00", 300),
}


def _step(split: str, task: str, seed: int, n_components: int) -> str:
    """One measure_isc.py invocation, writing to the repo tree and OUT_DIR."""
    _, max_rec = SPLIT_SETTINGS[split]
    out = f"{RESULTS_DIR}/isc_{split}_{task}.json"
    cap = f"--max-recordings={max_rec} " if max_rec else ""
    return (
        "PYTHONPATH=. uv run --group eeg python "
        "experiments/snr_scaling/measure_isc.py "
        f"--split={split} --task={task} "
        f"--n-components={n_components} --seed={seed} {cap}"
        f"--output={out} && cp {out} {OUT_DIR}/"
    )


def _hhmmss(total_seconds: int) -> str:
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_job(pairs: list[tuple[str, str]], partition: str, seed: int,
              n_components: int, name: str) -> Job:
    """One Job running every (split, task) in ``pairs`` sequentially."""
    steps = [_step(s, t, seed, n_components) for s, t in pairs]
    # Sum the per-split budgets so a packed job gets the same headroom the
    # separate jobs would have had.
    budget = 0
    for s, _ in pairs:
        h, m, sec = (int(v) for v in SPLIT_SETTINGS[s][0].split(":"))
        budget += h * 3600 + m * 60 + sec
    cmd = f"mkdir -p {OUT_DIR} && " + " && ".join(steps)
    return Job(
        name=name,
        cluster="delta",
        repo_path=REPO,
        partition=partition,
        time_limit=_hhmmss(budget),
        command=cmd,
        venv="__none__",
        branch="",
        env_vars=COMMON_ENV,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", default="test,val")
    p.add_argument("--tasks", default="ThePresent")
    p.add_argument("--partition", default="gpuA40x4",
                   help="Delta's 'cpu' partition needs a CPU allocation this "
                        "account does not have; keep a gpu* partition.")
    p.add_argument("--separate-jobs", action="store_true",
                   help="One job per (split, task) instead of one packed job.")
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("action", nargs="?", default="dry",
                   choices=["dry", "submit"])
    args = p.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [s for s in splits if s not in SPLIT_SETTINGS]
    if bad:
        sys.exit(f"Unknown split(s): {bad}. Choose from {list(SPLIT_SETTINGS)}.")

    pairs = [(s, t) for s in splits for t in tasks]
    if args.separate_jobs:
        jobs = [
            build_job([pair], args.partition, args.seed, args.n_components,
                      name=f"isc_{pair[0]}_{pair[1]}")
            for pair in pairs
        ]
    else:
        jobs = [build_job(pairs, args.partition, args.seed, args.n_components,
                          name="isc_" + "_".join(s for s, _ in pairs))]

    print(f"{len(jobs)} job(s) on partition={args.partition} "
          f"covering {len(pairs)} (split, task) pair(s):\n")
    for j in jobs:
        print(f"  {j.name}  [{j.time_limit}]")
        print(f"    {j.command}\n")

    if args.action != "submit":
        print("Dry run. Re-run with 'submit' to sbatch.")
        return
    for j in jobs:
        job_id = j.submit()
        print(f"submitted {j.name}: {job_id}")


if __name__ == "__main__":
    main()
