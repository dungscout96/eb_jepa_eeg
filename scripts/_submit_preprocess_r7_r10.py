"""Preprocess HBN releases R7-R10 (ThePresent + DespicableMe) on Delta.

Storage follows Delta's data-management guide
(https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html):

  /u         100 GB, snapshotted, "NOT intended as a source/destination for I/O
             during jobs" -> unusable, and only ~62 GB free anyway.
  /projects  500 GB, "shared data for a project, common data sets" -> the
             semantically correct home, but bbnv is at 494.7/500 GB AND
             371307/375000 files. Out on both counts; raising it is an
             allocation request that "may have a monetary fee".
  /work/hdd  "Area for computation, largest allocations, where I/O from jobs
             should occur." bbnv is allocated 48.83 TB with 1.5 TB used, and
             the guide lists it as NOT purged. -> destination.
  /tmp       purged after each job. Node-local only.

So output_dir and MNE_DATA both live under /work/hdd. The MNE_DATA move matters:
every other submit script in this repo points it at /u/dtyoung/mne_data, and
four releases of raw BIDS would both blow the 100 GB HOME quota and violate the
"not for job I/O" rule above.

R1-R6 are symlinked into the new root, so HBN_PREPROCESS_DIR remains a single
path and no loader code has to learn about multiple roots.

**One job per release, both tasks sequentially inside it.** ``load_or_download``
shares an MNE_DATA cache directory per dataset accession, so running the
ThePresent and DespicableMe jobs for the same release concurrently would race
two writers on one cache -- the same class of bug as concurrent ``git pull``
(neurolab skill pitfall 4). Different releases use different accessions and are
safe to run in parallel.

Release accessions verified against S3 dataset_description.json, including the
gap: R9 is ds005514 because **ds005513 does not exist**.

Usage:
    uv run --group eeg python scripts/_submit_preprocess_r7_r10.py          # dry run
    uv run --group eeg python scripts/_submit_preprocess_r7_r10.py submit
    uv run --group eeg python scripts/_submit_preprocess_r7_r10.py --only=R7 submit
"""
import argparse

from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
WORK = "/work/hdd/bbnv/dtyoung"
OUT_DIR = f"{WORK}/hbn_preprocessed"
CACHE_DIR = f"{WORK}/hbn_cache"      # raw BIDS downloads (the big one)
MNE_DATA = f"{WORK}/mne_data"

RELEASES = ["R7", "R8", "R9", "R10"]
TASKS = ["ThePresent", "DespicableMe"]

# Subject counts from each release's participants.tsv (pre-cached into OUT_DIR),
# used only to size the walltime request.
N_SUBJECTS = {"R7": 381, "R8": 257, "R9": 295, "R10": 533}


def build_job(release: str, tasks: list[str], partition: str,
              time_limit: str) -> Job:
    steps = [
        "PYTHONPATH=. uv run --group eeg python scripts/preprocess_hbn.py "
        f"release={release} task={t} output_dir={OUT_DIR}"
        for t in tasks
    ]
    return Job(
        name=f"prep_{release}",
        cluster="delta",
        repo_path=REPO,
        partition=partition,
        time_limit=time_limit,
        command=f"mkdir -p {OUT_DIR} {CACHE_DIR} {MNE_DATA} && " + " && ".join(steps),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_MODE": "disabled",
            # HBN_CACHE_DIR is the one that actually matters: load_or_download
            # passes cache_dir=DATA_DIR explicitly, and DATA_DIR is resolved from
            # HBN_CACHE_DIR, falling back to ~/.cache -- i.e. HOME, which is
            # capped at 100 GB and which the Delta guide says is "not intended
            # as a source/destination for I/O during jobs". Setting MNE_DATA
            # alone (as every other submit script in this repo does) would NOT
            # have redirected the raw downloads.
            "HBN_CACHE_DIR": CACHE_DIR,
            "MNE_DATA": MNE_DATA,
            "HBN_PREPROCESS_DIR": OUT_DIR,
            # data.eegdash.org was unreachable on 2026-08-02 (verified: failed
            # identically from Delta and from a laptop, so the service, not a
            # cluster firewall). "auto" would still try eegdash first and burn
            # its urllib3 retry ladder before falling back on every job, so pin
            # the S3 path. Drop this line once eegdash is healthy again.
            "HBN_DOWNLOAD_SOURCE": "s3",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # This account holds only bbnv-delta-gpu, so a gpu* partition is required
    # even though preprocessing is pure CPU (neurolab skill pitfall 1).
    p.add_argument("--partition", default="gpuA40x4")
    p.add_argument("--time-limit", default="10:00:00")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--only", default=None, help="One release, e.g. --only=R7")
    p.add_argument("action", nargs="?", default="dry", choices=["dry", "submit"])
    args = p.parse_args()

    releases = [args.only] if args.only else RELEASES
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"{len(releases)} job(s) x {len(tasks)} task(s) on {args.partition} "
          f"[{args.time_limit}]")
    print(f"  output_dir = {OUT_DIR}")
    print(f"  HBN_CACHE_DIR = {CACHE_DIR}   (raw BIDS)")
    print(f"  MNE_DATA   = {MNE_DATA}\n")
    for r in releases:
        print(f"  {r:>4}  {N_SUBJECTS.get(r, '?'):>4} subjects  ->  {', '.join(tasks)}")
    print()

    if args.action != "submit":
        print(build_job(releases[0], tasks, args.partition,
                        args.time_limit).command)
        print("\nDry run. Re-run with 'submit' to sbatch.")
        return

    for r in releases:
        job = build_job(r, tasks, args.partition, args.time_limit)
        print(f"submitted {job.name}: {job.submit()}")


if __name__ == "__main__":
    main()
