"""Build the demo npz exports (test + val) for the best scene_clip checkpoint.

Two ways to run this, both producing the same artifacts:

1. Interactive node (usually allocates faster than a queued batch job)::

       srun --account=bbnv-delta-gpu --partition=gpuA40x4-interactive \\
            --nodes=1 --gpus-per-node=1 --tasks=1 \\
            --tasks-per-node=16 --cpus-per-task=1 --mem=40g --pty bash
       cd /u/dtyoung/eb_jepa_eeg
       python demo/_submit_export.py interactive     # prints the exact commands

2. Batch::

       uv run --group eeg python demo/_submit_export.py [submit]

Each split runs export_retrieval_npz.py and then validates the npz by feeding it
straight back through retrieval.py --from-npz, whose JSON must reproduce
RESULTS.md §3.10. Copy both npz + check json back to the laptop and run
demo/build_demo.py.
"""
import sys

from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"

# Best checkpoint per experiments/clip_pretraining/scene_clip_from_checkpoint/
# RESULTS.md §3.10 (Delta job 20400451). config.yaml sits next to it.
CKPT_DIR = ("/work/hdd/bbnv/dtyoung/eb_jepa/scene_clip_from_checkpoint/"
            "jul22_warmstart_lr3e4_ep299")
OUT_DIR = "/work/hdd/bbnv/dtyoung/eb_jepa/demo_export"

SPLITS = ["test", "val"]

# Every published retrieval number used 0.5 s time buckets; the CLI default is
# 0.1, which would give N_pool ~406 instead of 101 and match nothing.
T_BUCKET_S = 0.5

ENV = {
    "WANDB_MODE": "disabled",
    "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/kkokate/hbn_preprocessed",
    "MNE_DATA": "/u/dtyoung/mne_data",
}


def stage(split: str, runner: str) -> str:
    npz = f"{OUT_DIR}/demo_{split}.npz"
    check = f"{OUT_DIR}/demo_{split}_check.json"
    return (
        f"echo '=== export {split} ===' && "
        f"PYTHONPATH=. {runner} demo/export_retrieval_npz.py "
        f"--config {CKPT_DIR}/config.yaml "
        f"--checkpoint {CKPT_DIR}/latest.pth.tar "
        f"--split {split} --encode-batch 64 "
        f"--output {npz} && "
        f"echo '=== validate {split} against RESULTS.md 3.10 ===' && "
        f"PYTHONPATH=. {runner} eb_jepa/evaluation/clip_probe/retrieval.py "
        f"--from-npz {npz} --t-bucket-s {T_BUCKET_S} --topks 1 5 10 "
        f"--output {check}"
    )


def build_job() -> Job:
    runner = "uv run --group eeg python"
    return Job(
        name="demo_retrieval_export",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:40:00",
        command=f"mkdir -p {OUT_DIR} && "
                + " && ".join(stage(s, runner) for s in SPLITS),
        venv="__none__",
        branch="",
        env_vars=ENV,
    )


def print_interactive():
    print("# On an interactive gpuA40x4-interactive allocation, from", REPO)
    for k, v in ENV.items():
        print(f"export {k}={v}")
    print(f"mkdir -p {OUT_DIR}")
    for s in SPLITS:
        print(stage(s, "uv run --group eeg python").replace(" && ", " \\\n    && "))
        print()
    print("# Then, from the laptop:")
    for s in SPLITS:
        print(f"scp dtyoung@delta:{OUT_DIR}/demo_{s}.npz demo/data/")
        print(f"scp dtyoung@delta:{OUT_DIR}/demo_{s}_check.json demo/data/")


if __name__ == "__main__":
    if "interactive" in sys.argv:
        print_interactive()
    else:
        job = build_job()
        if "submit" in sys.argv:
            print(f"Submitting {job.name} ({len(SPLITS)} splits)")
            print(f"job_id: {job.submit()}")
        else:
            print(f"Dry-run {job.name}. Add 'submit' to actually run, "
                  f"or 'interactive' for copy-paste commands.")
            print(job.submit(dry_run=True))
