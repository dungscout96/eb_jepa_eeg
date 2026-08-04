"""Submit the E0.2 K-averaging curve on Delta.

Unlike the E0.1 ISC job this needs a GPU and does not fit the 1-hour
interactive cap: it embeds the full train split (~703 recordings) to fit the
ridge heads, then the eval split, then re-encodes signal-space averages for
every (K, draw). Budget 3 h on the batch partition.

Defaults to the **val** split (293 subjects), not test:
  - K can be swept much further (empirical R(K) needs 2K <= n_recordings, so
    val allows K up to 146 vs 54 on test),
  - the val rho1 from E0.1 is the trustworthy one; the test estimate is
    downward-biased by CorrCA fit-set size (RESULTS.md 2.5).

Usage:
    uv run --group eeg python experiments/snr_scaling/_submit_k_averaging.py         # dry run
    uv run --group eeg python experiments/snr_scaling/_submit_k_averaging.py submit
"""
import argparse
import sys

from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
OUT_DIR = "/work/hdd/bbnv/dtyoung/eb_jepa/snr_scaling"
RESULTS_DIR = "experiments/snr_scaling"

# jul7 best from-scratch: soft-target tau=0.05, TP-only, seed 2026
# (RESULTS_jul7.md 3.4 -- test mean r = 0.1517).
EXP = "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip/jul7-tp_soft_target_clip_seed2026"

COMMON_ENV = {
    "WANDB_MODE": "disabled",
    "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
    "MNE_DATA": "/u/dtyoung/mne_data",
}


def build_job(split: str, ks: str, n_draws: int, seed: int, partition: str,
              time_limit: str, max_train: int | None,
              skip_signal: bool) -> Job:
    out = f"{RESULTS_DIR}/k_averaging_{split}.json"
    cap = f"--max-train-recordings={max_train} " if max_train else ""
    skip = "--skip-signal-space " if skip_signal else ""
    cmd = (
        f"mkdir -p {OUT_DIR} && "
        "PYTHONPATH=. uv run --group eeg python "
        "experiments/snr_scaling/k_averaging.py "
        f"--checkpoint {EXP}/latest.pth.tar --config {EXP}/config_TP.yaml "
        f"--eval-split={split} --device=cuda "
        f"--ks {ks} --n-draws={n_draws} --seed={seed} {cap}{skip}"
        f"--rho1-json={RESULTS_DIR}/isc_{split}_ThePresent.json "
        f"--output={out} && "
        f"cp {out} {OUT_DIR}/"
    )
    return Job(
        name=f"kavg_{split}",
        cluster="delta",
        repo_path=REPO,
        partition=partition,
        time_limit=time_limit,
        command=cmd,
        venv="__none__",
        branch="",
        env_vars=COMMON_ENV,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--ks", default="1 2 4 8 16 32 64 128")
    p.add_argument("--n-draws", type=int, default=20)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--partition", default="gpuA40x4")
    p.add_argument("--time-limit", default="03:00:00")
    p.add_argument("--max-train-recordings", type=int, default=None)
    p.add_argument("--skip-signal-space", action="store_true")
    p.add_argument("action", nargs="?", default="dry", choices=["dry", "submit"])
    args = p.parse_args()

    job = build_job(args.split, args.ks, args.n_draws, args.seed,
                    args.partition, args.time_limit,
                    args.max_train_recordings, args.skip_signal_space)
    print(f"{job.name}  [{job.time_limit}]  partition={args.partition}\n")
    print(f"  {job.command}\n")
    if args.action != "submit":
        print("Dry run. Re-run with 'submit' to sbatch.")
        return
    print(f"submitted {job.name}: {job.submit()}")


if __name__ == "__main__":
    main()
