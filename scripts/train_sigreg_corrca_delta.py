"""Train SIGReg + per-recording norm + CorrCA preprocessing.

Adds the missing cell from our config matrix: SIGReg with CorrCA spatial
filtering and per-recording normalization (the combo that CorrCA paired with
VICReg). All hyperparameters match retrain_best_configs_perrec_delta.py
except for --data.corrca_filters.

Configs (seed=2025, coeff mirrors our best SIGReg per temporal config):
  1. SIGReg nw1_ws1 coeff=1.0   — mirrors best sex-AUC SIGReg baseline
  2. SIGReg nw2_ws1 coeff=0.1   — mirrors best balanced SIGReg baseline
  3. SIGReg nw4_ws4 coeff=0.1   — mirrors overall-best SIGReg baseline

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/train_sigreg_corrca_delta.py --submit
"""

import sys

from neurolab.jobs import Job

CORRCA = "/projects/bbnv/kkokate/eb_jepa_eeg/corrca_filters.npz"

BASE_ARGS = (
    " --model.encoder_depth=2"
    " --optim.lr=5e-4"
    " --optim.lr_min=1e-6"
    " --optim.warmup_epochs=5"
    " --optim.epochs=100"
    " --optim.early_stopping_patience=20"
    " --loss.pred_loss_type=smooth_l1"
    " --loss.regularizer=sigreg"
    " --meta.seed=2025"
    " --data.num_workers=4"
    " --data.norm_mode=per_recording"
    f" --data.corrca_filters={CORRCA}"
    " --logging.wandb_group=sigreg_corrca_perrec"
)

RUNS = [
    # (nw, ws, bs, coeff, label, time_limit)
    (1, 1, 64, 1.0, "sigreg1.0_nw1_ws1_corrca", "01:30:00"),
    (2, 1, 64, 0.1, "sigreg0.1_nw2_ws1_corrca", "02:00:00"),
    (4, 4, 32, 0.1, "sigreg0.1_nw4_ws4_corrca", "10:00:00"),
]


def _make_cmd(nw, ws, bs, coeff):
    return (
        "PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py"
        f" --data.n_windows={nw}"
        f" --data.window_size_seconds={ws}"
        f" --data.batch_size={bs}"
        f" --loss.sigreg.coeff={coeff}"
        f"{BASE_ARGS}"
    )


def build_jobs():
    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )
    jobs = []
    for nw, ws, bs, coeff, label, tl in RUNS:
        job = Job(
            name=f"sgcc_{label}",  # sgcc = sigreg corrca
            cluster="delta",
            repo_path="/u/dtyoung/eb_jepa_eeg",
            command=git_cmd + _make_cmd(nw, ws, bs, coeff),
            venv="__none__",
            branch="",
            partition="gpuA40x4",
            time_limit=tl,
            mem_gb=64,
            gpus=1,
            env_vars={
                "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
                "WANDB_PROJECT": "eb_jepa",
            },
        )
        jobs.append((label, job, tl))
    return jobs


if __name__ == "__main__":
    submit = "--submit" in sys.argv
    jobs = build_jobs()
    print(f"Training {len(jobs)} SIGReg + per-rec + CorrCA configs.\n")
    for label, job, tl in jobs:
        print("=" * 72)
        print(f"Job: sgcc_{label}  [{tl}]")
        print("=" * 72)
        print(job.submit(dry_run=True))
        print()
    if submit:
        print("=" * 72)
        print("SUBMITTING")
        print("=" * 72)
        for label, job, _ in jobs:
            job_id = job.submit()
            print(f"  {job.name}: {job_id}")
    else:
        print(f"\n{len(jobs)} jobs ready. Re-run with --submit.")
