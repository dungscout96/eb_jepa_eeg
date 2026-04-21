"""Retrain the 7 documented best VICReg/SIGReg configs with early stopping.

Same per-run hyperparameters as the original Phase 1 sweeps — only change is
adding --optim.early_stopping_patience=20 (matching the CorrCA training
config in scripts/train_eeg_jepa.sbatch). This produces best.pth.tar files
at the val/reg_loss peak, comparable to CorrCA's best.pth.tar.

Configs (all seed=2025), drawn from docs/best_subject_trait_checkpoints.md:
  1. SIGReg nw1_ws1 coeff=1.0       — best overall sex AUC (0.620)
  2. SIGReg nw2_ws1 coeff=0.1       — best balanced SIGReg at nw2_ws1
  3. SIGReg nw4_ws4 coeff=0.1       — best overall subject-trait checkpoint
  4. VICReg+proj nw2_ws1 coeff=1.0  — best age-cls AUC among VICReg+proj
  5. VICReg+proj nw4_ws4 coeff=1.0  — best sex AUC for VICReg+proj
  6. VICReg+proj nw4_ws4 coeff=0.1  — best age-corr for VICReg+proj
  7. VICReg noproj  nw4_ws4 coeff=0.1 — best age-cls AUC overall

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/retrain_best_configs_delta.py            # dry run
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/retrain_best_configs_delta.py --submit
"""

import sys

from neurolab.jobs import Job

# Common training hyperparameters (match sweep_{sigreg,vicreg}_delta.py).
BASE_ARGS = (
    " --model.encoder_depth=2"
    " --optim.lr=5e-4"
    " --optim.lr_min=1e-6"
    " --optim.warmup_epochs=5"
    " --optim.epochs=100"
    " --optim.early_stopping_patience=20"  # <-- the key addition
    " --loss.pred_loss_type=smooth_l1"
    " --meta.seed=2025"
    " --data.num_workers=4"
    " --logging.wandb_group=retrain_best_es20"
)

# Each entry: (nw, ws, bs, loss_args, label, time_limit)
RUNS = [
    # SIGReg — override is --loss.sigreg.coeff (nested) and --loss.regularizer=sigreg
    (1, 1, 64,
     " --loss.regularizer=sigreg --loss.sigreg.coeff=1.0",
     "sigreg1.0_nw1_ws1",    "01:30:00"),
    (2, 1, 64,
     " --loss.regularizer=sigreg --loss.sigreg.coeff=0.1",
     "sigreg0.1_nw2_ws1",    "02:00:00"),
    (4, 4, 32,
     " --loss.regularizer=sigreg --loss.sigreg.coeff=0.1",
     "sigreg0.1_nw4_ws4",    "10:00:00"),
    # VICReg with trained projector (default use_projector=true)
    (2, 1, 64,
     " --loss.regularizer=vc --loss.std_coeff=1.0 --loss.cov_coeff=1.0",
     "vc1.0_proj_nw2_ws1",   "02:00:00"),
    (4, 4, 32,
     " --loss.regularizer=vc --loss.std_coeff=1.0 --loss.cov_coeff=1.0",
     "vc1.0_proj_nw4_ws4",   "10:00:00"),
    (4, 4, 32,
     " --loss.regularizer=vc --loss.std_coeff=0.1 --loss.cov_coeff=0.1",
     "vc0.1_proj_nw4_ws4",   "10:00:00"),
    # VICReg without projector
    (4, 4, 32,
     " --loss.regularizer=vc --loss.std_coeff=0.1 --loss.cov_coeff=0.1 --loss.use_projector=false",
     "vc0.1_noproj_nw4_ws4", "10:00:00"),
]


def _make_cmd(nw, ws, bs, loss_args):
    return (
        "PYTHONPATH=. uv run --group eeg python experiments/eeg_jepa/main.py"
        f" --data.n_windows={nw}"
        f" --data.window_size_seconds={ws}"
        f" --data.batch_size={bs}"
        f"{loss_args}{BASE_ARGS}"
    )


def build_jobs():
    """One job per run. Small configs (nw1/nw2) could be packed but keeping
    them separate makes it easy to resubmit a single run if something fails.
    """
    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )

    jobs = []
    for nw, ws, bs, loss_args, label, tl in RUNS:
        cmd = _make_cmd(nw, ws, bs, loss_args)
        job = Job(
            name=f"rt_{label}",
            cluster="delta",
            repo_path="/u/dtyoung/eb_jepa_eeg",
            command=git_cmd + cmd,
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

    print(f"Retraining {len(jobs)} best-config runs with "
          f"--optim.early_stopping_patience=20 (matches CorrCA).\n")
    for label, job, tl in jobs:
        print("=" * 72)
        print(f"Job: rt_{label}  [{tl}]")
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
        print(f"\n{len(jobs)} jobs ready. Re-run with --submit to send.")
