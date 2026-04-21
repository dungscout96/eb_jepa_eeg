"""Retrain the 7 best VICReg/SIGReg configs with --data.norm_mode=per_recording.

Companion to scripts/retrain_best_configs_delta.py, which retrains the same 7
configs with the default global norm. This variant uses per-recording norm
(matching the CorrCA training setup), to isolate the effect of input
normalization on subject- vs stimulus-encoding in the learned representation.

Global norm: z-score using a single train-set-wide mean/std per channel.
  → preserves subject-level amplitude/baseline differences in the input.
Per-recording norm: z-score each recording separately.
  → strips subject fingerprint at the input level; the encoder has to
    recover subject identity (if at all) from stimulus-locked dynamics.

Configs and hyperparameters are identical to retrain_best_configs_delta.py.

Usage
-----
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/retrain_best_configs_perrec_delta.py            # dry run
PYTHONPATH=~/Documents/Research/scalable-infra-for-EEG-research/neurolab \\
    python scripts/retrain_best_configs_perrec_delta.py --submit
"""

import sys

from neurolab.jobs import Job

BASE_ARGS = (
    " --model.encoder_depth=2"
    " --optim.lr=5e-4"
    " --optim.lr_min=1e-6"
    " --optim.warmup_epochs=5"
    " --optim.epochs=100"
    " --optim.early_stopping_patience=20"
    " --loss.pred_loss_type=smooth_l1"
    " --meta.seed=2025"
    " --data.num_workers=4"
    " --data.norm_mode=per_recording"   # <-- the only difference vs sibling script
    " --logging.wandb_group=retrain_best_es20_perrec"
)

# (nw, ws, bs, loss_args, label, time_limit) — same 7 configs as the global
# norm counterpart.
RUNS = [
    (1, 1, 64,
     " --loss.regularizer=sigreg --loss.sigreg.coeff=1.0",
     "sigreg1.0_nw1_ws1",    "01:30:00"),
    (2, 1, 64,
     " --loss.regularizer=sigreg --loss.sigreg.coeff=0.1",
     "sigreg0.1_nw2_ws1",    "02:00:00"),
    (4, 4, 32,
     " --loss.regularizer=sigreg --loss.sigreg.coeff=0.1",
     "sigreg0.1_nw4_ws4",    "10:00:00"),
    (2, 1, 64,
     " --loss.regularizer=vc --loss.std_coeff=1.0 --loss.cov_coeff=1.0",
     "vc1.0_proj_nw2_ws1",   "02:00:00"),
    (4, 4, 32,
     " --loss.regularizer=vc --loss.std_coeff=1.0 --loss.cov_coeff=1.0",
     "vc1.0_proj_nw4_ws4",   "10:00:00"),
    (4, 4, 32,
     " --loss.regularizer=vc --loss.std_coeff=0.1 --loss.cov_coeff=0.1",
     "vc0.1_proj_nw4_ws4",   "10:00:00"),
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
    git_cmd = (
        "sleep $((RANDOM % 60)) &&"
        " (git fetch origin && git checkout main && git pull --ff-only"
        " || (sleep $((RANDOM % 30 + 10)) && git fetch origin && git pull --ff-only)) &&\n"
    )

    jobs = []
    for nw, ws, bs, loss_args, label, tl in RUNS:
        cmd = _make_cmd(nw, ws, bs, loss_args)
        job = Job(
            name=f"rtpr_{label}",                   # "rtpr" = retrain per-rec
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

    print(f"Retraining {len(jobs)} configs with --data.norm_mode=per_recording "
          f"(matches CorrCA training setup).\n")
    for label, job, tl in jobs:
        print("=" * 72)
        print(f"Job: rtpr_{label}  [{tl}]")
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
