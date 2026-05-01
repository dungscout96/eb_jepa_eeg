"""Optuna TPE sweep over linear-movie-probe hyperparameters on a single
encoder/probe seed. Reads val test corrs from probe_eval.py JSON output
and optimizes joint stim regression.

Usage on Delta:
  PYTHONPATH=. python scripts/optuna_linear_probe.py \\
      --ckpt /projects/.../phaseD_nw4ws2_baseline_s42/latest.pth.tar \\
      --probe_seed 42 --n_trials 30 \\
      --out_dir /projects/bbnv/kkokate/eb_jepa_eeg/predictions/optuna_linear
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import optuna


CONDA_PY = "/u/kkokate/.conda/envs/eb_jepa/bin/python"


def parse_metrics_from_log(log_text: str) -> dict[str, float]:
    """Extract `probe_eval/test/reg_<feature>_corr` lines from probe_eval stdout."""
    out = {}
    for line in log_text.splitlines():
        m = re.search(r"probe_eval/test/(reg_\w+_corr|cls_\w+_auc):\s*([+-]?[0-9.]+(?:[eE][+-]?[0-9]+)?)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def run_one_trial(
    ckpt: str, probe_seed: int, optimizer: str, lr: float, weight_decay: float,
    input_batchnorm: bool, out_dir: Path, trial_idx: int,
) -> dict[str, float]:
    pred_dir = out_dir / f"trial_{trial_idx:03d}_{optimizer}_lr{lr:.0e}_wd{weight_decay:.0e}_bn{int(input_batchnorm)}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        CONDA_PY, "experiments/eeg_jepa/probe_eval.py",
        f"--checkpoint={ckpt}",
        "--n_windows=4", "--window_size_seconds=2",
        "--batch_size=64", "--num_workers=2",
        "--splits=val,test",
        "--norm_mode=per_recording",
        "--corrca_filters=corrca_filters.npz",
        "--keep_channels",
        "--linear_movie_probe",
        f"--probe_method={optimizer}",
        f"--probe_lr={lr}",
        f"--probe_weight_decay={weight_decay}",
        f"--seed={probe_seed}",
        f"--save_predictions_dir={pred_dir}",
    ]
    if input_batchnorm:
        cmd.append("--input_batchnorm")
    proc = subprocess.run(
        cmd,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
             "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed"},
        capture_output=True, text=True, timeout=900,
    )
    log = proc.stdout + "\n" + proc.stderr
    (pred_dir / "trial.log").write_text(log)
    metrics = parse_metrics_from_log(log)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probe_seed", type=int, default=42)
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{args.out_dir}/optuna.db"
    study = optuna.create_study(
        study_name="linear_probe_sweep",
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        optimizer = trial.suggest_categorical("optimizer", ["adamw", "lbfgs"])
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        input_batchnorm = trial.suggest_categorical("batchnorm", [True, False])

        try:
            metrics = run_one_trial(
                args.ckpt, args.probe_seed, optimizer, lr, weight_decay,
                input_batchnorm, args.out_dir, trial.number,
            )
        except subprocess.TimeoutExpired:
            return -1.0

        # Joint objective: sum of test reg corrs across the 4 stim features
        keys = [
            "reg_position_in_movie_corr", "reg_luminance_mean_corr",
            "reg_contrast_rms_corr", "reg_narrative_event_score_corr",
        ]
        score = sum(metrics.get(k, -1.0) for k in keys)
        for k in keys:
            trial.set_user_attr(k, metrics.get(k))
        print(f"[trial {trial.number}] {optimizer} lr={lr:.2e} wd={weight_decay:.2e} bn={input_batchnorm} → {score:+.4f}")
        return score

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print("\n=== Best trial ===")
    print(f"score={study.best_value:+.4f}  params={study.best_params}")
    print(f"per-feature: {study.best_trial.user_attrs}")

    (args.out_dir / "best.json").write_text(json.dumps({
        "value": study.best_value, "params": study.best_params,
        "user_attrs": study.best_trial.user_attrs,
    }, indent=2))


if __name__ == "__main__":
    main()
