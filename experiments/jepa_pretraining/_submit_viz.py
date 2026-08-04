"""One-shot Delta job: build split_modality npzs + PNG/PDFs for both trained
JEPA runs (LeJEPA + Laya at jul10-h200-bs64 seed=2025).

Runs on gpuA40x4 — inference-only, batch=64, small model, no OOM concern.
Two runs handled sequentially in a single sbatch to save queue overhead.

Usage:
    uv run --group eeg python experiments/jepa_pretraining/_submit_viz.py [submit]
"""
import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
LEJEPA_DIR = "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-h200-bs64_lejepa_reve_seed2025"
LAYA_DIR   = "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-h200-bs64_laya_seed2025"


def build_job() -> Job:
    def stage(run_dir: str, label: str) -> str:
        npz  = f"{run_dir}/split_modality.npz"
        png  = f"{run_dir}/split_modality.png"
        return (
            f"echo '=== {label}: build npz ===' && "
            "PYTHONPATH=. uv run --group eeg python "
            "experiments/jepa_pretraining/build_split_modality_npz.py "
            f"--config {run_dir}/config.yaml "
            f"--checkpoint {run_dir}/latest.pth.tar "
            f"--output {npz} "
            "--split val --n-recordings 30 --max-windows 1500 --encode-batch 64 && "
            f"echo '=== {label}: plot ===' && "
            "PYTHONPATH=. uv run --group eeg python -m "
            "eb_jepa.evaluation.visualization.plot_split_modality "
            f"--npz {npz} --output {png} "
            f"--title '{label} — jul10 H200 bs=64 seed=2025'"
        )
    return Job(
        name="jepa_viz_jul10",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",   # Inference-only, no OOM concern here
        time_limit="00:45:00",
        command=stage(LEJEPA_DIR, "LeJEPA (convex λ=0.05)") + " && " +
                stage(LAYA_DIR,   "Laya (additive λ=0.02)"),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_MODE": "disabled",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
            "MNE_DATA": "/u/dtyoung/mne_data",
        },
    )


if __name__ == "__main__":
    job = build_job()
    if "submit" in sys.argv:
        print(f"Submitting {job.name}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name}. Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
