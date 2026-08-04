"""One-shot Delta job: build shared-space npzs + split_modality PNGs for the
four CLIP fine-tunes warm-started from jul10 JEPA checkpoints.

Two loss variants × two encoder shapes = 4 runs:
  soft_target_clip_from_jepa {lejepa, laya}   (α=0.5, τ=0.05)
  cs_aligner_from_jepa       {lejepa, laya}   (cs_weight=1.0, kernel=median)

Uses plot_modality_gap.py --save-npz to produce the CLIP-projected shared-
space embeddings (via clip_head.project_eeg / project_vision), then feeds
each npz to plot_split_modality.py for the per-modality figure.

Runs on gpuA40x4 — inference-only, no OOM concern. Chained sequentially in
a single sbatch to keep queue overhead low.

Usage:
    uv run --group eeg python experiments/clip_pretraining/_submit_viz_from_jepa.py [submit]
"""
import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"

RUNS = [
    {
        "dir":   "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip_from_jepa/jul10_soft_from_lejepa_seed2025",
        "label": "Soft-Target CLIP  ← LeJEPA (jul10)",
    },
    {
        "dir":   "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip_from_jepa/jul10_soft_from_laya_seed2025",
        "label": "Soft-Target CLIP  ← Laya (jul10)",
    },
    {
        "dir":   "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10_cs_from_lejepa_seed2025",
        "label": "CS-Aligner CLIP  ← LeJEPA (jul10)",
    },
    {
        "dir":   "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10_cs_from_laya_seed2025",
        "label": "CS-Aligner CLIP  ← Laya (jul10)",
    },
]


def stage(run: dict) -> str:
    d = run["dir"]
    label = run["label"]
    npz = f"{d}/split_modality.npz"
    modality_gap_png = f"{d}/modality_gap.png"
    split_png = f"{d}/split_modality.png"
    return (
        f"echo '=== {label}: modality_gap + save-npz ===' && "
        "PYTHONPATH=. uv run --group eeg python "
        "experiments/clip_pretraining/cs_aligner/plot_modality_gap.py "
        f"--config {d}/config.yaml "
        f"--checkpoint-b {d}/latest.pth.tar "
        f"--label-b '{label}' "
        "--split val --n-recordings 30 --max-windows-per-panel 1500 "
        f"--save-npz {npz} "
        f"--output {modality_gap_png} && "
        f"echo '=== {label}: split_modality ===' && "
        "PYTHONPATH=. uv run --group eeg python -m "
        "eb_jepa.evaluation.visualization.plot_split_modality "
        f"--npz {npz} --output {split_png} "
        f"--title '{label}'"
    )


def build_job() -> Job:
    return Job(
        name="clip_viz_jul10",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:45:00",
        command=" && ".join(stage(r) for r in RUNS),
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
        print(f"Submitting {job.name} ({len(RUNS)} runs)")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name} ({len(RUNS)} runs). Add 'submit' to actually run.")
        print(job.submit(dry_run=True))
