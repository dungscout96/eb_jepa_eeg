"""One-shot Delta job: split_modality PNGs for the two multi-movie JEPA runs
(LeJEPA + Laya, jul10-multi seed=2025).

Uses build_split_modality_npz.py (JEPA-flavored — raw encoder embeddings
zero-padded to V-JEPA-2 dim; no clip_head needed) then plot_split_modality.
Multi-movie config → the "Movie" column in the output figure is finally
informative (2 colors) rather than trivial.

Runs on gpuA40x4 — inference-only, no OOM concern. Both runs chained
sequentially in a single sbatch.

Usage:
    uv run --group eeg python experiments/jepa_pretraining/_submit_viz_multi.py [submit]
"""
import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"

RUNS = [
    {
        "dir":   "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-multi_lejepa_reve_seed2025",
        "label": "LeJEPA multi-movie (convex SIGReg λ=0.05)",
    },
    {
        "dir":   "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-multi_laya_seed2025",
        "label": "Laya multi-movie (additive SIGReg λ=0.02)",
    },
]


def stage(run: dict) -> str:
    d = run["dir"]
    label = run["label"]
    npz = f"{d}/split_modality.npz"
    png = f"{d}/split_modality.png"
    return (
        f"echo '=== {label}: build npz ===' && "
        "PYTHONPATH=. uv run --group eeg python "
        "experiments/jepa_pretraining/build_split_modality_npz.py "
        f"--config {d}/config.yaml "
        f"--checkpoint {d}/latest.pth.tar "
        f"--output {npz} "
        "--split val --n-recordings 30 --max-windows 1500 --encode-batch 64 && "
        f"echo '=== {label}: plot ===' && "
        "PYTHONPATH=. uv run --group eeg python -m "
        "eb_jepa.evaluation.visualization.plot_split_modality "
        f"--npz {npz} --output {png} "
        f"--title '{label}'"
    )


def build_job() -> Job:
    return Job(
        name="jepa_viz_multi_jul10",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:45:00",
        command=" && ".join(stage(r) for r in RUNS),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_MODE": "disabled",
            "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/kkokate/hbn_preprocessed",
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
