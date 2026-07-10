"""Submit the modality-gap + embedding-structure t-SNE plots as a short Delta job.

Uses the best jul7 from-scratch checkpoint per
``experiments/clip_pretraining/soft_target_clip/RESULTS_jul7.md`` §3.4:
TP-only soft τ=0.05 400 ep, seed=2026 — the current best from-scratch record.
Panel (a) = random-init reference; panel (b) = that checkpoint.

Pipeline (single sbatch job, sequential):
  1. ``plot_modality_gap.py`` — Fig-1-style two-panel modality-gap t-SNE.
     Extracts EEG + V-JEPA-2 shared-space embeddings across BOTH ThePresent
     and DespicableMe (task override; checkpoint was TP-only trained but the
     encoder is task-agnostic), saves them + metadata to ``.npz``.
  2. ``plot_embedding_structure.py`` — 2x2 recoloring of the saved .npz
     by modality / movie / scene id / shot id.

Override the checkpoint / config / labels via CLI if you want to compare two
trained checkpoints instead (e.g. CS-Aligner vs soft-target once CS-Aligner
lands).

Usage:
    uv run --group eeg python _submit.py [submit]
    uv run --group eeg python _submit.py --ckpt-b=<path> --config=<path> \\
        --tasks=ThePresent,DespicableMe [submit]
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
EXP_DIR = "experiments/clip_pretraining/cs_aligner"

# Best-from-scratch checkpoint from soft_target_clip RESULTS_jul7.md §3.4.
BEST_EXP_DIR = (
    "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip/"
    "jul7-tp_soft_target_clip_seed2026"
)
BEST_CKPT = f"{BEST_EXP_DIR}/latest.pth.tar"
BEST_CONFIG = f"{BEST_EXP_DIR}/config_TP.yaml"
BEST_LABEL = "Soft-Target CLIP (jul7 best)"


def build_job(
    ckpt_a: str | None,
    label_a: str,
    ckpt_b: str | None,
    label_b: str,
    config: str,
    split: str,
    n_recordings: int,
    tasks: str,
    npz_name: str,
    gap_png: str,
    struct_png: str,
) -> Job:
    ckpt_a_arg = f" --checkpoint-a {ckpt_a}" if ckpt_a else ""
    ckpt_b_arg = f" --checkpoint-b {ckpt_b}" if ckpt_b else ""
    npz_path = f"{EXP_DIR}/{npz_name}"
    plot_gap = (
        "PYTHONPATH=. uv run --group eeg python"
        f" {EXP_DIR}/plot_modality_gap.py"
        f" --config {config}"
        f"{ckpt_a_arg}"
        f' --label-a "{label_a}"'
        f"{ckpt_b_arg}"
        f' --label-b "{label_b}"'
        f" --split {split}"
        f" --n-recordings {n_recordings}"
        f" --tasks {tasks}"
        f" --save-npz {npz_path}"
        f" --output {EXP_DIR}/{gap_png}"
    )
    plot_struct = (
        "PYTHONPATH=. uv run --group eeg python"
        f" {EXP_DIR}/plot_embedding_structure.py"
        f" --npz {npz_path}"
        f" --output {EXP_DIR}/{struct_png}"
        f' --title "{label_b} — embedding structure"'
    )
    split_png = struct_png.replace("embedding_structure", "split_modality")
    if split_png == struct_png:
        split_png = "split_modality.png"
    plot_split = (
        "PYTHONPATH=. uv run --group eeg python"
        f" {EXP_DIR}/plot_split_modality.py"
        f" --npz {npz_path}"
        f" --output {EXP_DIR}/{split_png}"
        f' --title "{label_b} — per-modality t-SNE"'
    )
    return Job(
        name="cs_aligner_modality_gap",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:30:00",
        command=f"{plot_gap} && {plot_struct} && {plot_split}",
        venv="__none__",
        branch="feature/cs-aligner",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
        },
    )


def parse_kv(args: list[str]) -> tuple[dict, bool]:
    kv, submit = {}, False
    for a in args:
        if a == "submit":
            submit = True
        elif a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            kv[k.replace("-", "_")] = v
        else:
            raise ValueError(f"bad arg {a!r}")
    return kv, submit


if __name__ == "__main__":
    kv, submit = parse_kv(sys.argv[1:])
    # Sentinel ``--ckpt-b=none`` (or empty) → random init for panel (b). Useful
    # to emit a random-init npz for split-modality baseline plots.
    ckpt_b_raw = kv.get("ckpt_b", BEST_CKPT)
    ckpt_b = None if ckpt_b_raw.lower() in ("none", "") else ckpt_b_raw
    job = build_job(
        ckpt_a=kv.get("ckpt_a"),
        label_a=kv.get("label_a", "Random init"),
        ckpt_b=ckpt_b,
        label_b=kv.get("label_b", BEST_LABEL),
        config=kv.get("config", BEST_CONFIG),
        split=kv.get("split", "val"),
        n_recordings=int(kv.get("n_recordings", 60)),
        tasks=kv.get("tasks", "ThePresent,DespicableMe"),
        npz_name=kv.get("npz_name", "panel_b.npz"),
        gap_png=kv.get("gap_png", "modality_gap.png"),
        struct_png=kv.get("struct_png", "embedding_structure.png"),
    )
    if submit:
        print(f"Submitting {job.name}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name}")
        print(job.submit(dry_run=True))
