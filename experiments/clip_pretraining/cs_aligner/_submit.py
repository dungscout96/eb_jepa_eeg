"""Submit the modality-gap t-SNE plot as a short Delta job.

Uses the best jul7 from-scratch checkpoint per
``experiments/clip_pretraining/soft_target_clip/RESULTS_jul7.md`` §3.4:
TP-only soft τ=0.05 400 ep, seed=2026 — the current best from-scratch record.
Panel (a) = random-init reference; panel (b) = that checkpoint.

Override the checkpoint / labels via CLI if you want to compare two trained
checkpoints instead (e.g. CS-Aligner vs soft-target once CS-Aligner lands).

Usage:
    uv run --group eeg python _submit.py [submit]
    uv run --group eeg python _submit.py --ckpt-a=<path> --label-a=<str> \\
        --ckpt-b=<path> --label-b=<str> [submit]
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
    ckpt_b: str,
    label_b: str,
    config: str,
    split: str = "val",
    n_recordings: int = 30,
    output: str = "modality_gap.png",
) -> Job:
    ckpt_a_arg = f" --checkpoint-a {ckpt_a}" if ckpt_a else ""
    return Job(
        name="cs_aligner_modality_gap",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit="00:20:00",
        command=(
            "PYTHONPATH=. uv run --group eeg python"
            f" {EXP_DIR}/plot_modality_gap.py"
            f" --config {config}"
            f"{ckpt_a_arg}"
            f' --label-a "{label_a}"'
            f" --checkpoint-b {ckpt_b}"
            f' --label-b "{label_b}"'
            f" --split {split}"
            f" --n-recordings {n_recordings}"
            f" --output {EXP_DIR}/{output}"
        ),
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
    job = build_job(
        ckpt_a=kv.get("ckpt_a"),
        label_a=kv.get("label_a", "Random init"),
        ckpt_b=kv.get("ckpt_b", BEST_CKPT),
        label_b=kv.get("label_b", BEST_LABEL),
        config=kv.get("config", BEST_CONFIG),
        split=kv.get("split", "val"),
        n_recordings=int(kv.get("n_recordings", 30)),
        output=kv.get("output", "modality_gap.png"),
    )
    if submit:
        print(f"Submitting {job.name}")
        print(f"job_id: {job.submit()}")
    else:
        print(f"Dry-run {job.name}")
        print(job.submit(dry_run=True))
