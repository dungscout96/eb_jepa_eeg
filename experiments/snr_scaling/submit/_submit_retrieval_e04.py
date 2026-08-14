"""Submit Top-K retrieval evaluation (time/shot/scene, e->v and v->e in one
pass) over e04_reve_scaling (depth-22) cells on Delta.

Depth-22 counterpart of _submit_retrieval.py. See _submit_traintest_e04.py's
docstring for why this reads a per-cell epoch from
``experiments/snr_scaling/e04_selection.json`` instead of a fixed EPOCH, and
why the DM config is a repo-relative path rather than a file under kkokate's
checkpoint root.

The evaluator itself is ``eb_jepa/evaluation/clip_probe/retrieval.py``; this
only chooses checkpoints, config and split. It computes e->v and v->e at three
pool granularities (time / shot / scene) in one pass, for topk in {1, 5, 10}.

Presets:

  within    All 28 cells, val AND test, on ThePresent.
  cross     The same 28 checkpoints evaluated on DespicableMe, TEST ONLY
            (matching e03's cross retrieval preset), via config_probe_DM_e04.yaml.
            Zero-shot: nothing is refit, so this is the strictest transfer test.

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_retrieval_e04.py within
    uv run --group eeg python experiments/snr_scaling/submit/_submit_retrieval_e04.py cross submit --chunks 4
"""
import argparse
import json
from pathlib import Path

from neurolab.jobs import Job

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

REPO = "/u/dtyoung/eb_jepa_eeg"
CKPT_ROOT = "/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling"
OUT_DIR = "experiments/snr_scaling/raw_results"
SELECTION = ROOT / "e04_selection.json"
DM_CONFIG = "experiments/snr_scaling/config/config_probe_DM_e04.yaml"
TOPKS = "1 5 10"
RANDOM_BASELINE_CELL = "e04_s1863_a101_nd"


def load_selection() -> dict[str, int]:
    sel = json.loads(SELECTION.read_text())
    epochs, bad = {}, []
    for slug, info in sel.items():
        if "selected_epoch" in info:
            epochs[slug] = info["selected_epoch"]
        else:
            bad.append(slug)
    if bad:
        print(f"WARNING: {len(bad)} cell(s) had no selected epoch, skipped: {bad}")
    return epochs


PRESETS = {
    "within": dict(splits=["val", "test"], config=None, prefix="e04_retr"),
    "cross": dict(splits=["test"], config=DM_CONFIG, prefix="xtask_retr_DM"),
}


def build_steps(preset: str, epochs: dict[str, int]) -> list[str]:
    """One shell step per (split, checkpoint). Each step is independently
    idempotent (guarded by its own skip-if-exists test), so callers must NOT
    split these strings on " && " -- each step already contains that token
    internally as part of its skip guard.
    """
    p = PRESETS[preset]
    steps = []
    for split in p["splits"]:
        tag = f"{p['prefix']}{split}" if p["prefix"].endswith("DM") else f"{p['prefix']}_{split}"
        for slug, epoch in epochs.items():
            cfg = p["config"] or f"{CKPT_ROOT}/{slug}/config_probe.yaml"
            out = f"{OUT_DIR}/{tag}_{slug}.json"
            steps.append(
                f"([ -f {out} ] && echo 'SKIP {slug}') || "
                "PYTHONPATH=. uv run --group eeg python "
                "eb_jepa/evaluation/clip_probe/retrieval.py "
                f"--checkpoint {CKPT_ROOT}/{slug}/epoch_{epoch}.pth.tar "
                f"--config {cfg} --split {split} --device cuda "
                f"--topks {TOPKS} --output {out}"
            )
        if preset == "within":
            out = f"{OUT_DIR}/{tag}_random.json"
            cfg = f"{CKPT_ROOT}/{RANDOM_BASELINE_CELL}/config_probe.yaml"
        else:
            out = f"{OUT_DIR}/{tag}_random_e04.json"
            cfg = p["config"]
        steps.append(
            f"([ -f {out} ] && echo 'SKIP random') || "
            "PYTHONPATH=. uv run --group eeg python "
            "eb_jepa/evaluation/clip_probe/retrieval.py "
            f"--random-baseline --config {cfg} --split {split} "
            f"--device cuda --topks {TOPKS} --output {out}"
        )
    return steps


def chunk(steps: list[str], n: int) -> list[list[str]]:
    n = max(1, min(n, len(steps)))
    size, extra = divmod(len(steps), n)
    out, i = [], 0
    for c in range(n):
        take = size + (1 if c < extra else 0)
        out.append(steps[i:i + take])
        i += take
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("action", nargs="?", default="dry", choices=["dry", "submit"])
    ap.add_argument("--partition", default="gpuA40x4")
    ap.add_argument("--time-limit", default="02:00:00")
    ap.add_argument("--chunks", type=int, default=1)
    args = ap.parse_args()

    epochs = load_selection()
    p = PRESETS[args.preset]
    steps = build_steps(args.preset, epochs)
    groups = chunk(steps, args.chunks)
    print(f"preset={args.preset}  splits={p['splits']}  cells={len(epochs)}  "
          f"-> {len(steps)} retrieval run(s), per-cell selected epoch, in {len(groups)} job(s)")
    print(f"  config: {p['config'] or 'per-cell config_probe.yaml (ThePresent)'}\n")

    for i, group in enumerate(groups):
        job = Job(
            name=f"retr_e04_{args.preset}" + (f"_{i}" if len(groups) > 1 else ""),
            cluster="delta",
            repo_path=REPO,
            partition=args.partition,
            time_limit=args.time_limit,
            command=f"mkdir -p {OUT_DIR} && " + " && ".join(group),
            venv="__none__",
            branch="",
            env_vars={
                "WANDB_MODE": "disabled",
                "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/dtyoung/hbn_preprocessed",
            },
        )
        if args.action != "submit":
            print(f"--- {job.name} ---")
            print(job.command.replace(" && ", " &&\n\n"), "\n")
        else:
            print(f"submitted {job.name}: {job.submit()}")

    if args.action != "submit":
        print("Dry run. Re-run with 'submit' to sbatch.")


if __name__ == "__main__":
    main()
