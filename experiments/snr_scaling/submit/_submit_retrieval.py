"""Submit Top-K retrieval evaluation over E0.3 cells on Delta.

Replaces three near-identical throwaway shell scripts (``retr.sh``,
``lowS_retr.sh``, ``xtask_retr.sh``) that lived only in a scratchpad and on the
cluster. Those encoded which checkpoints, which epoch and which config each
published number came from -- i.e. exactly the provenance the results depend on
-- and would have vanished with the scratchpad. Every retrieval number in
RESULTS.md 2.12/2.13 and RESULTS_cross_task.md was produced by one of the three
presets below.

The evaluator itself is ``eb_jepa/evaluation/clip_probe/retrieval.py``; this only
chooses checkpoints, config and split. It computes e->v and v->e at three pool
granularities (time / shot / scene) in one pass.

Presets:

  tp        RESULTS.md 2.12 -- the S curve on ThePresent, val AND test.
            Each cell uses its OWN config_probe.yaml (task=ThePresent).
  low-s     RESULTS.md 2.13 -- the low-S arm, 5 S x 3 draws, test only.
  cross     RESULTS_cross_task.md -- the same TP-trained checkpoints evaluated
            on DespicableMe, test only, via the SHARED config_probe_DM.yaml.
            Zero-shot: nothing is refit, so this is the strictest transfer test.
  dm-ref    The two DM-native reference cells, on DM. The upper anchor that makes
            the `cross` numbers interpretable -- without it, transfer can only be
            compared to a random baseline that is itself inflated by the
            modal-answer collapse (RESULTS_cross_task.md 3).

All presets evaluate **epoch 325**, the checkpoint chosen by smoothed selection
(see ``select_and_probe_e03.py``); raw-argmax selection on the 29-window val
diagnostic is too noisy to use (RESULTS.md 2.10).

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_retrieval.py tp
    uv run --group eeg python experiments/snr_scaling/submit/_submit_retrieval.py cross submit
"""
import argparse

from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling"
OUT_DIR = "experiments/snr_scaling/raw_results"   # jobs write measured JSONs straight here
EPOCH = 325
TOPKS = "1 5 10"
DM_CONFIG = f"{CKPT_ROOT}/config_probe_DM.yaml"

# The full-pool cell has no draw seed in its slug; every other S does.
def _slug(s: int, draw: int | None = 11) -> str:
    if s == 1863:
        return "e03_s1863_a101_nd"
    return f"e03_s{s}_a101_nd_d{draw}"


PRESETS = {
    "tp": dict(
        splits=["val", "test"],
        cells=[(_slug(s), None) for s in (400, 701, 1000, 1400, 1863)],
        config=None,                      # each cell's own config_probe.yaml
        prefix="e03_retr",
        random_from="e03_s1863_a101_nd",
    ),
    "low-s": dict(
        splits=["test"],
        cells=[(_slug(s, d), None) for s in (10, 20, 50, 100, 200)
               for d in (11, 22, 33)],
        config=None,
        prefix="e03_retr",
        random_from=None,                 # reuses the `tp` random baseline
    ),
    "cross": dict(
        splits=["test"],
        cells=[(_slug(s), None) for s in
               (10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863)],
        config=DM_CONFIG,
        prefix="xtask_retr_DM",
        random_from=DM_CONFIG,
    ),
    "dm-ref": dict(
        splits=["test"],
        cells=[("e03_s701_a85_dm", None), ("e03_s1841_a85_dm", None)],
        config=DM_CONFIG,
        prefix="xtask_retr_DM",
        random_from=None,                 # `cross` already wrote it
    ),
}


def build_command(preset: str) -> str:
    p = PRESETS[preset]
    steps = []
    for split in p["splits"]:
        # "xtask_retr_DM" + "test" -> xtask_retr_DMtest_<slug>.json, matching
        # the filenames the published numbers were read from.
        tag = f"{p['prefix']}{split}" if p["prefix"].endswith("DM") \
            else f"{p['prefix']}_{split}"
        for slug, _ in p["cells"]:
            cfg = p["config"] or f"{CKPT_ROOT}/{slug}/config_probe.yaml"
            out = f"{OUT_DIR}/{tag}_{slug}.json"
            steps.append(
                # Skip work already done so a re-run is idempotent -- these
                # sweeps get resubmitted after partial failures.
                f"([ -f {out} ] && echo 'SKIP {slug}') || "
                "PYTHONPATH=. uv run --group eeg python "
                "eb_jepa/evaluation/clip_probe/retrieval.py "
                f"--checkpoint {CKPT_ROOT}/{slug}/epoch_{EPOCH}.pth.tar "
                f"--config {cfg} --split {split} --device cuda "
                f"--topks {TOPKS} --output {out}"
            )
        if p["random_from"]:
            cfg = (p["random_from"] if p["random_from"].endswith(".yaml")
                   else f"{CKPT_ROOT}/{p['random_from']}/config_probe.yaml")
            out = f"{OUT_DIR}/{tag}_random.json"
            steps.append(
                f"([ -f {out} ] && echo 'SKIP random') || "
                "PYTHONPATH=. uv run --group eeg python "
                "eb_jepa/evaluation/clip_probe/retrieval.py "
                f"--random-baseline --config {cfg} --split {split} "
                f"--device cuda --topks {TOPKS} --output {out}"
            )
    return " && ".join(steps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("action", nargs="?", default="dry", choices=["dry", "submit"])
    ap.add_argument("--partition", default="gpuA40x4")
    ap.add_argument("--time-limit", default="02:00:00")
    args = ap.parse_args()

    p = PRESETS[args.preset]
    n = len(p["cells"]) * len(p["splits"]) + (len(p["splits"]) if p["random_from"] else 0)
    print(f"preset={args.preset}  splits={p['splits']}  cells={len(p['cells'])}  "
          f"-> {n} retrieval run(s) at epoch {EPOCH}")
    print(f"  config: {p['config'] or 'per-cell config_probe.yaml (ThePresent)'}\n")

    job = Job(
        name=f"retr_{args.preset}",
        cluster="delta",
        repo_path=REPO,
        partition=args.partition,
        time_limit=args.time_limit,
        command="mkdir -p experiments/snr_scaling/raw_results && " + build_command(args.preset),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_MODE": "disabled",
            # The unified root: R1-R6 symlinked, R7-R10 real. Retrieval only
            # reads the eval split, but resolve identically to the probe runs.
            "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/dtyoung/hbn_preprocessed",
        },
    )
    if args.action != "submit":
        print(job.command.replace(" && ", " &&\n\n"))
        print("\nDry run. Re-run with 'submit' to sbatch.")
        return
    print(f"submitted {job.name}: {job.submit()}")


if __name__ == "__main__":
    main()
