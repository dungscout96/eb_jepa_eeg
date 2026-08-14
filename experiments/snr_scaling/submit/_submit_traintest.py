"""Submit train->eval linear probes (Pearson r) over E0.3 cells on Delta.

Companion to ``_submit_retrieval.py``, and the same motivation: the ``e03_tt_*``
numbers in RESULTS.md 2.12/2.13 were produced by throwaway shell scripts that
lived only in a scratchpad and on the cluster, so the provenance of every
published Pearson r would have vanished with them. The ``tp`` and ``low-s``
presets below reconstruct those runs; ``cross`` and ``dm-ref`` are new.

The evaluator is ``eb_jepa/evaluation/clip_probe/probe_traintest.py``: it fits a
RidgeCV head on the TRAIN split and evaluates on a held-out split, unlike
``probe.py`` which cross-validates *within* the split it reports. Head-fitting
data is the full train pool for every cell, so only the encoder's cohort differs.

The ``tp``/``low-s`` flags were reconstructed from the artifacts' own recorded
metadata and match it exactly: bootstrap=2000, seed=42, all 12 features,
n_train_recordings=1863 (ThePresent extended train), n_test_recordings 293 (val)
/ 108 (test). Cross-task runs read the shared DespicableMe config, whose train
pool is 1841 recordings.

Presets:

  tp        RESULTS.md 2.12 -- the S curve on ThePresent, val AND test.
            Each cell uses its OWN config_probe.yaml (task=ThePresent).
  low-s     RESULTS.md 2.13 -- the low-S arm, test only, 1 draw per S
            (corroborating only; the 3-draw readouts are probe.py + retrieval).
  cross     RESULTS_cross_task.md -- TP-trained checkpoints evaluated on
            DespicableMe, val AND test, via the SHARED config_probe_DM.yaml.
            The ridge head IS refit on DM train, so this measures linear
            transfer of the features -- not the zero-shot alignment that
            `_submit_retrieval.py cross` measures.
  dm-ref    The two DM-native reference cells, on DM, val AND test. The upper
            anchor that turns a raw transfer r into a fraction of native.

All presets evaluate **epoch 325**, the checkpoint chosen by smoothed selection
(see ``select_and_probe_e03.py``).

Each run re-embeds the train split, so cost is ~12-15 min of GPU per run
regardless of split size. ``--chunks N`` splits the run list across N jobs;
every step is guarded by a skip-if-output-exists test, so a job that times out
can simply be resubmitted and will pick up where it stopped.

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_traintest.py cross
    uv run --group eeg python experiments/snr_scaling/submit/_submit_traintest.py cross submit --chunks 4
"""
import argparse
from pathlib import Path

from neurolab.jobs import Job

# Where `verify` reads artifacts from locally, after they are rsynced back.
RAW_DIR = Path(__file__).resolve().parents[1] / "raw_results"

REPO = "/u/dtyoung/eb_jepa_eeg"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e03_scaling"
OUT_DIR = "experiments/snr_scaling/raw_results"   # jobs write measured JSONs straight here
EPOCH = 325
BOOTSTRAP = 2000                                  # matches every existing e03_tt_* artifact
DM_CONFIG = f"{CKPT_ROOT}/config_probe_DM.yaml"

ENV = {
    "WANDB_MODE": "disabled",
    # The unified root: R1-R6 symlinked, R7-R10 real.
    "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/dtyoung/hbn_preprocessed",
    # REQUIRED, and silently so. The ridge head is fit on the TRAIN split, and
    # without this opt-in the loader falls back to the R1-R4 default -- 741
    # DespicableMe recordings instead of 1841, 703 ThePresent instead of 1863.
    # The job still completes and writes a plausible JSON; only
    # n_train_recordings gives it away. Every e03_tt_* artifact on disk records
    # the extended pool, so omitting this also breaks the comparison it feeds.
    "HBN_TRAIN_RELEASES": "R1,R2,R3,R4,R7,R8,R9,R10",
}

# What n_train_recordings must be, per task, if ENV took effect. Checked after
# the fact by the `verify` action.
#
# Both are MEASURED, not assumed. ThePresent=1863 is confirmed by all 17
# e03_tt_* artifacts. DespicableMe=1832 is confirmed by all 26 xtask_tt_DM*
# artifacts here and, independently, by 84 depth-22 e04 cross-task artifacts
# built by a different submitter -- so it is the pool, not a quirk of one run.
#
# It is NOT the 1841 that RESULTS_cross_task.md 1 quotes: that is the
# `max_subjects` cap requested for the DM-native training cells, and a cap that
# exceeds the pool is a silent no-op. The 9-recording gap is presumably a few
# DespicableMe recordings failing the annotation/duration checks that
# ThePresent's pass. Harmless for the comparison -- every cell fits its ridge
# head on the identical 1832 -- but worth pinning so a real regression to the
# 741-recording R1-R4 default cannot hide behind a fuzzy expectation.
EXPECTED_TRAIN_RECORDINGS = {"ThePresent": 1863, "DespicableMe": 1832}


# The full-pool cell has no draw seed in its slug; every other S does.
def _slug(s: int, draw: int | None = 11) -> str:
    if s == 1863:
        return "e03_s1863_a101_nd"
    return f"e03_s{s}_a101_nd_d{draw}"


PRESETS = {
    "tp": dict(
        splits=["val", "test"],
        cells=[_slug(s) for s in (400, 701, 1000, 1400, 1863)],
        config=None,                      # each cell's own config_probe.yaml
        prefix="e03_tt",
        random_from="e03_s1863_a101_nd",
    ),
    "low-s": dict(
        splits=["test"],
        cells=[_slug(s) for s in (10, 20, 50, 100, 200)],
        config=None,
        prefix="e03_tt",
        random_from=None,                 # reuses the `tp` random baseline
    ),
    "cross": dict(
        splits=["val", "test"],
        cells=[_slug(s) for s in
               (10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863)],
        config=DM_CONFIG,
        prefix="xtask_tt_DM",
        random_from=DM_CONFIG,
    ),
    "dm-ref": dict(
        splits=["val", "test"],
        cells=["e03_s701_a85_dm", "e03_s1841_a85_dm"],
        config=DM_CONFIG,
        prefix="xtask_tt_DM",
        random_from=None,                 # `cross` already writes it
    ),
}


def _tag(prefix: str, split: str) -> str:
    # "xtask_tt_DM" + "val" -> xtask_tt_DMval_<slug>.json; "e03_tt" + "val" ->
    # e03_tt_val_<slug>.json, matching the filenames already on disk.
    return f"{prefix}{split}" if prefix.endswith("DM") else f"{prefix}_{split}"


def build_steps(preset: str) -> list[str]:
    """One shell step per (split, checkpoint), each idempotent."""
    p = PRESETS[preset]
    steps = []
    for split in p["splits"]:
        tag = _tag(p["prefix"], split)
        for slug in p["cells"]:
            cfg = p["config"] or f"{CKPT_ROOT}/{slug}/config_probe.yaml"
            out = f"{OUT_DIR}/{tag}_{slug}.json"
            steps.append(
                f"([ -f {out} ] && echo 'SKIP {tag} {slug}') || "
                "PYTHONPATH=. uv run --group eeg python "
                "eb_jepa/evaluation/clip_probe/probe_traintest.py "
                f"--checkpoint {CKPT_ROOT}/{slug}/epoch_{EPOCH}.pth.tar "
                f"--config {cfg} --eval-split {split} --device cuda "
                f"--bootstrap {BOOTSTRAP} --output {out}"
            )
        if p["random_from"]:
            cfg = (p["random_from"] if p["random_from"].endswith(".yaml")
                   else f"{CKPT_ROOT}/{p['random_from']}/config_probe.yaml")
            out = f"{OUT_DIR}/{tag}_random.json"
            steps.append(
                f"([ -f {out} ] && echo 'SKIP {tag} random') || "
                "PYTHONPATH=. uv run --group eeg python "
                "eb_jepa/evaluation/clip_probe/probe_traintest.py "
                f"--random-baseline --config {cfg} --eval-split {split} "
                f"--device cuda --bootstrap {BOOTSTRAP} --output {out}"
            )
    return steps


def chunk(steps: list[str], n: int) -> list[list[str]]:
    """Contiguous split into at most n near-equal chunks (runs cost the same)."""
    n = max(1, min(n, len(steps)))
    size, extra = divmod(len(steps), n)
    out, i = [], 0
    for c in range(n):
        take = size + (1 if c < extra else 0)
        out.append(steps[i:i + take])
        i += take
    return out


def verify(preset: str, raw_dir) -> int:
    """Check every artifact this preset should have produced. Returns n bad.

    `sacct COMPLETED 0:0` proves nothing (neurolab pitfall 8), and neither does
    the file existing -- a run that lost HBN_TRAIN_RELEASES writes a perfectly
    well-formed JSON off the wrong training pool. n_train_recordings is the
    field that distinguishes them, so it is the field to check.
    """
    import json
    import re as _re

    p = PRESETS[preset]
    task = "DespicableMe" if p["config"] == DM_CONFIG else "ThePresent"
    want = EXPECTED_TRAIN_RECORDINGS[task]
    bad = 0
    for step in build_steps(preset):
        out = _re.search(r"--output (\S+)", step).group(1)
        path = raw_dir / out.split("/")[-1]
        if not path.exists():
            print(f"  MISSING  {path.name}")
            bad += 1
            continue
        d = json.loads(path.read_text())
        got = d["n_train_recordings"]
        if got != want:
            print(f"  WRONG POOL  {path.name}: n_train_recordings={got}, expected {want}")
            bad += 1
        elif len(d["features"]) != 12:
            print(f"  FEATURES  {path.name}: {len(d['features'])} of 12")
            bad += 1
    print(f"verify {preset} ({task}, expect n_train={want}): "
          f"{'all clean' if not bad else f'{bad} bad'}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("action", nargs="?", default="dry",
                    choices=["dry", "submit", "verify"])
    ap.add_argument("--partition", default="gpuA40x4")
    ap.add_argument("--time-limit", default="04:00:00")
    ap.add_argument("--chunks", type=int, default=1,
                    help="Split the runs across this many parallel jobs.")
    args = ap.parse_args()

    if args.action == "verify":
        raise SystemExit(1 if verify(args.preset, RAW_DIR) else 0)

    p = PRESETS[args.preset]
    steps = build_steps(args.preset)
    groups = chunk(steps, args.chunks)
    print(f"preset={args.preset}  splits={p['splits']}  cells={len(p['cells'])}  "
          f"-> {len(steps)} probe run(s) at epoch {EPOCH}, in {len(groups)} job(s)")
    print(f"  config: {p['config'] or 'per-cell config_probe.yaml (ThePresent)'}\n")

    for i, group in enumerate(groups):
        job = Job(
            name=f"tt_{args.preset}" + (f"_{i}" if len(groups) > 1 else ""),
            cluster="delta",
            repo_path=REPO,
            partition=args.partition,
            time_limit=args.time_limit,
            command=f"mkdir -p {OUT_DIR} && " + " && ".join(group),
            venv="__none__",
            branch="",
            env_vars=ENV,
        )
        if args.action != "submit":
            print(f"--- {job.name}  ({len(group)} runs) ---")
            print(job.command.replace(" && ", " &&\n\n"), "\n")
        else:
            print(f"submitted {job.name} ({len(group)} runs): {job.submit()}")

    if args.action != "submit":
        print("Dry run. Re-run with 'submit' to sbatch.")


if __name__ == "__main__":
    main()
