"""Submit train->eval linear probes (Pearson r) over e04_reve_scaling (depth-22)
cells on Delta.

Depth-22 counterpart of _submit_traintest.py, pointed at kkokate's
e04_reve_scaling checkpoints instead of e03_scaling. Two differences from the
e03 submitter, both deliberate:

  - **Per-cell epoch by default, not a fixed EPOCH.** e03's submitters
    hardcode epoch 325 for every cell. By default here each cell uses its OWN
    smoothed-selection epoch from ``experiments/snr_scaling/e04_selection.json``
    (produced by ``src/select_e04.py``), so both depth arms are early-stopped
    the same way without assuming they land on the same epoch. Pass
    ``--epoch 325`` to instead reproduce e03's fixed-epoch convention exactly
    -- needed for a depth-12 vs depth-22 delta that isn't confounded by the
    two arms using different selection protocols (see RESULTS_model_scaling.md's
    methodology caveat). Fixed-epoch outputs get an ``_ep<N>`` filename suffix
    so they never collide with the per-cell-selected ones.
  - **The DM config is a repo-relative path**, not a file living under
    kkokate's checkpoint root -- this submitter only reads from
    /work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling, never writes there.

The evaluator is ``eb_jepa/evaluation/clip_probe/probe_traintest.py``, same as
e03: fits a RidgeCV head on the TRAIN split, evaluates on a held-out split.

Presets:

  within    All 28 cells, val AND test, on ThePresent. Each cell uses its own
            config_probe.yaml (already sitting in its checkpoint dir).
  cross     The same 28 checkpoints evaluated on DespicableMe, val AND test,
            via the shared config_probe_DM_e04.yaml (depth-22 architecture).
            The ridge head IS refit on DM train -- this measures linear
            transfer of the features, not zero-shot alignment (that's
            _submit_retrieval_e04.py cross).

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_traintest_e04.py within
    uv run --group eeg python experiments/snr_scaling/submit/_submit_traintest_e04.py cross submit --chunks 6
    uv run --group eeg python experiments/snr_scaling/submit/_submit_traintest_e04.py within submit --epoch 325 --chunks 8
"""
import argparse
import json
from pathlib import Path

from neurolab.jobs import Job

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # experiments/snr_scaling
RAW_DIR = ROOT / "raw_results"  # where `verify` reads artifacts from locally

REPO = "/u/dtyoung/eb_jepa_eeg"
CKPT_ROOT = "/work/hdd/bbnv/kkokate/eb_jepa/e04_reve_scaling"
OUT_DIR = "experiments/snr_scaling/raw_results"
SELECTION = ROOT / "e04_selection.json"
DM_CONFIG = "experiments/snr_scaling/config/config_probe_DM_e04.yaml"
TP_CONFIG = "experiments/snr_scaling/config/config_probe_TP_e04.yaml"
BOOTSTRAP = 2000

# --- arms -------------------------------------------------------------------
# Both arms are the SAME depth-22 architecture evaluated by the same code
# against the same two configs, so they share every step-builder below; only
# the checkpoint root and the cell->epoch map differ.
#
#   e04      kkokate's e04_reve_scaling, 28 cells, per-cell smoothed selection
#            from e04_selection.json (or --epoch N to override).
#   addval   e05_addval -- the same recipe retrained with R5 folded into the
#            pretraining pool (see submit/_submit_e05_addval.py). R5 is no
#            longer held out for these cells, so smoothed val selection is
#            unavailable and they are pinned to a FIXED epoch (375, where 4 of
#            4 high-S e04 cells' own selection landed). Their slugs start with
#            `e05_`, so output filenames never collide with the e04 arm's.
#
# EXPECTED_TRAIN_RECORDINGS is deliberately NOT per-arm: the ridge head is fit
# on [R1..R4, R7..R10] for every cell in both arms, so 1863/1832 must hold for
# the addval cells too. If an addval artifact ever reports 2156, the probe
# read the pretraining config instead of the eval config and the comparison is
# void.
E05_CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e05_addval"
E05_DEFAULT_EPOCH = 375
E05_CELLS = (
    [f"e05_s1400_a101_av_d{d}" for d in (11, 22, 33)]
    + [f"e05_s1863_a101_av_d{d}" for d in (11, 22, 33)]
    + ["e05_s2156_a101_av"]
)

ENV = {
    "WANDB_MODE": "disabled",
    "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/dtyoung/hbn_preprocessed",
    # REQUIRED, and silently so -- same pitfall _submit_traintest.py hit for
    # e03: without this the "train" split (which the ridge head is fit on)
    # falls back to the R1-R4 default, 701/741 recordings instead of the
    # extended 1863/1841. The job still completes and writes a plausible
    # JSON; only n_train_recordings gives it away. See `verify` below and
    # eb_jepa/datasets/hbn.py's SPLIT_RELEASES comment.
    "HBN_TRAIN_RELEASES": "R1,R2,R3,R4,R7,R8,R9,R10",
}

# What n_train_recordings must be, per task, if ENV/train_releases took
# effect. Checked after the fact by the `verify` action.
#
# ThePresent=1863 matches e03's own convention exactly (48/48 e04 artifacts
# confirm it). DespicableMe is EMPIRICALLY 1832, not the 1841 _submit_traintest.py
# (e03) assumes -- measured consistently across all 58 e04 cross-task cells, so
# not noise. e03 has never actually produced an xtask_tt_DM* artifact to check
# its own assumption against (raw_results/ has none as of 2026-08-13), so 1841
# there is an unverified docstring estimate, not a measured value. The 9-cell
# gap is presumably a handful of DespicableMe recordings failing
# reject_recording()'s annotation/duration checks that ThePresent's don't --
# plausible and DM-specific, not a bug in this pipeline (same env var, same
# extended-release list, same code path as the ThePresent runs that DO land on
# 1863 exactly). Flagged upstream for e03's own constant to be reconciled.
EXPECTED_TRAIN_RECORDINGS = {"ThePresent": 1863, "DespicableMe": 1832}


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


def resolve_arm(arm: str, epoch: int | None) -> tuple[str, dict[str, int], str, str]:
    """-> (ckpt_root, {slug: epoch}, filename_suffix, human description)."""
    if arm == "e04":
        if epoch is not None:
            return (CKPT_ROOT, {s: epoch for s in load_selection()},
                    f"_ep{epoch}", f"fixed epoch {epoch}")
        return CKPT_ROOT, load_selection(), "", "per-cell selected epoch"
    # addval: no held-out val to select on, so a fixed epoch is the protocol,
    # not a fallback. Default 375; --epoch overrides. No filename suffix at the
    # default -- these slugs are unique to this arm, so nothing can collide,
    # and an unsuffixed name keeps the primary artifact obvious.
    ep = E05_DEFAULT_EPOCH if epoch is None else epoch
    return (E05_CKPT_ROOT, {s: ep for s in E05_CELLS},
            "" if ep == E05_DEFAULT_EPOCH else f"_ep{ep}",
            f"fixed epoch {ep} (no held-out val; see _submit_e05_addval.py)")


PRESETS = {
    "within": dict(
        splits=["val", "test"],
        # Repo-relative, not each cell's own config_probe.yaml -- every e04
        # cell's copy is functionally identical (verified 2026-08-13) and this
        # one declares train_releases explicitly. See config_probe_TP_e04.yaml.
        config=TP_CONFIG,
        prefix="e04_tt",
        random_from=TP_CONFIG,
    ),
    "cross": dict(
        splits=["val", "test"],
        config=DM_CONFIG,
        prefix="xtask_tt_DM",
        random_from=DM_CONFIG,
    ),
}


def _tag(prefix: str, split: str) -> str:
    return f"{prefix}{split}" if prefix.endswith("DM") else f"{prefix}_{split}"


# The addval arm reports TEST ONLY, on both presets. Its cells trained on R5,
# so a "val" number would be measured on data the encoder saw -- not a weaker
# result, a meaningless one. R6 is untouched by both arms and is the only split
# on which the two are comparable at all.
ARM_SPLITS = {"e04": None, "addval": ["test"]}


def build_steps(preset: str, epochs: dict[str, int], epoch_suffix: str = "",
                ckpt_root: str = CKPT_ROOT, arm: str = "e04") -> list[str]:
    p = PRESETS[preset]
    allowed = ARM_SPLITS[arm]
    splits = p["splits"] if allowed is None else [
        s for s in p["splits"] if s in allowed]
    steps = []
    for split in splits:
        tag = _tag(p["prefix"], split)
        for slug, epoch in epochs.items():
            out = f"{OUT_DIR}/{tag}_{slug}{epoch_suffix}.json"
            steps.append(
                f"([ -f {out} ] && echo 'SKIP {tag} {slug}') || "
                "PYTHONPATH=. uv run --group eeg python "
                "eb_jepa/evaluation/clip_probe/probe_traintest.py "
                f"--checkpoint {ckpt_root}/{slug}/epoch_{epoch}.pth.tar "
                f"--config {p['config']} --eval-split {split} --device cuda "
                f"--bootstrap {BOOTSTRAP} --output {out}"
            )
        # The random baseline loads no checkpoint and reads the same config, so
        # it is identical across arms AND epochs -- the addval arm reuses the
        # e04 artifact rather than recomputing a bit-identical one.
        if arm != "e04":
            continue
        rand_base = f"{tag}_random" if preset == "within" else f"{tag}_random_e04"
        out = f"{OUT_DIR}/{rand_base}.json"
        steps.append(
            f"([ -f {out} ] && echo 'SKIP {tag} random') || "
            "PYTHONPATH=. uv run --group eeg python "
            "eb_jepa/evaluation/clip_probe/probe_traintest.py "
            f"--random-baseline --config {p['random_from']} --eval-split {split} "
            f"--device cuda --bootstrap {BOOTSTRAP} --output {out}"
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


def verify(preset: str, epochs: dict[str, int], raw_dir: Path,
           epoch_suffix: str = "", ckpt_root: str = CKPT_ROOT,
           arm: str = "e04") -> int:
    """Check every artifact this preset should have produced. Returns n bad.

    `sacct COMPLETED 0:0` proves nothing (neurolab pitfall 8), and neither
    does the file existing -- a run that lost HBN_TRAIN_RELEASES writes a
    perfectly well-formed JSON off the wrong training pool.
    n_train_recordings is the field that distinguishes them.
    """
    import json
    import re as _re

    p = PRESETS[preset]
    task = "DespicableMe" if p["config"] == DM_CONFIG else "ThePresent"
    want = EXPECTED_TRAIN_RECORDINGS[task]
    bad = 0
    for step in build_steps(preset, epochs, epoch_suffix, ckpt_root, arm):
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
    ap.add_argument("--epoch", type=int, default=None,
                    help="Use this FIXED epoch for every cell instead of "
                         "per-cell selection (e.g. 325, matching e03's own "
                         "convention). Outputs get an _ep<N> filename suffix.")
    ap.add_argument("--arm", default="e04", choices=["e04", "addval"],
                    help="e04 = kkokate's e04_reve_scaling (default). "
                         "addval = the e05_addval cells, trained with R5 in "
                         "the pool; TEST SPLIT ONLY, fixed epoch "
                         f"{E05_DEFAULT_EPOCH}.")
    args = ap.parse_args()

    ckpt_root, epochs, epoch_suffix, epoch_desc = resolve_arm(args.arm, args.epoch)

    if args.action == "verify":
        raise SystemExit(1 if verify(args.preset, epochs, RAW_DIR, epoch_suffix,
                                     ckpt_root, args.arm) else 0)

    steps = build_steps(args.preset, epochs, epoch_suffix, ckpt_root, args.arm)
    groups = chunk(steps, args.chunks)
    p = PRESETS[args.preset]
    splits = ARM_SPLITS[args.arm] or p["splits"]
    print(f"arm={args.arm}  preset={args.preset}  splits={splits}  "
          f"cells={len(epochs)}  "
          f"-> {len(steps)} probe run(s), {epoch_desc}, in {len(groups)} job(s)")
    print(f"  config: {p['config'] or 'per-cell config_probe.yaml (ThePresent)'}\n")

    for i, group in enumerate(groups):
        job = Job(
            name=f"tt_{args.arm}_{args.preset}" + (f"_ep{args.epoch}" if args.epoch else "") + (f"_{i}" if len(groups) > 1 else ""),
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
