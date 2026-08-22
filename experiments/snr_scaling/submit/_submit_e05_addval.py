"""Submit the e05_addval arm: e04's depth-22 subject-scaling recipe, retrained
with the val release (R5) folded into the training pool.

WHY. ``RESULTS_model_scaling.md``'s headline is that e04's depth-22 curve rises
monotonically with S from 10 through 1400 and then DROPS at the full-pool
S=1863 cell -- consistently across both probe tasks, both splits and all three
retrieval granularities. That file also names the reason the drop cannot yet be
believed: **S=1863 is the whole R1-R4+R7-R10 pool, so it has exactly one
possible draw**, while every other S is a mean over three. Its selected epoch is
a single noisy pick rather than an average, and RESULTS.md 2.10 documents a
prior case where selection variance manufactured an apparent scaling artifact
in this very experiment.

Folding R5 in takes the ThePresent train pool from 1863 recordings to 2156, and
that buys two distinct things:

  1. **S=1863 becomes drawable, with three replicates.** 1863 < 2156, so
     ``max_subjects=1863`` is a real subsample rather than a no-op. This is the
     direct test of the drop: if three independent draws at S=1863 land at the
     S=1400 level, the drop was selection/draw variance; if they reproduce it,
     it is a population-level effect.
  2. **A genuinely new max-S point at S=2156**, the whole R5-inclusive pool --
     the first observation past the previous ceiling of the data.

CELLS (7 training runs, ~80 min each on gpuA40x4):

  S=1400 x draws {11,22,33}  POOL-STITCH CALIBRATION. e04 already has three
                             draws at S=1400, drawn from the 1863 pool. These
                             draw the same S from the 2156 pool. Subject draws
                             nest WITHIN a pool (permute once per seed, take
                             the first S), but the permuted list is a different
                             list once R5 is in it -- so d11 here is NOT e04's
                             d11 cohort. If the two agree, R5 subjects are worth
                             what the rest are and the curves stitch honestly;
                             if they do not, that difference IS the finding and
                             every S=1863/2156 comparison below must be read
                             through it. RESULTS.md 2.11 measured exactly this
                             kind of pool effect once already (R7-R10 subjects
                             worth 8-15%% less than R1-R4 at matched count), so
                             this is not a hypothetical.
  S=1863 x draws {11,22,33}  The money cells. See (1) above.
  S=2156 x 1 draw            Whole pool; only one draw exists, same convention
                             as e04's S=1863 and e03's S=1863.

PROTOCOL, and the one thing R5-in-train costs. Everything is held to e04:
``meta.seed=2026``, the REVE warm start (see ENCODER_INIT_FROM -- both of these
are CLI overrides that e04's on-disk config does NOT record), 400 epochs,
``epoch_size=703`` so every cell runs exactly 4400 steps, ``max_anchors=101``,
``save_every=25``. The cost is that R5 can no
longer serve as a held-out selection signal -- ``val/clip_scene_auc`` is now
in-sample for these cells, so the smoothed-argmax selection e04 uses
(``src/select_e04.py``) is unavailable. These cells are therefore evaluated at
a FIXED epoch 375, which is where 4 of 4 high-S e04 cells' own selection landed
(S=1400 d11/d33 and S=1863 at 375; S=1400 d22 at 350), so the comparison is
matched to within one 25-epoch checkpoint interval. ``save_every=25`` keeps the
whole epoch grid on disk, so a different epoch can be evaluated later without
retraining. All reported numbers are on TEST (R6), untouched by either arm.

The in-loop val diagnostic is deliberately left ON as a POSITIVE CONTROL: if R5
really entered the training pool, ``val/clip_scene_auc`` on these cells must run
visibly above e04's. The `verify` action checks the harder evidence -- the
recording count the job actually loaded.

Usage:
    uv run --group eeg python experiments/snr_scaling/submit/_submit_e05_addval.py
    uv run --group eeg python experiments/snr_scaling/submit/_submit_e05_addval.py sync
    uv run --group eeg python experiments/snr_scaling/submit/_submit_e05_addval.py submit
    uv run --group eeg python experiments/snr_scaling/submit/_submit_e05_addval.py verify
"""
import argparse
import subprocess
from pathlib import Path

from neurolab.jobs import Job
from neurolab.jobs.submit import ssh_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # experiments/snr_scaling
REPO_LOCAL = ROOT.parents[1]            # repo root, for `sync`

REPO = "/u/dtyoung/eb_jepa_eeg"
CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e05_addval"
BASE_CONFIG = "experiments/snr_scaling/config/clip_pretrain_e05_addval.yaml"
# NOT {REPO}/logs. neurolab resolves Job.log_dir against the sbatch WorkDir,
# which on Delta is the login home, not repo_path -- verified against
# `scontrol show job` (StdOut=/u/dtyoung/logs/<name>_<jobid>.out).
LOG_DIR = "/u/dtyoung/logs"

# Held identical to e04_reve_scaling. VERIFIED AGAINST THE ACTUAL INVOCATION
# (`wandb/latest-run/files/wandb-metadata.json` -> "args"), not against the
# on-disk config.yaml.
#
# THIS DISTINCTION COST A FULL SWEEP. The first version of this file was
# reconstructed from e04's config.yaml, which says `seed: 2025` and
# `encoder_init_from: null`. Both are wrong for the run that actually happened:
# every e04 cell overrides them on the CLI. Because the overrides never touch
# the file, a byte-for-byte diff of the two arms' config.yaml came back clean
# while the arms differed in the single most important way -- e04 warm-starts
# from a pretrained REVE encoder and the first e05 attempt trained from
# scratch. The symptom was a uniform ~37 %% deficit at matched S and a
# completely FLAT S curve (0.185 at S=1400, 1863 and 2156 alike), because a
# from-scratch depth-22 encoder at 4400 steps is so under-trained that subject
# count stops mattering. In-loop `clip_scene_auc` plateaued at 0.72 against
# e04's 0.84 -- and that was WITH R5 in train, where it should have been
# inflated.
#
# If you change anything here, check it against the wandb args of a real e04
# cell, not against its config.yaml.
SEED = 2026
# Every e04 cell starts from this checkpoint (5/5 sampled cells; only the
# encoder weights are copied -- optimizer, LR schedule and step counter start
# fresh, see clip_pretrain.py). Read-only, owned by kkokate.
ENCODER_INIT_FROM = "/work/hdd/bbnv/kkokate/eb_jepa/reve_base_eet_init.pth.tar"

# --- the from-scratch arm (--from-scratch) ----------------------------------
# The same 31 cells, the same 2156 pool, everything identical -- MINUS the warm
# start. It exists because the warm start turned out to dominate the comparison
# it was confounded with: at S=1400 the two depths are indistinguishable from
# scratch (0.1918 depth-12 vs 0.1853 depth-22) while warm-starting is worth
# +0.111, roughly 15x that difference.
#
# The first e05 sweep accidentally measured this condition at S>=1400 and its
# artifacts survive under raw_results/_invalid_e05_fromscratch -- but its
# CHECKPOINTS were overwritten in place by the warm-start retrain, so the full
# axis has to be trained rather than re-evaluated.
#
# Separate checkpoint root and a distinct `e05fs_` slug prefix, so neither a
# checkpoint nor an artifact can collide with the warm-started arm.
FS_CKPT_ROOT = "/work/hdd/bbnv/dtyoung/eb_jepa/e05_fromscratch"
FROM_SCRATCH = False           # set by main() from --from-scratch
# Set by main() from --epochs, exactly like FROM_SCRATCH above, because THREE
# separate completion checks derive from it: the screen's "is this run
# finished" (EPOCHS // SAVE_EVERY - 1), verify's expected checkpoint count, and
# the idempotence test for `epoch_{EPOCHS - SAVE_EVERY}.pth.tar`. Leaving it a
# constant while --epochs varied would have made an 800-epoch submission test
# for epoch_375 -- which every 400-epoch cell already has -- and silently skip
# all 31 jobs as "already done".
DEFAULT_EPOCHS = 400       # argparse default; EPOCHS is the resolved value
EPOCHS = DEFAULT_EPOCHS
SAVE_EVERY = 25
ANCHORS = 101
EPOCH_SIZE = 703          # -> 11 steps/epoch -> 4400 steps, every cell
WANDB_GROUP = "e05_addval"

# The pool the S axis is drawn from, in RECORDINGS. e04's own axis labels S by
# recording count (its "S=1863" is a max_subjects cap that exceeds the pool's
# subject count and is therefore a no-op meaning "everything"); this arm keeps
# that convention so the two axes are read the same way.
#
# MEASURED, not assumed: preprocessed ThePresent file counts on the unified
# root are R1-R4 = 703 and R7-R10 = 1183 (1886 raw), of which 1863 survive
# reject_recording() -- the figure all 48 e04 artifacts record. R5 holds 296
# raw files and yields 293, the count every e03/e04 val-split artifact records.
# 1863 + 293 = 2156. The `verify` action checks this against what the job
# actually loaded rather than trusting the arithmetic.
#
# CONFIRMED at runtime 2026-08-14, job 21138023:
#   HBN_TRAIN_RELEASES override: train = ['R1','R2','R3','R4','R5','R7','R8','R9','R10']
#   Scaling subsample: 2156 -> 1400 subjects (2156 -> 1400 recordings)
#   E0.3 scaling cell: ... -> 1400 recordings, 703 items/epoch, 11 steps/epoch
# So the pool is 2156 on BOTH counts -- one recording per subject, unlike the
# R1-R4-only pool where 703 recordings came from 701 subjects. S therefore
# means the same thing in subjects and in recordings for this whole arm.
POOL_RECORDINGS = 2156
E04_POOL_RECORDINGS = 1863

DRAW_SEEDS = [11, 22, 33]

# THE FULL S AXIS, drawn from the 2156 pool. This arm is now the *official*
# subject-scaling curve for the paper (decision 2026-08-17), superseding e04's
# 1863-pool axis rather than merely spot-checking its top end.
#
# The first seven cells (S=1400/1863 x 3 draws, plus the whole-pool 2156 cell)
# were run 2026-08-14..17 to test e04's S=1863 drop; the rest extend the same
# arm down the axis so one curve, one pool, covers 10 -> 2156. Draws still nest
# within a seed (permute the 2156-subject list once, take the first S), so
# 10 subset 20 subset ... subset 1863 for a given draw and the curve measures
# ADDED subjects.
#
# S values match e04/e03 exactly so the arms remain comparable point-for-point
# where anyone wants to compare them; only the pool differs.
# TRAINED (7 cells). The top of the axis, where e04's drop was.
CELLS = (
    [(1400, d) for d in DRAW_SEEDS]     # pool-stitch calibration vs e04
    + [(1863, d) for d in DRAW_SEEDS]   # the 3-draw replicate e04 cannot have
    + [(POOL_RECORDINGS, None)]         # new max-S point; one draw exists
)

# NOT trained -- opt in with `--full-axis`. Extends this arm down the whole S
# axis so one curve, one pool, covers 10 -> 2156, rather than splicing e04's
# 1863-pool cells onto the low end.
#
# Only worth the ~31 GPU-h of training if the measured pool offset at the
# S=1400 and S=1863 OVERLAP points (both arms at fixed epoch 325) turns out to
# be material for the readout being published. It is not, on the probe, at
# epoch 375: +0.0041 within-task, i.e. under half of e04's own between-draw sd.
# Check that at 325 before spending it -- and note that a spliced axis with a
# stated calibration is established practice here (see _submit_e03.py's
# EXTENDED_CELLS, which stitched R1-R4 onto R1-R10 the same way).
LOW_S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000]
FULL_AXIS_CELLS = [(s, d) for s in LOW_S_AXIS for d in DRAW_SEEDS]

# Files this arm adds that the cluster checkout will not have. The Delta
# checkout is NOT a git checkout of this branch (see the experiment README), so
# anything newly referenced must be rsynced or the job's `&&` chain
# short-circuits and "completes" in seconds with exit 0 -- neurolab pitfall 8,
# case 1, which has already happened once in this experiment.
SYNC_PATHS = [BASE_CONFIG]

ENV = {
    "WANDB_PROJECT": "eb_jepa",
    # R1-R6 symlinked, R7-R10 real. R5 lives here, so no new preprocessing is
    # needed for this arm -- verified 2026-08-14.
    "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/dtyoung/hbn_preprocessed",
    # Belt and braces. The authoritative declaration is data.train_releases in
    # the config (which takes precedence, see hbn.py::_resolve_releases); this
    # keeps any code path that reads only the env var on the same pool. The two
    # MUST agree -- if you edit one, edit the other.
    "HBN_TRAIN_RELEASES": "R1,R2,R3,R4,R5,R7,R8,R9,R10",
}


def ckpt_root() -> str:
    return FS_CKPT_ROOT if FROM_SCRATCH else CKPT_ROOT


def slug_prefix() -> str:
    return "e05fs" if FROM_SCRATCH else "e05"


def epoch_marker() -> str:
    """Slug segment separating budgets, so they cannot share a directory.

    A longer run writes epoch_25..epoch_375 too, so without this an 800-epoch
    submission would OVERWRITE the 400-epoch checkpoints in place -- and those
    are what every published from-scratch number was measured from. The
    convention matches the existing 800-epoch arm, which keeps its `_ep800`
    cells beside the 400-epoch ones in one root.
    """
    return "" if EPOCHS == DEFAULT_EPOCHS else f"_ep{EPOCHS}"


def slug(subjects: int, draw: int | None) -> str:
    s = f"{slug_prefix()}_s{subjects}_a{ANCHORS}_av{epoch_marker()}"
    return s if draw is None else f"{s}_d{draw}"


# Cells whose seed-2026 run failed to converge and were re-run at another seed.
# NOT a rescue invented here: e04 hit the identical failure three times
# (e04_s1000_nd_d33, e04_s200_nd_d22, e04_s400_nd_d33 -- each kept as a
# `_FAILED_seed2026` directory with a full 15-checkpoint grid) and re-ran each
# at --meta.seed=7. The failure is unmistakable and is judged on the artifacts,
# never on a preference for the answer:
#
#   e05_s1400_av_d11 scored e2v@1 time = 0.017 / 0.017 / 0.018 at epochs
#   300 / 350 / 375 -- flat, and at the 0.010 random baseline -- while its two
#   siblings sat at 0.072-0.074 and its final training loss was 3.9 against
#   their 2.8. It is an optimisation failure, not an unlucky cohort: the
#   S=1863 d11 cell, whose cohort is a strict SUPERSET of this one, trained
#   fine (0.077).
# MECHANISM, measured rather than guessed. A failed cell is not diverging late
# -- it never leaves the collapsed solution. InfoNCE at batch_size=64 has
# ln(64) = 4.1589 as its chance value, and every failure observed on this arm
# sits at 4.02-4.16 FLAT from epoch 1 to 400. A healthy cell at the same S
# leaves immediately (S=400 d22: 4.21 -> 3.89 -> 3.47 -> 2.90 -> ... -> 1.21).
# So the failure is collapse at initialisation, and the seed decides whether the
# run escapes the basin.
#
# Three of 31 cells collapsed at seed 2026 (~10 %, matching e04's 3/28). Two
# escaped at seed 7. (400, 11) collapsed at BOTH 2026 and 7, so it gets one
# final attempt at 2025 -- and that is the cap. If it collapses a third time,
# S=400 is reported at n=2 with the failure disclosed, NOT retried until it
# works: past a small fixed budget, "try seeds until one trains" stops being a
# fix for a known instability and starts being selection on the outcome.
COLLAPSED_LOSS = 4.1589        # ln(64); batch_size=64
RERUN_SEED = {
    (1400, 11): 7,
    (701, 33): 7,
    (1000, 11): 7,
    (400, 11): 2025,           # third and final attempt; 2026 and 7 both collapsed
}


def build_job(subjects: int, draw: int | None, partition: str,
              time_limit: str, epochs: int) -> Job:
    name = slug(subjects, draw)
    # The reseed map was measured on the warm-started arm; a from-scratch run is
    # a different optimisation and must be screened on its own terms.
    seed = SEED if FROM_SCRATCH else RERUN_SEED.get((subjects, draw), SEED)
    exp_dir = f"{ckpt_root()}/{name}"

    # Patch a COPY inside the job so the on-disk frozen config is never mutated
    # by concurrent runs (e03's convention).
    patch = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"c.data.max_subjects = {subjects}; "
        f"c.data.subsample_seed = {draw if draw is not None else SEED}; "
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')"
    )
    # The probe/retrieval snapshot must see the FULL data, and must fit its
    # ridge head on the SAME pool as every e04 cell -- so it clears the scaling
    # knobs AND reverts train_releases to e04's [R1..R4, R7..R10]. Only the
    # encoder's pretraining cohort is allowed to differ between arms; if the
    # head-fit pool moved too, a delta could not be attributed to either.
    probe_patch = (
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        "c.data.max_subjects = None; "
        "c.data.max_anchors = None; "
        "c.data.epoch_size = None; "
        "c.data.train_releases = ['R1','R2','R3','R4','R7','R8','R9','R10']; "
        f"OmegaConf.save(c, '{exp_dir}/config_probe.yaml')"
    )
    return Job(
        name=name,
        cluster="delta",
        repo_path=REPO,
        partition=partition,
        time_limit=time_limit,
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp {BASE_CONFIG} {exp_dir}/config.yaml && "
            f"PYTHONPATH=. uv run --group eeg python -c \"{patch}\" && "
            f"PYTHONPATH=. uv run --group eeg python -c \"{probe_patch}\" && "
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={epochs}"
            f" --meta.seed={seed}"
            # Load-bearing, and invisible in the on-disk config -- see the
            # ENCODER_INIT_FROM comment. Deliberately ABSENT for the
            # --from-scratch arm, which is that arm's entire purpose.
            + ("" if FROM_SCRATCH
               else f" --meta.encoder_init_from={ENCODER_INIT_FROM}")
            + f" --folder={exp_dir}"
            + f" --logging.save_every={SAVE_EVERY}"
            + f" --logging.wandb_group={WANDB_GROUP}{'_fromscratch' if FROM_SCRATCH else ''}"
        ),
        venv="__none__",
        branch="",
        env_vars=ENV,
    )


def sync() -> None:
    """Copy this arm's new files to the Delta checkout. See SYNC_PATHS."""
    print(f"syncing {len(SYNC_PATHS)} path(s) to delta:{REPO}")
    for p in SYNC_PATHS:
        print(f"  {p}")
    tar = subprocess.run(
        ["tar", "czf", "-", *SYNC_PATHS], cwd=REPO_LOCAL,
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ssh", "delta", f"cd {REPO} && tar xzf -"],
        input=tar.stdout, check=True,
    )
    for p in SYNC_PATHS:
        got = ssh_run("delta", f"test -f {REPO}/{p} && echo OK || echo MISSING")
        print(f"  {'OK ' if 'OK' in got.stdout else 'MISSING'} {p}")


def screen() -> int:
    """Flag non-converged cells by final training loss vs same-S siblings.

    The cheapest and earliest detector available, and the one this arm needed
    three times. Why loss and why relative:

    - **Loss, not a score.** It needs no evaluation, so a failure is caught
      before ~40 min of probe GPU is spent on a dead checkpoint. A failed cell
      still writes a full 15-checkpoint grid and a well-formed artifact.
    - **Relative to the same S, not an absolute threshold.** Loss scales with
      cohort size here (S=10 sits near 0.70, S=1863 near 2.8), because more
      distinct subjects make the contrastive task harder. A fixed cutoff that
      catches a failure at S=1000 would miss the same failure at S=50.

    Measured separation on this arm: the three genuine failures sit at
    1.67-3.09x their S-group median, while the worst healthy cell is 1.19x. The
    1.30x threshold is comfortably inside that gap rather than tuned to it.

    The relative test is the SCREEN; the mechanism is collapse. A failed cell
    sits at ln(batch_size) = 4.1589, InfoNCE's chance value, flat for the whole
    run -- see COLLAPSED_LOSS. Cells within a few percent of that never learned
    anything, whatever their S.

    WHAT THIS DOES NOT CATCH. The instability is a spectrum and this screen only
    detects one end. `e04_s1863_a101_nd` -- the cell whose apparent drop started
    this whole experiment -- finished at loss 3.32 against its alternate-seed
    twin's 2.62, scoring ~18 %% low on the probe. That is 1.22x its comparison
    median: BELOW the 1.30x threshold, and overlapping the worst healthy cell
    here (1.19x). Lowering the threshold does not separate them; the
    distributions genuinely overlap. A clean screen result therefore means "no
    cell collapsed", NOT "every cell trained well". Only replication catches
    partial failure -- treat any n=1 point as provisional no matter how clean
    it looks.
    """
    import re as _re
    import statistics as _st

    # Only judge FINISHED runs. A run still in flight has a loss that is
    # mid-descent, and comparing it against completed siblings' final losses
    # flags it as collapsed when it is merely early -- this exact false
    # positive fired once on (400, 11) at 45 of ~75 minutes, reading 2.49
    # against siblings' 1.26/1.30. Completion is the full checkpoint grid.
    n_done = EPOCHS // SAVE_EVERY - 1
    out = ssh_run(
        "delta",
        f'for d in {ckpt_root()}/{slug_prefix()}_s*_a101_av*/; do '
        # An unmatched glob is passed through LITERALLY by bash, so a
        # missing or empty root would otherwise yield a bogus cell named
        # `e05fs_s*_a101_av*` and still exit 0.
        '[ -d "$d" ] || continue; '
        'c=$(basename $d); case "$c" in *FAILED*) continue;; esac; '
        f'k=$(ls "$d"/epoch_*.pth.tar 2>/dev/null | wc -l); '
        f'[ "$k" -lt {n_done} ] && {{ echo "$c INCOMPLETE $k"; continue; }}; '
        f'f=$(ls -t {LOG_DIR}/${{c}}_*.out 2>/dev/null | head -1); '
        'echo "$c $(grep -oE \'loss=[0-9.]+\' $f 2>/dev/null '
        '| tail -1 | tr -d \'loss=\')"; done').stdout

    groups: dict[int, list[tuple[str, float]]] = {}
    incomplete = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[1] == "INCOMPLETE":
            incomplete.append((parts[0], parts[2] if len(parts) > 2 else "?"))
            continue
        try:
            loss = float(parts[1])
        except ValueError:
            continue
        m = _re.search(r"_s(\d+)_", parts[0])
        if m:
            groups.setdefault(int(m.group(1)), []).append((parts[0], loss))

    flagged = []
    for S in sorted(groups):
        med = _st.median([v for _c, v in groups[S]])
        for cell, loss in sorted(groups[S]):
            ratio = loss / med if med else 1.0
            bad = ratio > 1.30
            if bad:
                flagged.append((cell, loss, ratio))
            print(f"  S={S:<5} {cell:<26} loss={loss:.4f}  "
                  f"{ratio:.2f}x median{'   *** NOT CONVERGED' if bad else ''}")
    print()
    if incomplete:
        print("still training -- NOT judged (a mid-descent loss looks collapsed "
              f"next to a finished sibling's; need {n_done} checkpoints):")
        for cell, k in incomplete:
            print(f"  {cell}  ({k}/{n_done} checkpoints)")
        print()
    if flagged:
        print(f"{len(flagged)} of {sum(len(v) for v in groups.values())} cells "
              "did not converge. Add each to RERUN_SEED, move its checkpoint "
              "dir aside as <cell>_FAILED_seed<seed>, move its raw_results "
              "artifacts aside, and resubmit:")
        for cell, _l, _r in flagged:
            print(f"  {cell}")
    elif not groups:
        # "all 0 cells converged" reads as success. It is not; it means the
        # root is empty or the glob matched nothing.
        print(f"NO CELLS FOUND under {ckpt_root()} -- nothing was screened.")
        return 1
    else:
        print(f"all {sum(len(v) for v in groups.values())} cells converged.")
    return len(flagged)


def verify() -> int:
    """Check what each cell ACTUALLY trained on. Returns n bad.

    `sacct COMPLETED 0:0` proves nothing here -- the batch script has no
    `set -e` and ends on an echo, so its exit status is always that echo's
    (neurolab pitfall 8). Two things are checked instead, both from artifacts:

      - the checkpoint grid exists (epoch_25..375), and
      - the WARM START took. e04 initialises its encoder from
        reve_base_eet_init.pth.tar and load_encoder_weights is strict=False,
        so a failed init leaves the encoder random and only logs about it.
        A from-scratch cell looks completely normal from the outside -- it was
        this that invalidated the first e05 sweep. And
      - the pool the loader reported. clip_pretrain.py logs
        "E0.3 scaling cell: max_subjects=S ... -> N recordings", and N is the
        only thing that distinguishes a cell that got R5 from one that
        silently fell back to e04's 1863. A run that lost the R5 declaration
        still writes perfectly well-formed checkpoints.
    """
    bad = 0
    for subjects, draw in CELLS:
        name = slug(subjects, draw)
        # A cell that has been re-run has one log per job id. Always read the
        # NEWEST, or a stale run's lines answer questions about the current
        # one -- which is exactly how a superseded from-scratch run could keep
        # looking fine here.
        log_file = ssh_run(
            "delta", f"ls -t {LOG_DIR}/{name}_*.out 2>/dev/null | head -1"
        ).stdout.strip() or f"{LOG_DIR}/{name}_NONE.out"
        n_ckpt = ssh_run(
            "delta", f"ls {ckpt_root()}/{name}/epoch_*.pth.tar 2>/dev/null | wc -l"
        ).stdout.strip()
        n_tmp = ssh_run(
            "delta", f"ls {ckpt_root()}/{name}/*.tmp 2>/dev/null | wc -l"
        ).stdout.strip()
        log = ssh_run(
            "delta",
            f"grep -h -A4 'E0.3 scaling cell' {log_file} 2>/dev/null | tr -s ' \\n' ' ' | tail -c 200",
        ).stdout.strip()
        # 15, not 16: the loop is 0-indexed to epoch 399 and saves when
        # epoch %% SAVE_EVERY == 0, so the grid is epoch_25..epoch_375 and
        # there is no epoch_400. `latest.pth.tar` holds the final state but
        # does not match the epoch_* glob. Confirmed against e04's own cells.
        want_ckpt = EPOCHS // SAVE_EVERY - 1
        # "Scaling subsample: N -> S subjects (M -> K recordings)" -- N is the
        # POOL's unique-subject count, the cleanest evidence that R5 landed in
        # it. Absent when max_subjects >= N (a no-op cap), which is exactly the
        # S=2156 whole-pool cell; there the "E0.3 scaling cell" line's own
        # recording count is the pool size instead.
        pool = ssh_run(
            "delta",
            f"grep -h -A1 'Scaling subsample:' {log_file} 2>/dev/null | head -2 | tr -s ' \\n' ' '",
        ).stdout.strip()
        # The warm start. load_encoder_weights uses strict=False, so a name or
        # shape mismatch leaves most of the encoder RANDOM and logs it rather
        # than raising -- indistinguishable downstream from a cell that simply
        # learned badly. missing=0 is the only proof the init took.
        warm = ssh_run(
            "delta",
            f"grep -h -A4 'encoder tensors' {log_file} 2>/dev/null "
            "| tr -s ' \\n' ' ' | tail -c 200",
        ).stdout.strip()

        problems = []
        if n_ckpt != str(want_ckpt):
            problems.append(f"{n_ckpt}/{want_ckpt} ckpts")
        if n_tmp != "0":
            problems.append(f"{n_tmp} stray .tmp (died mid-write)")
        if not warm:
            problems.append("NO WARM START -- trained from scratch, not "
                            "comparable to e04")
        elif "missing=0" not in warm:
            problems.append(f"warm start incomplete: ...{warm[-70:]}")
        if not log:
            problems.append("no scaling-cell log line")
        elif subjects >= POOL_RECORDINGS:
            if f"-> {POOL_RECORDINGS} recordings" not in log:
                problems.append(f"pool != {POOL_RECORDINGS}: ...{log[-60:]}")
        elif not pool:
            problems.append("no subsample log line (cap was a no-op?)")
        else:
            # e04's 1863-recording pool holds ~1855 unique subjects, so
            # anything under 2000 means R5 never made it in and this cell
            # silently reproduced an e04 cell.
            n_pool_subj = int(pool.split("Scaling subsample:")[1].split("->")[0])
            if n_pool_subj < 2000:
                problems.append(
                    f"pool has only {n_pool_subj} subjects -- R5 missing, "
                    "this is an e04 cell")
        if problems:
            bad += 1
            print(f"  BAD   {name}: {'; '.join(problems)}")
        else:
            print(f"  ok    {name}: {n_ckpt} ckpts | {pool[-70:] or log[-70:]}")
    print(f"verify: {'all clean' if not bad else f'{bad} of {len(CELLS)} bad'}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="dry",
                    choices=["dry", "sync", "submit", "verify", "screen"])
    ap.add_argument("--partition", default="gpuA40x4")
    # e04's own cells took ~80 min end to end (14 min dataset scan + ~4.4 min
    # per 25 epochs, measured from the checkpoint mtimes). 3 h leaves room for
    # the larger pool's longer scan and a slower node.
    ap.add_argument("--time-limit", default="03:00:00")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help="Training length. A value other than the default puts the\n"
                         "cells in their own _ep<N> directories, so a longer run\n"
                         "cannot overwrite the checkpoints of a shorter one.")
    ap.add_argument("--only", default=None,
                    help="One cell, e.g. --only=1863_d11 or --only=2156.")
    ap.add_argument("--force", action="store_true",
                    help="Resubmit cells whose final checkpoint already "
                         "exists. Off by default: --full-axis includes the 7 "
                         "cells already trained and verified, and retraining "
                         "them would overwrite good checkpoints (and burn "
                         "~9 GPU-h) for nothing.")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Train the SAME 31 cells with NO warm start, into "
                         "FS_CKPT_ROOT under `e05fs_` slugs. Isolates what the "
                         "REVE initialisation buys across the whole subject "
                         "axis. Implies --full-axis.")
    ap.add_argument("--full-axis", action="store_true",
                    help="Also train LOW_S_AXIS (S=10..1000 x 3 draws, 24 "
                         "cells) so the whole 10->2156 curve comes from one "
                         "pool instead of splicing e04's low end onto it. "
                         "~31 GPU-h train + ~36 GPU-h eval, ~312 GB.")
    args = ap.parse_args()

    global FROM_SCRATCH, EPOCHS
    FROM_SCRATCH = args.from_scratch
    EPOCHS = args.epochs
    if FROM_SCRATCH:
        args.full_axis = True

    if args.action == "screen":
        raise SystemExit(1 if screen() else 0)
    if args.action == "verify":
        raise SystemExit(1 if verify() else 0)
    if args.action == "sync":
        sync()
        return

    cells = CELLS + (FULL_AXIS_CELLS if args.full_axis else [])
    if args.only:
        parts = args.only.split("_d")
        s_ = int(parts[0])
        d_ = int(parts[1]) if len(parts) > 1 else None
        cells = [(s_, d_)]

    # Idempotence. --full-axis names the whole arm, which includes the 7 cells
    # already trained and verified; resubmitting those would overwrite good
    # checkpoints and re-burn ~9 GPU-h. A cell counts as done if its last saved
    # checkpoint exists.
    if args.action == "submit" and not args.force:
        done = ssh_run(
            "delta",
            # `|| true`: on a first run the root does not exist, the glob
            # matches nothing, and the final `test -f` exits 1 -- which ssh_run
            # would raise on. A missing root simply means nothing is done yet.
            f"for d in {ckpt_root()}/*/; do "
            f"test -f \"$d/epoch_{EPOCHS - SAVE_EVERY}.pth.tar\" && "
            "basename \"$d\"; done 2>/dev/null || true").stdout.split()
        keep = [(s_, d_) for s_, d_ in cells if slug(s_, d_) not in done]
        if len(keep) != len(cells):
            print(f"skipping {len(cells) - len(keep)} cell(s) that already have "
                  f"epoch_{EPOCHS - SAVE_EVERY}.pth.tar (--force to override)\n")
        cells = keep
        if not cells:
            print("nothing to do.")
            return

    steps = args.epochs * -(-EPOCH_SIZE // 64)
    print(f"{len(cells)} cell(s), {args.epochs} ep = {steps} steps each on "
          f"{args.partition} [{args.time_limit}]")
    print(f"pool = {POOL_RECORDINGS} recordings "
          f"(e04's {E04_POOL_RECORDINGS} + R5's 293)\n")
    print(f"  {'cell':<28}{'S':>7}{'draw':>7}  note")
    for s_, d in cells:
        note = ("whole pool, one draw" if s_ >= POOL_RECORDINGS
                else "3-draw replicate of e04's single-draw cell"
                if s_ == E04_POOL_RECORDINGS else "pool-stitch calibration")
        print(f"  {slug(s_, d):<28}{s_:>7}{str(d or '-'):>7}  {note}")
    print()

    if args.action != "submit":
        s_, d = cells[0]
        job = build_job(s_, d, args.partition, args.time_limit, args.epochs)
        print(job.command.replace(" && ", " &&\n\n"))
        print(f"\nenv: {ENV}")
        print("\nDry run. Run 'sync' first (the cluster checkout does not have "
              f"{BASE_CONFIG}), then 'submit'.")
        return

    # Create the root up front. Each job's `mkdir -p <exp_dir>` would create it
    # as a side effect, but several read paths (screen, verify, the skip guard)
    # glob it before any job has started, and a missing root made two of them
    # misbehave -- one raised, one silently reported a literal glob as a cell.
    # The parent carries setgid (drwxrws--- delta_bbnv), so group ownership is
    # inherited; nothing is chmod-ed here.
    ssh_run("delta", f"mkdir -p {ckpt_root()} && ls -ld {ckpt_root()}")

    for s_, d in cells:
        job = build_job(s_, d, args.partition, args.time_limit, args.epochs)
        print(f"submitted {job.name}: {job.submit()}")


if __name__ == "__main__":
    main()
