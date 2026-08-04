"""Submit N independent Delta jobs — one per (evaluator, checkpoint) — so
SLURM runs them in parallel across the gpuA40x4 pool.

Evaluators:
  probe.py             — per-window Ridge, 5-fold GroupKFold on val split.
                          Runs on all 10 checkpoints.
  probe_traintest.py   — SSL-standard linear eval: fit on train, eval on test.
                          Runs on all 10 checkpoints.
  retrieval.py         — Top-K retrieval; needs clip_head → 4 CLIP fine-tunes only.

Total: 10 + 10 + 4 = 24 jobs, each ~5-25 min. With A40 pool availability,
wall-clock ≈ single slowest evaluation (typically probe_traintest, ~25 min).

Every probe uses the encoder-shape-matched single-movie config (task=TP)
so multi-movie encoders get evaluated on the SAME 108-recording TP val /
test set as the single-movie ones — apples-to-apples. Retrieval uses each
CLIP run's own config (needs loss.proj_dim for MovieCLIPHead).

Usage:
    uv run --group eeg python experiments/jepa_pretraining/_submit_probe.py [submit]

    submit : append to actually sbatch each job. Otherwise dry-run summary.
"""
import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
OUT_DIR = "/work/hdd/bbnv/dtyoung/eb_jepa/probe_results_jul10"

# Single-movie configs — task=ThePresent — used as encoder-shape templates.
LEJEPA_CFG = "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-h200-bs64_lejepa_reve_seed2025/config.yaml"
LAYA_CFG   = "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-h200-bs64_laya_seed2025/config.yaml"

# CLIP-checkpoint configs — needed for retrieval.py (MovieCLIPHead reads loss.proj_dim).
SOFT_LEJEPA_CFG = "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip_from_jepa/jul10_soft_from_lejepa_seed2025/config.yaml"
SOFT_LAYA_CFG   = "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip_from_jepa/jul10_soft_from_laya_seed2025/config.yaml"
CS_LEJEPA_CFG   = "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10_cs_from_lejepa_seed2025/config.yaml"
CS_LAYA_CFG     = "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10_cs_from_laya_seed2025/config.yaml"

# (name, checkpoint | None, encoder-shape config, retrieval config | None)
CHECKPOINTS = [
    ("random_lejepa",       None, LEJEPA_CFG, None),
    ("random_laya",         None, LAYA_CFG,   None),
    ("jepa_lejepa_single",
     "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-h200-bs64_lejepa_reve_seed2025/latest.pth.tar",
     LEJEPA_CFG, None),
    ("jepa_lejepa_multi",
     "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-multi_lejepa_reve_seed2025/latest.pth.tar",
     LEJEPA_CFG, None),
    ("jepa_laya_single",
     "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-h200-bs64_laya_seed2025/latest.pth.tar",
     LAYA_CFG, None),
    ("jepa_laya_multi",
     "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-multi_laya_seed2025/latest.pth.tar",
     LAYA_CFG, None),
    ("soft_from_lejepa",
     "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip_from_jepa/jul10_soft_from_lejepa_seed2025/latest.pth.tar",
     LEJEPA_CFG, SOFT_LEJEPA_CFG),
    ("soft_from_laya",
     "/work/hdd/bbnv/dtyoung/eb_jepa/soft_target_clip_from_jepa/jul10_soft_from_laya_seed2025/latest.pth.tar",
     LAYA_CFG, SOFT_LAYA_CFG),
    ("cs_from_lejepa",
     "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10_cs_from_lejepa_seed2025/latest.pth.tar",
     LEJEPA_CFG, CS_LEJEPA_CFG),
    ("cs_from_laya",
     "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10_cs_from_laya_seed2025/latest.pth.tar",
     LAYA_CFG, CS_LAYA_CFG),
]


COMMON_ENV = {
    "WANDB_MODE": "disabled",
    "HBN_PREPROCESS_DIR": "/projects/bbnv/kkokate/hbn_preprocessed",
    "MNE_DATA": "/u/dtyoung/mne_data",
}


def _base_job(name: str, time_limit: str, command: str) -> Job:
    return Job(
        name=name,
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit=time_limit,
        command=command,
        venv="__none__",
        branch="",
        env_vars=COMMON_ENV,
    )


def _ckpt_arg(ckpt: str | None) -> str:
    return f"--checkpoint {ckpt}" if ckpt else ""


def _rand_flag(ckpt: str | None) -> str:
    return "--random-baseline" if ckpt is None else ""


def job_probe(name: str, ckpt: str | None, cfg: str) -> Job:
    out = f"{OUT_DIR}/{name}__probe_val.json"
    cmd = (
        f"mkdir -p {OUT_DIR} && "
        "PYTHONPATH=. uv run --group eeg python "
        "eb_jepa/evaluation/clip_probe/probe.py "
        f"{_ckpt_arg(ckpt)} --config {cfg} "
        "--split val --cv-splits 5 --encode-batch 64 "
        f"--output {out} {_rand_flag(ckpt)}"
    )
    return _base_job(f"probe_val_{name}", "00:30:00", cmd)


def job_traintest(name: str, ckpt: str | None, cfg: str) -> Job:
    out = f"{OUT_DIR}/{name}__probe_traintest.json"
    cmd = (
        f"mkdir -p {OUT_DIR} && "
        "PYTHONPATH=. uv run --group eeg python "
        "eb_jepa/evaluation/clip_probe/probe_traintest.py "
        f"{_ckpt_arg(ckpt)} --config {cfg} "
        "--eval-split test --encode-batch 64 "
        f"--output {out} {_rand_flag(ckpt)}"
    )
    return _base_job(f"probe_tt_{name}", "01:30:00", cmd)


def job_retrieval(name: str, ckpt: str, cfg: str) -> Job:
    out = f"{OUT_DIR}/{name}__retrieval_val.json"
    cmd = (
        f"mkdir -p {OUT_DIR} && "
        "PYTHONPATH=. uv run --group eeg python "
        "eb_jepa/evaluation/clip_probe/retrieval.py "
        f"--checkpoint {ckpt} --config {cfg} "
        "--split val --encode-batch 64 "
        f"--output {out}"
    )
    return _base_job(f"retrieval_{name}", "00:30:00", cmd)


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for name, ckpt, cfg, _ in CHECKPOINTS:
        jobs.append(job_probe(name, ckpt, cfg))
    for name, ckpt, cfg, _ in CHECKPOINTS:
        jobs.append(job_traintest(name, ckpt, cfg))
    for name, ckpt, _, rcfg in CHECKPOINTS:
        if rcfg is not None:
            jobs.append(job_retrieval(name, ckpt, rcfg))
    return jobs


if __name__ == "__main__":
    jobs = build_jobs()
    submit = "submit" in sys.argv
    n_probe = len(CHECKPOINTS)
    n_traintest = len(CHECKPOINTS)
    n_retrieval = sum(1 for _, _, _, r in CHECKPOINTS if r is not None)
    header = f"{n_probe} probes + {n_traintest} traintest + {n_retrieval} retrieval = {len(jobs)} independent jobs"
    if not submit:
        print(f"Dry-run: {header}")
        for j in jobs:
            print(f"  {j.name}  ({j.time_limit})")
        print("Add 'submit' to actually sbatch each.")
        sys.exit(0)
    print(f"Submitting: {header}")
    job_ids: list[str] = []
    for j in jobs:
        jid = j.submit()
        job_ids.append(str(jid))
        print(f"  {j.name:35s}  →  job_id={jid}")
    print(f"\nAll job IDs: {','.join(job_ids)}")
