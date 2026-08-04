"""Submit probe + traintest + retrieval on the two multi-movie CS-Aligner
CLIP fine-tunes (warm-started from multi-movie JEPA).

Six jobs total (3 evaluators × 2 encoders), each an independent sbatch so
they run in parallel on gpuA40x4. Evaluated on the TP-only val/test set
using the encoder-shape config from the single-movie JEPA runs (LEJEPA_CFG,
LAYA_CFG) — apples-to-apples with the existing jul10 probe results.

Retrieval uses the single-source CS-Aligner CLIP config for each shape
(same encoder + head architecture as the multi-source checkpoints since we
hand-mirrored shapes), so the clip_head loads without shape mismatch.

Usage:
    uv run --group eeg python experiments/jepa_pretraining/_submit_probe_multimulti.py [submit]
"""
import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
OUT_DIR = "/work/hdd/bbnv/dtyoung/eb_jepa/probe_results_jul10"

LEJEPA_CFG = "/work/hdd/bbnv/dtyoung/eb_jepa/lejepa_reve/jul10-h200-bs64_lejepa_reve_seed2025/config.yaml"
LAYA_CFG   = "/work/hdd/bbnv/dtyoung/eb_jepa/laya/jul10-h200-bs64_laya_seed2025/config.yaml"
CS_LEJEPA_CFG = "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10_cs_from_lejepa_seed2025/config.yaml"
CS_LAYA_CFG   = "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10_cs_from_laya_seed2025/config.yaml"

CHECKPOINTS = [
    ("cs_multi_from_lejepa_multi",
     "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10-multi_cs_from_lejepa_multi_seed2025/latest.pth.tar",
     LEJEPA_CFG, CS_LEJEPA_CFG),
    ("cs_multi_from_laya_multi",
     "/work/hdd/bbnv/dtyoung/eb_jepa/cs_aligner_from_jepa/jul10-multi_cs_from_laya_multi_seed2025/latest.pth.tar",
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


def job_probe(name: str, ckpt: str, cfg: str) -> Job:
    out = f"{OUT_DIR}/{name}__probe_val.json"
    cmd = (
        f"mkdir -p {OUT_DIR} && "
        "PYTHONPATH=. uv run --group eeg python "
        "eb_jepa/evaluation/clip_probe/probe.py "
        f"--checkpoint {ckpt} --config {cfg} "
        "--split val --cv-splits 5 --encode-batch 64 "
        f"--output {out}"
    )
    return _base_job(f"probe_val_{name}", "00:30:00", cmd)


def job_traintest(name: str, ckpt: str, cfg: str) -> Job:
    out = f"{OUT_DIR}/{name}__probe_traintest.json"
    cmd = (
        f"mkdir -p {OUT_DIR} && "
        "PYTHONPATH=. uv run --group eeg python "
        "eb_jepa/evaluation/clip_probe/probe_traintest.py "
        f"--checkpoint {ckpt} --config {cfg} "
        "--eval-split test --encode-batch 64 "
        f"--output {out}"
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
    for name, ckpt, cfg, rcfg in CHECKPOINTS:
        jobs.append(job_probe(name, ckpt, cfg))
        jobs.append(job_traintest(name, ckpt, cfg))
        jobs.append(job_retrieval(name, ckpt, rcfg))
    return jobs


if __name__ == "__main__":
    jobs = build_jobs()
    submit = "submit" in sys.argv
    if not submit:
        print(f"Dry-run: {len(jobs)} jobs")
        for j in jobs:
            print(f"  {j.name}  ({j.time_limit})")
        print("Add 'submit' to actually sbatch each.")
        sys.exit(0)
    print(f"Submitting {len(jobs)} jobs")
    job_ids: list[str] = []
    for j in jobs:
        jid = j.submit()
        job_ids.append(str(jid))
        print(f"  {j.name:40s}  →  job_id={jid}")
    print(f"\nAll job IDs: {','.join(job_ids)}")
