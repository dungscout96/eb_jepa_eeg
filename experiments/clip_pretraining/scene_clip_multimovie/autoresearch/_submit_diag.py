"""Diagnostic runs to separate capacity-vs-compute-vs-domain-shift.

Four parallel jobs, all against iter 1 baseline (embed=512 depth=12 400 ep):

  iter6_wider  : embed 512->768, heads 12, freqs 5. Depth=12, 400 ep. ~100M params.
  iter7_deeper : depth 12->24. Embed=512, 400 ep. ~92M params.
  iter8_both   : embed 512->768 AND depth 12->24. 400 ep. ~200M params (>2x).
  iter9_2xcompute: iter1 config unchanged, epochs 400->800. ~46M params.

If iter 6/7/8 recover TP toward single-movie iter12 ep500's +0.0505: capacity
was the bottleneck. If iter 9 does: compute was. If none: domain shift is
the real driver of the multi-movie per-domain regression.

Wall estimates (per-epoch cost estimates x epochs + probes + slack):
  iter6:  ~20 s/ep * 400 = 133 min + 15 probes + 15 slack = 163 min -> 3h wall
  iter7:  ~22 s/ep * 400 = 147 min + 15 + 15 = 177 min -> 3h wall
  iter8:  ~40 s/ep * 400 = 267 min + 15 + 15 = 297 min -> 5.5h wall
  iter9:  ~11 s/ep * 800 = 147 min + 15 + 15 = 177 min -> 3h wall

Usage:
    uv run --group eeg python _submit_diag.py all [submit]
    uv run --group eeg python _submit_diag.py iter6 submit
"""

import sys
from neurolab.jobs import Job

REPO = "/u/dtyoung/eb_jepa_eeg"
AUTORESEARCH_DIR = "experiments/clip_pretraining/scene_clip_multimovie/autoresearch"
CKPT_ROOT = "/u/dtyoung/eb_jepa_eeg/checkpoints/autoresearch/multimovie_jul2"

# Diagnostic variants: (iter_num, patch_script, epochs, wall)
VARIANTS = {
    "iter6": {
        "iter_num": 6,
        "patch": (
            "c.model.encoder_embed_dim = 768; "
            "c.model.encoder_heads = 12; "
            "c.model.freqs = 5; "
        ),
        "epochs": 400,
        "wall": "03:00:00",
        "name_suffix": "_wider",
    },
    "iter7": {
        "iter_num": 7,
        "patch": "c.model.encoder_depth = 24; ",
        "epochs": 400,
        "wall": "03:00:00",
        "name_suffix": "_deeper",
    },
    "iter8": {
        "iter_num": 8,
        "patch": (
            "c.model.encoder_embed_dim = 768; "
            "c.model.encoder_heads = 12; "
            "c.model.freqs = 5; "
            "c.model.encoder_depth = 24; "
        ),
        "epochs": 400,
        "wall": "06:00:00",  # 5.5h expected, 6h margin
        "name_suffix": "_both",
    },
    "iter9": {
        "iter_num": 9,
        "patch": "",
        "epochs": 800,
        "wall": "03:00:00",
        "name_suffix": "_2xcompute",
    },
}


def build_job(key: str) -> Job:
    v = VARIANTS[key]
    iter_num = v["iter_num"]
    exp_dir = f"{CKPT_ROOT}/iter{iter_num}{v['name_suffix']}"
    output_TP = f"{AUTORESEARCH_DIR}/probe_val_iter{iter_num}{v['name_suffix']}_TP.json"
    output_DM = f"{AUTORESEARCH_DIR}/probe_val_iter{iter_num}{v['name_suffix']}_DM.json"
    patch_cmd = (
        "PYTHONPATH=. uv run --group eeg python -c \""
        "from omegaconf import OmegaConf; "
        f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
        f"{v['patch']}"
        f"OmegaConf.save(c, '{exp_dir}/config.yaml')\" && "
    ) if v["patch"] else ""

    return Job(
        name=f"auto_mm_iter{iter_num}{v['name_suffix']}",
        cluster="delta",
        repo_path=REPO,
        partition="gpuA40x4",
        time_limit=v["wall"],
        command=(
            f"mkdir -p {exp_dir} && "
            f"cp experiments/clip_pretraining/scene_clip_multimovie/autoresearch/clip_pretrain.yaml {exp_dir}/config.yaml && "
            f"{patch_cmd}"
            # Snapshot single-task variants for per-movie probing
            "PYTHONPATH=. uv run --group eeg python -c \""
            "from omegaconf import OmegaConf; "
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            "c.data.task = 'ThePresent'; "
            f"OmegaConf.save(c, '{exp_dir}/config_TP.yaml'); "
            f"c = OmegaConf.load('{exp_dir}/config.yaml'); "
            "c.data.task = 'DespicableMe'; "
            f"OmegaConf.save(c, '{exp_dir}/config_DM.yaml')\" && "
            # Train
            "PYTHONPATH=. uv run --group eeg python -m eb_jepa.training.clip_pretrain"
            f" --fname={exp_dir}/config.yaml"
            f" --optim.epochs={v['epochs']}"
            f" --folder={exp_dir}"
            " --logging.save_every=50"
            f" --logging.wandb_group=auto_mm_iter{iter_num}{v['name_suffix']}"
            " && "
            # Probe TP
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            f" --config {exp_dir}/config_TP.yaml"
            " --split val --cv-splits 5"
            f" --output {output_TP}"
            " && "
            # Probe DM
            "PYTHONPATH=. uv run --group eeg python"
            " eb_jepa/evaluation/clip_probe/probe.py"
            f" --checkpoint {exp_dir}/latest.pth.tar"
            f" --config {exp_dir}/config_DM.yaml"
            " --split val --cv-splits 5"
            f" --output {output_DM}"
        ),
        venv="__none__",
        branch="",
        env_vars={
            "WANDB_PROJECT": "eb_jepa",
            "HBN_PREPROCESS_DIR": "/work/hdd/bbnv/kkokate/hbn_preprocessed",
        },
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: _submit_diag.py {{all|{'|'.join(VARIANTS.keys())}}} [submit]")
        sys.exit(1)
    target = sys.argv[1]
    submit = len(sys.argv) > 2 and sys.argv[2] == "submit"
    keys = list(VARIANTS.keys()) if target == "all" else [target]
    for key in keys:
        job = build_job(key)
        print("=" * 60)
        print(f"{'SUBMITTING' if submit else 'DRY RUN'} — {job.name} (wall={VARIANTS[key]['wall']}, ep={VARIANTS[key]['epochs']})")
        print("=" * 60)
        if submit:
            print(f"job_id: {job.submit()}")
        else:
            print(job.submit(dry_run=True))
