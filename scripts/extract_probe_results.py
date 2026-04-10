"""Extract and summarize Phase 1 probe eval results from SLURM logs."""
import re
import glob
from collections import defaultdict
import numpy as np

files = sorted(glob.glob("/u/dtyoung/logs/ps1_*_17465*.out"))
results = []

for f in files:
    text = open(f).read()
    parts = text.split("=== Probe Eval:")

    for block in parts[1:]:
        # Extract config from "eeg_jepa_bs64_..._nw1_ws1s_seed2025 ==="
        header = block.split("\n")[0].strip().rstrip(" ===")
        m = re.search(r"nw(\d+)_ws(\d+)s_seed(\d+)", header)
        if not m:
            continue
        nw, ws, seed = m.group(1), m.group(2), m.group(3)

        metrics = {}
        for line in block.split("\n"):
            for key in [
                "val/subject_trait/bal_acc",
                "val/subject_trait/auc",
            ]:
                if key + ":" in line:
                    try:
                        val = float(line.split(":")[-1].strip())
                        metrics[key.split("/", 1)[-1]] = val
                    except ValueError:
                        pass

        if metrics:
            results.append((int(nw), int(ws), seed, metrics))

results.sort(key=lambda x: (x[0], x[1], x[2]))

KEYS_BA = [
    "subject_trait/bal_acc",
    "subject_trait/auc",
]
KEYS_R = []
SHORT = ["subj_ba", "subj_auc"]

hdr = f"{'config':<16} {'seed':>5} |"
for s in SHORT:
    hdr += f" {s:>8}"
print(hdr)
print("-" * len(hdr))

prev_cfg = ""
for nw, ws, seed, m in results:
    cfg = f"nw{nw}_ws{ws}"
    if cfg != prev_cfg and prev_cfg:
        print()
    prev_cfg = cfg
    vals = [m.get(k, 0) for k in KEYS_BA + KEYS_R]
    row = f"{cfg:<16} {seed:>5} |"
    for v in vals:
        row += f" {v:>8.4f}"
    print(row)

# Means per config
print("\n" + "=" * 120)
hdr2 = f"{'config':<16} {'n':>3} |"
for s in SHORT:
    hdr2 += f" {s:>8}"
print(hdr2)
print("-" * len(hdr2))

config_results = defaultdict(list)
for nw, ws, seed, m in results:
    config_results[(nw, ws)].append(m)

for (nw, ws), ms in sorted(config_results.items()):
    cfg = f"nw{nw}_ws{ws}"
    n = len(ms)
    means = []
    for k in KEYS_BA + KEYS_R:
        means.append(np.mean([m.get(k, 0) for m in ms]))
    row = f"{cfg:<16} {n:>3} |"
    for v in means:
        row += f" {v:>8.4f}"
    print(row)
