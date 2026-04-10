"""Extract subject-trait probe results from ps1 logs."""
import re
import glob
from collections import defaultdict
import numpy as np

files = sorted(glob.glob("/u/dtyoung/logs/ps1_*_17482*.out"))
results = []

for f in files:
    text = open(f).read()
    parts = text.split("=== Probe Eval:")

    for block in parts[1:]:
        header = block.split("\n")[0].strip().rstrip(" ===")
        m = re.search(r"nw(\d+)_ws(\d+)s_seed(\d+)", header)
        if not m:
            continue
        nw, ws, seed = m.group(1), m.group(2), m.group(3)

        metrics = {}
        for line in block.split("\n"):
            for key in [
                "val/subject/age_reg/mae",
                "val/subject/age_reg/corr",
                "val/subject/age_reg/r2",
                "val/subject/sex/bal_acc",
                "val/subject/sex/auc",
                "val/subject/age_gt_",
            ]:
                if key in line and ":" in line:
                    full_key = line.split(":")[0].strip().replace("probe_eval/", "")
                    try:
                        val = float(line.split(":")[-1].strip())
                        metrics[full_key] = val
                    except ValueError:
                        pass

        if metrics:
            results.append((int(nw), int(ws), seed, metrics))

results.sort(key=lambda x: (x[0], x[1], x[2]))

# Find the age_gt key dynamically
age_gt_key = None
for _, _, _, m in results:
    for k in m:
        if "age_gt_" in k and "bal_acc" in k:
            age_gt_key = k
            break
    if age_gt_key:
        break

age_gt_auc_key = age_gt_key.replace("bal_acc", "auc") if age_gt_key else None

SHORT = ["age_ba", "age_auc", "sex_ba", "sex_auc", "age_mae", "age_corr", "age_r2"]
hdr = f"{'config':<12} {'seed':>5} |"
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

    age_ba = m.get(age_gt_key, float("nan")) if age_gt_key else float("nan")
    age_auc = m.get(age_gt_auc_key, float("nan")) if age_gt_auc_key else float("nan")
    sex_ba = m.get("val/subject/sex/bal_acc", float("nan"))
    sex_auc = m.get("val/subject/sex/auc", float("nan"))
    age_mae = m.get("val/subject/age_reg/mae", float("nan"))
    age_corr = m.get("val/subject/age_reg/corr", float("nan"))
    age_r2 = m.get("val/subject/age_reg/r2", float("nan"))

    vals = [age_ba, age_auc, sex_ba, sex_auc, age_mae, age_corr, age_r2]
    row = f"{cfg:<12} {seed:>5} |"
    for v in vals:
        row += f" {v:>8.4f}" if not np.isnan(v) else f" {'nan':>8}"
    print(row)

# Means
print("\n" + "=" * 100)
hdr2 = f"{'config':<12} {'n':>3} |"
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

    def mean_key(key):
        vals = [m.get(key, float("nan")) for m in ms]
        valid = [v for v in vals if not np.isnan(v)]
        return np.mean(valid) if valid else float("nan")

    age_ba = mean_key(age_gt_key) if age_gt_key else float("nan")
    age_auc = mean_key(age_gt_auc_key) if age_gt_auc_key else float("nan")
    sex_ba = mean_key("val/subject/sex/bal_acc")
    sex_auc = mean_key("val/subject/sex/auc")
    age_mae = mean_key("val/subject/age_reg/mae")
    age_corr = mean_key("val/subject/age_reg/corr")
    age_r2 = mean_key("val/subject/age_reg/r2")

    vals = [age_ba, age_auc, sex_ba, sex_auc, age_mae, age_corr, age_r2]
    row = f"{cfg:<12} {n:>3} |"
    for v in vals:
        row += f" {v:>8.4f}" if not np.isnan(v) else f" {'nan':>8}"
    print(row)
