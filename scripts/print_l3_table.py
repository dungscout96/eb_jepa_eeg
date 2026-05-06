"""Print L1 + L3 table from a bootstrap-aggregate JSON."""

import json
import sys
from pathlib import Path

KEYS = [
    "reg_luminance_mean_corr",
    "reg_contrast_rms_corr",
    "reg_position_in_movie_corr",
    "reg_narrative_event_score_corr",
    "cls_luminance_mean_auc",
    "cls_contrast_rms_auc",
    "cls_position_in_movie_auc",
    "cls_narrative_event_score_auc",
    "age_reg_corr",
    "sex_auc",
    "movie_id_top1",
    "movie_id_top5",
]


def fmt(stat):
    if stat is None:
        return "      n/a       "
    return f"{stat['mean']:+.4f}±{stat['std']:.4f}"


def print_table(path):
    d = json.load(open(path))
    l1 = d.get("L1_5seed", {})
    l3 = d.get("L3", {})
    title = Path(path).stem
    print(f"=== {title} ===")
    for k in KEYS:
        a = l1.get(k)
        b = l3.get(k)
        if a or b:
            print(f"  {k:35s} L1={fmt(a)}  L3={fmt(b)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print_table(p)
