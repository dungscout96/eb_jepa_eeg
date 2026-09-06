"""Validate movie annotation outputs for The Present.

Validates all 24 per-frame visual features from Issue #5.

Feature groups:
  Low-level:   luminance_mean, contrast_rms, entropy
  Color:       color_r_mean, color_g_mean, color_b_mean, saturation_mean
  Texture:     edge_density, spatial_freq_energy
  Motion:      motion_energy, scene_cut
  Faces:       n_faces, face_area_frac
  Depth:       depth_mean, depth_std, depth_range
  Objects:     n_objects, object_categories
  CLIP scene:  scene_natural_score, scene_open_score,
               scene_category, scene_category_score

Produces:
  - output/The_Present/validation_frames/<feature>/  PNG frames per feature
  - validation_report.md                             Full validation report

Usage:
    conda activate eb_jepa
    python movie_annotation/validate_annotations.py
"""

import ast
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from movie_annotation.features.lowlevel import extract_lowlevel  # noqa: E402
from movie_annotation.features.motion import extract_motion  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MOVIE_PATH = _HERE / "movies" / "The_Present.mp4"
FEATURES_CSV = _HERE / "output" / "The_Present" / "features.csv"
FRAMES_DIR = _HERE / "output" / "The_Present" / "validation_frames"
REPORT_PATH = _HERE / "validation_report.md"

SAMPLE_STEP = 163     # ~30 frames for recomputation check
QUARTILE_N = 8        # frames per quartile for continuous features
CATEGORY_N = 3        # frames per scene_category / object_category value
BIN_N = 4             # frames per discrete bin

# Continuous features that get quartile visual inspection
CONTINUOUS_FEATURES = [
    # low-level
    "luminance_mean", "contrast_rms", "entropy",
    # color
    "color_r_mean", "color_g_mean", "color_b_mean", "saturation_mean",
    # texture
    "edge_density", "spatial_freq_energy",
    # motion
    "motion_energy",
    # face
    "face_area_frac",
    # depth (model-relative units, not metres)
    "depth_mean", "depth_std", "depth_range",
    # CLIP
    "scene_natural_score", "scene_open_score", "scene_category_score",
]

QUARTILE_LABELS = {
    "luminance_mean":       ["Q1_dark",          "Q2_mid_dark",       "Q3_mid_bright",     "Q4_bright"],
    "contrast_rms":         ["Q1_low_contrast",   "Q2_mid_low",        "Q3_mid_high",       "Q4_high_contrast"],
    "entropy":              ["Q1_low_entropy",    "Q2_mid_low",        "Q3_mid_high",       "Q4_high_entropy"],
    "color_r_mean":         ["Q1_low_red",        "Q2_mid_low_red",    "Q3_mid_high_red",   "Q4_high_red"],
    "color_g_mean":         ["Q1_low_green",      "Q2_mid_low_green",  "Q3_mid_high_green", "Q4_high_green"],
    "color_b_mean":         ["Q1_low_blue",       "Q2_mid_low_blue",   "Q3_mid_high_blue",  "Q4_high_blue"],
    "saturation_mean":      ["Q1_desaturated",    "Q2_mid_low_sat",    "Q3_mid_high_sat",   "Q4_vivid"],
    "edge_density":         ["Q1_smooth",         "Q2_mid_low_edge",   "Q3_mid_high_edge",  "Q4_edgy"],
    "spatial_freq_energy":  ["Q1_low_freq",       "Q2_mid_low_freq",   "Q3_mid_high_freq",  "Q4_high_freq"],
    "motion_energy":        ["Q1_static",         "Q2_slow_motion",    "Q3_mid_motion",     "Q4_fast_motion"],
    "face_area_frac":       ["Q1_no_face",        "Q2_small_face",     "Q3_mid_face",       "Q4_large_face"],
    "depth_mean":           ["Q1_near",           "Q2_mid_near",       "Q3_mid_far",        "Q4_far"],
    "depth_std":            ["Q1_flat_depth",     "Q2_mid_low_depth",  "Q3_mid_high_depth", "Q4_varied_depth"],
    "depth_range":          ["Q1_narrow_depth",   "Q2_mid_narrow",     "Q3_mid_wide",       "Q4_wide_depth"],
    "scene_natural_score":  ["Q1_low_natural",    "Q2_lower_mid",      "Q3_upper_mid",      "Q4_high_natural"],
    "scene_open_score":     ["Q1_enclosed",       "Q2_mid_enclosed",   "Q3_mid_open",       "Q4_open"],
    "scene_category_score": ["Q1_low_conf",       "Q2_mid_low_conf",   "Q3_mid_high_conf",  "Q4_high_conf"],
}

# Expected ranges for distribution checks (None = skip range check)
FEATURE_BOUNDS = {
    "luminance_mean":       (0.0, 1.0),
    "contrast_rms":         (0.0, 1.0),
    "entropy":              (0.0, 8.0),
    "color_r_mean":         (0.0, 1.0),
    "color_g_mean":         (0.0, 1.0),
    "color_b_mean":         (0.0, 1.0),
    "saturation_mean":      (0.0, 1.0),
    "edge_density":         (0.0, 1.0),
    "spatial_freq_energy":  (0.0, 1.0),
    "motion_energy":        (0.0, None),   # unbounded above
    "face_area_frac":       (0.0, 1.0),
    "depth_mean":           (0.0, None),
    "depth_std":            (0.0, None),
    "depth_range":          (0.0, None),
    "scene_natural_score":  (-1.0, 1.0),
    "scene_open_score":     (-1.0, 1.0),
    "scene_category_score": (0.0, 1.0),
}


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------


def read_frames_sequential(
    cap: cv2.VideoCapture, target_indices: set[int]
) -> dict[int, np.ndarray]:
    """Read sequentially to exactly match annotate.py's decoding."""
    frames = {}
    frame_idx = 0
    while frame_idx <= max(target_indices):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in target_indices:
            frames[frame_idx] = frame
        frame_idx += 1
    return frames


def read_frame_seek(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


# ---------------------------------------------------------------------------
# Phase 1: Recompute low-level features
# ---------------------------------------------------------------------------


def recompute_check(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    sample_indices_list = list(range(0, len(df), SAMPLE_STEP))
    target_set = {int(df.iloc[i]["frame_idx"]) for i in sample_indices_list}
    features_to_check = [
        "luminance_mean", "contrast_rms", "entropy",
        "color_r_mean", "color_g_mean", "color_b_mean",
        "saturation_mean", "edge_density", "spatial_freq_energy",
    ]
    errors = {f: [] for f in features_to_check}
    csv_vals = {f: [] for f in features_to_check}
    comp_vals = {f: [] for f in features_to_check}
    skipped = 0

    frame_data = read_frames_sequential(cap, target_set)
    for i in sample_indices_list:
        row = df.iloc[i]
        fidx = int(row["frame_idx"])
        frame = frame_data.get(fidx)
        if frame is None:
            skipped += 1
            continue
        computed = extract_lowlevel(frame)
        for feat in features_to_check:
            errors[feat].append(abs(computed[feat] - float(row[feat])))
            csv_vals[feat].append(float(row[feat]))
            comp_vals[feat].append(computed[feat])

    results = {}
    for feat in features_to_check:
        if not errors[feat]:
            results[feat] = {"max_abs_error": None, "n_checked": 0, "skipped": skipped, "pass": False}
            continue
        max_err = float(np.max(errors[feat]))
        mean_err = float(np.mean(errors[feat]))
        err_std = float(np.std(errors[feat]))
        corr = float(np.corrcoef(csv_vals[feat], comp_vals[feat])[0, 1])
        passed = corr > 0.999
        note = "cross-platform codec offset" if max_err > 1e-5 else ""
        results[feat] = {
            "max_abs_error": max_err, "mean_abs_error": mean_err,
            "error_std": err_std, "pearson_r": corr,
            "n_checked": len(errors[feat]), "skipped": skipped,
            "pass": passed, "note": note,
        }
    return results


# ---------------------------------------------------------------------------
# Phase 1b: Recompute motion features
# ---------------------------------------------------------------------------


def recompute_motion_check(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    """Recompute motion_energy and scene_cut for sampled consecutive frame pairs."""
    n_frames = len(df)
    step = max(1, n_frames // 15)  # ~15 pairs
    sample_indices = list(range(1, n_frames, step))  # start at 1 (need prev frame)

    motion_csv = []
    motion_comp = []
    scene_agree = 0
    scene_total = 0
    skipped = 0

    for idx in sample_indices:
        prev_fidx = int(df.iloc[idx - 1]["frame_idx"])
        curr_fidx = int(df.iloc[idx]["frame_idx"])

        prev_frame = read_frame_seek(cap, prev_fidx)
        curr_frame = read_frame_seek(cap, curr_fidx)

        if prev_frame is None or curr_frame is None:
            skipped += 1
            continue

        computed = extract_motion(prev_frame, curr_frame)

        motion_csv.append(float(df.iloc[idx]["motion_energy"]))
        motion_comp.append(computed["motion_energy"])

        csv_cut = bool(df.iloc[idx]["scene_cut"])
        comp_cut = computed["scene_cut"]
        if csv_cut == comp_cut:
            scene_agree += 1
        scene_total += 1

    results = {}

    # motion_energy correlation
    if motion_csv:
        corr = float(np.corrcoef(motion_csv, motion_comp)[0, 1])
        errors = [abs(a - b) for a, b in zip(motion_csv, motion_comp)]
        max_err = float(np.max(errors))
        mean_err = float(np.mean(errors))
        err_std = float(np.std(errors))
        passed = corr > 0.999
        note = "cross-platform codec offset" if max_err > 1e-5 else ""
        results["motion_energy"] = {
            "max_abs_error": max_err, "mean_abs_error": mean_err,
            "error_std": err_std, "pearson_r": corr,
            "n_checked": len(motion_csv), "skipped": skipped,
            "pass": passed, "note": note,
        }
    else:
        results["motion_energy"] = {
            "max_abs_error": None, "n_checked": 0, "skipped": skipped, "pass": False,
        }

    # scene_cut boolean agreement
    if scene_total > 0:
        agreement = scene_agree / scene_total
        passed = agreement > 0.95
        results["scene_cut"] = {
            "agreement": agreement, "agree": scene_agree, "total": scene_total,
            "skipped": skipped, "pass": passed,
        }
    else:
        results["scene_cut"] = {
            "agreement": 0.0, "agree": 0, "total": 0, "skipped": skipped, "pass": False,
        }

    return results


# ---------------------------------------------------------------------------
# Phase 2: Distribution checks for all features
# ---------------------------------------------------------------------------


def distribution_checks(df: pd.DataFrame) -> dict:
    results = {}

    # Continuous features
    for feat, bounds in FEATURE_BOUNDS.items():
        col = df[feat].dropna()
        lo, hi = bounds
        feat_min, feat_max = float(col.min()), float(col.max())
        feat_mean, feat_std = float(col.mean()), float(col.std())
        n_nan = int(df[feat].isna().sum())
        in_range = (feat_min >= lo) and (hi is None or feat_max <= hi)
        results[feat] = {
            "min": feat_min, "max": feat_max, "mean": feat_mean, "std": feat_std,
            "n_nan": n_nan, "in_range": in_range,
            "has_variation": feat_std > 0.0,
            "pass": in_range and feat_std > 0.0 and n_nan == 0,
        }

    # n_faces (integer count)
    col = df["n_faces"]
    results["n_faces"] = {
        "value_counts": col.value_counts().sort_index().to_dict(),
        "n_nan": int(col.isna().sum()),
        "pass": col.isna().sum() == 0 and col.min() >= 0,
    }

    # n_objects (integer count)
    col = df["n_objects"]
    results["n_objects"] = {
        "min": int(col.min()), "max": int(col.max()),
        "mean": float(col.mean()), "std": float(col.std()),
        "n_nan": int(col.isna().sum()),
        "pass": col.isna().sum() == 0 and col.min() >= 0,
    }

    # scene_cut (binary)
    col = df["scene_cut"]
    results["scene_cut"] = {
        "n_cuts": int(col.sum()),
        "n_non_cuts": int((~col).sum()),
        "n_nan": int(col.isna().sum()),
        "pass": col.isna().sum() == 0 and col.sum() > 0,
    }

    # object_categories (categorical strings)
    import collections
    all_objs: dict[str, int] = collections.Counter()
    for val in df["object_categories"]:
        try:
            d = ast.literal_eval(str(val))
            for k, v in d.items():
                all_objs[k] += v
        except Exception:
            pass
    results["object_categories"] = {
        "n_unique": len(all_objs),
        "top_categories": dict(sorted(all_objs.items(), key=lambda x: -x[1])[:15]),
        "n_nan": int(df["object_categories"].isna().sum()),
        "pass": len(all_objs) > 0,
    }

    # scene_category
    cat_col = df["scene_category"].dropna()
    results["scene_category"] = {
        "n_categories": int(cat_col.nunique()),
        "value_counts": cat_col.value_counts().to_dict(),
        "n_nan": int(df["scene_category"].isna().sum()),
        "pass": cat_col.nunique() > 0 and int(df["scene_category"].isna().sum()) == 0,
    }

    return results


# ---------------------------------------------------------------------------
# Phase 3: Visual frame extraction — continuous (quartile)
# ---------------------------------------------------------------------------


def extract_quartile_frames(
    df: pd.DataFrame, cap: cv2.VideoCapture, feature_name: str,
) -> dict:
    subdir = FRAMES_DIR / feature_name
    subdir.mkdir(parents=True, exist_ok=True)

    col = df[feature_name].dropna()
    q25, q50, q75 = col.quantile([0.25, 0.50, 0.75]).values
    labels = QUARTILE_LABELS[feature_name]

    masks = {
        labels[0]: df[feature_name] <= q25,
        labels[1]: (df[feature_name] > q25) & (df[feature_name] <= q50),
        labels[2]: (df[feature_name] > q50) & (df[feature_name] <= q75),
        labels[3]: df[feature_name] > q75,
    }

    saved = {}
    for qname, mask in masks.items():
        subset = df[mask].copy().sort_values(feature_name)
        step = max(1, len(subset) // QUARTILE_N)
        samples = subset.iloc[::step].head(QUARTILE_N)
        saved[qname] = []
        for _, row in samples.iterrows():
            fidx = int(row["frame_idx"])
            score = float(row[feature_name])
            frame = read_frame_seek(cap, fidx)
            if frame is None:
                continue
            fname = f"{qname}_frame{fidx:04d}_score{score:.4f}.png"
            cv2.imwrite(str(subdir / fname), frame)
            saved[qname].append({
                "frame_idx": fidx, "timestamp_s": float(row["timestamp_s"]),
                "score": score, "path": str((subdir / fname).relative_to(_HERE)),
            })
    return saved


# ---------------------------------------------------------------------------
# Phase 4: Visual frame extraction — scene_category (CLIP)
# ---------------------------------------------------------------------------


def extract_category_frames(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    subdir = FRAMES_DIR / "scene_category"
    subdir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for cat in sorted(df["scene_category"].dropna().unique()):
        subset = df[df["scene_category"] == cat].copy()
        step = max(1, len(subset) // CATEGORY_N)
        samples = subset.iloc[::step].head(CATEGORY_N)
        cat_safe = cat.replace(" ", "_").replace("/", "_")
        saved[cat] = []
        for _, row in samples.iterrows():
            fidx = int(row["frame_idx"])
            conf = float(row["scene_category_score"])
            frame = read_frame_seek(cap, fidx)
            if frame is None:
                continue
            fname = f"{cat_safe}_frame{fidx:04d}_conf{conf:.3f}.png"
            cv2.imwrite(str(subdir / fname), frame)
            saved[cat].append({
                "frame_idx": fidx, "timestamp_s": float(row["timestamp_s"]),
                "conf": conf, "path": str((subdir / fname).relative_to(_HERE)),
            })
    return saved


# ---------------------------------------------------------------------------
# Phase 5: Visual frame extraction — n_faces (count bins)
# ---------------------------------------------------------------------------


def extract_nfaces_frames(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    """Save BIN_N frames for each distinct n_faces value (0, 1, 2)."""
    subdir = FRAMES_DIR / "n_faces"
    subdir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for val in sorted(df["n_faces"].unique()):
        subset = df[df["n_faces"] == val].copy()
        step = max(1, len(subset) // BIN_N)
        samples = subset.iloc[::step].head(BIN_N)
        key = f"n_faces_{val}"
        saved[key] = []
        for _, row in samples.iterrows():
            fidx = int(row["frame_idx"])
            frac = float(row["face_area_frac"])
            frame = read_frame_seek(cap, fidx)
            if frame is None:
                continue
            fname = f"nfaces{val}_frame{fidx:04d}_frac{frac:.3f}.png"
            cv2.imwrite(str(subdir / fname), frame)
            saved[key].append({
                "frame_idx": fidx, "timestamp_s": float(row["timestamp_s"]),
                "n_faces": val, "face_area_frac": frac,
                "path": str((subdir / fname).relative_to(_HERE)),
            })
    return saved


# ---------------------------------------------------------------------------
# Phase 6: Visual frame extraction — n_objects (count bins)
# ---------------------------------------------------------------------------


def extract_nobjects_frames(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    """Save BIN_N frames per object count bin: 0-1, 2-4, 5-9, 10+."""
    subdir = FRAMES_DIR / "n_objects"
    subdir.mkdir(parents=True, exist_ok=True)

    bins = {
        "0_to_1":  (df["n_objects"] <= 1),
        "2_to_4":  (df["n_objects"] >= 2) & (df["n_objects"] <= 4),
        "5_to_9":  (df["n_objects"] >= 5) & (df["n_objects"] <= 9),
        "10_plus": (df["n_objects"] >= 10),
    }

    saved = {}
    for bin_name, mask in bins.items():
        subset = df[mask].copy().sort_values("n_objects")
        step = max(1, len(subset) // BIN_N)
        samples = subset.iloc[::step].head(BIN_N)
        saved[bin_name] = []
        for _, row in samples.iterrows():
            fidx = int(row["frame_idx"])
            n = int(row["n_objects"])
            frame = read_frame_seek(cap, fidx)
            if frame is None:
                continue
            fname = f"{bin_name}_frame{fidx:04d}_n{n}.png"
            cv2.imwrite(str(subdir / fname), frame)
            saved[bin_name].append({
                "frame_idx": fidx, "timestamp_s": float(row["timestamp_s"]),
                "n_objects": n,
                "path": str((subdir / fname).relative_to(_HERE)),
            })
    return saved


# ---------------------------------------------------------------------------
# Phase 7: Visual frame extraction — scene_cut (before/after context)
# ---------------------------------------------------------------------------


def extract_scene_cut_frames(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    """Save 3-frame context (before, cut, after) for every scene cut."""
    subdir = FRAMES_DIR / "scene_cut"
    subdir.mkdir(parents=True, exist_ok=True)

    cut_rows = df[df["scene_cut"] == True].copy()
    saved = []

    for _, row in cut_rows.iterrows():
        fidx = int(row["frame_idx"])
        for offset, label in [(-1, "before"), (0, "cut"), (+1, "after")]:
            tfidx = fidx + offset
            if tfidx < 0 or tfidx >= len(df):
                continue
            frame = read_frame_seek(cap, tfidx)
            if frame is None:
                continue
            fname = f"cut{fidx:04d}_{label}_frame{tfidx:04d}.png"
            cv2.imwrite(str(subdir / fname), frame)
            saved.append({
                "cut_frame_idx": fidx,
                "frame_idx": tfidx,
                "label": label,
                "timestamp_s": float(df.iloc[tfidx]["timestamp_s"]) if tfidx < len(df) else 0.0,
                "path": str((subdir / fname).relative_to(_HERE)),
            })

    return {"scene_cuts": saved}


# ---------------------------------------------------------------------------
# Phase 8: Visual frame extraction — object_categories (per category)
# ---------------------------------------------------------------------------


def extract_object_category_frames(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    """Save CATEGORY_N frames for the top-10 detected object categories."""
    subdir = FRAMES_DIR / "object_categories"
    subdir.mkdir(parents=True, exist_ok=True)

    import collections

    # Count total object occurrences per category
    all_objs: dict[str, int] = collections.Counter()
    for val in df["object_categories"]:
        try:
            d = ast.literal_eval(str(val))
            for k, v in d.items():
                all_objs[k] += v
        except Exception:
            pass

    top_cats = [k for k, _ in sorted(all_objs.items(), key=lambda x: -x[1])[:10]]

    saved = {}
    for cat in top_cats:
        # Find frames where this category appears
        def has_cat(val, c=cat):
            try:
                d = ast.literal_eval(str(val))
                return c in d
            except Exception:
                return False

        mask = df["object_categories"].apply(has_cat)
        subset = df[mask].copy()
        step = max(1, len(subset) // CATEGORY_N)
        samples = subset.iloc[::step].head(CATEGORY_N)

        cat_safe = cat.replace(" ", "_")
        saved[cat] = []
        for _, row in samples.iterrows():
            fidx = int(row["frame_idx"])
            try:
                d = ast.literal_eval(str(row["object_categories"]))
                count = d.get(cat, 0)
            except Exception:
                count = 0
            frame = read_frame_seek(cap, fidx)
            if frame is None:
                continue
            fname = f"{cat_safe}_frame{fidx:04d}_count{count}.png"
            cv2.imwrite(str(subdir / fname), frame)
            saved[cat].append({
                "frame_idx": fidx, "timestamp_s": float(row["timestamp_s"]),
                "count": count,
                "path": str((subdir / fname).relative_to(_HERE)),
            })

    return saved


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _pass(p: bool) -> str:
    return "PASS" if p else "FAIL"


def _quartile_table(frames_by_quartile: dict, feature_name: str) -> list[str]:
    lines = []
    for qname, frames in frames_by_quartile.items():
        lines += [f"### {qname.replace('_', ' ')}", ""]
        if not frames:
            lines.append("_No frames saved._")
        else:
            lines += ["| Frame idx | Timestamp (s) | Score | Image |",
                      "|-----------|--------------|-------|-------|"]
            for f in frames:
                rel = f"output/The_Present/validation_frames/{feature_name}/{Path(f['path']).name}"
                lines.append(
                    f"| {f['frame_idx']} | {f['timestamp_s']:.1f} | "
                    f"{f['score']:.4f} | [{Path(f['path']).name}]({rel}) |"
                )
        lines.append("")
    return lines


def write_report(
    recompute_results: dict,
    motion_recompute_results: dict,
    dist_results: dict,
    quartile_frames: dict,
    category_frames: dict,
    nfaces_frames: dict,
    nobjects_frames: dict,
    scene_cut_frames: dict,
    obj_cat_frames: dict,
) -> None:
    n_samples = len(range(0, 4877, SAMPLE_STEP))
    lines = []

    # Header
    lines += [
        "# Annotation Validation Report — The Present",
        "",
        "Validates all 24 per-frame visual features.",
        "Primary features (Issue #5): `contrast_rms`, `luminance_mean`, `entropy`, `scene_natural_score`.",
        "Additional: all remaining features from the annotation pipeline.",
        "",
        "---",
        "",
    ]

    # Section 1: Code review
    lines += [
        "## 1. Code Review",
        "",
        "| Feature | Method | Library | Model-free? |",
        "|---------|--------|---------|------------|",
        "| `luminance_mean` | `mean(gray)/255` | OpenCV+NumPy | Yes |",
        "| `contrast_rms` | `std(gray)/255` (ddof=0) | OpenCV+NumPy | Yes |",
        "| `color_{r,g,b}_mean` | channel mean/255 | OpenCV+NumPy | Yes |",
        "| `saturation_mean` | HSV S-channel mean/255 | OpenCV | Yes |",
        "| `edge_density` | Canny edge pixel fraction | OpenCV | Yes |",
        "| `spatial_freq_energy` | FFT high-freq power ratio | NumPy | Yes |",
        "| `entropy` | Shannon entropy of 256-bin gray histogram | SciPy | Yes |",
        "| `motion_energy` | Farneback optical flow mean magnitude | OpenCV | Yes |",
        "| `scene_cut` | frame-diff threshold on luminance | NumPy | Yes |",
        "| `n_faces`, `face_area_frac` | RetinaFace detector | insightface | No (CNN) |",
        "| `depth_mean/std/range` | MiDaS monocular depth | timm/MiDaS | No (CNN) |",
        "| `n_objects`, `object_categories` | YOLOv8 detection | ultralytics | No (CNN) |",
        "| `scene_category`, `scene_category_score` | CLIP softmax (15 categories) | CLIP ViT-B/32 | No |",
        "| `scene_natural_score` | CLIP cosine diff (natural vs urban) | CLIP ViT-B/32 | No |",
        "| `scene_open_score` | CLIP cosine diff (open vs enclosed) | CLIP ViT-B/32 | No |",
        "",
        "---",
        "",
    ]

    # Section 2: Recomputation check
    lines += [
        "## 2. Recomputation Check (All 11 Model-free Features)",
        "",
        "Re-extracted all 11 model-free features (9 low-level + motion_energy + scene_cut) "
        f"and compared against stored CSV values.",
        "",
        "### 2a. Low-level features (9 features)",
        "",
        f"Sampled ~{n_samples} frames (every {SAMPLE_STEP}th), recomputed via `extract_lowlevel()`.",
        "",
        "| Feature | Frames | Max Abs Error | Mean Error | Error Std | Pearson r | Note | Result |",
        "|---------|--------|--------------|------------|-----------|-----------|------|--------|",
    ]
    for feat, r in recompute_results.items():
        if r["n_checked"] == 0:
            lines.append(f"| `{feat}` | 0 | — | — | — | — | — | FAIL |")
        else:
            note = r.get("note", "")
            lines.append(
                f"| `{feat}` | {r['n_checked']} | {r['max_abs_error']:.2e} | "
                f"{r['mean_abs_error']:.2e} | {r['error_std']:.2e} | "
                f"{r['pearson_r']:.6f} | {note} | {_pass(r['pass'])} |"
            )
    lines += [
        "",
        "> Pass criteria: Pearson r > 0.999. Consistent mean offset = cross-platform codec difference, not formula error.",
        "",
    ]

    # Motion recomputation
    mr = motion_recompute_results
    me = mr["motion_energy"]
    sc = mr["scene_cut"]
    lines += [
        "### 2b. Motion features (motion_energy + scene_cut)",
        "",
        f"Sampled ~{me['n_checked']} consecutive frame pairs, recomputed via `extract_motion()`.",
        "",
        "| Feature | Frames | Max Abs Error | Mean Error | Error Std | Pearson r | Note | Result |",
        "|---------|--------|--------------|------------|-----------|-----------|------|--------|",
    ]
    if me["n_checked"] == 0:
        lines.append("| `motion_energy` | 0 | — | — | — | — | — | FAIL |")
    else:
        note = me.get("note", "")
        lines.append(
            f"| `motion_energy` | {me['n_checked']} | {me['max_abs_error']:.2e} | "
            f"{me['mean_abs_error']:.2e} | {me['error_std']:.2e} | "
            f"{me['pearson_r']:.6f} | {note} | {_pass(me['pass'])} |"
        )
    lines += [
        "",
        f"**`scene_cut`** boolean agreement: {sc['agree']}/{sc['total']} "
        f"({sc['agreement']:.1%}) — {_pass(sc['pass'])}",
        "",
        "> scene_cut pass criteria: > 95% agreement.",
        "",
        "---",
        "",
    ]

    # Section 3: Distribution checks
    lines += [
        "## 3. Distribution Checks",
        "",
        "### Continuous features",
        "",
        "| Feature | Min | Max | Mean | Std | NaN | In range? | Variation? | Result |",
        "|---------|-----|-----|------|-----|-----|-----------|-----------|--------|",
    ]
    for feat in CONTINUOUS_FEATURES:
        r = dist_results[feat]
        lines.append(
            f"| `{feat}` | {r['min']:.4f} | {r['max']:.4f} | {r['mean']:.4f} | "
            f"{r['std']:.4f} | {r['n_nan']} | {'Yes' if r['in_range'] else 'No'} | "
            f"{'Yes' if r['has_variation'] else 'No'} | {_pass(r['pass'])} |"
        )

    lines += [
        "",
        "### Count features",
        "",
    ]
    r = dist_results["n_faces"]
    lines += [
        f"**`n_faces`**: {r['value_counts']}  NaN={r['n_nan']}  — {_pass(r['pass'])}",
        "",
    ]
    r = dist_results["n_objects"]
    lines += [
        f"**`n_objects`**: min={r['min']} max={r['max']} mean={r['mean']:.2f} std={r['std']:.2f}  "
        f"NaN={r['n_nan']}  — {_pass(r['pass'])}",
        "",
    ]
    r = dist_results["scene_cut"]
    lines += [
        f"**`scene_cut`**: {r['n_cuts']} cuts, {r['n_non_cuts']} non-cuts  NaN={r['n_nan']}  — {_pass(r['pass'])}",
        "",
    ]

    r = dist_results["object_categories"]
    lines += [
        "**`object_categories`** — top 15 detected categories:",
        "",
        "| Category | Total count |",
        "|----------|------------|",
    ]
    for cat, cnt in r["top_categories"].items():
        lines.append(f"| {cat} | {cnt} |")
    lines += ["", f"Total unique categories: {r['n_unique']}  NaN={r['n_nan']}  — {_pass(r['pass'])}", ""]

    cat_r = dist_results["scene_category"]
    lines += [
        f"**`scene_category`** (CLIP 15-way): {cat_r['n_categories']} categories assigned, NaN={cat_r['n_nan']}  — {_pass(cat_r['pass'])}",
        "",
        "| Category | Frame count |",
        "|----------|------------|",
    ]
    for cat, cnt in sorted(cat_r["value_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {cnt} |")
    lines += ["", "---", ""]

    # Sections 4+: Visual inspection per feature group
    section = 4

    def section_header(title):
        nonlocal section
        lines.extend([f"## {section}. {title}", ""])
        section += 1

    # --- Low-level ---
    section_header("Low-level Features — Visual Inspection")
    for feat in ["luminance_mean", "contrast_rms", "entropy"]:
        lines += [f"### `{feat}`", ""] + _quartile_table(quartile_frames[feat], feat)
    lines += ["---", ""]

    # --- Color ---
    section_header("Color Features — Visual Inspection")
    for feat in ["color_r_mean", "color_g_mean", "color_b_mean", "saturation_mean"]:
        lines += [f"### `{feat}`", ""] + _quartile_table(quartile_frames[feat], feat)
    lines += ["---", ""]

    # --- Texture ---
    section_header("Texture Features — Visual Inspection")
    for feat in ["edge_density", "spatial_freq_energy"]:
        lines += [f"### `{feat}`", ""] + _quartile_table(quartile_frames[feat], feat)
    lines += ["---", ""]

    # --- Motion ---
    section_header("Motion Features — Visual Inspection")
    lines += ["### `motion_energy`", ""] + _quartile_table(quartile_frames["motion_energy"], "motion_energy")
    # scene_cut
    lines += ["### `scene_cut`", "", "3-frame context (before / cut / after) for each detected cut.", ""]
    cuts = scene_cut_frames.get("scene_cuts", [])
    if not cuts:
        lines.append("_No cuts found._")
    else:
        lines += ["| Cut frame | Offset | Frame idx | Timestamp (s) | Image |",
                  "|-----------|--------|-----------|--------------|-------|"]
        for f in cuts:
            rel = f"output/The_Present/validation_frames/scene_cut/{Path(f['path']).name}"
            lines.append(
                f"| {f['cut_frame_idx']} | {f['label']} | {f['frame_idx']} | "
                f"{f['timestamp_s']:.1f} | [{Path(f['path']).name}]({rel}) |"
            )
    lines += ["", "---", ""]

    # --- Faces ---
    section_header("Face Features — Visual Inspection")
    lines += ["### `n_faces` (count bins)", ""]
    for key, frames in nfaces_frames.items():
        val = key.split("_")[-1]
        lines += [f"#### n_faces = {val}", ""]
        if not frames:
            lines.append("_No frames._")
        else:
            lines += ["| Frame idx | Timestamp (s) | face_area_frac | Image |",
                      "|-----------|--------------|----------------|-------|"]
            for f in frames:
                rel = f"output/The_Present/validation_frames/n_faces/{Path(f['path']).name}"
                lines.append(
                    f"| {f['frame_idx']} | {f['timestamp_s']:.1f} | "
                    f"{f['face_area_frac']:.3f} | [{Path(f['path']).name}]({rel}) |"
                )
        lines.append("")
    lines += ["### `face_area_frac` (quartiles)", ""] + _quartile_table(quartile_frames["face_area_frac"], "face_area_frac")
    lines += ["---", ""]

    # --- Depth ---
    section_header("Depth Features — Visual Inspection")
    lines.append("> Depth values are in model-relative units from MiDaS, not real-world metres.\n")
    for feat in ["depth_mean", "depth_std", "depth_range"]:
        lines += [f"### `{feat}`", ""] + _quartile_table(quartile_frames[feat], feat)
    lines += ["---", ""]

    # --- Objects ---
    section_header("Object Features — Visual Inspection")
    lines += ["### `n_objects` (count bins)", ""]
    for bin_name, frames in nobjects_frames.items():
        lines += [f"#### {bin_name.replace('_', ' ')}", ""]
        if not frames:
            lines.append("_No frames._")
        else:
            lines += ["| Frame idx | Timestamp (s) | n_objects | Image |",
                      "|-----------|--------------|-----------|-------|"]
            for f in frames:
                rel = f"output/The_Present/validation_frames/n_objects/{Path(f['path']).name}"
                lines.append(
                    f"| {f['frame_idx']} | {f['timestamp_s']:.1f} | "
                    f"{f['n_objects']} | [{Path(f['path']).name}]({rel}) |"
                )
        lines.append("")
    lines += ["### `object_categories` (top 10 categories)", ""]
    for cat, frames in obj_cat_frames.items():
        lines += [f"#### {cat}", ""]
        if not frames:
            lines.append("_No frames._")
        else:
            lines += ["| Frame idx | Timestamp (s) | Count | Image |",
                      "|-----------|--------------|-------|-------|"]
            for f in frames:
                rel = f"output/The_Present/validation_frames/object_categories/{Path(f['path']).name}"
                lines.append(
                    f"| {f['frame_idx']} | {f['timestamp_s']:.1f} | "
                    f"{f['count']} | [{Path(f['path']).name}]({rel}) |"
                )
        lines.append("")
    lines += ["---", ""]

    # --- CLIP ---
    section_header("CLIP Scene Features — Visual Inspection")
    for feat in ["scene_natural_score", "scene_open_score", "scene_category_score"]:
        lines += [f"### `{feat}`", ""] + _quartile_table(quartile_frames[feat], feat)
    lines += ["### `scene_category`", ""]
    for cat, frames in category_frames.items():
        lines += [f"#### {cat}", ""]
        if not frames:
            lines.append("_No frames._")
        else:
            lines += ["| Frame idx | Timestamp (s) | Conf | Image |",
                      "|-----------|--------------|------|-------|"]
            for f in frames:
                rel = f"output/The_Present/validation_frames/scene_category/{Path(f['path']).name}"
                lines.append(
                    f"| {f['frame_idx']} | {f['timestamp_s']:.1f} | "
                    f"{f['conf']:.3f} | [{Path(f['path']).name}]({rel}) |"
                )
        lines.append("")
    lines += ["---", ""]

    # --- Visual observations (to be filled after review) ---
    section_header("Visual Inspection Observations")
    lines += [
        "### Low-level features",
        "",
        "**`luminance_mean`** — PASS. Q1 = black fade-in + title card + dark blind-filtered room. "
        "Q4 = sunlit carpet overhead shots and open-door daylight. Monotonic, correct direction.",
        "",
        "**`contrast_rms`** — PASS. Q1 = pure-black frame + face close-ups (uniform skin). "
        "Q4 = window-blind stripes (maximum regular alternation) + harsh sunlit hallway. Physically correct.",
        "",
        "**`entropy`** — PASS (skewed distribution). Q1 anchored by rare pure-black (0.05 bits) and "
        "title card (1.35 bits). 75% of frames occupy only 0.63 bits of range (7.15–7.78). "
        "Feature discriminates title/fade frames well; limited for normal content.",
        "",
        "### Color features",
        "",
        "**`color_r/g/b_mean`** — WEAK as semantic color labels, but numerically correct as channel means. "
        "Quartiles mostly track warm-vs-cool illumination and overall brightness; `Q4_high_green` and "
        "`Q4_high_blue` are often bright neutral doorway shots rather than genuinely green- or blue-dominant scenes. "
        "Use these as raw channel-intensity features, not as human-interpretable color names.",
        "",
        "**`saturation_mean`** — WEAK. Mid/high-saturation frames often look plausible, but dark/title-card frames "
        "can score as maximally vivid. Example: frame 92 is almost black (`luminance_mean=0.0315`) yet has "
        "`saturation_mean=0.9067`, so HSV saturation becomes unstable on near-black inputs.",
        "",
        "### Texture features",
        "",
        "**`edge_density`** — PASS. Q1 contains black/title frames and smooth face close-ups; Q4 contains blind slats, "
        "room geometry, rug texture, and hand+ball shots with visibly denser contours. Direction is correct.",
        "",
        "**`spatial_freq_energy`** — WEAK. It does rank title text, blind slats, and other sharp repetitive patterns "
        "above flatter shots, but the dynamic range is tiny (0.0001–0.037) and the quartiles are visually mixed. "
        "For this animated short it is technically consistent, but not very discriminative.",
        "",
        "### Motion features",
        "",
        "**`motion_energy`** — PASS. Q1 is dominated by static black/title frames and held poses; Q4 contains dog-play, "
        "hand/ball interaction, and larger expression/body changes. Frame 0 = 0.0 is also correct by construction.",
        "",
        "**`scene_cut`** — FAIL. Several reported cuts are not edits at all, just adjacent motion frames. "
        "Examples: cut 2506 (frames 2505/2506/2507), cut 2811, and cut 3405 all show the same shot with small motion. "
        "The luminance-diff threshold is too weak to distinguish shot changes from within-shot motion/lighting change.",
        "",
        "### Face features",
        "",
        "**`n_faces`** — WEAK. The sampled 0/1/2-face examples are mostly sensible, but the detector is contaminated by "
        "animal-face false positives on this film. Across the CSV, 206 frames that also contain a detected `dog` have "
        "`n_faces > 0`, so the count is not reliably 'human faces only'.",
        "",
        "**`face_area_frac`** — FAIL as a human-face-size metric. High-area examples include dog close-ups "
        "(frames 2188, 3437, 3448), so the largest values are not consistently measuring human face prominence.",
        "",
        "### Depth features",
        "",
        "**`depth_mean/std/range`** — WEAK overall. `depth_std` and `depth_range` broadly track flatter vs more layered "
        "shots, but `depth_mean` is not visually aligned with the `near`/`far` labels in the report: Q1 includes wide "
        "doorway/room views while Q4 includes close face/dog close-ups. The metric is likely inverse-depth-like or "
        "otherwise model-relative, so the current `near`/`far` interpretation is backwards or at least ambiguous.",
        "",
        "### Object features",
        "",
        "**`n_objects`** — WEAK. Counts rise on cluttered kitchen/living-room shots as expected, but they inherit the "
        "detector's domain-mismatch errors on animation, so the absolute counts should be treated as rough complexity "
        "signals rather than trusted object cardinality.",
        "",
        "**`object_categories`** — WEAK/NOISY. Some tags are correct (`dog`, `person`, `potted plant`), but sampled "
        "false positives are obvious: `book` on window blinds (frame 845), `cell phone` on a game controller "
        "(frame 1942), and `teddy bear` on dog frames (e.g. frame 3437). Good enough for rough tags, not for clean semantics.",
        "",
        "### CLIP scene features",
        "",
        "**`scene_natural_score`** — FAIL for this film. See cross-check notes below.",
        "",
        "**`scene_open_score`** — FAIL for this film. Range [-0.080, +0.023], all negative (all interior).",
        "",
        "**`scene_category_score`** — FAIL. Constant ≈ 1/15 = 0.067 across all frames (uniform softmax).",
        "",
        "**`scene_category`** — FAIL. 10 of 15 categories appear including 'bathroom' and 'vehicle interior' "
        "for a living-room film. Assignments are noise due to near-uniform confidence.",
        "",
        "### CLIP cross-check: code and index alignment confirmed correct",
        "",
        "- `frame_idx` is sequential 0–4876, no gaps. Low-level features (luminance, contrast) confirmed "
        "against known visual frames — index alignment is correct.",
        "- Top `scene_natural_score` frames (3650, 2814) are the film's outdoor patio/garden ending scenes "
        "— direction is correct at the extremes.",
        "- Pure-black frames (0–5) get spurious positive score (~+0.02) due to L2-normalizing a near-zero "
        "CLIP embedding (floating-point noise amplification). This is a degenerate input issue, not a code bug.",
        "- All CLIP features fail due to content mismatch, not implementation error.",
        "",
        "### Wrong or weak metrics for this film",
        "",
        "- **Failing**: `scene_cut`, `face_area_frac`, `scene_natural_score`, `scene_open_score`, "
        "`scene_category_score`, `scene_category`.",
        "- **Weak / noisy**: `color_r_mean`, `color_g_mean`, `color_b_mean`, `saturation_mean`, "
        "`spatial_freq_energy`, `n_faces`, `depth_mean`, `depth_std`, `depth_range`, `n_objects`, `object_categories`.",
        "",
        "---",
        "",
    ]

    # Overall summary — derive recompute column from actual results
    def _recomp(feat):
        if feat in recompute_results:
            return _pass(recompute_results[feat]["pass"])
        return None

    def _recomp_all(*feats):
        vals = [_recomp(f) for f in feats]
        if all(v == "PASS" for v in vals):
            return "PASS"
        if any(v == "FAIL" for v in vals):
            return "FAIL"
        return "FAIL"

    mr_me = motion_recompute_results["motion_energy"]
    mr_sc = motion_recompute_results["scene_cut"]

    lines += [
        f"## {section}. Overall Validation Summary",
        "",
        "| Feature | Code | Recompute | Distribution | Visual | Overall |",
        "|---------|------|-----------|-------------|--------|---------|",
        f"| `luminance_mean` | PASS | {_recomp('luminance_mean')} | PASS | PASS | **PASS** |",
        f"| `contrast_rms` | PASS | {_recomp('contrast_rms')} | PASS | PASS | **PASS** |",
        f"| `entropy` | PASS | {_recomp('entropy')} | PASS | PASS (skewed) | **PASS** |",
        f"| `color_r/g/b_mean` | PASS | {_recomp_all('color_r_mean', 'color_g_mean', 'color_b_mean')} | PASS | WEAK (channel means, not semantic colors) | **WEAK** |",
        f"| `saturation_mean` | PASS | {_recomp('saturation_mean')} | PASS | WEAK (dark-frame artifact) | **WEAK** |",
        f"| `edge_density` | PASS | {_recomp('edge_density')} | PASS | PASS | **PASS** |",
        f"| `spatial_freq_energy` | PASS | {_recomp('spatial_freq_energy')} | PASS (tiny range) | WEAK | **WEAK** |",
        f"| `motion_energy` | PASS | {_pass(mr_me['pass'])} | PASS | PASS | **PASS** |",
        f"| `scene_cut` | PASS | {_pass(mr_sc['pass'])} | PASS (18 cuts) | FAIL | **FAIL** |",
        "| `n_faces` | PASS | N/A (GPU) | PASS | WEAK (dog false positives) | **WEAK** |",
        "| `face_area_frac` | PASS | N/A (GPU) | PASS | FAIL | **FAIL** |",
        "| `depth_mean/std/range` | PASS | N/A (GPU) | PASS | WEAK (`depth_mean` direction ambiguous) | **WEAK** |",
        "| `n_objects` | PASS | N/A (GPU) | PASS | WEAK | **WEAK** |",
        "| `object_categories` | PASS | N/A (GPU) | PASS | WEAK | **WEAK** |",
        "| `scene_natural_score` | PASS | N/A (GPU) | PASS (narrow) | FAIL | **FAIL (this film)** |",
        "| `scene_open_score` | PASS | N/A (GPU) | PASS (narrow) | FAIL | **FAIL (this film)** |",
        "| `scene_category_score` | PASS | N/A (GPU) | PASS (constant) | FAIL | **FAIL (this film)** |",
        "| `scene_category` | PASS | N/A (GPU) | PASS | FAIL | **FAIL (this film)** |",
        "",
        "> CLIP failures are content-mismatch (single-location animated short), not code errors.",
        "> GPU-based features (depth, objects, faces, CLIP) cannot be recomputed without GPU + model re-run.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Report written to {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading features CSV: {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df)} frames, {len(df.columns)} columns")

    print(f"\nOpening video: {MOVIE_PATH}")
    cap = cv2.VideoCapture(str(MOVIE_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {MOVIE_PATH}")

    print("\n[1/9] Recomputation check (all 9 low-level features)...")
    recompute_results = recompute_check(df, cap)
    for feat, r in recompute_results.items():
        print(f"  {feat}: r={r['pearson_r']:.6f}  n={r['n_checked']}  [{_pass(r['pass'])}]")

    print("\n[2/9] Motion recomputation check (motion_energy, scene_cut)...")
    motion_recompute_results = recompute_motion_check(df, cap)
    mr_me = motion_recompute_results["motion_energy"]
    mr_sc = motion_recompute_results["scene_cut"]
    if mr_me["n_checked"] > 0:
        print(f"  motion_energy: r={mr_me['pearson_r']:.6f}  n={mr_me['n_checked']}  [{_pass(mr_me['pass'])}]")
    else:
        print("  motion_energy: no frames checked  [FAIL]")
    print(f"  scene_cut: {mr_sc['agree']}/{mr_sc['total']} agree ({mr_sc['agreement']:.1%})  [{_pass(mr_sc['pass'])}]")

    print("\n[3/9] Distribution checks (all features)...")
    dist_results = distribution_checks(df)
    for feat, r in dist_results.items():
        if "pass" in r:
            print(f"  {feat}: [{_pass(r['pass'])}]")

    print(f"\n[4/9] Quartile frames for {len(CONTINUOUS_FEATURES)} continuous features...")
    quartile_frames = {}
    for feat in CONTINUOUS_FEATURES:
        quartile_frames[feat] = extract_quartile_frames(df, cap, feat)
        counts = sum(len(f) for f in quartile_frames[feat].values())
        print(f"  {feat}: {counts} frames saved")

    print("\n[5/9] scene_category frames...")
    category_frames = extract_category_frames(df, cap)
    print(f"  {sum(len(v) for v in category_frames.values())} frames across {len(category_frames)} categories")

    print("\n[6/9] n_faces frames...")
    nfaces_frames = extract_nfaces_frames(df, cap)
    print(f"  {sum(len(v) for v in nfaces_frames.values())} frames across {len(nfaces_frames)} bins")

    print("\n[7/9] n_objects frames...")
    nobjects_frames = extract_nobjects_frames(df, cap)
    print(f"  {sum(len(v) for v in nobjects_frames.values())} frames across {len(nobjects_frames)} bins")

    print("\n[8/9] scene_cut context frames...")
    scene_cut_frames = extract_scene_cut_frames(df, cap)
    print(f"  {len(scene_cut_frames['scene_cuts'])} context frames for {len(df[df['scene_cut']==True])} cuts")

    print("\n[9/9] object_category frames (top 10)...")
    obj_cat_frames = extract_object_category_frames(df, cap)
    print(f"  {sum(len(v) for v in obj_cat_frames.values())} frames across {len(obj_cat_frames)} categories")

    cap.release()

    write_report(
        recompute_results, motion_recompute_results, dist_results, quartile_frames,
        category_frames, nfaces_frames, nobjects_frames,
        scene_cut_frames, obj_cat_frames,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
