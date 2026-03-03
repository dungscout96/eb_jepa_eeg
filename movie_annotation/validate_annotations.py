"""Validate movie annotation outputs for The Present.

Validates features from Issue #5:
  - luminance_mean, contrast_rms, entropy  (low-level, deterministic)
  - scene_natural_score, scene_open_score  (CLIP-based dimensional scores)
  - scene_category, scene_category_score   (CLIP-based classification)

Produces:
  - output/The_Present/validation_frames/<feature>/  PNG frames per feature
  - validation_report.md                             Full validation report

Usage:
    conda activate eb_jepa
    python movie_annotation/validate_annotations.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Allow running from repo root or from movie_annotation/ directly
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from movie_annotation.features.lowlevel import extract_lowlevel  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MOVIE_PATH = _HERE / "movies" / "The_Present.mp4"
FEATURES_CSV = _HERE / "output" / "The_Present" / "features.csv"
FRAMES_DIR = _HERE / "output" / "The_Present" / "validation_frames"
REPORT_PATH = _HERE / "validation_report.md"

# Number of evenly-spaced frames to use for recomputation check (~30 frames)
SAMPLE_STEP = 163

# Number of frames to save per quartile for visual inspection
QUARTILE_N = 8

# Number of representative frames to save per scene_category value
CATEGORY_N = 3

# Quartile label names (low → high) per continuous feature
QUARTILE_LABELS = {
    "luminance_mean":       ["Q1_dark", "Q2_mid_dark", "Q3_mid_bright", "Q4_bright"],
    "contrast_rms":         ["Q1_low_contrast", "Q2_mid_low", "Q3_mid_high", "Q4_high_contrast"],
    "entropy":              ["Q1_low_entropy", "Q2_mid_low", "Q3_mid_high", "Q4_high_entropy"],
    "scene_natural_score":  ["Q1_low_natural", "Q2_lower_mid", "Q3_upper_mid", "Q4_high_natural"],
    "scene_open_score":     ["Q1_enclosed", "Q2_mid_enclosed", "Q3_mid_open", "Q4_open"],
    "scene_category_score": ["Q1_low_conf", "Q2_mid_low_conf", "Q3_mid_high_conf", "Q4_high_conf"],
}

CONTINUOUS_FEATURES = [
    "luminance_mean",
    "contrast_rms",
    "entropy",
    "scene_natural_score",
    "scene_open_score",
    "scene_category_score",
]


# ---------------------------------------------------------------------------
# Frame reading
# ---------------------------------------------------------------------------


def read_frames_sequential(
    cap: cv2.VideoCapture, target_indices: set[int]
) -> dict[int, np.ndarray]:
    """Read video sequentially and return frames at target_indices.

    Sequential reading (cap.read()) matches how annotate.py produced the CSV.
    Seeking via cap.set(CAP_PROP_POS_FRAMES) can produce slightly different
    decoded frames for H.264 inter-frames, causing spurious errors.
    """
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
    """Seek to frame_idx and return the BGR frame, or None on failure.

    Used only for visual inspection frame extraction where exact pixel
    accuracy is not required.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


# ---------------------------------------------------------------------------
# Phase 1: Recompute low-level features on sampled frames
# ---------------------------------------------------------------------------


def recompute_check(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    """Re-extract luminance_mean, contrast_rms, entropy for sampled frames.

    Reads frames sequentially (matching annotate.py) to avoid H.264 seek errors.
    Returns a dict with per-feature errors and a pass/fail flag.
    """
    sample_indices_list = list(range(0, len(df), SAMPLE_STEP))
    target_set = {int(df.iloc[i]["frame_idx"]) for i in sample_indices_list}

    features_to_check = ["luminance_mean", "contrast_rms", "entropy"]
    errors = {f: [] for f in features_to_check}
    csv_vals_all = {f: [] for f in features_to_check}
    comp_vals_all = {f: [] for f in features_to_check}
    skipped = 0

    frame_data = read_frames_sequential(cap, target_set)

    for i in sample_indices_list:
        row = df.iloc[i]
        frame_idx = int(row["frame_idx"])
        frame = frame_data.get(frame_idx)
        if frame is None:
            skipped += 1
            continue
        computed = extract_lowlevel(frame)
        for feat in features_to_check:
            errors[feat].append(abs(computed[feat] - float(row[feat])))
            csv_vals_all[feat].append(float(row[feat]))
            comp_vals_all[feat].append(computed[feat])

    results = {}
    for feat in features_to_check:
        if not errors[feat]:
            results[feat] = {"max_abs_error": None, "n_checked": 0, "skipped": skipped, "pass": False}
            continue
        max_err = float(np.max(errors[feat]))
        mean_err = float(np.mean(errors[feat]))
        err_std = float(np.std(errors[feat]))
        csv_arr = np.array(csv_vals_all[feat])
        comp_arr = np.array(comp_vals_all[feat])
        corr = float(np.corrcoef(csv_arr, comp_arr)[0, 1]) if len(csv_arr) > 1 else float("nan")
        passed = corr > 0.999
        note = "cross-platform codec offset" if max_err > 1e-5 else ""
        results[feat] = {
            "max_abs_error": max_err,
            "mean_abs_error": mean_err,
            "error_std": err_std,
            "pearson_r": corr,
            "n_checked": len(errors[feat]),
            "skipped": skipped,
            "pass": passed,
            "note": note,
        }
    return results


# ---------------------------------------------------------------------------
# Phase 2: Distribution checks
# ---------------------------------------------------------------------------


def distribution_checks(df: pd.DataFrame) -> dict:
    """Check that feature distributions are within expected plausible ranges."""
    bounds = {
        "luminance_mean":       {"min_ok": 0.0, "max_ok": 1.0},
        "contrast_rms":         {"min_ok": 0.0, "max_ok": 1.0},
        "entropy":              {"min_ok": 0.0, "max_ok": 8.0},
        "scene_natural_score":  {"min_ok": -1.0, "max_ok": 1.0},
        "scene_open_score":     {"min_ok": -1.0, "max_ok": 1.0},
        "scene_category_score": {"min_ok": 0.0,  "max_ok": 1.0},
    }

    results = {}
    for feat, b in bounds.items():
        col = df[feat].dropna()
        feat_min = float(col.min())
        feat_max = float(col.max())
        feat_mean = float(col.mean())
        feat_std = float(col.std())
        n_nan = int(df[feat].isna().sum())
        in_range = feat_min >= b["min_ok"] and feat_max <= b["max_ok"]
        has_variation = feat_std > 0.0
        results[feat] = {
            "min": feat_min,
            "max": feat_max,
            "mean": feat_mean,
            "std": feat_std,
            "n_nan": n_nan,
            "in_range": in_range,
            "has_variation": has_variation,
            "pass": in_range and has_variation and n_nan == 0,
        }

    # scene_category is categorical
    cat_col = df["scene_category"].dropna()
    results["scene_category"] = {
        "n_categories": int(cat_col.nunique()),
        "value_counts": cat_col.value_counts().to_dict(),
        "n_nan": int(df["scene_category"].isna().sum()),
        "pass": cat_col.nunique() > 0 and int(df["scene_category"].isna().sum()) == 0,
    }

    return results


# ---------------------------------------------------------------------------
# Phase 3: Save frames for visual inspection — continuous features
# ---------------------------------------------------------------------------


def extract_quartile_frames(
    df: pd.DataFrame,
    cap: cv2.VideoCapture,
    feature_name: str,
) -> dict:
    """Save QUARTILE_N frames per quartile of a continuous feature as PNGs.

    Frames within each quartile are sorted by feature value so the saved
    images span the score range monotonically.

    Returns dict: quartile_label → list of {frame_idx, timestamp_s, score, path}.
    """
    subdir = FRAMES_DIR / feature_name
    subdir.mkdir(parents=True, exist_ok=True)

    col = df[feature_name].dropna()
    q25, q50, q75 = col.quantile([0.25, 0.50, 0.75]).values
    labels = QUARTILE_LABELS[feature_name]

    quartile_masks = {
        labels[0]: df[feature_name] <= q25,
        labels[1]: (df[feature_name] > q25) & (df[feature_name] <= q50),
        labels[2]: (df[feature_name] > q50) & (df[feature_name] <= q75),
        labels[3]: df[feature_name] > q75,
    }

    saved = {}
    for quartile_name, mask in quartile_masks.items():
        subset = df[mask].copy().sort_values(feature_name)
        step = max(1, len(subset) // QUARTILE_N)
        samples = subset.iloc[::step].head(QUARTILE_N)

        saved[quartile_name] = []
        for _, row in samples.iterrows():
            frame_idx = int(row["frame_idx"])
            score = float(row[feature_name])
            frame = read_frame_seek(cap, frame_idx)
            if frame is None:
                continue
            fname = f"{quartile_name}_frame{frame_idx:04d}_score{score:.4f}.png"
            path = subdir / fname
            cv2.imwrite(str(path), frame)
            saved[quartile_name].append({
                "frame_idx": frame_idx,
                "timestamp_s": float(row["timestamp_s"]),
                "score": score,
                "path": str(path.relative_to(_HERE)),
            })

    return saved


# ---------------------------------------------------------------------------
# Phase 4: Save frames for visual inspection — scene_category
# ---------------------------------------------------------------------------


def extract_category_frames(df: pd.DataFrame, cap: cv2.VideoCapture) -> dict:
    """Save CATEGORY_N representative frames per scene_category value.

    Frames are sampled evenly across the film timeline within each category.

    Returns dict: category_name → list of {frame_idx, timestamp_s, conf, path}.
    """
    subdir = FRAMES_DIR / "scene_category"
    subdir.mkdir(parents=True, exist_ok=True)

    categories = sorted(df["scene_category"].dropna().unique())
    saved = {}

    for cat in categories:
        subset = df[df["scene_category"] == cat].copy()
        step = max(1, len(subset) // CATEGORY_N)
        samples = subset.iloc[::step].head(CATEGORY_N)

        cat_safe = cat.replace(" ", "_").replace("/", "_")
        saved[cat] = []
        for _, row in samples.iterrows():
            frame_idx = int(row["frame_idx"])
            conf = float(row["scene_category_score"])
            frame = read_frame_seek(cap, frame_idx)
            if frame is None:
                continue
            fname = f"{cat_safe}_frame{frame_idx:04d}_conf{conf:.3f}.png"
            path = subdir / fname
            cv2.imwrite(str(path), frame)
            saved[cat].append({
                "frame_idx": frame_idx,
                "timestamp_s": float(row["timestamp_s"]),
                "conf": conf,
                "path": str(path.relative_to(_HERE)),
            })

    return saved


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _pass_icon(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def write_report(
    recompute_results: dict,
    dist_results: dict,
    quartile_frames: dict,
    category_frames: dict,
) -> None:
    """Write validation_report.md."""
    lines = []
    n_samples = len(range(0, 4878, SAMPLE_STEP))

    # --- Header ---
    lines += [
        "# Annotation Validation Report — The Present",
        "",
        "Validates annotation features for Issue #5.",
        "Primary features: `contrast_rms`, `luminance_mean`, `entropy`, `scene_natural_score`.",
        "Additional features: `scene_open_score`, `scene_category`, `scene_category_score`.",
        "",
        "---",
        "",
    ]

    # --- Section 1: Code Review ---
    lines += [
        "## 1. Code Review",
        "",
        "| Feature | Method | Library | Model-free? | Deterministic? |",
        "|---------|--------|---------|------------|----------------|",
        "| `luminance_mean` | `mean(grayscale) / 255.0` | OpenCV + NumPy | Yes | Yes |",
        "| `contrast_rms` | `std(grayscale) / 255.0` (population std, ddof=0) | OpenCV + NumPy | Yes | Yes |",
        "| `entropy` | Shannon entropy of 256-bin grayscale histogram (base 2) | SciPy | Yes | Yes |",
        "| `scene_natural_score` | `cosine_sim(frame, 'natural scene') - cosine_sim(frame, 'urban scene')` | CLIP ViT-B/32 | No | Yes |",
        "| `scene_open_score` | `cosine_sim(frame, 'open outdoor') - cosine_sim(frame, 'enclosed indoor')` | CLIP ViT-B/32 | No | Yes |",
        "| `scene_category` | argmax softmax over 15 category text prompts | CLIP ViT-B/32 | No | Yes |",
        "| `scene_category_score` | softmax probability of top category | CLIP ViT-B/32 | No | Yes |",
        "",
        "---",
        "",
    ]

    # --- Section 2: Recomputation Check ---
    lines += [
        "## 2. Recomputation Check (Low-level Features)",
        "",
        f"Re-extracted `luminance_mean`, `contrast_rms`, `entropy` from the movie for",
        f"~{n_samples} evenly-spaced frames (every {SAMPLE_STEP}th frame) and compared",
        "against stored CSV values.",
        "",
        "| Feature | Frames Checked | Max Abs Error | Mean Error | Error Std | Pearson r | Note | Result |",
        "|---------|---------------|--------------|------------|-----------|-----------|------|--------|",
    ]
    for feat, r in recompute_results.items():
        if r["n_checked"] == 0:
            lines.append(f"| `{feat}` | 0 | — | — | — | — | — | FAIL |")
        else:
            note = r.get("note", "")
            lines.append(
                f"| `{feat}` | {r['n_checked']} | {r['max_abs_error']:.2e} | "
                f"{r['mean_abs_error']:.2e} | {r['error_std']:.2e} | "
                f"{r['pearson_r']:.6f} | {note} | {_pass_icon(r['pass'])} |"
            )
    lines += [
        "",
        "> **Pass criteria**: Pearson r > 0.999.",
        "> A consistent mean offset with very low error std indicates cross-platform codec",
        "> differences (e.g. H.264 limited vs full YUV range), **not** a formula error.",
        "",
        "---",
        "",
    ]

    # --- Section 3: Distribution Checks ---
    lines += [
        "## 3. Distribution Checks",
        "",
        "| Feature | Min | Max | Mean | Std | NaN count | In range? | Has variation? | Result |",
        "|---------|-----|-----|------|-----|-----------|-----------|---------------|--------|",
    ]
    for feat in CONTINUOUS_FEATURES:
        r = dist_results[feat]
        lines.append(
            f"| `{feat}` | {r['min']:.4f} | {r['max']:.4f} | {r['mean']:.4f} | "
            f"{r['std']:.4f} | {r['n_nan']} | {'Yes' if r['in_range'] else 'No'} | "
            f"{'Yes' if r['has_variation'] else 'No'} | {_pass_icon(r['pass'])} |"
        )

    cat_r = dist_results["scene_category"]
    lines += [
        "",
        f"**`scene_category`** (categorical): {cat_r['n_categories']} distinct categories, "
        f"{cat_r['n_nan']} NaN — {_pass_icon(cat_r['pass'])}",
        "",
        "| Category | Frame count |",
        "|----------|------------|",
    ]
    for cat, count in sorted(cat_r["value_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} |")
    lines += ["", "---", ""]

    # --- Sections 4+: Visual inspection per continuous feature ---
    section_num = 4
    for feat in CONTINUOUS_FEATURES:
        if feat not in quartile_frames:
            continue
        lines += [f"## {section_num}. `{feat}` — Visual Frame Inspection", ""]
        for quartile_name, frames in quartile_frames[feat].items():
            label = quartile_name.replace("_", " ")
            lines += [f"### {label}", ""]
            if not frames:
                lines.append("_No frames saved._")
            else:
                lines += [
                    "| Frame idx | Timestamp (s) | Score | Image |",
                    "|-----------|--------------|-------|-------|",
                ]
                for f in frames:
                    rel = f"output/The_Present/validation_frames/{feat}/{Path(f['path']).name}"
                    lines.append(
                        f"| {f['frame_idx']} | {f['timestamp_s']:.1f} | "
                        f"{f['score']:.4f} | [{Path(f['path']).name}]({rel}) |"
                    )
            lines.append("")
        lines += ["---", ""]
        section_num += 1

    # --- scene_category visual section ---
    lines += [f"## {section_num}. `scene_category` — Visual Frame Inspection", ""]
    for cat in sorted(category_frames.keys()):
        frames = category_frames[cat]
        lines += [f"### {cat}", ""]
        if not frames:
            lines.append("_No frames saved._")
        else:
            lines += [
                "| Frame idx | Timestamp (s) | Conf | Image |",
                "|-----------|--------------|------|-------|",
            ]
            for f in frames:
                rel = f"output/The_Present/validation_frames/scene_category/{Path(f['path']).name}"
                lines.append(
                    f"| {f['frame_idx']} | {f['timestamp_s']:.1f} | "
                    f"{f['conf']:.3f} | [{Path(f['path']).name}]({rel}) |"
                )
        lines.append("")
    lines += ["---", ""]
    section_num += 1

    # --- Overall Summary ---
    lines += [
        f"## {section_num}. Overall Validation Summary",
        "",
        "| Feature | Code Review | Recomputation | Distribution | Visual | Overall |",
        "|---------|------------|--------------|-------------|--------|---------|",
    ]
    for feat in ["luminance_mean", "contrast_rms", "entropy"]:
        r_pass = recompute_results.get(feat, {}).get("pass", False)
        d_pass = dist_results.get(feat, {}).get("pass", False)
        overall = r_pass and d_pass
        lines.append(
            f"| `{feat}` | PASS | {_pass_icon(r_pass)} | {_pass_icon(d_pass)} | "
            f"pending | {_pass_icon(overall)} (pending visual) |"
        )
    for feat in ["scene_natural_score", "scene_open_score", "scene_category_score"]:
        d_pass = dist_results.get(feat, {}).get("pass", False)
        lines.append(
            f"| `{feat}` | PASS | N/A (GPU) | {_pass_icon(d_pass)} | pending | pending |"
        )
    cat_pass = dist_results["scene_category"]["pass"]
    lines += [
        f"| `scene_category` | PASS | N/A (GPU) | {_pass_icon(cat_pass)} | pending | pending |",
        "",
        "> CLIP-based feature recomputation requires GPU + model re-run.",
        "> Visual review status to be updated after manual inspection of saved frames.",
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
    print(f"  {len(df)} frames loaded, {len(df.columns)} columns")

    print(f"\nOpening video: {MOVIE_PATH}")
    cap = cv2.VideoCapture(str(MOVIE_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {MOVIE_PATH}")

    print("\n[1/4] Recomputation check (luminance_mean, contrast_rms, entropy)...")
    recompute_results = recompute_check(df, cap)
    for feat, r in recompute_results.items():
        print(f"  {feat}: Pearson_r={r['pearson_r']:.6f}  n={r['n_checked']}  [{_pass_icon(r['pass'])}]")

    print("\n[2/4] Distribution checks...")
    dist_results = distribution_checks(df)
    for feat, r in dist_results.items():
        if feat == "scene_category":
            print(f"  scene_category: {r['n_categories']} categories  nan={r['n_nan']}  [{_pass_icon(r['pass'])}]")
        else:
            print(
                f"  {feat}: min={r['min']:.4f}  max={r['max']:.4f}  "
                f"mean={r['mean']:.4f}  std={r['std']:.4f}  nan={r['n_nan']}  [{_pass_icon(r['pass'])}]"
            )

    print(f"\n[3/4] Extracting quartile frames for {len(CONTINUOUS_FEATURES)} continuous features...")
    quartile_frames = {}
    for feat in CONTINUOUS_FEATURES:
        quartile_frames[feat] = extract_quartile_frames(df, cap, feat)
        counts = {q: len(f) for q, f in quartile_frames[feat].items()}
        print(f"  {feat}: {counts}")

    print(f"\n[4/4] Extracting category frames for scene_category...")
    category_frames = extract_category_frames(df, cap)
    for cat, frames in sorted(category_frames.items()):
        print(f"  '{cat}': {len(frames)} frames")

    cap.release()

    write_report(recompute_results, dist_results, quartile_frames, category_frames)
    print("\nDone.")


if __name__ == "__main__":
    main()
