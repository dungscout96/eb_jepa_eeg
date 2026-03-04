"""Motion and scene-cut features using optical flow."""

import cv2
import numpy as np

# Threshold for scene-cut detection via histogram correlation
SCENE_CUT_THRESHOLD = 0.3


def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute mean optical flow magnitude between two consecutive grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(magnitude.mean())


def detect_scene_cut(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> bool:
    """Detect scene cut by comparing color histograms between consecutive frames."""
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)

    # Compute hue-saturation histogram
    hist_prev = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_curr = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)

    correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
    return bool(correlation < SCENE_CUT_THRESHOLD)


def extract_motion(prev_bgr: np.ndarray | None, curr_bgr: np.ndarray) -> dict:
    """Extract motion energy and scene-cut flag.

    Args:
        prev_bgr: Previous frame (BGR) or None for the first frame.
        curr_bgr: Current frame (BGR).

    Returns dict with keys: motion_energy, scene_cut
    """
    if prev_bgr is None:
        return {"motion_energy": 0.0, "scene_cut": False}

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    motion_energy = compute_optical_flow(prev_gray, curr_gray)
    scene_cut = detect_scene_cut(prev_bgr, curr_bgr)

    return {"motion_energy": motion_energy, "scene_cut": scene_cut}
