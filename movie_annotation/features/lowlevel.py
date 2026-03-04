"""Low-level visual features: luminance, contrast, color, edges, spatial frequency, entropy."""

import cv2
import numpy as np
from scipy.stats import entropy as shannon_entropy


def extract_lowlevel(frame_bgr: np.ndarray) -> dict:
    """Extract low-level visual features from a single BGR frame.

    Returns dict with keys:
        luminance_mean, contrast_rms, color_r_mean, color_g_mean, color_b_mean,
        saturation_mean, edge_density, spatial_freq_energy, entropy
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Luminance: mean brightness
    luminance_mean = float(gray.mean() / 255.0)

    # RMS contrast: std of grayscale normalized to [0, 1]
    contrast_rms = float(gray.std() / 255.0)

    # Mean RGB (OpenCV is BGR)
    color_b_mean = float(frame_bgr[:, :, 0].mean() / 255.0)
    color_g_mean = float(frame_bgr[:, :, 1].mean() / 255.0)
    color_r_mean = float(frame_bgr[:, :, 2].mean() / 255.0)

    # Saturation from HSV
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation_mean = float(hsv[:, :, 1].mean() / 255.0)

    # Edge density via Canny
    edges = cv2.Canny(frame_bgr, 100, 200)
    edge_density = float(edges.astype(bool).sum() / (h * w))

    # Spatial frequency energy: ratio of high-frequency power
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift) ** 2
    total_power = magnitude.sum()
    # Create mask for low frequencies (central 25% of spectrum)
    cy, cx = h // 2, w // 2
    ry, rx = h // 4, w // 4
    mask = np.ones((h, w), dtype=bool)
    mask[cy - ry : cy + ry, cx - rx : cx + rx] = False
    high_freq_power = magnitude[mask].sum()
    spatial_freq_energy = float(high_freq_power / total_power) if total_power > 0 else 0.0

    # Shannon entropy of grayscale histogram
    hist, _ = np.histogram(gray.astype(np.uint8), bins=256, range=(0, 256))
    hist_prob = hist / hist.sum()
    img_entropy = float(shannon_entropy(hist_prob, base=2))

    return {
        "luminance_mean": luminance_mean,
        "contrast_rms": contrast_rms,
        "color_r_mean": color_r_mean,
        "color_g_mean": color_g_mean,
        "color_b_mean": color_b_mean,
        "saturation_mean": saturation_mean,
        "edge_density": edge_density,
        "spatial_freq_energy": spatial_freq_energy,
        "entropy": img_entropy,
    }
