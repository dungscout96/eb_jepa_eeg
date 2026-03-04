"""Monocular depth estimation using Intel DPT-Large (MiDaS)."""

import numpy as np
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor


class DepthEstimator:
    """Batched monocular depth estimation with DPT-Large."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.model.to(device).eval()

    @torch.no_grad()
    def extract_batch(self, frames_bgr: list[np.ndarray]) -> list[dict]:
        """Extract depth features for a batch of BGR frames.

        Returns list of dicts with keys: depth_mean, depth_std, depth_range
        """
        # Convert BGR numpy arrays to RGB PIL images
        pil_images = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        predictions = outputs.predicted_depth  # (B, H, W)

        results = []
        for i in range(predictions.shape[0]):
            depth_map = predictions[i].cpu().numpy()
            results.append({
                "depth_mean": float(depth_map.mean()),
                "depth_std": float(depth_map.std()),
                "depth_range": float(depth_map.max() - depth_map.min()),
            })

        return results
