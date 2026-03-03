"""Object detection using facebook/detr-resnet-50."""

import json

import numpy as np
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

DETECTION_THRESHOLD = 0.7


class ObjectDetector:
    """Batched object detection with DETR."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.to(device).eval()

    @torch.no_grad()
    def extract_batch(self, frames_bgr: list[np.ndarray]) -> list[dict]:
        """Extract object detection features for a batch of BGR frames.

        Returns list of dicts with keys: n_objects, object_categories
        """
        pil_images = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Process each image's detections
        target_sizes = torch.tensor([img.size[::-1] for img in pil_images]).to(self.device)
        batch_results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=DETECTION_THRESHOLD,
        )

        results = []
        for detections in batch_results:
            labels = detections["labels"].cpu().numpy()
            category_names = [self.model.config.id2label[lid] for lid in labels]
            category_counts = {}
            for name in category_names:
                category_counts[name] = category_counts.get(name, 0) + 1

            results.append({
                "n_objects": len(labels),
                "object_categories": json.dumps(category_counts),
            })

        return results
