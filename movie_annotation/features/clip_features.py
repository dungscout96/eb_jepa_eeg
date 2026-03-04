"""CLIP embeddings and zero-shot scene classification."""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Scene categories for zero-shot classification
SCENE_CATEGORIES = [
    "indoor room",
    "outdoor street",
    "outdoor nature landscape",
    "office or workspace",
    "kitchen",
    "bedroom",
    "living room",
    "bathroom",
    "school or classroom",
    "sports field or stadium",
    "restaurant or cafe",
    "store or shop",
    "vehicle interior",
    "stage or theater",
    "fantasy or animated world",
]

# Prompts for dimensional scene scores
SCENE_NATURAL_PROMPT = "a photograph of a natural scene with trees, grass, or water"
SCENE_URBAN_PROMPT = "a photograph of an urban scene with buildings and roads"
SCENE_OPEN_PROMPT = "a photograph of an open outdoor scene with a wide view"
SCENE_ENCLOSED_PROMPT = "a photograph of an enclosed indoor scene"


def _to_tensor(output):
    """Extract tensor from model output (handles transformers 5.x API change)."""
    if isinstance(output, torch.Tensor):
        return output
    # BaseModelOutputWithPooling or similar — use pooler_output or last_hidden_state
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise TypeError(f"Unexpected output type: {type(output)}")


class CLIPFeatureExtractor:
    """CLIP-based visual embeddings and zero-shot scene classification."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(device).eval()

        # Pre-encode text prompts
        self._encode_text_prompts()

    @torch.no_grad()
    def _encode_text_prompts(self):
        """Pre-encode all text prompts for reuse across frames."""
        # Scene category prompts
        scene_texts = [f"a photograph of a {cat}" for cat in SCENE_CATEGORIES]
        inputs = self.processor(text=scene_texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.scene_text_embeds = _to_tensor(self.model.get_text_features(**inputs))
        self.scene_text_embeds = self.scene_text_embeds / self.scene_text_embeds.norm(
            dim=-1, keepdim=True,
        )

        # Dimensional prompts (natural vs urban, open vs enclosed)
        dim_texts = [
            SCENE_NATURAL_PROMPT, SCENE_URBAN_PROMPT,
            SCENE_OPEN_PROMPT, SCENE_ENCLOSED_PROMPT,
        ]
        inputs = self.processor(text=dim_texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.dim_text_embeds = _to_tensor(self.model.get_text_features(**inputs))
        self.dim_text_embeds = self.dim_text_embeds / self.dim_text_embeds.norm(
            dim=-1, keepdim=True,
        )

    @torch.no_grad()
    def extract_batch(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
        """Extract CLIP embeddings and scene classification for a batch.

        Args:
            frames_bgr: List of BGR numpy frames.

        Returns:
            (embeddings, features) where:
            - embeddings: np.ndarray of shape (batch, 512)
            - features: list of dicts with scene classification keys
        """
        pil_images = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        image_embeds = _to_tensor(self.model.get_image_features(**inputs))
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Scene category classification
        scene_sims = (image_embeds_norm @ self.scene_text_embeds.T)  # (B, n_categories)
        scene_probs = scene_sims.softmax(dim=-1)
        top_indices = scene_probs.argmax(dim=-1).cpu().numpy()
        top_scores = scene_probs.max(dim=-1).values.cpu().numpy()

        # Dimensional scores
        dim_sims = (image_embeds_norm @ self.dim_text_embeds.T)  # (B, 4)
        dim_sims_np = dim_sims.cpu().numpy()

        embeddings = image_embeds.cpu().numpy()  # raw (un-normalized) embeddings

        features = []
        for i in range(len(frames_bgr)):
            # Natural vs urban: difference of similarities
            natural_score = float(dim_sims_np[i, 0] - dim_sims_np[i, 1])
            # Open vs enclosed: difference of similarities
            open_score = float(dim_sims_np[i, 2] - dim_sims_np[i, 3])

            features.append({
                "scene_category": SCENE_CATEGORIES[top_indices[i]],
                "scene_category_score": float(top_scores[i]),
                "scene_natural_score": natural_score,
                "scene_open_score": open_score,
            })

        return embeddings, features
