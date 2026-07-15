"""
Image embedder -- wraps a pretrained MobileNetV3-Small as a fixed feature
extractor (no fine-tuning, no training loop). This is what makes few-shot
classification with only 5-10 images per class viable: the CNN already
learned general visual features from ImageNet, we're just re-using its
representation space and comparing distances in it, not training new
weights.

MobileNetV3-Small chosen specifically for latency: ~2.5M params vs ~11M for
ResNet18, meaningfully faster CPU inference at a small accuracy cost -- the
right tradeoff for a "5-10 images, fast predictions" use case.
"""
from __future__ import annotations

import io
import time

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# Standard ImageNet normalization -- required since the backbone was
# pretrained on ImageNet-normalized inputs.
_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageEmbedder:
    """Loads a pretrained MobileNetV3-Small once, reuses it for all embeddings."""

    def __init__(self):
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = models.mobilenet_v3_small(weights=weights)
        # Strip the final classification layer -- we want the feature vector
        # (576-dim) feeding into it, not the 1000-class ImageNet prediction.
        backbone.classifier = torch.nn.Identity()
        backbone.eval()
        self.model = backbone
        self.embedding_dim = 576

    @torch.no_grad()
    def embed(self, image_bytes: bytes) -> np.ndarray:
        """Embed a single image (raw bytes) into a 576-dim feature vector."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = _PREPROCESS(img).unsqueeze(0)  # add batch dim
        embedding = self.model(tensor).squeeze(0).numpy()
        # L2-normalize so cosine similarity == dot product downstream
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    @torch.no_grad()
    def embed_with_timing(self, image_bytes: bytes) -> tuple[np.ndarray, float]:
        """Same as embed(), but also returns wall-clock latency in ms."""
        start = time.perf_counter()
        embedding = self.embed(image_bytes)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return embedding, elapsed_ms


# Singleton -- loading the pretrained weights takes ~1-2s, don't repeat per request
_embedder_instance: ImageEmbedder | None = None


def get_embedder() -> ImageEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = ImageEmbedder()
    return _embedder_instance
