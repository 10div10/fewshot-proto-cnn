"""
Prototype store -- implements the core of Prototypical Networks inference:
each class is represented by a single "prototype" vector, computed as the
mean of its few-shot example embeddings. A new image is classified by
finding the nearest prototype (cosine similarity, since embeddings are
L2-normalized so cosine == dot product).

This is deliberately training-free: no backprop, no epochs, no GPU needed.
Adding a class is just: embed 5-10 images, average them, store the vector.
That's what makes it fast to set up and fast to predict with.
"""
from __future__ import annotations

import json
import os

import numpy as np


class PrototypeStore:
    def __init__(self):
        # class_name -> prototype vector (mean of its reference embeddings)
        self.prototypes: dict[str, np.ndarray] = {}
        # class_name -> number of reference images used (for transparency)
        self.reference_counts: dict[str, int] = {}

    def add_class(self, class_name: str, embeddings: list[np.ndarray]) -> None:
        """
        Build (or overwrite) a class prototype from its few-shot reference
        embeddings. Recommended: 5-10 images per class -- enough to average
        out lighting/angle noise, few enough to stay fast and cheap to collect.
        """
        if not embeddings:
            raise ValueError(f"Cannot create a prototype for '{class_name}' with zero images.")
        stacked = np.stack(embeddings)
        prototype = stacked.mean(axis=0)
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototype = prototype / norm
        self.prototypes[class_name] = prototype
        self.reference_counts[class_name] = len(embeddings)

    def remove_class(self, class_name: str) -> bool:
        existed = class_name in self.prototypes
        self.prototypes.pop(class_name, None)
        self.reference_counts.pop(class_name, None)
        return existed

    def predict(self, embedding: np.ndarray, top_k: int = 3) -> dict:
        """
        Classify a query embedding by cosine similarity to every stored
        prototype. Returns the best match plus a ranked list, so low-confidence
        predictions are visible rather than silently returning a bad guess.
        """
        if not self.prototypes:
            return {"error": "No classes registered yet. Add at least one class first."}

        scores = {
            name: float(np.dot(embedding, proto))  # cosine sim, both are unit vectors
            for name, proto in self.prototypes.items()
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        best_class, best_score = ranked[0]
        return {
            "predicted_class": best_class,
            "confidence": round(best_score, 4),
            "top_k": [
                {"class": name, "score": round(score, 4)}
                for name, score in ranked[:top_k]
            ],
        }

    def list_classes(self) -> dict:
        return {
            name: {"reference_images": self.reference_counts.get(name, 0)}
            for name in self.prototypes
        }

    def save(self, path: str) -> None:
        """Persist prototypes to disk as JSON (vectors) + metadata."""
        payload = {
            "prototypes": {k: v.tolist() for k, v in self.prototypes.items()},
            "reference_counts": self.reference_counts,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    def load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No prototype store found at {path}")
        with open(path) as f:
            payload = json.load(f)
        self.prototypes = {k: np.array(v) for k, v in payload["prototypes"].items()}
        self.reference_counts = payload.get("reference_counts", {})
