"""
Few-Shot State Change Classifier module.
Detects physical state changes such as vault door opened/closed or printer lid opened (Section 5.8).
Uses k-Nearest Neighbors on visual embeddings with zero external training dependencies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from trace.vision.embed import VisualDNAExtractor


class StateClassifier:
    """
    Few-shot asset state classifier (e.g. Safe Door Open vs Closed).
    """

    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors
        self.extractor = VisualDNAExtractor()
        # Stored training samples: {asset_id: [(embedding_vec, state_label), ...]}
        self.dataset: Dict[str, List[Tuple[np.ndarray, str]]] = {}

    def add_sample(self, asset_id: str, crop_bgr: np.ndarray, state_label: str) -> None:
        """Adds a reference sample for an asset state."""
        vec = self.extractor.extract(crop_bgr)
        self.dataset.setdefault(asset_id, []).append((vec, state_label))

    def predict(self, asset_id: str, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Predicts state label and confidence score.
        Returns: (predicted_state, confidence)
        """
        samples = self.dataset.get(asset_id, [])
        if not samples:
            return "DEFAULT", 1.0

        live_vec = self.extractor.extract(crop_bgr)

        # Calculate cosine similarities to all reference samples
        similarities = []
        for sample_vec, label in samples:
            sim = self.extractor.cosine_similarity(live_vec, sample_vec)
            similarities.append((sim, label))

        # Sort descending by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[: min(self.k, len(similarities))]

        # Tally weighted votes
        vote_weights: Dict[str, float] = {}
        total_w = 0.0
        for sim, label in top_k:
            # Softmax or rectified weight
            w = max(0.001, (sim + 1.0) / 2.0)
            vote_weights[label] = vote_weights.get(label, 0.0) + w
            total_w += w

        best_label = max(vote_weights.items(), key=lambda x: x[1])[0]
        confidence = vote_weights[best_label] / max(1e-6, total_w)

        return best_label, float(confidence)

    def seed_default_branch_samples(self, asset_id: str = "safe-02") -> None:
        """
        Seeds 15 mock training crops per state (CLOSED vs OPEN) matching Section 5.8
        so STATE_CHANGED triggers reliably out-of-the-box on the safe door showpiece.
        """
        w, h = 120, 140

        # 1. Closed Door Samples (Steel door, dial in center)
        for i in range(15):
            img = np.full((h, w, 3), 60 + i, dtype=np.uint8)
            cv2.rectangle(img, (10, 10), (w - 10, h - 10), (80 + i, 85 + i, 90 + i), -1)
            cv2.circle(img, (w // 2, h // 2), 20, (160, 165, 170), -1)
            cv2.circle(img, (w // 2, h // 2), 6, (30, 30, 30), -1)
            cv2.rectangle(img, (w - 25, h // 2 - 20), (w - 15, h // 2 + 20), (180, 185, 190), -1)
            self.add_sample(asset_id, img, "CLOSED")

        # 2. Open Door Samples (Dark vault interior, open door swung right, shelves, gold bar)
        for i in range(15):
            img = np.full((h, w, 3), 20 + i, dtype=np.uint8) # Dark safe cavity
            # Interior shelves
            cv2.line(img, (15, 45), (w - 30, 45), (100, 100, 100), 2)
            cv2.line(img, (15, 90), (w - 30, 90), (100, 100, 100), 2)
            # Gold bar / cash stacks on shelf
            cv2.rectangle(img, (30, 25), (75, 43), (40, 180, 230), -1) # Gold metallic bar
            # Swung open door on right side
            cv2.rectangle(img, (w - 25, 5), (w - 5, h - 5), (90, 95, 100), -1)
            self.add_sample(asset_id, img, "OPEN")

