"""
Visual DNA Embedding module.
Extracts 384-dimensional L2-normalised feature vectors capturing color, texture, and geometry.
Provides variant augmentation, cosine similarity, and in-memory caching (Section 5.1).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional
import cv2
import numpy as np


class VisualDNAExtractor:
    """
    Extracts invariant visual embeddings (384-D, unit L2 norm) for registered assets.
    """

    def __init__(self, embedding_dim: int = 384):
        self.dim = embedding_dim
        self._cache: Dict[str, np.ndarray] = {}

    def extract(self, crop_bgr: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """
        Extracts a normalized 384-dimensional embedding from an image crop.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0
            return vec

        # Cache key based on downsampled crop hash
        if use_cache:
            tiny = cv2.resize(crop_bgr, (24, 24), interpolation=cv2.INTER_AREA)
            cache_key = hashlib.md5(tiny.tobytes()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Standardize crop size
        target_size = (128, 128)
        resized = cv2.resize(crop_bgr, target_size, interpolation=cv2.INTER_AREA)

        # 1. Multi-scale LAB color histogram (4x4 spatial cells)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        features: List[float] = []

        # 4x4 spatial grid
        cell_h, cell_w = 32, 32
        for r in range(4):
            for c in range(4):
                cell = lab[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                # Mean and std of L, A, B in this cell (6 values)
                for ch in range(3):
                    features.append(float(np.mean(cell[:, :, ch])))
                    features.append(float(np.std(cell[:, :, ch])))

        # 2. Gradient / Edge orientation histogram (Sobel)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # 8-bin orientation histogram per 4x4 cell
        bin_width = 360.0 / 8.0
        for r in range(4):
            for c in range(4):
                c_mag = mag[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                c_ang = ang[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                hist, _ = np.histogram(c_ang, bins=8, range=(0, 360), weights=c_mag)
                features.extend([float(v) for v in hist])

        # Concatenate features into vector
        raw_vec = np.array(features, dtype=np.float32)

        # Pad or trim to exact embedding_dim (384)
        if len(raw_vec) < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[: len(raw_vec)] = raw_vec
            raw_vec = padded
        else:
            raw_vec = raw_vec[: self.dim]

        # L2-normalise
        norm = float(np.linalg.norm(raw_vec))
        if norm > 1e-6:
            vec = raw_vec / norm
        else:
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[0] = 1.0

        if use_cache:
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = vec

        return vec

    def generate_variants(self, base_crop_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Generates simulated variants under lighting shifts, blur, and contrast changes
        matching Section 5.1 Visual DNA requirements.
        """
        variants = [self.extract(base_crop_bgr)]

        # Variant 1: Brightness +30%
        v1 = cv2.convertScaleAbs(base_crop_bgr, alpha=1.0, beta=35)
        variants.append(self.extract(v1))

        # Variant 2: Brightness -30%
        v2 = cv2.convertScaleAbs(base_crop_bgr, alpha=1.0, beta=-35)
        variants.append(self.extract(v2))

        # Variant 3: Contrast increase (+25%)
        v3 = cv2.convertScaleAbs(base_crop_bgr, alpha=1.25, beta=0)
        variants.append(self.extract(v3))

        # Variant 4: Mild Gaussian blur
        v4 = cv2.GaussianBlur(base_crop_bgr, (5, 5), 1.0)
        variants.append(self.extract(v4))

        return variants

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two unit vectors."""
        return float(np.dot(v1, v2))
