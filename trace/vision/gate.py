"""
Change Gate module.
High-speed CLAHE + SSIM + Delta-E pixel verification.
Eliminates redundant deep feature evaluations for static scenes (Section 5.5).
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


def compute_ssim_grayscale(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two single-channel images.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    c1 = 6.5025   # (0.01 * 255)^2
    c2 = 58.5225  # (0.03 * 255)^2

    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(i1 * i1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(i2 * i2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = num / (den + 1e-7)
    return float(np.mean(ssim_map))


class ChangeGate:
    """
    Evaluates whether an asset ROI has experienced visual change.
    """

    def __init__(self, ssim_threshold: float = 0.93, delta_e_threshold: float = 6.0):
        self.ssim_threshold = ssim_threshold
        self.delta_e_threshold = delta_e_threshold
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def evaluate(
        self,
        live_crop_bgr: np.ndarray,
        baseline_crop_bgr: np.ndarray,
        mask_non_person: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, float]:
        """
        Applies CLAHE on L channel of LAB, calculates SSIM and Delta E on visible pixels.
        Returns: (is_unchanged, ssim, delta_e)
        """
        if live_crop_bgr.shape[:2] != baseline_crop_bgr.shape[:2]:
            h, w = baseline_crop_bgr.shape[:2]
            live_crop_bgr = cv2.resize(live_crop_bgr, (w, h))

        # Convert to LAB color space
        live_lab = cv2.cvtColor(live_crop_bgr, cv2.COLOR_BGR2LAB)
        base_lab = cv2.cvtColor(baseline_crop_bgr, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel to normalize lighting variations
        live_l = self.clahe.apply(live_lab[:, :, 0])
        base_l = self.clahe.apply(base_lab[:, :, 0])

        # Compute SSIM on normalized L channel
        ssim = compute_ssim_grayscale(live_l, base_l)

        # Compute Delta E (CIE76 color difference)
        diff_lab = live_lab.astype(np.float32) - base_lab.astype(np.float32)
        delta_e_map = np.sqrt(np.sum(diff_lab ** 2, axis=2))

        if mask_non_person is not None and np.any(mask_non_person > 0):
            valid_mask = mask_non_person > 0
            if np.any(valid_mask):
                delta_e = float(np.mean(delta_e_map[valid_mask]))
            else:
                delta_e = float(np.mean(delta_e_map))
        else:
            delta_e = float(np.mean(delta_e_map))

        # Unchanged if SSIM is high and color shift is low
        is_unchanged = (ssim >= self.ssim_threshold) and (delta_e <= self.delta_e_threshold)

        return is_unchanged, ssim, delta_e
