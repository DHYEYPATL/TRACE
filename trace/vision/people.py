"""
People Segmentation and Privacy Masking module.
Generates temporal person masks, calculates asset occlusion ratios,
and enforces positive-evidence visibility floors (Section 5.4).
"""

from __future__ import annotations

import collections
from typing import List, Optional, Tuple
import cv2
import numpy as np


class PeopleSegmenter:
    """
    Detects humans to calculate occlusion and mask customer/staff identities.
    Integrates OpenCV HOG detector with temporal dilation and multi-frame union.
    """

    def __init__(self, mask_dilation_px: int = 15, history_len: int = 3):
        self.mask_dilation_px = mask_dilation_px
        self.history = collections.deque(maxlen=history_len)

        # OpenCV built-in pedestrian detector (fast CPU inference)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

        # Morphological dilation kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (mask_dilation_px * 2 + 1, mask_dilation_px * 2 + 1)
        )

    def segment(self, frame_bgr: np.ndarray, extra_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Produces binary mask of people in the frame (255 where person present).
        Combines pedestrian detector with temporal accumulation over 3 frames.
        """
        h, w = frame_bgr.shape[:2]
        curr_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Include injected demo perturbation if present
        if extra_mask is not None:
            curr_mask = cv2.bitwise_or(curr_mask, extra_mask)

        # 2. Fast CPU pedestrian detection (downscaled for speed)
        rects = []
        scale = 0.5
        target_w = int(w * scale)
        target_h = int(h * scale)

        # OpenCV default HOG detector has fixed window size 64x128.
        # Passing an image smaller than this causes memory corruption / crash in 32-bit OpenCV.
        try:
            if target_w >= 64 and target_h >= 128:
                small = cv2.resize(frame_bgr, (target_w, target_h))
                rects, _ = self.hog.detectMultiScale(
                    small, winStride=(8, 8), padding=(8, 8), scale=1.05
                )
            elif w >= 64 and h >= 128:
                rects, _ = self.hog.detectMultiScale(
                    frame_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05
                )
                scale = 1.0
        except Exception:
            rects = []

        for rx, ry, rw, rh in rects:
            # Scale coordinates back up
            bx = int(rx / scale)
            by = int(ry / scale)
            bw = int(rw / scale)
            bh = int(rh / scale)
            # Create smooth elliptical body mask
            center = (bx + bw // 2, by + bh // 2)
            axes = (bw // 2, bh // 2)
            cv2.ellipse(curr_mask, center, axes, 0, 0, 360, 255, -1)

        # 3. Dilate mask by 15px
        dilated = cv2.dilate(curr_mask, self.kernel)
        
        # Filter history to keep only masks matching current dimensions
        valid_history = [m for m in self.history if m.shape == (h, w)]
        self.history = collections.deque(valid_history, maxlen=self.history.maxlen)
        self.history.append(dilated)

        # 4. Temporal OR across the last 3 frames (Section 5.4)
        accum_mask = np.zeros((h, w), dtype=np.uint8)
        for m in self.history:
            accum_mask = cv2.bitwise_or(accum_mask, m)

        return accum_mask

    @staticmethod
    def calculate_visibility_and_occlusion(
        person_mask: np.ndarray,
        region_pts: List[List[int]],
        frame_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Calculates:
        - occlusion_ratio: fraction of region covered by people (0.0 to 1.0)
        - visibility: fraction of region completely clear of people (0.0 to 1.0)
        """
        h, w = frame_shape[:2]
        region_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(region_pts, dtype=np.int32)
        cv2.fillPoly(region_mask, [pts], 255)

        total_region_px = int(np.count_nonzero(region_mask))
        if total_region_px == 0:
            return 0.0, 1.0

        occluded_px = int(np.count_nonzero(cv2.bitwise_and(region_mask, person_mask)))
        occlusion_ratio = float(occluded_px / total_region_px)
        visibility = max(0.0, 1.0 - occlusion_ratio)

        return occlusion_ratio, visibility
