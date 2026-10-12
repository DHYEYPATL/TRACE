"""
New Object Detection module.
Maintains running background model in monitored zones and detects persistent foreign blobs (Section 5.7).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class NewObjectDetector:
    """
    Monitors zones for unexpected unregistered objects (Demand 6 / NEW_OBJECT).
    """

    def __init__(self, persistence_s: float = 8.0, min_area_fraction: float = 0.005):
        self.persistence_s = persistence_s
        self.min_area_fraction = min_area_fraction

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        self.blob_tracker: Dict[str, Dict[str, Any]] = {}
        self.next_blob_id: int = 1

    def update(
        self,
        frame_bgr: np.ndarray,
        person_mask: np.ndarray,
        registered_regions: List[List[List[int]]],
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Detects persistent foreground objects outside registered regions and person masks.
        """
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        min_blob_area = frame_area * self.min_area_fraction

        # 1. Background subtraction
        fg_mask = self.bg_subtractor.apply(frame_bgr, learningRate=0.001)

        # 2. Exclude person mask
        if person_mask is not None and np.any(person_mask > 0):
            fg_mask[person_mask > 0] = 0

        # 3. Exclude registered asset regions
        reg_mask = np.zeros((h, w), dtype=np.uint8)
        for r_pts in registered_regions:
            cv2.fillPoly(reg_mask, [np.array(r_pts, dtype=np.int32)], 255)
        # Dilate slightly so edges of assets don't trigger false new object
        reg_mask = cv2.dilate(reg_mask, np.ones((15, 15), np.uint8))
        fg_mask[reg_mask > 0] = 0

        # 4. Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # 5. Extract contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_detected_blobs: List[Tuple[int, int, int, int]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_blob_area:
                x, y, bw, bh = cv2.boundingRect(c)
                current_detected_blobs.append((x, y, bw, bh))

        # 6. Track persistence across time
        now = timestamp
        active_new_objects = []

        matched_keys = set()
        for bx, by, bw, bh in current_detected_blobs:
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            matched_key = None
            best_dist_sq = 60.0 ** 2
            for key, data in self.blob_tracker.items():
                if key in matched_keys:
                    continue
                tcx, tcy = data["centroid"]
                dist_sq = (bcx - tcx) ** 2 + (bcy - tcy) ** 2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    matched_key = key

            if matched_key:
                data = self.blob_tracker[matched_key]
                data["centroid"] = (bcx, bcy)
                data["bbox"] = (bx, by, bw, bh)
                data["last_seen"] = now
                matched_keys.add(matched_key)

                # Has it persisted for >= persistence_s?
                if (now - data["first_seen"]) >= self.persistence_s:
                    is_new = not data.get("alert_dispatched", False)
                    data["status"] = "CONFIRMED"
                    data["alert_dispatched"] = True
                    active_new_objects.append({
                        "id": matched_key,
                        "bbox": (bx, by, bw, bh),
                        "persistence_s": round(now - data["first_seen"], 1),
                        "first_seen": data["first_seen"],
                        "is_new_confirmation": is_new,
                    })
            else:
                new_key = f"new-obj-{self.next_blob_id}"
                self.next_blob_id += 1
                self.blob_tracker[new_key] = {
                    "centroid": (bcx, bcy),
                    "bbox": (bx, by, bw, bh),
                    "first_seen": now,
                    "last_seen": now,
                    "status": "TRACKED",
                    "alert_dispatched": False,
                }
                matched_keys.add(new_key)

        # Evict old blobs not seen in last 3.0 seconds
        stale_keys = [k for k, v in self.blob_tracker.items() if (now - v["last_seen"]) > 3.0]
        for k in stale_keys:
            del self.blob_tracker[k]

        return active_new_objects
